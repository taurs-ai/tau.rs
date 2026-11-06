//! Stateless BF16 prefill convolution: tensor contracts and the CUDA launch.
//! GDN owns activation and history updates; this operator reads all inputs.

use super::cuda_driver_error;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape,
    Tensor,
};
use core::ffi::c_void;
use half::bf16;

/// Calculates the minimum backing-storage length needed by a tensor view.
///
/// The last accessed index is `offset + sum((dimension - 1) * stride)`. Adding
/// one gives the exclusive end, which the CUDA operator compares with the actual
/// allocation length. This accounts for gaps in strided views, unlike counting
/// logical tensor elements alone. Empty dimensions or overflowing arithmetic
/// return `None`, as do addresses beyond the kernel's signed 64-bit index range.
fn prefill_storage_end(layout: &Layout) -> Option<usize> {
    layout
        .dims()
        .iter()
        .zip(layout.stride())
        .try_fold(
            layout.start_offset().checked_add(1)?,
            |end, (&dim, &stride)| {
                end.checked_add(dim.checked_sub(1)?.checked_mul(stride)?)
            },
        )
        .filter(|&end| i64::try_from(end).is_ok())
}

/// Checks whether the tensor shapes and memory layouts match the native prefill
/// convolution kernel. A layout describes dimensions, element strides (the gap
/// between neighboring values along each dimension), and a starting offset.
///
/// This checks metadata only: it does not read tensor values or launch GPU work.
/// Dtypes and devices are checked by the caller; the CUDA operator separately
/// checks that the calculated accesses fit inside the actual backing allocations.
fn prefill_layouts_supported(
    input: &Layout,
    history: &Layout,
    weight: &Layout,
) -> bool {
    // The projected QKV input must have shape [batch, channels, tokens].
    let Ok((batch, channels, tokens)) = input.shape().dims3() else {
        return false;
    };
    // The kernel handles one sequence with nonempty channels and multiple tokens.
    // Single-token calls use the existing decode or reference path.
    batch == 1
        && channels > 0
        && tokens > 1
        // Each channel has its own four-tap filter and four raw history values.
        // Matching channel counts prevent reading another channel's state/weights.
        && history.dims() == [1, channels, 4]
        && weight.dims() == [channels, 1, 4]
        // Weights are indexed as channel * 4 + tap, so they must be packed.
        // Input and history may be strided views: the kernel receives their strides.
        && weight.is_contiguous()
        // One output is produced per channel/token pair. Its total count must
        // fit the launcher's signed 32-bit limit, without overflowing the product.
        && channels
            .checked_mul(tokens)
            .is_some_and(|n| i32::try_from(n).is_ok())
        && [input, history, weight].into_iter().all(|layout| {
            // Include the view's starting offset and all strides when calculating
            // its last reachable element. Reject arithmetic overflow and values
            // that cannot be represented by the kernel's signed 64-bit indexing.
            prefill_storage_end(layout).is_some()
                && layout.stride().iter().all(|&s| i64::try_from(s).is_ok())
        })
}

/// Decides whether GDN may select the native prefill convolution for these tensors.
///
/// All three tensors must be BF16 on the same CUDA device, and their layouts must
/// satisfy the kernel's shape/indexing rules. Input and history can be strided
/// views with nonzero offsets. A `false` result lets GDN choose its Candle
/// reference implementation before any native allocation or launch occurs.
pub(in crate::components) fn supports_prefill_convolution(
    input: &Tensor,
    history: &Tensor,
    weight: &Tensor,
) -> bool {
    input.device().is_cuda()
        && [history, weight]
            .into_iter()
            .all(|t| input.device().same_device(t.device()))
        && [input, history, weight]
            .into_iter()
            .all(|t| t.dtype() == DType::BF16)
        && prefill_layouts_supported(input.layout(), history.layout(), weight.layout())
}

/// Runs native prefill convolution through Candle's three-input operator API.
///
/// The result contains BF16 convolution values before activation. The inputs
/// and history are read-only; GDN applies activation and updates history outside
/// this helper. `apply_op3_no_bwd` invokes the CUDA adapter without building a
/// backward operation, since this path is used for inference.
pub(in crate::components) fn prefill_convolution_cuda(
    input: &Tensor,
    history: &Tensor,
    weight: &Tensor,
) -> CandleResult<Tensor> {
    if !supports_prefill_convolution(input, history, weight) {
        candle::bail!("unsupported BF16 GDN prefill convolution contract")
    }
    input.apply_op3_no_bwd(history, weight, &PrefillConvolution)
}

/// Stateless adapter between Candle tensors and the native CUDA convolution.
///
/// `prefill_convolution_cuda` passes a value of this type to `apply_op3_no_bwd`,
/// with input, history and weights as the three tensor arguments. Candle then
/// calls this adapter's `cuda_fwd` implementation, which checks their storage,
/// allocates the output and launches the kernel on Candle's active stream.
///
/// This unit struct has no fields because each call supplies all three inputs.
/// It neither stores weights nor owns mutable history; the surrounding GDN code
/// updates history after the operation. CPU dispatch is rejected by `cpu_fwd`.
struct PrefillConvolution;

impl CustomOp3 for PrefillConvolution {
    /// Supplies the label Candle uses to identify this operation in diagnostics.
    fn name(&self) -> &'static str {
        "tau-bf16-gdn-prefill-convolution"
    }

    /// Rejects accidental CPU dispatch. GDN selects its portable reference path
    /// before calling this adapter when the tensors do not support native CUDA.
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("GDN prefill convolution is CUDA-only")
    }

    /// Converts Candle's storage/layout arguments into a native convolution call.
    ///
    /// `xs`, `hs` and `ws` hold input, history and weights; the matching `*l`
    /// arguments describe how their logical views sit in those allocations.
    /// After validating access bounds, allocate a separate BF16 output, enqueue
    /// the kernel and return its storage with shape [1, channels, tokens].
    /// Activation and history updates remain the surrounding GDN's responsibility.
    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        hs: &CudaStorage,
        hl: &Layout,
        ws: &CudaStorage,
        wl: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        // Recheck the kernel contract at the raw-storage boundary, before taking
        // pointers. Every argument must also belong to the same CUDA device.
        if !prefill_layouts_supported(xl, hl, wl)
            || xs.device.id() != hs.device.id()
            || xs.device.id() != ws.device.id()
        {
            candle::bail!(
                "GDN prefill convolution layout/device changed after dispatch"
            )
        }
        // Typed views verify that the backing storage really contains BF16.
        // These borrow GPU memory; they do not copy values to the CPU.
        let x = xs.as_cuda_slice::<bf16>()?;
        let h = hs.as_cuda_slice::<bf16>()?;
        let w = ws.as_cuda_slice::<bf16>()?;
        // A valid shape alone does not prove a view fits its allocation. Include
        // offsets and strides, then compare the exclusive end with storage size.
        for (layout, len) in [(xl, x.len()), (hl, h.len()), (wl, w.len())] {
            if prefill_storage_end(layout).is_none_or(|end| end > len) {
                candle::bail!("GDN prefill convolution view exceeds storage")
            }
        }
        let (_, channels, tokens) = xl.shape().dims3()?;
        // Move each pointer to its view's first element without copying data.
        // The kernel receives input/history strides to find subsequent elements.
        let x = x.slice(xl.start_offset()..);
        let h = h.slice(hl.start_offset()..);
        let w = w.slice(wl.start_offset()..);
        let device = xs.device.clone();
        // The output is contiguous and separate from all inputs. It can start
        // uninitialized because the kernel writes every channel/token position.
        // SAFETY: The kernel writes every element of the contiguous output.
        let mut output = unsafe { device.alloc::<bf16>(channels * tokens)? };
        let stream = device.cuda_stream();
        // Pointer guards track these reads/writes on Candle's stream. Keep them
        // alive until the launch has been enqueued so access ordering is recorded.
        let (xp, x_guard) = x.device_ptr(&stream);
        let (hp, h_guard) = h.device_ptr(&stream);
        let (wp, w_guard) = w.device_ptr(&stream);
        let (op, o_guard) = output.device_ptr_mut(&stream);
        // The earlier layout checks make the integer casts below safe. Queue one
        // convolution over all channels/tokens; do not synchronize the CPU here.
        // A launch error is returned to the caller, without retrying the reference.
        // SAFETY: Validated BF16 shapes/devices, offsets, strides, integer ranges
        // and backing bounds above. Output is disjoint; all pointer guards live
        // through enqueue on Candle's active stream. No host synchronization.
        unsafe {
            tau_cuda_kernels::launch_bf16_gdn_prefill_convolution(
                xp as *const c_void,
                hp as *const c_void,
                wp as *const c_void,
                op as *mut c_void,
                channels as i32,
                tokens as i32,
                xl.stride()[1] as i64,
                xl.stride()[2] as i64,
                hl.stride()[1] as i64,
                hl.stride()[2] as i64,
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(cuda_driver_error)?;
        // Release pointer access guards after enqueue, then transfer the output
        // allocation into Candle storage. Wrapping it performs no additional copy;
        // later work on the same stream consumes it after convolution completes.
        drop((x_guard, h_guard, w_guard, o_guard));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from((1, channels, tokens)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Storage};

    /// Rejects overflowing indices and views beyond their backing allocations
    /// before the convolution kernel can access device memory.
    #[test]
    fn prefill_convolution_rejects_overflow_and_out_of_bounds_views() {
        let history = Layout::contiguous((1, 7, 4));
        let weight = Layout::contiguous((7, 1, 4));
        let overflow = Layout::new(Shape::from((1, 7, 8)), vec![0, usize::MAX, 1], 0);
        assert!(!prefill_layouts_supported(&overflow, &history, &weight));
        let overflow = Layout::new(Shape::from((1, 7, 8)), vec![0, 8, 1], usize::MAX);
        assert!(!prefill_layouts_supported(&overflow, &history, &weight));
        let device = Device::new_cuda(0).unwrap();
        let tensor = Tensor::zeros((1, 7, 4), DType::BF16, &device).unwrap();
        let (storage, _) = tensor.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            unreachable!()
        };
        for bad_input in [
            Layout::contiguous((1, 7, 8)),
            Layout::contiguous_with_offset((1, 7, 4), 1),
        ] {
            let error = PrefillConvolution
                .cuda_fwd(storage, &bad_input, storage, &history, storage, &weight)
                .unwrap_err();
            assert!(error.to_string().contains("exceeds storage"));
        }
    }
}
