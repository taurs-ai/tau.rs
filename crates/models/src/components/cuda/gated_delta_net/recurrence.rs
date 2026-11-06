//! Stateless FP32 recurrence over prepared Q/K/V, alpha and beta.
//!
//! One fresh packed output owns readout and final state as disjoint views. State
//! inputs are read-only; the model/decoder retain sequence and graph ownership.

use super::cuda_driver_error;
use candle::Result as CandleResult;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Shape, Tensor};
use core::ffi::c_void;

pub(in crate::components) fn supports_prefill_recurrence(query: &Tensor) -> bool {
    query.device().is_cuda()
        && query.dtype() == DType::F32
        && query.dims4().is_ok_and(|(b, t, h, k)| {
            b == 1
                && t > 1
                && h > 0
                && k == 128
                && t.checked_mul(h)
                    .and_then(|n| n.checked_mul(386))
                    .is_some_and(|n| i32::try_from(n).is_ok())
        })
}

/// Preparation stays in Candle initially, preserving all existing cast points.
pub(in crate::components) fn prefill_recurrence_cuda(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    if !supports_prefill_recurrence(query)
        || key.dims() != query.dims()
        || value.dims() != query.dims()
    {
        candle::bail!("unsupported GDN prefill recurrence contract")
    }
    let packed = Tensor::cat(
        &[query, key, value, &alpha.unsqueeze(3)?, &beta.unsqueeze(3)?],
        3,
    )?;
    prefill_packed_cuda(&packed, initial)
}

pub(super) fn prefill_packed_cuda(
    packed: &Tensor,
    initial: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    let (_, tokens, heads, _) = packed.dims4()?;
    let result = packed.apply_op2_no_bwd(initial, &PrefillRecurrence)?;
    let outputs = tokens * heads * 128;
    Ok((
        result
            .narrow(0, 0, outputs)?
            .reshape((1, tokens, heads, 128))?,
        result
            .narrow(0, outputs, heads * 128 * 128)?
            .reshape((1, heads, 128, 128))?,
    ))
}

struct PrefillRecurrence;

impl CustomOp2 for PrefillRecurrence {
    fn name(&self) -> &'static str {
        "tau-gdn-prefill-recurrence"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("GDN native prefill recurrence requires CUDA")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        initial: &CudaStorage,
        sl: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let (b, tokens, heads, width) = il.shape().dims4()?;
        let count = tokens
            .checked_add(128)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(128))
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or_else(|| candle::Error::Msg("GDN output size overflow".to_owned()))?;
        if b != 1
            || tokens <= 1
            || heads == 0
            || width != 386
            || sl.dims() != [1, heads, 128, 128]
            || input.device.id() != initial.device.id()
        {
            candle::bail!("invalid GDN recurrence layout/device")
        }
        let view = |layout: &Layout, length: usize| {
            layout
                .contiguous_offsets()
                .filter(|&(_, end)| end <= length)
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "GDN recurrence requires bounded contiguous views".to_owned(),
                    )
                })
        };
        let input = input.as_cuda_slice::<f32>()?;
        let initial_slice = initial.as_cuda_slice::<f32>()?;
        let (start, end) = view(il, input.len())?;
        let input_slice = input.slice(start..end);
        let (start, end) = view(sl, initial_slice.len())?;
        let initial_slice = initial_slice.slice(start..end);
        let device = initial.device.clone();
        // SAFETY: Every readout and state element is initialized by the kernel.
        let mut output = unsafe { device.alloc::<f32>(count)? };
        let stream = device.cuda_stream();
        let (ip, ig) = input_slice.device_ptr(&stream);
        let (sp, sg) = initial_slice.device_ptr(&stream);
        let (op, og) = output.device_ptr_mut(&stream);
        // SAFETY: Shapes, bounds, dtypes and devices were checked above. Output is
        // fresh and disjoint, and all pointer guards live through stream enqueue.
        unsafe {
            tau_cuda_kernels::launch_gdn_prefill_recurrence(
                ip as *const f32,
                sp as *const f32,
                op as *mut f32,
                tokens as i32,
                heads as i32,
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(cuda_driver_error)?;
        drop((ig, sg, og));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from(count),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Storage};

    #[test]
    fn recurrence_rejects_out_of_bounds_views_before_launch() {
        let device = Device::new_cuda(0).unwrap();
        let input = Tensor::zeros((1, 2, 4, 386), DType::F32, &device).unwrap();
        let initial = Tensor::zeros((1, 4, 128, 128), DType::F32, &device).unwrap();
        let (ig, il) = input.storage_and_layout();
        let (sg, sl) = initial.storage_and_layout();
        let (Storage::Cuda(i), Storage::Cuda(s)) = (&*ig, &*sg) else {
            unreachable!()
        };
        let bad = Layout::contiguous_with_offset((1, 2, 4, 386), 1);
        assert!(
            PrefillRecurrence
                .cuda_fwd(i, &bad, s, sl)
                .unwrap_err()
                .to_string()
                .contains("bounded contiguous")
        );
        let bad = Layout::contiguous_with_offset((1, 4, 128, 128), 1);
        assert!(
            PrefillRecurrence
                .cuda_fwd(i, il, s, &bad)
                .unwrap_err()
                .to_string()
                .contains("bounded contiguous")
        );
        let bad =
            Layout::new(Shape::from((1, usize::MAX, 4, 386)), vec![0, 0, 0, 0], 0);
        assert!(PrefillRecurrence.cuda_fwd(i, &bad, s, sl).is_err());
    }
}
