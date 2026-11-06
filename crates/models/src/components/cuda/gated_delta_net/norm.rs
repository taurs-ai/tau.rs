//! GDN prefill readout rounding and gated RMSNorm, with FP32 master weights.

use super::cuda_driver_error;
use candle::Result as CandleResult;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Shape, Tensor};
use core::ffi::c_void;
use half::bf16;

pub(in crate::components) fn supports_prefill_norm(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> bool {
    input.device().is_cuda()
        && input.device().same_device(gate.device())
        && input.device().same_device(weight.device())
        && input.dtype() == DType::F32
        && gate.dtype() == DType::BF16
        && weight.dtype() == DType::F32
        && input.dims() == gate.dims()
        && input.dims().last() == Some(&128)
        && weight.dims() == [128]
        && [input, gate, weight].iter().all(|t| t.is_contiguous())
        && input.elem_count() > 0
        && i32::try_from(input.elem_count()).is_ok()
        && epsilon.is_finite()
        && epsilon > 0.
        && epsilon <= f32::MAX as f64
}

pub(in crate::components) fn prefill_norm_cuda(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> CandleResult<Tensor> {
    if !supports_prefill_norm(input, gate, weight, epsilon) {
        candle::bail!("unsupported GDN prefill gated norm")
    }
    input.apply_op3_no_bwd(
        gate,
        weight,
        &Norm {
            epsilon: epsilon as f32,
        },
    )
}

struct Norm {
    epsilon: f32,
}

impl CustomOp3 for Norm {
    fn name(&self) -> &'static str {
        "tau-gdn-prefill-gated-norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("GDN native norm requires CUDA")
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        gate: &CudaStorage,
        gl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let count = il.shape().elem_count();
        if il.dims() != gl.dims()
            || il.dims().last() != Some(&128)
            || wl.dims() != [128]
            || count == 0
            || i32::try_from(count).is_err()
            || input.device.id() != gate.device.id()
            || input.device.id() != weight.device.id()
        {
            candle::bail!("invalid GDN prefill norm layout/device")
        }
        let view = |layout: &Layout, len| {
            layout
                .contiguous_offsets()
                .filter(|&(_, end)| end <= len)
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "GDN norm requires bounded contiguous views".to_owned(),
                    )
                })
        };
        let x = input.as_cuda_slice::<f32>()?;
        let g = gate.as_cuda_slice::<bf16>()?;
        let w = weight.as_cuda_slice::<f32>()?;
        let (start, end) = view(il, x.len())?;
        let x = x.slice(start..end);
        let (start, end) = view(gl, g.len())?;
        let g = g.slice(start..end);
        let (start, end) = view(wl, w.len())?;
        let w = w.slice(start..end);
        let device = input.device.clone();
        // SAFETY: The kernel initializes every BF16 output element.
        let mut output = unsafe { device.alloc::<bf16>(count)? };
        let stream = device.cuda_stream();
        let (xp, xg) = x.device_ptr(&stream);
        let (gp, gg) = g.device_ptr(&stream);
        let (wp, wg) = w.device_ptr(&stream);
        let (op, og) = output.device_ptr_mut(&stream);
        // SAFETY: Checked shapes, typed views, bounds and devices; output is
        // disjoint and pointer guards remain live until stream enqueue finishes.
        unsafe {
            tau_cuda_kernels::launch_gdn_prefill_norm(
                xp as *const f32,
                gp as *const c_void,
                wp as *const f32,
                op as *mut c_void,
                (count / 128) as i32,
                self.epsilon,
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(cuda_driver_error)?;
        drop((xg, gg, wg, og));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            il.shape().clone(),
        ))
    }
}
