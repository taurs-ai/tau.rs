//! Explicit FP32 sigmoid/product with BF16 input/output; dense defaults are separate.
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor,
};
use half::bf16;
struct SigmoidMul;
impl CustomOp2 for SigmoidMul {
    /// Identifies failures at the explicit precision boundary.
    fn name(&self) -> &'static str {
        "tau-sigmoid-mul-fp32"
    }
    /// Retains portable reference selection with the component owner.
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("CUDA-only sigmoid")
    }
    /// Uses the same pointwise implementation as prepared attention execution.
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        y: &CudaStorage,
        yl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if x.device.id() != y.device.id() {
            candle::bail!("sigmoid device mismatch")
        }
        let count = xl.shape().elem_count();
        let n = i32::try_from(count).map_err(candle::Error::wrap)?;
        let (start, end) = xl.contiguous_offsets().ok_or_else(|| {
            candle::Error::Msg("noncontiguous sigmoid input".to_owned())
        })?;
        let (ys, ye) = yl.contiguous_offsets().ok_or_else(|| {
            candle::Error::Msg("noncontiguous sigmoid gate".to_owned())
        })?;
        let device = x.device.clone();
        let stream = device.cuda_stream();
        let x = x.as_cuda_slice::<bf16>()?.slice(start..end);
        let y = y.as_cuda_slice::<bf16>()?.slice(ys..ye);
        let (xp, xg) = x.device_ptr(&stream);
        let (yp, yg) = y.device_ptr(&stream);
        // SAFETY: The pointwise kernel initializes every output element.
        let mut output = unsafe { device.alloc::<bf16>(count)? };
        let (out, og) = output.device_ptr_mut(&stream);
        // SAFETY: Wrapper checks shape/dtype/device; contiguous ranges and integer
        // bounds are checked above, and owners survive stream-ordered execution.
        unsafe {
            tau_cuda_kernels::launch_decode_elementwise(
                7,
                xp as *const _,
                yp as *const _,
                out as *mut _,
                std::ptr::null(),
                n,
                n,
                0,
                0,
                stream.cu_stream().cast(),
            )
        }
        .map_err(candle::Error::wrap)?;
        drop((og, yg, xg));
        // The second storage carries the same device checked before slicing.
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            xl.shape().clone(),
        ))
    }
}

/// Enqueues the opted-in FP32 sigmoid/multiply without rounding the sigmoid.
pub(in crate::components) fn sigmoid_mul_fp32(
    input: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    if input.shape() != gate.shape()
        || !input.device().is_cuda()
        || !input.device().same_device(gate.device())
        || input.dtype() != DType::BF16
        || gate.dtype() != DType::BF16
        || !input.is_contiguous()
        || !gate.is_contiguous()
    {
        candle::bail!("unsupported FP32 sigmoid contract")
    }
    input.apply_op2_no_bwd(gate, &SigmoidMul)
}
