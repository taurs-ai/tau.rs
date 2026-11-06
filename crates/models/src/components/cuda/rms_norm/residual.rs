//! Fused residual/RMSNorm contract and one allocation with two retained views.

use super::centered::supports_centered_rms_norm;
use candle::backend::BackendStorage;
use candle::cuda_backend::CudaDType;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use candle::{
    CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape,
    Tensor,
};
use core::ffi::c_void;
use half::bf16;
use tau_cuda_kernels::{RmsNormDType, RmsNormVectorBytes};

/// Three read-only inputs, one packed allocation containing two disjoint outputs.
/// The wrapper below exposes contiguous views without extra GPU copies.
#[derive(Debug, Clone, Copy)]
struct FusedResidualRmsNormCuda {
    epsilon: f32,
    preferred_vector_bytes: RmsNormVectorBytes,
}

impl FusedResidualRmsNormCuda {
    #[allow(clippy::too_many_arguments)]
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        branch: &CudaStorage,
        branch_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        dtype: RmsNormDType,
    ) -> CandleResult<(CudaStorage, Shape)> {
        if input.device.id() != branch.device.id()
            || input.device.id() != weight.device.id()
        {
            candle::bail!("fused residual RMSNorm requires one CUDA device")
        }
        let shape = input_layout.shape();
        let columns = shape.dims().last().copied().unwrap_or(0);
        if columns == 0
            || branch_layout.shape() != shape
            || weight_layout.shape().dims1()? != columns
        {
            candle::bail!(
                "fused residual RMSNorm input/branch/weight shapes do not match"
            )
        }
        let count = shape.elem_count();
        let rows = i32::try_from(count / columns).map_err(|_| {
            candle::Error::Msg("fused RMSNorm row count exceeds i32".to_owned())
        })?;
        let columns = i32::try_from(columns).map_err(|_| {
            candle::Error::Msg("fused RMSNorm width exceeds i32".to_owned())
        })?;
        let packed_count = count.checked_mul(2).ok_or_else(|| {
            candle::Error::Msg("fused RMSNorm output size overflow".to_owned())
        })?;
        let packed_bytes = packed_count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                candle::Error::Msg("fused RMSNorm output bytes overflow".to_owned())
            })?;
        let offsets = |layout: &Layout| {
            layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "fused RMSNorm requires contiguous storage".to_owned(),
                )
            })
        };
        let (start, end) = offsets(input_layout)?;
        let input_slice = input.as_cuda_slice::<T>()?.slice(start..end);
        let (start, end) = offsets(branch_layout)?;
        let branch_slice = branch.as_cuda_slice::<T>()?.slice(start..end);
        let (start, end) = offsets(weight_layout)?;
        let weight_slice = weight.as_cuda_slice::<T>()?.slice(start..end);
        let device = input.device.clone();
        // SAFETY: The fused kernel initializes both halves before either is read.
        let mut output = unsafe { device.alloc::<T>(packed_count)? };
        let stream = device.cuda_stream();
        let (input_ptr, input_guard) = input_slice.device_ptr(&stream);
        let (branch_ptr, branch_guard) = branch_slice.device_ptr(&stream);
        let (weight_ptr, weight_guard) = weight_slice.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let normalized_ptr = output_ptr
            .checked_add((packed_bytes / 2) as u64)
            .ok_or_else(|| {
                candle::Error::Msg("fused RMSNorm output pointer overflow".to_owned())
            })?;
        // SAFETY: Views, dtypes, devices, sizes and offsets are checked. The two
        // output halves are disjoint and fresh; guards cover the asynchronous enqueue.
        unsafe {
            tau_cuda_kernels::launch_fused_residual_rms_norm(
                input_ptr as *const c_void,
                branch_ptr as *const c_void,
                weight_ptr as *const c_void,
                output_ptr as *mut c_void,
                normalized_ptr as *mut c_void,
                rows,
                columns,
                self.epsilon,
                dtype,
                stream.cu_stream() as *mut c_void,
                self.preferred_vector_bytes,
            )
        }
        .map_err(|error| candle::Error::Cuda(Box::new(error)))?;
        drop((output_guard, weight_guard, branch_guard, input_guard));
        let mut dimensions = vec![2];
        dimensions.extend_from_slice(shape.dims());
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from(dimensions),
        ))
    }
}

impl CustomOp3 for FusedResidualRmsNormCuda {
    fn name(&self) -> &'static str {
        "tau-fused-residual-rms-norm-cuda"
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
        candle::bail!("fused residual RMSNorm is CUDA-only; use the component fallback")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        branch: &CudaStorage,
        bl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        if input.dtype() != branch.dtype() || input.dtype() != weight.dtype() {
            candle::bail!("fused residual RMSNorm requires matching storage dtypes")
        }
        match input.dtype() {
            DType::F32 => {
                self.launch::<f32>(input, il, branch, bl, weight, wl, RmsNormDType::F32)
            }
            DType::BF16 => self.launch::<bf16>(
                input,
                il,
                branch,
                bl,
                weight,
                wl,
                RmsNormDType::BF16,
            ),
            dtype => candle::bail!("fused residual RMSNorm does not support {dtype:?}"),
        }
    }
}

pub(in crate::components) fn supports_fused_residual_rms_norm(
    input: &Tensor,
    branch: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> bool {
    supports_centered_rms_norm(input, weight, epsilon)
        && input.shape() == branch.shape()
        && input.dtype() == branch.dtype()
        && input.device().same_device(branch.device())
        && branch.is_contiguous()
        && input
            .elem_count()
            .checked_mul(2)
            .and_then(|count| count.checked_mul(input.dtype().size_in_bytes()))
            .is_some()
}

/// Owns one packed allocation; each returned view independently retains it.
pub(in crate::components) fn fused_residual_rms_norm_cuda(
    input: &Tensor,
    branch: &Tensor,
    weight: &Tensor,
    epsilon: f64,
    preferred_vector_bytes: RmsNormVectorBytes,
) -> CandleResult<(Tensor, Tensor)> {
    if !supports_fused_residual_rms_norm(input, branch, weight, epsilon) {
        candle::bail!("unsupported fused residual RMSNorm contract")
    }
    let packed = input.apply_op3_no_bwd(
        branch,
        weight,
        &FusedResidualRmsNormCuda {
            epsilon: epsilon as f32,
            preferred_vector_bytes,
        },
    )?;
    Ok((
        packed.narrow(0, 0, 1)?.squeeze(0)?,
        packed.narrow(0, 1, 1)?.squeeze(0)?,
    ))
}
