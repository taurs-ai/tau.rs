//! Centered RMSNorm contract, typed storage checks and asynchronous CUDA launch.

use candle::backend::BackendStorage;
use candle::cuda_backend::CudaDType;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape,
    Tensor,
};
use core::ffi::c_void;
use half::bf16;
use tau_cuda_kernels::{RmsNormDType, RmsNormVectorBytes};

#[derive(Debug, Clone, Copy)]
struct CenteredRmsNormCuda {
    epsilon: f32,
    preferred_vector_bytes: RmsNormVectorBytes,
}

impl CenteredRmsNormCuda {
    /// Validates tensor storage and enqueues the typed RMSNorm CUDA kernel.
    fn launch<T>(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        weight_storage: &CudaStorage,
        weight_layout: &Layout,
        dtype: RmsNormDType,
    ) -> CandleResult<(CudaStorage, Shape)>
    where
        T: CudaDType + DeviceRepr,
    {
        if input_storage.device.id() != weight_storage.device.id() {
            candle::bail!("centered RMSNorm input and weight must use one CUDA device")
        }
        let input_shape = input_layout.shape();
        let columns = input_shape.dims().last().copied().ok_or_else(|| {
            candle::Error::Msg(
                "centered RMSNorm input must have rank at least one".to_owned(),
            )
        })?;
        if columns == 0 || weight_layout.shape().dims1()? != columns {
            candle::bail!(
                "centered RMSNorm weight must have shape [{columns}], got {:?}",
                weight_layout.shape()
            )
        }
        let element_count = input_shape.elem_count();
        let rows = element_count / columns;
        let rows = i32::try_from(rows).map_err(|_| {
            candle::Error::Msg("centered RMSNorm row count exceeds i32".to_owned())
        })?;
        let columns = i32::try_from(columns).map_err(|_| {
            candle::Error::Msg("centered RMSNorm width exceeds i32".to_owned())
        })?;
        let (input_start, input_end) =
            input_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "centered RMSNorm input must be contiguous".to_owned(),
                )
            })?;
        let (weight_start, weight_end) =
            weight_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "centered RMSNorm weight must be contiguous".to_owned(),
                )
            })?;

        let input = input_storage
            .as_cuda_slice::<T>()?
            .slice(input_start..input_end);
        let weight = weight_storage
            .as_cuda_slice::<T>()?
            .slice(weight_start..weight_end);
        let device = input_storage.device.clone();
        // SAFETY: The native kernel initializes every output element before the
        // returned storage can be observed.
        let mut output = unsafe { device.alloc::<T>(element_count)? };
        let stream = device.cuda_stream();
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (weight_ptr, weight_guard) = weight.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);

        // SAFETY: Shapes, dtypes, devices, contiguous offsets, pointer lifetimes,
        // and the active Candle stream are validated above. The guards remain alive
        // until after the asynchronous launch has been enqueued.
        unsafe {
            tau_cuda_kernels::launch_centered_rms_norm_configured(
                input_ptr as *const c_void,
                weight_ptr as *const c_void,
                output_ptr as *mut c_void,
                rows,
                columns,
                self.epsilon,
                dtype,
                stream.cu_stream() as *mut c_void,
                self.preferred_vector_bytes,
            )
        }
        .map_err(|error| {
            candle::Error::Cuda(Box::<dyn std::error::Error + Send + Sync>::from(
                error.to_string(),
            ))
        })?;
        drop(output_guard);
        drop(weight_guard);
        drop(input_guard);

        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            input_shape.clone(),
        ))
    }
}

impl CustomOp2 for CenteredRmsNormCuda {
    /// Returns the operator name used in Candle diagnostics.
    fn name(&self) -> &'static str {
        "tau-centered-rms-norm-cuda"
    }

    /// Rejects accidental CPU dispatch because this bridge is CUDA-only.
    fn cpu_fwd(
        &self,
        _input_storage: &CpuStorage,
        _input_layout: &Layout,
        _weight_storage: &CpuStorage,
        _weight_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("{} is a CUDA-only bring-up operator", self.name())
    }

    /// Selects the native FP32 or BF16 launcher for matching CUDA storage.
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        weight_storage: &CudaStorage,
        weight_layout: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        if input_storage.dtype() != weight_storage.dtype() {
            candle::bail!(
                "centered RMSNorm input and weight dtypes must match, got {:?} and \
                 {:?}",
                input_storage.dtype(),
                weight_storage.dtype()
            )
        }
        match input_storage.dtype() {
            DType::F32 => self.launch::<f32>(
                input_storage,
                input_layout,
                weight_storage,
                weight_layout,
                RmsNormDType::F32,
            ),
            DType::BF16 => self.launch::<bf16>(
                input_storage,
                input_layout,
                weight_storage,
                weight_layout,
                RmsNormDType::BF16,
            ),
            dtype => {
                candle::bail!("centered RMSNorm CUDA kernel does not support {dtype:?}")
            }
        }
    }
}

/// Reports whether tensors satisfy the native centered-RMSNorm launch contract.
pub(in crate::components) fn supports_centered_rms_norm(
    input: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> bool {
    let Some(&columns) = input.dims().last() else {
        return false;
    };
    let element_count = input.elem_count();
    let rows = element_count.checked_div(columns);

    input.device().is_cuda()
        && input.device().same_device(weight.device())
        && input.dtype() == weight.dtype()
        && matches!(input.dtype(), DType::F32 | DType::BF16)
        && input.is_contiguous()
        && weight.is_contiguous()
        && weight.dims() == [columns]
        && columns > 0
        && i32::try_from(columns).is_ok()
        && rows.is_some_and(|rows| rows > 0 && i32::try_from(rows).is_ok())
        && epsilon.is_finite()
        && epsilon > 0.0
        && epsilon <= f32::MAX as f64
}

/// Applies centered RMSNorm through Candle's no-backward custom-operator boundary.
pub(in crate::components) fn centered_rms_norm_cuda(
    input: &Tensor,
    weight: &Tensor,
    epsilon: f64,
    preferred_vector_bytes: RmsNormVectorBytes,
) -> CandleResult<Tensor> {
    if !epsilon.is_finite() || epsilon <= 0.0 || epsilon > f32::MAX as f64 {
        candle::bail!("centered RMSNorm epsilon must be a positive finite f32")
    }
    input.apply_op2_no_bwd(
        weight,
        &CenteredRmsNormCuda {
            epsilon: epsilon as f32,
            preferred_vector_bytes,
        },
    )
}
