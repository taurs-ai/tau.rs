//! CUDA custom-operator bridge for fused SiLU-and-multiply.
//!
//! The operator consumes a contiguous packed gate/up projection, launches on
//! Candle's active stream, and returns a Candle-owned tensor with half the last
//! dimension. Contract checks let callers retain a portable Candle fallback.

use candle::backend::BackendStorage;
use candle::cuda_backend::CudaDType;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use candle::{
    CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result as CandleResult, Shape,
    Tensor,
};
use core::ffi::c_void;
use half::bf16;
use tau_cuda_kernels::SiluAndMulDType;

#[derive(Debug, Clone, Copy)]
struct SiluAndMulCuda;

impl SiluAndMulCuda {
    /// Validates packed storage and enqueues the typed CUDA implementation.
    fn launch<T>(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        dtype: SiluAndMulDType,
    ) -> CandleResult<(CudaStorage, Shape)>
    where
        T: CudaDType + DeviceRepr,
    {
        let input_shape = input_layout.shape();
        let packed_width = input_shape.dims().last().copied().ok_or_else(|| {
            candle::Error::Msg(
                "SiLU-and-multiply input must have rank at least one".to_owned(),
            )
        })?;
        if packed_width == 0 || packed_width % 2 != 0 {
            candle::bail!(
                "SiLU-and-multiply input width must be positive and even, got \
                 {packed_width}"
            )
        }
        let columns = packed_width / 2;
        let rows = input_shape.elem_count() / packed_width;
        let rows = i32::try_from(rows).map_err(|_| {
            candle::Error::Msg("SiLU-and-multiply row count exceeds i32".to_owned())
        })?;
        let columns = i32::try_from(columns).map_err(|_| {
            candle::Error::Msg("SiLU-and-multiply output width exceeds i32".to_owned())
        })?;
        let (input_start, input_end) =
            input_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "SiLU-and-multiply input must be contiguous".to_owned(),
                )
            })?;
        let input = input_storage
            .as_cuda_slice::<T>()?
            .slice(input_start..input_end);
        let output_element_count = input_shape.elem_count() / 2;
        let device = input_storage.device.clone();
        // SAFETY: The native kernel writes every output element before the
        // returned storage can be observed.
        let mut output = unsafe { device.alloc::<T>(output_element_count)? };
        let stream = device.cuda_stream();
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);

        // SAFETY: The shape, contiguous layout, dtype, allocation sizes, pointer
        // lifetimes, and active Candle stream are validated above. Both guards
        // remain alive until the asynchronous launch has been enqueued.
        unsafe {
            tau_cuda_kernels::launch_silu_and_mul(
                input_ptr as *const c_void,
                output_ptr as *mut c_void,
                rows,
                columns,
                dtype,
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(|error| {
            candle::Error::Cuda(Box::<dyn std::error::Error + Send + Sync>::from(
                error.to_string(),
            ))
        })?;
        drop(output_guard);
        drop(input_guard);

        let mut output_dims = input_shape.dims().to_vec();
        let last_dimension = output_dims
            .last_mut()
            .expect("rank was checked before creating the CUDA output");
        *last_dimension /= 2;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from(output_dims),
        ))
    }
}

impl CustomOp1 for SiluAndMulCuda {
    /// Returns the operation name used in Candle diagnostics.
    fn name(&self) -> &'static str {
        "tau-silu-and-mul-cuda"
    }

    /// Rejects accidental CPU dispatch because this bridge is CUDA-only.
    fn cpu_fwd(
        &self,
        _input_storage: &CpuStorage,
        _input_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("{} is a CUDA-only operator", self.name())
    }

    /// Selects the native FP32 or BF16 launcher for matching CUDA storage.
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        match input_storage.dtype() {
            DType::F32 => {
                self.launch::<f32>(input_storage, input_layout, SiluAndMulDType::F32)
            }
            DType::BF16 => {
                self.launch::<bf16>(input_storage, input_layout, SiluAndMulDType::BF16)
            }
            dtype => candle::bail!(
                "SiLU-and-multiply CUDA kernel does not support {dtype:?}"
            ),
        }
    }
}

/// Reports whether a tensor satisfies the native fused-operation contract.
pub(in crate::components) fn supports_silu_and_mul(input: &Tensor) -> bool {
    let Some(&packed_width) = input.dims().last() else {
        return false;
    };
    let rows = input
        .elem_count()
        .checked_div(packed_width)
        .filter(|_| packed_width > 0);
    let columns = packed_width / 2;

    input.device().is_cuda()
        && matches!(input.dtype(), DType::F32 | DType::BF16)
        && input.is_contiguous()
        && packed_width > 0
        && packed_width % 2 == 0
        && columns > 0
        && i32::try_from(columns).is_ok()
        && rows.is_some_and(|rows| rows > 0 && i32::try_from(rows).is_ok())
}

/// Applies fused SiLU-and-multiply through Candle's no-backward boundary.
pub(in crate::components) fn silu_and_mul_cuda(input: &Tensor) -> CandleResult<Tensor> {
    input.apply_op1_no_bwd(&SiluAndMulCuda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Computes the packed SwiGLU operation with ordinary Candle primitives.
    fn reference(input: &Tensor) -> CandleResult<Tensor> {
        let last_dimension = input.rank() - 1;
        let columns = input.dim(last_dimension)? / 2;
        let gate = input.narrow(last_dimension, 0, columns)?.silu()?;
        let up = input.narrow(last_dimension, columns, columns)?;
        gate * up
    }

    /// Compares custom and Candle paths for both supported CUDA storage dtypes.
    #[test]
    fn custom_and_reference_silu_and_mul_match() {
        let device = Device::new_cuda(0)
            .expect("SiLU-and-multiply comparison test requires CUDA GPU 0");
        let values = [
            [1.0f32, -2.0, 0.5, 3.0, 2.0, 0.25, -4.0, 1.5],
            [-0.5, 4.0, -3.0, 2.0, 1.25, -2.0, 0.75, 0.5],
        ];

        // Candle rounds the BF16 SiLU result before its separate multiply,
        // whereas the fused kernel rounds once after the product.
        for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::BF16, 5e-2)] {
            let input = Tensor::new(&values, &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            assert!(supports_silu_and_mul(&input));

            let custom = silu_and_mul_cuda(&input)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let reference = reference(&input)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();

            for (custom_row, reference_row) in custom.iter().zip(reference) {
                for (&custom, reference) in custom_row.iter().zip(reference_row) {
                    assert!(
                        (custom - reference).abs() <= tolerance,
                        "custom value {custom} differs from reference {reference} by \
                         more than {tolerance}"
                    );
                }
            }
        }
    }

    /// Rejects odd packed widths before attempting a native launch.
    #[test]
    fn rejects_odd_packed_width() {
        let device = Device::new_cuda(0)
            .expect("SiLU-and-multiply contract test requires CUDA GPU 0");
        let input = Tensor::zeros((2, 7), DType::F32, &device).unwrap();
        assert!(!supports_silu_and_mul(&input));
    }
}
