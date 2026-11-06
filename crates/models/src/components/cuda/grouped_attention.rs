//! One-row BF16 QK matmul over a transposed, capacity-backed key cache.
//! Candle's standard matmul requires a tightly packed matrix leading dimension;
//! this bridge supplies the retained capacity as cuBLAS's leading dimension.
use candle::cuda_backend::cudarc::{
    cublas,
    driver::{DevicePtr, DevicePtrMut},
};
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor,
};
use half::bf16;

struct CachedQk;

impl CustomOp2 for CachedQk {
    fn name(&self) -> &'static str {
        "tau-cached-qk"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("cached QK is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        qs: &CudaStorage,
        ql: &Layout,
        ks: &CudaStorage,
        kl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, heads, rows, dim) = ql.shape().dims4()?;
        let (kb, kh, kd, length) = kl.shape().dims4()?;
        let stride = kl.stride()[2];
        if batch != 1
            || rows != 1
            || heads == 0
            || dim == 0
            || length == 0
            || (kb, kh, kd) != (1, heads, dim)
            || ql.stride()[3] != 1
            || ql.stride()[1] < dim
            || kl.stride()[3] != 1
            || stride < length
            || dim.checked_mul(stride).is_none_or(|n| kl.stride()[1] < n)
        {
            candle::bail!("unsupported cached-QK shape or strides")
        }
        let q = qs.as_cuda_slice::<bf16>()?;
        let k = ks.as_cuda_slice::<bf16>()?;
        let bounds = |layout: &Layout, width: usize, last_row: usize, size: usize| {
            layout.stride()[1]
                .checked_mul(heads - 1)
                .and_then(|n| n.checked_add(last_row))
                .and_then(|n| n.checked_add(width))
                .and_then(|n| n.checked_add(layout.start_offset()))
                .is_some_and(|end| end <= size)
        };
        let last_row = stride
            .checked_mul(dim - 1)
            .ok_or_else(|| candle::Error::Msg("cached-QK size overflow".to_owned()))?;
        if !bounds(ql, dim, 0, q.len()) || !bounds(kl, length, last_row, k.len()) {
            candle::bail!("cached-QK view exceeds storage")
        }
        let int = |v| {
            i32::try_from(v).map_err(|_| {
                candle::Error::Msg("cached-QK dimension exceeds i32".to_owned())
            })
        };
        let long = |v| {
            i64::try_from(v).map_err(|_| {
                candle::Error::Msg("cached-QK stride exceeds i64".to_owned())
            })
        };
        let (m, d, h, lda) = (int(length)?, int(dim)?, int(heads)?, int(stride)?);
        let (stride_a, stride_b) = (long(kl.stride()[1])?, long(ql.stride()[1])?);
        let count = heads.checked_mul(length).ok_or_else(|| {
            candle::Error::Msg("cached-QK output size overflow".to_owned())
        })?;
        let q = q.slice(ql.start_offset()..);
        let k = k.slice(kl.start_offset()..);
        let device = qs.device.clone();
        // SAFETY: beta=0 initializes every element of this [heads, 1, length] output.
        let mut output = unsafe { device.alloc::<bf16>(count)? };
        let stream = device.cuda_stream();
        let (qp, q_guard) = q.device_ptr(&stream);
        let (kp, k_guard) = k.device_ptr(&stream);
        let (op, o_guard) = output.device_ptr_mut(&stream);
        let (alpha, beta) = (1f32, 0f32);
        use cublas::sys::{
            cublasComputeType_t as Compute, cublasGemmAlgo_t as Algo,
            cublasOperation_t as Op, cudaDataType_t as Type,
        };
        let compute = if candle::cuda_backend::gemm_reduced_precision_bf16() {
            Compute::CUBLAS_COMPUTE_32F_FAST_16BF
        } else {
            Compute::CUBLAS_COMPUTE_32F
        };
        // SAFETY: Shapes, offsets, strides and allocation bounds are checked above.
        // The wrapper checks devices/dtypes. Candle owns the active-stream handle;
        // pointer guards retain inputs/output until this asynchronous call is queued.
        unsafe {
            cublas::result::gemm_strided_batched_ex(
                *device.cublas_handle().handle(),
                Op::CUBLAS_OP_N,
                Op::CUBLAS_OP_N,
                m,
                1,
                d,
                (&alpha as *const f32).cast(),
                kp as *const _,
                Type::CUDA_R_16BF,
                lda,
                stride_a,
                qp as *const _,
                Type::CUDA_R_16BF,
                d,
                stride_b,
                (&beta as *const f32).cast(),
                op as *mut _,
                Type::CUDA_R_16BF,
                m,
                i64::from(m),
                h,
                compute,
                Algo::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(|e| candle::Error::Msg(format!("cached-QK cuBLAS: {e}")))?;
        drop((q_guard, k_guard, o_guard));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from((1, heads, 1, length)),
        ))
    }
}

pub(in crate::components) fn cached_qk(
    query: &Tensor,
    key_transposed: &Tensor,
) -> Result<Tensor> {
    if !query.device().same_device(key_transposed.device())
        || query.dtype() != DType::BF16
        || key_transposed.dtype() != DType::BF16
    {
        candle::bail!("cached-QK requires BF16 inputs on the same CUDA device")
    }
    query.apply_op2_no_bwd(key_transposed, &CachedQk)
}
