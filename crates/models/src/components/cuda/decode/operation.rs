//! Typed execution of prepared operations into preallocated output slots.
use super::{PreparedBlas, error};
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::{
    cublas,
    driver::{DevicePtr, DevicePtrMut},
};
use candle::{
    CpuStorage, CudaStorage, DType, InplaceOp1, Layout, Result, Storage, Tensor,
};
use core::ffi::c_void;
use half::bf16;
use std::rc::Rc;
pub(super) enum Kind {
    MoeShared {
        rows: i32,
        width: i32,
    },
    Linear {
        m: i32,
        k: i32,
    },
    Norm {
        rows: i32,
        width: i32,
        epsilon: f32,
    },
    Silu {
        width: i32,
    },
    Elementwise {
        kind: i32,
        n: i32,
        width: i32,
        capacity: i32,
        transposed: i32,
    },
    #[cfg(test)]
    Rope {
        n: i32,
        dim: i32,
        rotary: i32,
    },
    AttentionPrepare {
        heads: i32,
        kv_heads: i32,
        dim: i32,
        capacity: i32,
        rotary: i32,
        transposed: i32,
        fp32_rope: i32,
        query_epsilon: f32,
        key_epsilon: f32,
    },
    Attention {
        heads: i32,
        kv_heads: i32,
        dim: i32,
        capacity: i32,
        transposed: i32,
        block_size: i32,
        bf16_products: i32,
    },
}

pub(super) struct DecodeOp {
    pub(super) inputs: Vec<Tensor>,
    pub(super) kind: Kind,
    pub(super) blas: Option<Rc<PreparedBlas>>,
}

impl InplaceOp1 for DecodeOp {
    /// Identifies prepared launches in the tensor runtime's error diagnostics.
    fn name(&self) -> &'static str {
        "tau-prepared-decode"
    }
    /// Rejects CPU storage; the recorded operations require native CUDA launches.
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        candle::bail!("prepared decode is CUDA-only")
    }

    /// Resolves retained tensor views to device pointers and queues the selected
    /// native kernel or cuBLAS projection into the existing output slot. Pointer
    /// offsets preserve views; the private stream orders output and scratch writes.
    fn cuda_fwd(&self, output: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let device = output.device.clone();
        let stream = device.cuda_stream();
        let inputs = self
            .inputs
            .iter()
            .map(Tensor::storage_and_layout)
            .collect::<Vec<_>>();
        // All allocations predate capture on this private, event-tracking-disabled
        // device. Owners retain every tensor and serialize updates on one stream;
        // no other stream can read or write these output slots concurrently.
        // Attention also owns its fourth retained tensor as mutable FP32 scratch;
        // both tile and combine kernels use it only on this serialized stream.
        // Attention preparation similarly writes its retained K/V cache inputs.
        // These buffers are exclusively owned, disjoint, and used on this same
        // private stream; no other device operation can observe a partial append.
        let pointers = inputs
            .iter()
            .map(|(storage, layout)| {
                let Storage::Cuda(storage) = &**storage else {
                    candle::bail!("expected CUDA storage")
                };
                if storage.device.id() != device.id() {
                    candle::bail!("prepared device mismatch")
                }
                let offset = layout.start_offset();
                Ok(match storage.dtype() {
                    DType::BF16 => {
                        storage.as_cuda_slice::<bf16>()?.device_ptr(&stream).0
                            + (offset * 2) as u64
                    }
                    DType::F32 => {
                        storage.as_cuda_slice::<f32>()?.device_ptr(&stream).0
                            + (offset * 4) as u64
                    }
                    DType::U32 => {
                        storage.as_cuda_slice::<u32>()?.device_ptr(&stream).0
                            + (offset * 4) as u64
                    }
                    _ => candle::bail!("prepared dtype mismatch"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (out, guard) = output.as_cuda_slice_mut::<bf16>()?.device_ptr_mut(&stream);
        let out = (out + (layout.start_offset() * 2) as u64) as *mut c_void;
        let p = |i: usize| pointers[i] as *const c_void;
        let meta = || *pointers.last().unwrap() as *const u32;
        let cuda_stream = stream.cu_stream() as *mut c_void;
        // SAFETY: Builders validate device/dtype/shape contracts and retain every
        // input/output allocation. Metadata is validated before each real replay.
        unsafe {
            match self.kind {
                Kind::MoeShared { rows, width } => {
                    tau_cuda_kernels::launch_moe_shared_combine(
                        p(0),
                        p(1),
                        p(2),
                        p(3),
                        out,
                        rows,
                        width,
                        cuda_stream,
                    )
                    .map_err(error)?;
                }
                Kind::Linear { m, k } => {
                    use cublas::sys::{
                        cublasComputeType_t as Compute, cublasGemmAlgo_t as Algo,
                        cublasOperation_t as Op, cudaDataType_t as Type,
                    };
                    let (alpha, beta) = (1f32, 0f32);
                    let blas = self.blas.as_ref().ok_or_else(|| {
                        error("prepared projection has no owned cuBLAS workspace")
                    })?;
                    let compute = if candle::cuda_backend::gemm_reduced_precision_bf16()
                    {
                        Compute::CUBLAS_COMPUTE_32F_FAST_16BF
                    } else {
                        Compute::CUBLAS_COMPUTE_32F
                    };
                    // Match the eager linear backend's vendor entrypoint,
                    // geometry and strides, changing only output ownership.
                    cublas::result::gemm_strided_batched_ex(
                        *blas.handle.handle(),
                        Op::CUBLAS_OP_T,
                        Op::CUBLAS_OP_N,
                        m,
                        1,
                        k,
                        (&alpha as *const f32).cast(),
                        p(1),
                        Type::CUDA_R_16BF,
                        k,
                        i64::from(m) * i64::from(k),
                        p(0),
                        Type::CUDA_R_16BF,
                        k,
                        i64::from(k),
                        (&beta as *const f32).cast(),
                        out,
                        Type::CUDA_R_16BF,
                        m,
                        i64::from(m),
                        1,
                        compute,
                        Algo::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                    .map_err(error)?;
                }
                Kind::Norm {
                    rows,
                    width,
                    epsilon,
                } => {
                    tau_cuda_kernels::launch_centered_rms_norm_configured(
                        p(0),
                        p(1),
                        out,
                        rows,
                        width,
                        epsilon,
                        tau_cuda_kernels::RmsNormDType::BF16,
                        cuda_stream,
                        tau_cuda_kernels::RmsNormVectorBytes::for_row_stride(
                            width as usize * 2,
                        ),
                    )
                    .map_err(error)?;
                }
                Kind::Silu { width } => {
                    tau_cuda_kernels::launch_silu_and_mul(
                        p(0),
                        out,
                        1,
                        width,
                        tau_cuda_kernels::SiluAndMulDType::BF16,
                        cuda_stream,
                    )
                    .map_err(error)?;
                }
                Kind::Elementwise {
                    kind,
                    n,
                    width,
                    capacity,
                    transposed,
                } => {
                    tau_cuda_kernels::launch_decode_elementwise(
                        kind,
                        p(0),
                        if pointers.len() > 1 {
                            p(1)
                        } else {
                            std::ptr::null()
                        },
                        out,
                        if matches!(kind, 0 | 4) {
                            meta()
                        } else {
                            std::ptr::null()
                        },
                        n,
                        width,
                        capacity,
                        transposed,
                        cuda_stream,
                    )
                    .map_err(error)?;
                }
                #[cfg(test)]
                Kind::Rope { n, dim, rotary } => {
                    tau_cuda_kernels::launch_decode_rope(
                        p(0),
                        p(1),
                        p(2),
                        out,
                        meta(),
                        n,
                        dim,
                        rotary,
                        cuda_stream,
                    )
                    .map_err(error)?;
                }
                Kind::AttentionPrepare {
                    heads,
                    kv_heads,
                    dim,
                    capacity,
                    rotary,
                    transposed,
                    fp32_rope,
                    query_epsilon,
                    key_epsilon,
                } => {
                    tau_cuda_kernels::launch_decode_attention_prepare(
                        p(0),
                        p(1),
                        p(2),
                        p(3),
                        p(4),
                        p(5),
                        p(6),
                        out,
                        p(7) as *mut c_void,
                        p(8) as *mut c_void,
                        meta(),
                        heads,
                        kv_heads,
                        dim,
                        capacity,
                        rotary,
                        transposed,
                        fp32_rope,
                        query_epsilon,
                        key_epsilon,
                        cuda_stream,
                    )
                    .map_err(error)?;
                }
                Kind::Attention {
                    heads,
                    kv_heads,
                    dim,
                    capacity,
                    transposed,
                    block_size,
                    bf16_products,
                } => {
                    tau_cuda_kernels::launch_decode_attention(
                        p(0),
                        p(1),
                        p(2),
                        out,
                        meta(),
                        p(3) as *mut f32,
                        heads,
                        kv_heads,
                        dim,
                        capacity,
                        transposed,
                        block_size,
                        bf16_products,
                        cuda_stream,
                    )
                    .map_err(error)?;
                }
            }
        }
        drop(guard);
        Ok(())
    }
}
