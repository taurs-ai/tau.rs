//! Native CUDA kernels for tau.rs inference.
//!
//! This crate owns native inference implementations with explicit pointer,
//! allocation-lifetime, and stream contracts. Tensor bridges adapt the caller's
//! storage to these interfaces. Use maintained vendor primitives for general
//! operations when they meet the measured requirements.
//!
//! Native kernels are always built and linked. Runtime device and input checks
//! select them only for supported CUDA tensors.

mod cuda_ffi;
mod moe;
pub use moe::{
    MOE_PREFILL_EXPERTS, MOE_PREFILL_META_WORDS, MOE_PREFILL_TILE, MoeDecodeConfig,
    MoeDecodeOptions, MoeDecodeProfile, MoeDecodeResolution, MoeDecodeSelection,
    MoePrefillConfig, MoePrefillTile, launch_fused_bf16_moe_decode,
    launch_linear_bf16_fp32, launch_moe_decode_projection, launch_moe_prefill,
    launch_moe_prefill_dispatch, launch_moe_routing_fp32, launch_moe_shared_combine,
    moe_prefill_capacity, moe_prefill_capacity_for_tile,
};

mod greedy;
pub use greedy::{GreedyDType, greedy_argmax_workspace_words, launch_greedy_argmax};
mod decode;
pub use decode::{
    launch_decode_attention, launch_decode_attention_prepare,
    launch_decode_elementwise, launch_decode_rope,
};
mod gdn_prefill;
pub use gdn_prefill::{
    launch_gdn_prefill_norm, launch_gdn_prefill_prepare, launch_gdn_prefill_recurrence,
};

/// Enqueues four-tap BF16 depthwise prefill convolution, before activation.
///
/// # Safety
/// Inputs/history are read-only BF16 [1,C,T]/[1,C,4] views with the supplied
/// nonnegative element strides; weights are contiguous [C,1,4]. Pointers include
/// storage offsets and must cover every strided access. Output is a disjoint
/// writable C*T BF16 allocation. Dimensions satisfy C>0, T>1, C*T<=i32::MAX.
/// All buffers belong to the supplied CUDA stream's device; pointer guards must
/// survive enqueue, and stream ordering must preserve their execution lifetimes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_bf16_gdn_prefill_convolution(
    input: *const core::ffi::c_void,
    history: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    channels: i32,
    tokens: i32,
    input_channel_stride: i64,
    input_token_stride: i64,
    history_channel_stride: i64,
    history_token_stride: i64,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented storage and stream contract.
    let code = unsafe {
        cuda_ffi::tau_launch_bf16_gdn_prefill_convolution(
            input,
            history,
            weight,
            output,
            channels,
            tokens,
            input_channel_stride,
            input_token_stride,
            history_channel_stride,
            history_token_stride,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "BF16 GDN prefill convolution",
        })
    }
}

/// Storage type understood by the centered RMSNorm bring-up kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RmsNormDType {
    /// IEEE single-precision input, weight, and output.
    F32 = 0,
    /// BFloat16 input, weight, and output with FP32 reduction.
    BF16 = 1,
}

/// Upper bound on each thread's RMSNorm memory access, selected before execution.
/// These portable widths are supported by the compiled CUDA kernels; this is not
/// a GPU-reported SIMD width or a claim about the fastest width on every device.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RmsNormVectorBytes {
    Scalar = 0,
    Bytes4 = 4,
    Bytes8 = 8,
    #[default]
    Bytes16 = 16,
}

impl RmsNormVectorBytes {
    /// Prefers wide accesses without padding rows or crossing row boundaries.
    /// The launcher also checks actual pointer alignment, which is unknown here.
    pub fn for_row_stride(row_bytes: usize) -> Self {
        if row_bytes == 0 {
            Self::Scalar
        } else if row_bytes.is_multiple_of(16) {
            Self::Bytes16
        } else if row_bytes.is_multiple_of(8) {
            Self::Bytes8
        } else if row_bytes.is_multiple_of(4) {
            Self::Bytes4
        } else {
            Self::Scalar
        }
    }
}

#[cfg(test)]
mod vector_policy_tests {
    use super::RmsNormVectorBytes;

    #[test]
    fn heuristic_prefers_widest_row_aligned_access() {
        for (stride, expected) in [
            (0, RmsNormVectorBytes::Scalar),
            (2, RmsNormVectorBytes::Scalar),
            (6, RmsNormVectorBytes::Scalar),
            (4, RmsNormVectorBytes::Bytes4),
            (12, RmsNormVectorBytes::Bytes4),
            (8, RmsNormVectorBytes::Bytes8),
            (24, RmsNormVectorBytes::Bytes8),
            (16, RmsNormVectorBytes::Bytes16),
            (2560 * 2, RmsNormVectorBytes::Bytes16),
            (5120 * 4, RmsNormVectorBytes::Bytes16),
        ] {
            assert_eq!(RmsNormVectorBytes::for_row_stride(stride), expected);
        }
    }
}

/// Storage type understood by the fused SiLU-and-multiply kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SiluAndMulDType {
    /// IEEE single-precision packed input and output.
    F32 = 0,
    /// BFloat16 packed input and output with FP32 activation arithmetic.
    BF16 = 1,
}

/// Error returned when the native CUDA launcher rejects a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchError {
    code: i32,
    kernel: &'static str,
}

impl LaunchError {
    /// Returns the native CUDA or vendor-library status reported by the launcher.
    pub fn code(self) -> i32 {
        self.code
    }
}

impl std::fmt::Display for LaunchError {
    /// Formats the native status returned by the CUDA or cuBLAS launcher.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} native launch failed with status code {}",
            self.kernel, self.code
        )
    }
}

impl std::error::Error for LaunchError {}

/// Enqueues centered RMSNorm on an existing CUDA stream.
///
/// The pointers must address device allocations on the stream's CUDA context.
/// Input and output contain `rows * columns` elements; `weight` contains
/// `columns` elements. All allocations use `dtype`. The call validates launch
/// configuration but does not synchronize the stream.
///
/// # Safety
///
/// The caller must keep every allocation and its stream-synchronization guard
/// alive until this function returns and must ensure the pointer, shape, dtype,
/// and stream contracts described above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_centered_rms_norm(
    input: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    rows: i32,
    columns: i32,
    epsilon: f32,
    dtype: RmsNormDType,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the documented pointer/stream contracts.
    unsafe {
        launch_centered_rms_norm_configured(
            input,
            weight,
            output,
            rows,
            columns,
            epsilon,
            dtype,
            stream,
            RmsNormVectorBytes::Bytes16,
        )
    }
}

/// Launches with a previously selected vector-width ceiling. Misaligned pointers
/// use narrower accesses or the scalar kernel; this never benchmarks at runtime.
///
/// # Safety
/// The pointer, allocation lifetime, shape and stream contracts are identical to
/// [`launch_centered_rms_norm`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_centered_rms_norm_configured(
    input: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    rows: i32,
    columns: i32,
    epsilon: f32,
    dtype: RmsNormDType,
    stream: *mut core::ffi::c_void,
    preferred_bytes: RmsNormVectorBytes,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the raw CUDA pointer and stream contracts.
    let code = unsafe {
        cuda_ffi::tau_launch_centered_rms_norm_configured(
            input,
            weight,
            output,
            rows,
            columns,
            epsilon,
            dtype as i32,
            stream,
            preferred_bytes as i32,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "centered RMSNorm",
        })
    }
}

/// Adds two tensors in their storage dtype and normalizes their rounded sum.
/// Both outputs contain `rows * columns` elements; weight contains `columns`.
///
/// # Safety
/// All pointers must use `dtype` on the stream's device and remain valid until
/// queued work completes. Outputs must not overlap each other or any input.
/// Inputs may alias each other. Keep stream-synchronization guards alive through
/// enqueue, as with [`launch_centered_rms_norm`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_fused_residual_rms_norm(
    input: *const core::ffi::c_void,
    branch: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    residual_output: *mut core::ffi::c_void,
    output: *mut core::ffi::c_void,
    rows: i32,
    columns: i32,
    epsilon: f32,
    dtype: RmsNormDType,
    stream: *mut core::ffi::c_void,
    preferred_bytes: RmsNormVectorBytes,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the documented pointer/shape/lifetime contracts.
    let code = unsafe {
        cuda_ffi::tau_launch_fused_residual_rms_norm(
            input,
            branch,
            weight,
            residual_output,
            output,
            rows,
            columns,
            epsilon,
            dtype as i32,
            stream,
            preferred_bytes as i32,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "fused residual RMSNorm",
        })
    }
}

/// Enqueues fused `SiLU(gate) * up` for a packed gate/up tensor.
///
/// The input contains `rows` contiguous vectors with `columns * 2` elements.
/// Each vector stores the gate half first and the up half second. The output
/// contains `rows * columns` elements in the same storage dtype. The call
/// validates launch configuration but does not synchronize the stream.
///
/// # Safety
///
/// The caller must keep both device allocations and the stream-synchronization
/// guards alive until this function returns. The pointers, dimensions, dtype,
/// and CUDA stream must satisfy the layout described above.
pub unsafe fn launch_silu_and_mul(
    input: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    rows: i32,
    columns: i32,
    dtype: SiluAndMulDType,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the raw CUDA pointer, shape, dtype, and stream
    // contracts documented above.
    let code = unsafe {
        cuda_ffi::tau_launch_silu_and_mul(
            input,
            output,
            rows,
            columns,
            dtype as i32,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "SiLU-and-multiply",
        })
    }
}

/// Enqueues stateful single-token Gated DeltaNet decode with BF16 input/output.
///
/// `qkvz` contains the packed projected Q/K/V/Z vector and `ba` contains packed
/// beta/decay projections. The launcher updates the BF16 convolution history and
/// FP32 recurrent matrix in place, writes the fused gated-normalization result,
/// and leaves all operations asynchronous on `stream`. `negative_a` and
/// `norm_weight` contain FP32 parameters; projections, convolution weights,
/// `dt_bias`, convolution history, the convolved Q/K/V buffer, and output are BF16.
/// For 128x128 heads, `recurrence_output` is a separate BF16 scratch buffer of
/// `num_value_heads * value_head_dim` elements, retained across calls/replays.
/// It must not alias inputs, state, or output. Other head sizes may pass null.
///
/// # Safety
///
/// Every pointer must address a contiguous device allocation on `stream` with
/// lengths implied by the dimensions. State and output pointers must be writable,
/// parameter dtypes must match the function description, and all allocations and
/// synchronization guards must remain alive until this function returns.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_fused_bf16_gated_delta_net_decode(
    qkvz: *const core::ffi::c_void,
    ba: *const core::ffi::c_void,
    conv_weight: *const core::ffi::c_void,
    dt_bias: *const core::ffi::c_void,
    negative_a: *const core::ffi::c_void,
    norm_weight: *const f32,
    conv_state: *mut core::ffi::c_void,
    recurrent_state: *mut core::ffi::c_void,
    convolved_qkv: *mut core::ffi::c_void,
    recurrence_output: *mut core::ffi::c_void,
    output: *mut core::ffi::c_void,
    conv_width: i32,
    conv_kernel_size: i32,
    num_key_heads: i32,
    num_value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    epsilon: f32,
    reference_reduction: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the raw CUDA pointer, state, dimension, and stream
    // contracts documented above.
    let code = unsafe {
        cuda_ffi::tau_launch_fused_bf16_gated_delta_net_decode(
            qkvz,
            ba,
            conv_weight,
            dt_bias,
            negative_a,
            norm_weight,
            conv_state,
            recurrent_state,
            convolved_qkv,
            recurrence_output,
            output,
            conv_width,
            conv_kernel_size,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            epsilon,
            reference_reduction,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "fused BF16 Gated DeltaNet decode",
        })
    }
}
