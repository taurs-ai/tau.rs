//! Rust launch wrappers and safety contracts for GDN's prefill stages.
//!
//! Prefill processes known prompt tokens before one-token decode begins. These
//! wrappers prepare recurrence inputs, advance memory across the chunk, and gate
//! the resulting activations. All native C ABI declarations live in `cuda_ffi.rs`;
//! this module documents the caller's buffer contracts and converts native status
//! codes to Rust `Result` values. Candle owns the buffers and CUDA stream.

use crate::{LaunchError, cuda_ffi};
use core::ffi::c_void;

/// Applies BF16 rounding, FP32 RMSNorm and SiLU gating to all prefill rows.
///
/// # Safety
/// Input is contiguous FP32 [rows,128], gate BF16 [rows,128], weight FP32 [128].
/// Output is disjoint BF16 [rows,128]. rows>0 fits i32; epsilon is finite/positive.
/// Storage belongs to stream's device and remains alive through execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gdn_prefill_norm(
    input: *const f32,
    gate: *const c_void,
    weight: *const f32,
    output: *mut c_void,
    rows: i32,
    epsilon: f32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented tensor and stream contract.
    let code = unsafe {
        cuda_ffi::tau_launch_gdn_prefill_norm(
            input, gate, weight, output, rows, epsilon, stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "GDN prefill gated norm",
        })
    }
}

/// Normalizes grouped Q/K and packs recurrence inputs on the caller's stream.
///
/// # Safety
/// Mixed is a bounded BF16 [1,tokens,(2*key_heads+value_heads)*128] view with
/// supplied nonnegative strides. Alpha is contiguous FP32 [1,tokens,value_heads],
/// beta is BF16 with that shape. Output is disjoint FP32 [tokens,value_heads,386].
/// Positive dimensions and products fit i32; value_heads is divisible by key_heads.
/// All storage belongs to stream's device and remains alive through execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gdn_prefill_prepare(
    mixed: *const c_void,
    alpha: *const f32,
    beta: *const c_void,
    output: *mut f32,
    tokens: i32,
    key_heads: i32,
    value_heads: i32,
    token_stride: i64,
    channel_stride: i64,
    round_normalized: i32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented tensor and stream contract.
    let code = unsafe {
        cuda_ffi::tau_launch_gdn_prefill_prepare(
            mixed,
            alpha,
            beta,
            output,
            tokens,
            key_heads,
            value_heads,
            token_stride,
            channel_stride,
            round_normalized,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "GDN prefill preparation",
        })
    }
}

/// Enqueues sequential multi-token recurrence without allocation or synchronization.
///
/// # Safety
/// Input is contiguous FP32 [tokens, heads, 386], containing scaled Q, normalized
/// K, V (128 each), alpha and beta. Initial state is read-only FP32 [heads,128,128].
/// Output is disjoint FP32 [tokens*heads*128 + heads*128*128], with readout followed
/// by final state. tokens>1 and heads>0 fit i32. All buffers belong to the supplied
/// stream's device and remain alive until execution completes.
pub unsafe fn launch_gdn_prefill_recurrence(
    input: *const f32,
    initial: *const f32,
    output: *mut f32,
    tokens: i32,
    heads: i32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented storage and stream contract.
    let code = unsafe {
        cuda_ffi::tau_launch_gdn_prefill_recurrence(
            input, initial, output, tokens, heads, stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "GDN prefill recurrence",
        })
    }
}
