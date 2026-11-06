//! Native greedy reduction ABI; the caller owns policy, buffers and readback.

use crate::{LaunchError, cuda_ffi};
use core::ffi::c_void;

/// Storage read by the native argmax. Both variants compare exact FP32 values.
#[derive(Clone, Copy)]
#[repr(i32)]
pub enum GreedyDType {
    F32 = 0,
    BF16 = 1,
}

/// Returns the native reduction's scratch size in U32 words, including tail blocks.
/// Keeping this calculation native makes allocation and launch geometry agree.
pub fn greedy_argmax_workspace_words(vocabulary: u32) -> usize {
    // SAFETY: This pure size query accepts every U32 count and dereferences nothing.
    unsafe { cuda_ffi::tau_greedy_argmax_workspace_words(vocabulary) }
}

/// Enqueues vocabulary reduction and final selection on the supplied CUDA stream.
/// Result words are `[token_id, valid]`; an all-NaN vocabulary writes valid=0.
/// Equal values choose the greatest token ID, with FP32 total ordering of zeros.
///
/// # Safety
/// Input covers `vocabulary>0` contiguous elements of `dtype`. Scratch contains at
/// least `greedy_argmax_workspace_words(vocabulary)` U32 words; result contains
/// two. All three allocations are disjoint, belong to stream's device and remain
/// alive until both kernels complete. The caller orders previous logit writes,
/// scratch/result reuse and host readback on this stream. Pointer access guards
/// must cover enqueue when the tensor backend tracks allocation dependencies.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_greedy_argmax(
    logits: *const c_void,
    dtype: GreedyDType,
    vocabulary: u32,
    scratch: *mut u32,
    scratch_words: usize,
    result: *mut u32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented bounds and stream lifetimes.
    let code = unsafe {
        cuda_ffi::tau_launch_greedy_argmax(
            logits,
            dtype as i32,
            vocabulary,
            scratch,
            scratch_words,
            result,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "greedy argmax",
        })
    }
}
