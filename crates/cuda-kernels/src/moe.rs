//! Native MoE precision/routing/dispatch launch contracts.

mod config;
mod decode_config;
mod decode_selection;
pub use config::{MoePrefillConfig, MoePrefillTile};
pub use decode_config::{MoeDecodeConfig, MoeDecodeProfile};
pub use decode_selection::{MoeDecodeOptions, MoeDecodeResolution, MoeDecodeSelection};
mod prefill;
pub use prefill::{
    MOE_PREFILL_EXPERTS, MOE_PREFILL_META_WORDS, MOE_PREFILL_TILE, launch_moe_prefill,
    launch_moe_prefill_dispatch, moe_prefill_capacity, moe_prefill_capacity_for_tile,
};

use super::{LaunchError, cuda_ffi};

/// Queue the exact projection backend used inside routed decode, into FP32.
/// Operator validation uses this boundary to time either projection separately;
/// routing and BF16/weighting epilogues remain with the pipeline composer.
///
/// # Safety
/// Device pointer arrays cover `batches` disjoint outputs [columns], inputs
/// [inner] and weights [columns,inner]. All owners survive ordered execution on
/// stream. Profile/device resources are validated before capture; optional
/// receipts cover five i32 words per batch. Logical M is always one.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_decode_projection(
    inputs: *const core::ffi::c_void,
    weights: *const core::ffi::c_void,
    outputs: *mut core::ffi::c_void,
    batches: i32,
    columns: i32,
    inner: i32,
    profile: &MoeDecodeProfile,
    receipts: *mut i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: Caller upholds the retained device-pointer array/shape contract.
    let code = unsafe {
        cuda_ffi::tau_launch_moe_decode_gemm(
            inputs, weights, outputs, batches, columns, inner, profile, receipts,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "MoE decode projection",
        })
    }
}

/// Queues a BF16 shared/routed [M,H] sum with FP32 gate dot/sigmoid/FMA.
///
/// # Safety
/// Input/shared/routed/output cover [M,H], weight covers H; H<=8192. All buffers
/// use the stream device, output is disjoint, and owners survive ordered use.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_shared_combine(
    input: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    shared: *const core::ffi::c_void,
    routed: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    rows: i32,
    width: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented geometry/stream contract.
    let code = unsafe {
        cuda_ffi::tau_launch_moe_shared_combine(
            input, weight, shared, routed, output, rows, width, stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "MoE shared combine",
        })
    }
}

/// Queues FP32 softmax/top-8 for BF16 [M,256], writing F32 [M,16].
///
/// # Safety
/// Inputs/output must cover their shapes on the stream device and be disjoint.
/// Owners/guards survive enqueue; stream ordering protects execution lifetimes.
pub unsafe fn launch_moe_routing_fp32(
    logits: *const core::ffi::c_void,
    output: *mut f32,
    rows: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented buffer/stream contract.
    let code =
        unsafe { cuda_ffi::tau_launch_moe_routing_fp32(logits, output, rows, stream) };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "MoE FP32 routing",
        })
    }
}

/// Queues a contiguous BF16 [M,K] times [N,K]^T projection into FP32 [M,N].
///
/// # Safety
/// Buffers must cover their full shapes on the stream's device. The output is
/// disjoint; all owners/guards survive enqueue and execution is stream ordered.
pub unsafe fn launch_linear_bf16_fp32(
    input: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    output: *mut f32,
    rows: i32,
    columns: i32,
    inner: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller upholds the documented storage/stream contract.
    let code = unsafe {
        cuda_ffi::tau_launch_linear_bf16_fp32(
            input, weight, output, rows, columns, inner, stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "BF16 linear FP32",
        })
    }
}
/// Enqueues the complete BF16 routed-expert decode pipeline for one token.
///
/// The launcher selects and normalizes the top eight experts from 256 router
/// logits, prepares device pointer arrays, runs packed gate/up and down
/// projections with FP32 MMA accumulation, applies fused SiLU gating,
/// weights down accumulators before BF16 storage, then sums them in FP32. Every pointer addresses
/// device memory on `stream`; workspace sizes follow the shape contract below.
/// The function enqueues work asynchronously and does not synchronize.
///
/// # Safety
///
/// The caller must provide contiguous BF16 input, logits, weights, intermediate
/// buffers, and output with the documented shapes. `selected_ids` holds eight
/// `i32` values, `selected_weights` holds eight `f32` values, and
/// `pointer_workspace` holds 48 pointer-sized entries. `gate_up_fp32` holds
/// [8,2*I] FP32 values and `expert_fp32` holds [8,H]. All buffers use the active
/// stream device, remain disjoint, and survive enqueue with stream-ordered use.
/// Profiles have been validated for this device before capture. Optional test
/// `receipts` covers 80 i32 words: eight five-word actual profiles per projection;
/// production passes null and performs no receipt allocation or readback.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_fused_bf16_moe_decode(
    input: *const core::ffi::c_void,
    router_logits: *const core::ffi::c_void,
    gate_up_weights: *const core::ffi::c_void,
    down_weights: *const core::ffi::c_void,
    selected_ids: *mut core::ffi::c_void,
    selected_weights: *mut core::ffi::c_void,
    pointer_workspace: *mut core::ffi::c_void,
    gate_up_output: *mut core::ffi::c_void,
    activated_output: *mut core::ffi::c_void,
    expert_output: *mut core::ffi::c_void,
    gate_up_fp32: *mut core::ffi::c_void,
    expert_fp32: *mut core::ffi::c_void,
    output: *mut core::ffi::c_void,
    hidden_size: i32,
    intermediate_size: i32,
    num_experts: i32,
    experts_per_token: i32,
    gate_up_config: &crate::MoeDecodeProfile,
    down_config: &crate::MoeDecodeProfile,
    receipts: *mut i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the device pointer, workspace, and
    // stream contracts documented above.
    let code = unsafe {
        cuda_ffi::tau_launch_fused_bf16_moe_decode(
            input,
            router_logits,
            gate_up_weights,
            down_weights,
            selected_ids,
            selected_weights,
            pointer_workspace,
            gate_up_output,
            activated_output,
            expert_output,
            gate_up_fp32,
            expert_fp32,
            output,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            gate_up_config,
            down_config,
            receipts,
            stream,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "fused BF16 MoE decode",
        })
    }
}
