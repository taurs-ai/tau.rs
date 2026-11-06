//! Tensor bridges for tau.rs's native CUDA kernels.
//!
//! Bridges validate tensor storage, device placement, and stream lifetimes.
//! Components call them only when their input contract matches and
//! optimized-kernel mode is enabled.

pub(crate) mod decode;
mod fused_moe;
mod gated_delta_net;
mod grouped_attention;
mod linear;
mod routing;
mod shared_expert;
mod sigmoid;
// The low-level primitive is exercised by capture tests until model integration.
#[allow(dead_code)]
pub(crate) mod graph;
mod rms_norm;
mod silu_and_mul;

pub(super) use fused_moe::{FusedMoeRunner, MoePrefillRunner};
pub(super) use gated_delta_net::{
    GatedDeltaNetRunner, mixed_prefill_cuda, prefill_convolution_cuda,
    prefill_norm_cuda, prefill_recurrence_cuda, supports_mixed_prefill,
    supports_prefill_convolution, supports_prefill_norm, supports_prefill_recurrence,
};
pub(super) use grouped_attention::cached_qk;
pub(super) use linear::linear_bf16_fp32;
pub(super) use rms_norm::{
    centered_rms_norm_cuda, fused_residual_rms_norm_cuda, supports_centered_rms_norm,
    supports_fused_residual_rms_norm,
};
pub(super) use routing::moe_routes_fp32;
pub(super) use shared_expert::moe_shared_combine;
pub(super) use sigmoid::sigmoid_mul_fp32;
pub(super) use silu_and_mul::{silu_and_mul_cuda, supports_silu_and_mul};
