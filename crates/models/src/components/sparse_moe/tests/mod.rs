//! Component composition, backend, diagnostic and timing checks.

mod block;
mod cuda;
mod diagnostics;
mod grouped;
mod prefill;
mod prepared;

use super::*;
use candle::{DType, Device};
use std::collections::HashMap;

/// Supplies a compact configuration with every optional behavior enabled.
pub(super) fn valid_config() -> SparseMoeConfig {
    SparseMoeConfig {
        hidden_size: 2,
        expert_intermediate_size: 2,
        expert_activation: Activation::Relu,
        num_experts: 2,
        experts_per_token: 1,
        router_bias: false,
        router_score_function: RouterScoreFunction::Softmax,
        normalize_selected_weights: true,
        shared_expert: Some(SharedExpertConfig {
            intermediate_size: 2,
            activation: Activation::Relu,
            gate: Some(SharedExpertGateConfig {
                activation: Activation::Sigmoid,
                bias: false,
            }),
        }),
        weight_layout: SparseMoeWeightLayout::Individual,
    }
}
