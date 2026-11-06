//! Configuration and composition shared by dense and MoE Qwen3.5 models.
//!
//! The family façades retain exact checkpoint dispatch, wrapper layouts, and
//! feed-forward choices. This module owns the hybrid text-decoder contract common
//! to both variants.

mod config;
mod decoder;
mod model;
mod tensors;

pub(crate) use config::{HybridLayerType, HybridTextConfig};
pub(crate) use model::HybridCausalLm;
pub(crate) use tensors::hybrid_text_tensor_specs;
