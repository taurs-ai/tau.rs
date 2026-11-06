//! Tensor requirements shared by dense and MoE Qwen3.5 text decoders.
//!
//! Wrapper prefixes and feed-forward layouts remain family-specific. This module
//! derives the common embedding, normalization, token-mixer, and language-model
//! head requirements around a callback that contributes each layer's MLP tensors.
//! Only Gated DeltaNet `linear_attn.A_log` and `linear_attn.norm.weight` may use
//! FP32 storage in a BF16 checkpoint; all other consumed text tensors must match
//! the declared checkpoint dtype, including `dt_bias` and other normalizers.

use super::{HybridLayerType, HybridTextConfig};
use crate::tensor_contract::TensorSpec;

/// Derives the shared text tensor contract and delegates feed-forward tensors.
pub(crate) fn hybrid_text_tensor_specs(
    config: &HybridTextConfig,
    model_prefix: &str,
    mut add_feed_forward: impl FnMut(&mut Vec<TensorSpec>, &str),
) -> Vec<TensorSpec> {
    let mut required = Vec::new();
    required.push(TensorSpec::new(
        format!("{model_prefix}.embed_tokens.weight"),
        vec![config.vocab_size, config.hidden_size],
    ));
    required.push(TensorSpec::new(
        format!("{model_prefix}.norm.weight"),
        vec![config.hidden_size],
    ));
    if !config.tie_word_embeddings {
        required.push(TensorSpec::new(
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
        ));
    }

    for (layer_index, &layer_type) in config.layer_types.iter().enumerate() {
        let layer_prefix = format!("{model_prefix}.layers.{layer_index}");
        required.push(TensorSpec::new(
            format!("{layer_prefix}.input_layernorm.weight"),
            vec![config.hidden_size],
        ));
        required.push(TensorSpec::new(
            format!("{layer_prefix}.post_attention_layernorm.weight"),
            vec![config.hidden_size],
        ));
        add_feed_forward(&mut required, &layer_prefix);

        match layer_type {
            HybridLayerType::LinearAttention => {
                add_linear_attention_specs(&mut required, config, &layer_prefix)
            }
            HybridLayerType::FullAttention => {
                add_full_attention_specs(&mut required, config, &layer_prefix)
            }
        }
    }
    required
}

/// Adds Gated DeltaNet projection, convolution, normalization, and state weights.
fn add_linear_attention_specs(
    required: &mut Vec<TensorSpec>,
    config: &HybridTextConfig,
    layer_prefix: &str,
) {
    let prefix = format!("{layer_prefix}.linear_attn");
    let convolution_width = config.linear_convolution_width();
    let value_width = config.linear_value_width();
    let value_heads = config.linear_num_value_heads;
    for (name, shape) in [
        (
            "in_proj_qkv.weight",
            vec![convolution_width, config.hidden_size],
        ),
        ("in_proj_z.weight", vec![value_width, config.hidden_size]),
        ("in_proj_a.weight", vec![value_heads, config.hidden_size]),
        ("in_proj_b.weight", vec![value_heads, config.hidden_size]),
        (
            "conv1d.weight",
            vec![convolution_width, 1, config.linear_conv_kernel_dim],
        ),
        ("A_log", vec![value_heads]),
        ("dt_bias", vec![value_heads]),
        ("norm.weight", vec![config.linear_value_head_dim]),
        ("out_proj.weight", vec![config.hidden_size, value_width]),
    ] {
        let spec = TensorSpec::new(format!("{prefix}.{name}"), shape);
        required.push(if matches!(name, "A_log" | "norm.weight") {
            spec.with_f32_storage()
        } else {
            spec
        });
    }
}

/// Adds gated grouped-query attention projections and per-head normalizers.
fn add_full_attention_specs(
    required: &mut Vec<TensorSpec>,
    config: &HybridTextConfig,
    layer_prefix: &str,
) {
    let prefix = format!("{layer_prefix}.self_attn");
    let query_width = config.query_width();
    let key_value_width = config.key_value_width();
    for (name, shape) in [
        ("q_proj.weight", vec![2 * query_width, config.hidden_size]),
        ("k_proj.weight", vec![key_value_width, config.hidden_size]),
        ("v_proj.weight", vec![key_value_width, config.hidden_size]),
        ("o_proj.weight", vec![config.hidden_size, query_width]),
        ("q_norm.weight", vec![config.head_dim]),
        ("k_norm.weight", vec![config.head_dim]),
    ] {
        required.push(TensorSpec::new(format!("{prefix}.{name}"), shape));
    }

    if config.attention_bias {
        for (name, shape) in [
            ("q_proj.bias", vec![2 * query_width]),
            ("k_proj.bias", vec![key_value_width]),
            ("v_proj.bias", vec![key_value_width]),
            ("o_proj.bias", vec![config.hidden_size]),
        ] {
            required.push(TensorSpec::new(format!("{prefix}.{name}"), shape));
        }
    }
}
