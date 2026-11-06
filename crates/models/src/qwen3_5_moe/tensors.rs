//! Safetensors-header validation for Qwen3.5 MoE text execution.
//!
//! The validator checks every tensor consumed by the causal/individual-expert and
//! conditional-generation/packed-expert constructors before Candle allocates model
//! weights. Vision, MTP, and other auxiliary subtrees remain accepted and lazy.

use super::config::{
    LoadedQwen3_5MoeConfig, Qwen3_5MoeCheckpointKind, Qwen3_5MoeConfig,
};
use crate::qwen3_5_common::hybrid_text_tensor_specs;
use crate::tensor_contract::{TensorSpec, validate_tensor_contract};
use crate::{ModelLoadError, ModelShard};

/// Validates every Qwen3.5 MoE text tensor before model construction begins.
pub(crate) fn validate_qwen3_5_moe_tensor_contract(
    config: &LoadedQwen3_5MoeConfig,
    shards: &[ModelShard],
) -> Result<(), ModelLoadError> {
    let required = required_tensors(config);
    validate_tensor_contract("Qwen3.5 MoE", config.weight_dtype, shards, &required)
}

/// Derives the complete text contract for the selected wrapper and expert layout.
fn required_tensors(config: &LoadedQwen3_5MoeConfig) -> Vec<TensorSpec> {
    let model = &config.model;
    let hybrid = &model.hybrid;
    let model_prefix = match config.checkpoint_kind {
        Qwen3_5MoeCheckpointKind::CausalLm => "model",
        Qwen3_5MoeCheckpointKind::ConditionalGeneration => "model.language_model",
    };
    hybrid_text_tensor_specs(hybrid, model_prefix, |required, prefix| {
        add_sparse_moe_requirements(required, model, config.checkpoint_kind, prefix);
    })
}

/// Adds the router, routed experts, shared expert, and shared gate requirements.
fn add_sparse_moe_requirements(
    required: &mut Vec<TensorSpec>,
    config: &Qwen3_5MoeConfig,
    checkpoint_kind: Qwen3_5MoeCheckpointKind,
    layer_prefix: &str,
) {
    let hybrid = &config.hybrid;
    let prefix = format!("{layer_prefix}.mlp");
    required.push(TensorSpec::new(
        format!("{prefix}.gate.weight"),
        vec![config.num_experts, hybrid.hidden_size],
    ));
    for (name, shape) in [
        (
            "shared_expert.gate_proj.weight",
            vec![config.shared_expert_intermediate_size, hybrid.hidden_size],
        ),
        (
            "shared_expert.up_proj.weight",
            vec![config.shared_expert_intermediate_size, hybrid.hidden_size],
        ),
        (
            "shared_expert.down_proj.weight",
            vec![hybrid.hidden_size, config.shared_expert_intermediate_size],
        ),
        ("shared_expert_gate.weight", vec![1, hybrid.hidden_size]),
    ] {
        required.push(TensorSpec::new(format!("{prefix}.{name}"), shape));
    }

    let experts_prefix = format!("{prefix}.experts");
    match checkpoint_kind {
        Qwen3_5MoeCheckpointKind::CausalLm => {
            for expert_index in 0..config.num_experts {
                let expert_prefix = format!("{experts_prefix}.{expert_index}");
                for (name, shape) in [
                    (
                        "gate_proj.weight",
                        vec![config.moe_intermediate_size, hybrid.hidden_size],
                    ),
                    (
                        "up_proj.weight",
                        vec![config.moe_intermediate_size, hybrid.hidden_size],
                    ),
                    (
                        "down_proj.weight",
                        vec![hybrid.hidden_size, config.moe_intermediate_size],
                    ),
                ] {
                    required.push(TensorSpec::new(
                        format!("{expert_prefix}.{name}"),
                        shape,
                    ));
                }
            }
        }
        Qwen3_5MoeCheckpointKind::ConditionalGeneration => {
            required.push(TensorSpec::new(
                format!("{experts_prefix}.gate_up_proj"),
                vec![
                    config.num_experts,
                    2 * config.moe_intermediate_size,
                    hybrid.hidden_size,
                ],
            ));
            required.push(TensorSpec::new(
                format!("{experts_prefix}.down_proj"),
                vec![
                    config.num_experts,
                    hybrid.hidden_size,
                    config.moe_intermediate_size,
                ],
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::config::{
        ARCHITECTURE, CONDITIONAL_ARCHITECTURE, CONDITIONAL_MODEL_TYPE, MODEL_TYPE,
        load_qwen3_5_moe_config,
    };
    use crate::safetensors::load_safetensors_shards;
    use serde_json::{Value, json};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Builds valid shared text metadata for MoE tensor-contract tests.
    fn valid_text_config() -> Value {
        json!({
            "architectures": [ARCHITECTURE],
            "model_type": MODEL_TYPE,
            "dtype": "bfloat16",
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "use_cache": true,
            "tie_word_embeddings": false,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ]
        })
    }

    /// Writes wrapper metadata for the selected MoE checkpoint layout.
    fn temporary_config(kind: Qwen3_5MoeCheckpointKind) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen3-5-moe-tensors-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let contents = match kind {
            Qwen3_5MoeCheckpointKind::CausalLm => valid_text_config(),
            Qwen3_5MoeCheckpointKind::ConditionalGeneration => json!({
                "architectures": [CONDITIONAL_ARCHITECTURE],
                "model_type": CONDITIONAL_MODEL_TYPE,
                "tie_word_embeddings": false,
                "text_config": valid_text_config(),
                "vision_config": { "model_type": "qwen3_5_moe_vision" }
            }),
        };
        fs::write(root.join("config.json"), contents.to_string()).unwrap();
        root
    }

    /// Loads one generated MoE configuration and removes its temporary directory.
    fn loaded_config(kind: Qwen3_5MoeCheckpointKind) -> LoadedQwen3_5MoeConfig {
        let root = temporary_config(kind);
        let (architecture, model_type) = match kind {
            Qwen3_5MoeCheckpointKind::CausalLm => (ARCHITECTURE, MODEL_TYPE),
            Qwen3_5MoeCheckpointKind::ConditionalGeneration => {
                (CONDITIONAL_ARCHITECTURE, CONDITIONAL_MODEL_TYPE)
            }
        };
        let config = load_qwen3_5_moe_config(&root, architecture, model_type).unwrap();
        fs::remove_dir_all(root).unwrap();
        config
    }

    /// Verifies tensor requirements for causal checkpoints with individual experts.
    #[test]
    fn derives_causal_individual_expert_contract() {
        let config = loaded_config(Qwen3_5MoeCheckpointKind::CausalLm);
        let required = required_tensors(&config);
        assert!(required.iter().any(|requirement| {
            requirement.name() == "model.layers.0.mlp.experts.7.gate_proj.weight"
                && requirement.shape() == [4, 16]
        }));
        assert!(
            !required.iter().any(|requirement| requirement
                .name()
                .ends_with("experts.gate_up_proj"))
        );
    }

    /// Verifies tensor requirements for conditional checkpoints with packed experts.
    #[test]
    fn derives_conditional_packed_expert_contract() {
        let config = loaded_config(Qwen3_5MoeCheckpointKind::ConditionalGeneration);
        let required = required_tensors(&config);
        assert!(required.iter().any(|requirement| {
            requirement.name()
                == "model.language_model.layers.0.mlp.experts.gate_up_proj"
                && requirement.shape() == [8, 8, 16]
        }));
        assert!(
            !required.iter().any(|requirement| requirement
                .name()
                .contains("mlp.experts.0.gate_proj"))
        );
    }

    /// Resolves the generated MoE artifact used by the ignored contract test.
    fn local_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_MOE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.5-MoE-Tiny-Dev")
            })
    }

    /// Verifies the generated artifact satisfies the complete MoE tensor contract.
    #[test]
    #[ignore = "validates the local generated Qwen3.5 MoE development model"]
    fn validates_generated_qwen3_5_moe_tensor_contract() {
        let root = local_test_model_path();
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        let shards = load_safetensors_shards(&root).unwrap();

        validate_qwen3_5_moe_tensor_contract(&config, &shards).unwrap();
    }
}
