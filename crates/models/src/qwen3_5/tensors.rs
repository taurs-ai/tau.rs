//! Safetensors-header validation for dense Qwen3.5 text execution.
//!
//! The validator checks the exact names, storage dtypes, and shapes that model
//! construction will consume. Vision, MTP, and other auxiliary checkpoint
//! subtrees are deliberately tolerated and remain lazy in the text-only runtime.

use super::config::{LoadedQwen3_5Config, Qwen3_5Config};
use crate::qwen3_5_common::hybrid_text_tensor_specs;
use crate::tensor_contract::{TensorSpec, validate_tensor_contract};
use crate::{ModelLoadError, ModelShard};

/// Validates every dense text tensor before Candle model construction begins.
pub(crate) fn validate_qwen3_5_tensor_contract(
    config: &LoadedQwen3_5Config,
    shards: &[ModelShard],
) -> Result<(), ModelLoadError> {
    let required = required_tensors(&config.model);
    validate_tensor_contract("dense Qwen3.5", config.weight_dtype, shards, &required)
}

/// Derives the complete text tensor contract from architecture dimensions.
fn required_tensors(config: &Qwen3_5Config) -> Vec<TensorSpec> {
    let hybrid = &config.hybrid;
    hybrid_text_tensor_specs(hybrid, "model.language_model", |required, prefix| {
        required.push(TensorSpec::new(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![config.intermediate_size, hybrid.hidden_size],
        ));
        required.push(TensorSpec::new(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![config.intermediate_size, hybrid.hidden_size],
        ));
        required.push(TensorSpec::new(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hybrid.hidden_size, config.intermediate_size],
        ));
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5::config::{ARCHITECTURE, MODEL_TYPE, load_qwen3_5_config};
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Writes the dense fixture configuration used by tensor-contract tests.
    fn temporary_config() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen3-5-dense-tensors-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::json!({
                "architectures": [ARCHITECTURE],
                "model_type": MODEL_TYPE,
                "tie_word_embeddings": false,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "dtype": "bfloat16",
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "intermediate_size": 32,
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
                    "partial_rotary_factor": 0.5,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention"
                    ],
                    "full_attention_interval": 4,
                    "attn_output_gate": true,
                    "mamba_ssm_dtype": "float32"
                }
            })
            .to_string(),
        )
        .unwrap();
        root
    }

    /// Loads the dense configuration from a prepared temporary directory.
    fn loaded_config(root: &Path) -> LoadedQwen3_5Config {
        load_qwen3_5_config(root, ARCHITECTURE, MODEL_TYPE).unwrap()
    }

    /// Verifies tensor names and shapes derived from the dense fixture config.
    #[test]
    fn derives_fixture_tensor_shapes_from_config() {
        let root = temporary_config();
        let config = loaded_config(&root);
        let requirements = required_tensors(&config.model)
            .into_iter()
            .map(|requirement| {
                (requirement.name().to_owned(), requirement.shape().to_vec())
            })
            .collect::<HashMap<_, _>>();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            requirements["model.language_model.layers.0.\
                          linear_attn.in_proj_qkv.weight"],
            vec![32, 16]
        );
        assert_eq!(
            requirements["model.language_model.layers.3.self_attn.q_proj.weight"],
            vec![32, 16]
        );
        assert_eq!(
            requirements["model.language_model.layers.3.mlp.down_proj.weight"],
            vec![16, 32]
        );
    }
}
