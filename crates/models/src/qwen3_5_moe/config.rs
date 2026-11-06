//! Configuration parsing and validation for Qwen3.5 MoE checkpoints.
//!
//! This module is the boundary between Hugging Face JSON metadata and model
//! construction. It owns the exact architecture dispatch key, every construction
//! field consumed by the model, structural validation, and runtime dtype
//! compatibility checks. Keeping these concerns separate lets the model module
//! focus on tensor layout and execution.

use crate::ModelLoadError;
use crate::components::{
    RouterScoreFunction, SharedExpertConfig, SharedExpertGateConfig, SparseMoeConfig,
    SparseMoeWeightLayout,
};
use crate::dtype::{ModelDType, supported_dtypes_from_torch_dtype};
use crate::metadata::{optional_json_string, parse_json_file};
use crate::qwen3_5_common::HybridTextConfig;
use candle::Device;
use candle_nn::Activation;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

pub(crate) const ARCHITECTURE: &str = "Qwen3_5MoeForCausalLM";
pub(crate) const MODEL_TYPE: &str = "qwen3_5_moe_text";
pub(crate) const CONDITIONAL_ARCHITECTURE: &str = "Qwen3_5MoeForConditionalGeneration";
pub(crate) const CONDITIONAL_MODEL_TYPE: &str = "qwen3_5_moe";
pub(crate) const TEXT_MODEL_TYPE: &str = MODEL_TYPE;

/// Top-level checkpoint wrapper and its serialized decoder layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen3_5MoeCheckpointKind {
    CausalLm,
    ConditionalGeneration,
}

impl Qwen3_5MoeCheckpointKind {
    /// Resolves only the two exact architecture/model-type pairs tau.rs supports.
    fn from_dispatch(
        architecture: &str,
        model_type: &str,
    ) -> Result<Self, ModelLoadError> {
        match (architecture, model_type) {
            (ARCHITECTURE, TEXT_MODEL_TYPE) => Ok(Self::CausalLm),
            (CONDITIONAL_ARCHITECTURE, CONDITIONAL_MODEL_TYPE) => {
                Ok(Self::ConditionalGeneration)
            }
            _ => Err(ModelLoadError::InvalidConfig(format!(
                "model construction supports {ARCHITECTURE}/{TEXT_MODEL_TYPE} \
                 and {CONDITIONAL_ARCHITECTURE}/{CONDITIONAL_MODEL_TYPE}, got \
                 {architecture}/{model_type}"
            ))),
        }
    }

    /// Returns the user-facing architecture name for diagnostics.
    pub(super) const fn architecture(self) -> &'static str {
        match self {
            Self::CausalLm => ARCHITECTURE,
            Self::ConditionalGeneration => CONDITIONAL_ARCHITECTURE,
        }
    }
}

/// Construction fields required by the Qwen3.5 MoE text decoder.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Qwen3_5MoeConfig {
    #[serde(flatten)]
    pub(crate) hybrid: HybridTextConfig,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) shared_expert_intermediate_size: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_experts: usize,
}

impl Qwen3_5MoeConfig {
    /// Translates Qwen's expert fields into a reusable sparse-MoE config.
    pub(crate) fn sparse_moe_config(
        &self,
        weight_layout: SparseMoeWeightLayout,
    ) -> SparseMoeConfig {
        SparseMoeConfig {
            hidden_size: self.hybrid.hidden_size,
            expert_intermediate_size: self.moe_intermediate_size,
            expert_activation: self.hybrid.hidden_act,
            num_experts: self.num_experts,
            experts_per_token: self.num_experts_per_tok,
            router_bias: false,
            router_score_function: RouterScoreFunction::Softmax,
            normalize_selected_weights: true,
            shared_expert: Some(SharedExpertConfig {
                intermediate_size: self.shared_expert_intermediate_size,
                activation: self.hybrid.hidden_act,
                gate: Some(SharedExpertGateConfig {
                    activation: Activation::Sigmoid,
                    bias: false,
                }),
            }),
            weight_layout,
        }
    }

    /// Adds sparse-expert requirements to the shared hybrid config validation.
    fn validate(&self) -> Result<(), ModelLoadError> {
        self.hybrid.validate("Qwen3.5 MoE")?;
        self.sparse_moe_config(SparseMoeWeightLayout::Individual)
            .validate()
            .map_err(ModelLoadError::InvalidConfig)?;
        Ok(())
    }
}

/// Parsed construction config and dtypes permitted by checkpoint metadata.
///
/// A permitted dtype is still validated against the selected backend. BF16 is
/// accepted on CUDA and rejected on Candle CPU before weights are allocated.
#[derive(Debug)]
pub(crate) struct LoadedQwen3_5MoeConfig {
    pub(crate) model: Qwen3_5MoeConfig,
    /// Declared bulk storage dtype; tensor contracts permit specific FP32 weights.
    pub(super) weight_dtype: ModelDType,
    pub(crate) supported_dtypes: Vec<ModelDType>,
    pub(super) checkpoint_kind: Qwen3_5MoeCheckpointKind,
}

/// Resolves the two Hugging Face dtype spellings within one config object.
fn dtype_metadata(value: &Value) -> Result<Option<&str>, ModelLoadError> {
    let torch_dtype = optional_json_string(value, "torch_dtype")?;
    let dtype = optional_json_string(value, "dtype")?;
    if let (Some(torch_dtype), Some(dtype)) = (torch_dtype, dtype)
        && torch_dtype != dtype
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "config dtype {dtype:?} conflicts with torch_dtype {torch_dtype:?}"
        )));
    }
    Ok(torch_dtype.or(dtype))
}

/// Parses the declared bulk storage dtype; per-tensor exceptions are checked later.
fn weight_dtype(dtype: Option<&str>) -> Result<ModelDType, ModelLoadError> {
    match dtype {
        Some("bfloat16" | "bf16") => Ok(ModelDType::BF16),
        Some("float32" | "f32") => Ok(ModelDType::F32),
        Some(dtype) => Err(ModelLoadError::InvalidConfig(format!(
            "Qwen3.5 MoE checkpoint dtype {dtype:?} is unsupported; expected \
             bfloat16 or float32"
        ))),
        None => Err(ModelLoadError::InvalidConfig(
            "Qwen3.5 MoE config must declare dtype or torch_dtype".to_owned(),
        )),
    }
}

/// Returns and validates the nested text config used by the multimodal wrapper.
fn conditional_text_config(value: &Value) -> Result<Value, ModelLoadError> {
    let text_config = value.get("text_config").ok_or_else(|| {
        ModelLoadError::InvalidConfig(format!(
            "{CONDITIONAL_ARCHITECTURE} config must include text_config"
        ))
    })?;
    let text_model_type = text_config
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ModelLoadError::InvalidConfig(
                "Qwen3.5 MoE text_config must include model_type".to_owned(),
            )
        })?;
    if text_model_type != TEXT_MODEL_TYPE {
        return Err(ModelLoadError::InvalidConfig(format!(
            "Qwen3.5 MoE text_config model_type must be {TEXT_MODEL_TYPE}, got \
             {text_model_type}"
        )));
    }
    Ok(text_config.clone())
}

/// Parses and validates either supported Qwen3.5 MoE `config.json` layout.
pub(crate) fn load_qwen3_5_moe_config(
    root: &Path,
    architecture: &str,
    model_type: &str,
) -> Result<LoadedQwen3_5MoeConfig, ModelLoadError> {
    let checkpoint_kind =
        Qwen3_5MoeCheckpointKind::from_dispatch(architecture, model_type)?;

    let path = root.join("config.json");
    let value = parse_json_file(&path)?;
    let text_value = match checkpoint_kind {
        Qwen3_5MoeCheckpointKind::CausalLm => value.clone(),
        Qwen3_5MoeCheckpointKind::ConditionalGeneration => {
            conditional_text_config(&value)?
        }
    };
    let outer_dtype = dtype_metadata(&value)?;
    let text_dtype = dtype_metadata(&text_value)?;
    if let (Some(outer_dtype), Some(text_dtype)) = (outer_dtype, text_dtype)
        && outer_dtype != text_dtype
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "outer config dtype {outer_dtype:?} conflicts with text_config dtype \
             {text_dtype:?}"
        )));
    }
    let declared_dtype = text_dtype.or(outer_dtype);
    let weight_dtype = weight_dtype(declared_dtype)?;
    let supported_dtypes = supported_dtypes_from_torch_dtype(declared_dtype);
    let model: Qwen3_5MoeConfig = serde_json::from_value(text_value)
        .map_err(|source| ModelLoadError::Json(path, source.to_string()))?;
    model.validate()?;

    Ok(LoadedQwen3_5MoeConfig {
        model,
        weight_dtype,
        supported_dtypes,
        checkpoint_kind,
    })
}

/// Validates the requested dtype against checkpoint metadata.
pub(crate) fn validate_qwen3_5_moe_dtype(
    config: &LoadedQwen3_5MoeConfig,
    requested: ModelDType,
) -> Result<(), ModelLoadError> {
    if config.supported_dtypes.contains(&requested) {
        return Ok(());
    }
    Err(ModelLoadError::UnsupportedDType {
        architecture: config.checkpoint_kind.architecture().to_owned(),
        requested,
        supported: config.supported_dtypes.clone(),
    })
}

/// Rejects BF16 on Candle's CPU backend before large weights are allocated.
///
/// CUDA/BF16 passes this construction-time check and is covered by an explicit
/// artifact-backed load/forward regression. Keep backend validation distinct from
/// checkpoint dtype validation so unsupported CPU combinations fail early.
pub(crate) fn validate_qwen3_5_moe_device_dtype(
    device: &Device,
    requested: ModelDType,
) -> Result<(), ModelLoadError> {
    if requested == ModelDType::BF16 && device.is_cpu() {
        return Err(ModelLoadError::UnsupportedDTypeOnDevice {
            requested,
            device: "cpu".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Writes a disposable `config.json` and returns its containing directory.
    fn temporary_config(contents: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen3-5-moe-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("config.json"), contents).unwrap();
        root
    }

    /// Supplies a compact but structurally complete hybrid MoE configuration.
    fn valid_config() -> &'static str {
        r#"{
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "model_type": "qwen3_5_moe_text",
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
        }"#
    }

    /// Confirms modern dtype metadata and every hybrid construction field parse.
    #[test]
    fn parses_hybrid_qwen3_5_moe_config_and_dtype() {
        let root = temporary_config(valid_config());
        let config = load_qwen3_5_moe_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.model.hybrid.num_hidden_layers, 4);
        assert_eq!(config.model.hybrid.layer_types.len(), 4);
        assert_eq!(config.model.num_experts, 8);
        assert_eq!(config.model.num_experts_per_tok, 2);
        assert_eq!(config.weight_dtype, ModelDType::BF16);
        assert_eq!(
            config.model.hybrid.rope_parameters.partial_rotary_factor,
            0.5
        );
        let sparse_moe = config
            .model
            .sparse_moe_config(SparseMoeWeightLayout::Individual);
        assert_eq!(sparse_moe.hidden_size, 16);
        assert_eq!(sparse_moe.expert_intermediate_size, 4);
        assert_eq!(sparse_moe.num_experts, 8);
        assert_eq!(sparse_moe.experts_per_token, 2);
        assert!(!sparse_moe.router_bias);
        assert_eq!(
            sparse_moe.router_score_function,
            RouterScoreFunction::Softmax
        );
        assert!(sparse_moe.normalize_selected_weights);
        let shared_expert = sparse_moe.shared_expert.unwrap();
        assert_eq!(shared_expert.intermediate_size, 4);
        assert_eq!(shared_expert.activation, Activation::Silu);
        let shared_gate = shared_expert.gate.unwrap();
        assert_eq!(shared_gate.activation, Activation::Sigmoid);
        assert!(!shared_gate.bias);
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
        validate_qwen3_5_moe_dtype(&config, ModelDType::BF16).unwrap();
    }

    /// Confirms the production multimodal wrapper delegates to its text config.
    #[test]
    fn parses_conditional_generation_text_config_and_bf16() {
        let text_config: Value = serde_json::from_str(valid_config()).unwrap();
        let contents = serde_json::json!({
            "architectures": [CONDITIONAL_ARCHITECTURE],
            "model_type": CONDITIONAL_MODEL_TYPE,
            "text_config": text_config,
            "vision_config": {
                "model_type": "qwen3_5_moe_vision",
                "hidden_size": 8
            }
        })
        .to_string();
        let root = temporary_config(&contents);

        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            config.checkpoint_kind,
            Qwen3_5MoeCheckpointKind::ConditionalGeneration
        );
        assert_eq!(config.model.hybrid.num_hidden_layers, 4);
        assert_eq!(config.model.num_experts, 8);
        assert_eq!(config.weight_dtype, ModelDType::BF16);
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
        validate_qwen3_5_moe_dtype(&config, ModelDType::BF16).unwrap();
    }

    /// Requires a supported storage dtype so header validation is unambiguous.
    #[test]
    fn rejects_missing_or_unsupported_checkpoint_dtype() {
        let mut config: Value = serde_json::from_str(valid_config()).unwrap();
        config.as_object_mut().unwrap().remove("dtype");
        let root = temporary_config(&config.to_string());
        let error =
            load_qwen3_5_moe_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("must declare dtype"));

        let mut config: Value = serde_json::from_str(valid_config()).unwrap();
        config["dtype"] = Value::String("float16".to_owned());
        let root = temporary_config(&config.to_string());
        let error =
            load_qwen3_5_moe_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("checkpoint dtype"));
        assert!(error.to_string().contains("unsupported"));
    }

    /// Confirms a development model cannot accidentally omit a token-mixer family.
    #[test]
    fn rejects_layer_schedule_without_both_token_mixers() {
        let contents =
            valid_config().replace("\"full_attention\"", "\"linear_attention\"");
        let root = temporary_config(&contents);
        let error =
            load_qwen3_5_moe_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains("requires both linear_attention"));
    }

    /// Confirms Qwen-specific RoPE algorithms are rejected at the config boundary.
    #[test]
    fn rejects_unsupported_qwen_rope_type() {
        let contents = valid_config()
            .replace("\"rope_type\": \"default\"", "\"rope_type\": \"dynamic\"");
        let root = temporary_config(&contents);
        let error =
            load_qwen3_5_moe_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains("supports default RoPE"));
        assert!(error.to_string().contains("dynamic"));
    }

    /// Confirms similarly named Qwen architectures cannot enter this loader.
    #[test]
    fn rejects_inexact_dispatch_key() {
        let root = temporary_config(valid_config());
        let error = load_qwen3_5_moe_config(&root, "Qwen3MoeForCausalLM", "qwen3_moe")
            .unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains(ARCHITECTURE));
        assert!(error.to_string().contains("Qwen3MoeForCausalLM/qwen3_moe"));
    }

    /// Confirms Candle CPU rejects BF16 while retaining its FP32 fallback.
    #[test]
    fn bf16_rejects_candle_cpu_backend() {
        let error = validate_qwen3_5_moe_device_dtype(&Device::Cpu, ModelDType::BF16)
            .unwrap_err();
        assert!(matches!(
            error,
            ModelLoadError::UnsupportedDTypeOnDevice {
                requested: ModelDType::BF16,
                ..
            }
        ));
        validate_qwen3_5_moe_device_dtype(&Device::Cpu, ModelDType::F32).unwrap();
    }
}
