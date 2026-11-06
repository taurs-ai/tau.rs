//! Configuration parsing and validation for dense Qwen3.5 checkpoints.
//!
//! Dense Qwen3.5 is serialized as a conditional-generation wrapper around a
//! nested text decoder. This module validates that exact wrapper contract and
//! every configuration choice consumed by tau.rs's text-only runtime before
//! safetensors are materialized on the selected device.

use crate::ModelLoadError;
use crate::dtype::{ModelDType, supported_dtypes_from_torch_dtype};
use crate::metadata::{optional_json_string, parse_json_file};
use crate::qwen3_5_common::{HybridLayerType, HybridTextConfig};
use candle::Device;
use candle_nn::Activation;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

pub(crate) const ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
pub(crate) const MODEL_TYPE: &str = "qwen3_5";
pub(crate) const TEXT_MODEL_TYPE: &str = "qwen3_5_text";

/// Construction fields required by the dense Qwen3.5 text decoder.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(super) struct Qwen3_5Config {
    #[serde(flatten)]
    pub(super) hybrid: HybridTextConfig,
    pub(super) intermediate_size: usize,
    #[serde(default)]
    pub(super) partial_rotary_factor: Option<f64>,
    pub(super) full_attention_interval: usize,
    pub(super) attn_output_gate: bool,
    pub(super) mamba_ssm_dtype: String,
}

impl Qwen3_5Config {
    /// Adds dense-only requirements to the shared hybrid config validation.
    fn validate(&self) -> Result<(), ModelLoadError> {
        if self.intermediate_size == 0 {
            return Err(ModelLoadError::InvalidConfig(
                "dense Qwen3.5 dimensions and vocabulary must be non-zero".to_owned(),
            ));
        }
        self.hybrid.validate("dense Qwen3.5")?;
        if self.full_attention_interval == 0 {
            return Err(ModelLoadError::InvalidConfig(
                "dense Qwen3.5 full_attention_interval must be non-zero".to_owned(),
            ));
        }
        for (layer_index, &layer_type) in self.hybrid.layer_types.iter().enumerate() {
            let expected = if (layer_index + 1) % self.full_attention_interval == 0 {
                HybridLayerType::FullAttention
            } else {
                HybridLayerType::LinearAttention
            };
            if layer_type != expected {
                return Err(ModelLoadError::InvalidConfig(format!(
                    "dense Qwen3.5 layer {layer_index} is {layer_type:?}, expected \
                     {expected:?} for full_attention_interval {}",
                    self.full_attention_interval
                )));
            }
        }
        if self.hybrid.hidden_act != Activation::Silu {
            return Err(ModelLoadError::InvalidConfig(format!(
                "dense Qwen3.5 currently supports hidden_act=silu, got {:?}",
                self.hybrid.hidden_act
            )));
        }
        if !self.attn_output_gate {
            return Err(ModelLoadError::InvalidConfig(
                "dense Qwen3.5 requires attn_output_gate=true".to_owned(),
            ));
        }
        if self.mamba_ssm_dtype != "float32" {
            return Err(ModelLoadError::InvalidConfig(format!(
                "dense Qwen3.5 requires mamba_ssm_dtype=\"float32\", got {:?}",
                self.mamba_ssm_dtype
            )));
        }
        if let Some(partial_rotary_factor) = self.partial_rotary_factor
            && partial_rotary_factor
                != self.hybrid.rope_parameters.partial_rotary_factor
        {
            return Err(ModelLoadError::InvalidConfig(format!(
                "dense Qwen3.5 partial_rotary_factor {partial_rotary_factor} \
                 conflicts with rope_parameters.partial_rotary_factor {}",
                self.hybrid.rope_parameters.partial_rotary_factor
            )));
        }
        Ok(())
    }
}

/// Parsed dense construction config plus checkpoint and runtime dtype contracts.
#[derive(Debug)]
pub(crate) struct LoadedQwen3_5Config {
    pub(super) model: Qwen3_5Config,
    /// Declared bulk storage dtype; tensor contracts permit specific FP32 weights.
    pub(super) weight_dtype: ModelDType,
    pub(crate) supported_dtypes: Vec<ModelDType>,
}

impl LoadedQwen3_5Config {
    /// Returns the maximum text sequence length accepted by the decoder.
    pub(crate) fn context_window(&self) -> usize {
        self.model.hybrid.max_position_embeddings
    }
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
            "dense Qwen3.5 checkpoint dtype {dtype:?} is unsupported; expected \
             bfloat16 or float32"
        ))),
        None => Err(ModelLoadError::InvalidConfig(
            "dense Qwen3.5 config must declare dtype or torch_dtype".to_owned(),
        )),
    }
}

/// Extracts an optional boolean while preserving a useful configuration error.
fn optional_json_bool(
    value: &Value,
    field: &str,
) -> Result<Option<bool>, ModelLoadError> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };
    value.as_bool().map(Some).ok_or_else(|| {
        ModelLoadError::InvalidConfig(format!(
            "config field {field:?} must be a boolean"
        ))
    })
}

/// Returns the nested text config and verifies its architecture-specific type.
fn conditional_text_config(value: &Value) -> Result<&Value, ModelLoadError> {
    let text_config = value.get("text_config").ok_or_else(|| {
        ModelLoadError::InvalidConfig(format!(
            "{ARCHITECTURE} config must include text_config"
        ))
    })?;
    let text_model_type = text_config
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ModelLoadError::InvalidConfig(
                "dense Qwen3.5 text_config must include model_type".to_owned(),
            )
        })?;
    if text_model_type != TEXT_MODEL_TYPE {
        return Err(ModelLoadError::InvalidConfig(format!(
            "dense Qwen3.5 text_config model_type must be {TEXT_MODEL_TYPE}, got \
             {text_model_type}"
        )));
    }
    Ok(text_config)
}

/// Parses and validates the exact dense conditional-generation config layout.
pub(crate) fn load_qwen3_5_config(
    root: &Path,
    architecture: &str,
    model_type: &str,
) -> Result<LoadedQwen3_5Config, ModelLoadError> {
    if (architecture, model_type) != (ARCHITECTURE, MODEL_TYPE) {
        return Err(ModelLoadError::InvalidConfig(format!(
            "dense model construction supports {ARCHITECTURE}/{MODEL_TYPE}, got \
             {architecture}/{model_type}"
        )));
    }

    let path = root.join("config.json");
    let value = parse_json_file(&path)?;
    let text_value = conditional_text_config(&value)?;
    let outer_dtype = dtype_metadata(&value)?;
    let text_dtype = dtype_metadata(text_value)?;
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
    let model: Qwen3_5Config = serde_json::from_value(text_value.clone())
        .map_err(|source| ModelLoadError::Json(path, source.to_string()))?;

    if let Some(outer_tie_word_embeddings) =
        optional_json_bool(&value, "tie_word_embeddings")?
        && outer_tie_word_embeddings != model.hybrid.tie_word_embeddings
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "outer config tie_word_embeddings={outer_tie_word_embeddings} conflicts \
             with text_config tie_word_embeddings={}",
            model.hybrid.tie_word_embeddings
        )));
    }
    model.validate()?;

    Ok(LoadedQwen3_5Config {
        model,
        weight_dtype,
        supported_dtypes,
    })
}

/// Validates the requested runtime dtype against checkpoint metadata.
pub(crate) fn validate_qwen3_5_dtype(
    config: &LoadedQwen3_5Config,
    requested: ModelDType,
) -> Result<(), ModelLoadError> {
    if config.supported_dtypes.contains(&requested) {
        return Ok(());
    }
    Err(ModelLoadError::UnsupportedDType {
        architecture: ARCHITECTURE.to_owned(),
        requested,
        supported: config.supported_dtypes.clone(),
    })
}

/// Rejects BF16 on Candle CPU before any dense model weights are allocated.
pub(crate) fn validate_qwen3_5_device_dtype(
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

    /// Writes a temporary dense Qwen3.5 configuration for parser tests.
    fn temporary_config(contents: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen3-5-dense-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("config.json"), contents).unwrap();
        root
    }

    /// Builds a valid nested text configuration used as the test baseline.
    fn valid_text_config() -> Value {
        serde_json::json!({
            "model_type": TEXT_MODEL_TYPE,
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
                "partial_rotary_factor": 0.5,
                "mrope_interleaved": true,
                "mrope_section": [1, 1]
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
        })
    }

    /// Serializes a valid conditional-generation wrapper configuration.
    fn valid_config() -> String {
        serde_json::json!({
            "architectures": [ARCHITECTURE],
            "model_type": MODEL_TYPE,
            "tie_word_embeddings": false,
            "text_config": valid_text_config(),
            "vision_config": { "model_type": "qwen3_5_vision" }
        })
        .to_string()
    }

    /// Produces a valid wrapper with one nested text field replaced.
    fn config_with_text_change(field: &str, value: Value) -> String {
        let mut outer: Value = serde_json::from_str(&valid_config()).unwrap();
        outer["text_config"][field] = value;
        outer.to_string()
    }

    /// Verifies exact dense wrapper dispatch and nested field parsing.
    #[test]
    fn parses_exact_dense_nested_config_and_dtype() {
        let root = temporary_config(&valid_config());
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.model.hybrid.hidden_size, 16);
        assert_eq!(config.model.intermediate_size, 32);
        assert_eq!(config.model.hybrid.num_hidden_layers, 4);
        assert_eq!(config.weight_dtype, ModelDType::BF16);
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
        validate_qwen3_5_dtype(&config, ModelDType::F32).unwrap();
        validate_qwen3_5_dtype(&config, ModelDType::BF16).unwrap();
    }

    /// Verifies that approximate dense architecture/type keys are rejected.
    #[test]
    fn rejects_inexact_dense_dispatch_keys() {
        let root = temporary_config(&valid_config());
        for (architecture, model_type) in [
            ("Qwen3_5ForCausalLM", "qwen3_5_text"),
            ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe"),
            (ARCHITECTURE, "qwen3_5_moe"),
        ] {
            let error =
                load_qwen3_5_config(&root, architecture, model_type).unwrap_err();
            assert!(error.to_string().contains(ARCHITECTURE));
            assert!(error.to_string().contains(architecture));
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// Verifies that the nested text model type must match Qwen3.5.
    #[test]
    fn rejects_wrong_nested_text_model_type() {
        let contents = config_with_text_change(
            "model_type",
            Value::String("qwen3_5_moe_text".to_owned()),
        );
        let root = temporary_config(&contents);
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains(TEXT_MODEL_TYPE));
        assert!(error.to_string().contains("qwen3_5_moe_text"));
    }

    /// Verifies that conflicting wrapper and text metadata is rejected.
    #[test]
    fn rejects_conflicting_wrapper_metadata() {
        let mut outer: Value = serde_json::from_str(&valid_config()).unwrap();
        outer["dtype"] = Value::String("float32".to_owned());
        let root = temporary_config(&outer.to_string());
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(
            error
                .to_string()
                .contains("conflicts with text_config dtype")
        );

        let mut outer: Value = serde_json::from_str(&valid_config()).unwrap();
        outer["tie_word_embeddings"] = Value::Bool(true);
        let root = temporary_config(&outer.to_string());
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("tie_word_embeddings"));
    }

    /// Verifies rejection of unsupported text-execution configuration choices.
    #[test]
    fn rejects_unsupported_text_execution_choices() {
        for (field, value, expected) in [
            ("intermediate_size", serde_json::json!(0), "non-zero"),
            (
                "attention_dropout",
                serde_json::json!(0.1),
                "attention_dropout=0",
            ),
            ("use_cache", serde_json::json!(false), "use_cache=true"),
            (
                "attn_output_gate",
                serde_json::json!(false),
                "attn_output_gate=true",
            ),
            (
                "mamba_ssm_dtype",
                serde_json::json!("bfloat16"),
                "mamba_ssm_dtype",
            ),
            ("hidden_act", serde_json::json!("relu"), "hidden_act=silu"),
        ] {
            let root = temporary_config(&config_with_text_change(field, value));
            let error =
                load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
            fs::remove_dir_all(&root).unwrap();
            assert!(
                error.to_string().contains(expected),
                "unexpected error for {field}: {error}"
            );
        }
    }

    /// Verifies schedule, layer count, and RoPE consistency checks.
    #[test]
    fn rejects_inconsistent_layer_and_rope_metadata() {
        let root = temporary_config(&config_with_text_change(
            "layer_types",
            serde_json::json!([
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention"
            ]),
        ));
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("full_attention_interval"));

        let root = temporary_config(&config_with_text_change(
            "partial_rotary_factor",
            serde_json::json!(0.25),
        ));
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("conflicts with rope_parameters"));
    }

    /// Verifies rejection of absent and unsupported checkpoint dtypes.
    #[test]
    fn rejects_missing_or_unsupported_checkpoint_dtype() {
        let mut outer: Value = serde_json::from_str(&valid_config()).unwrap();
        outer["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("dtype");
        let root = temporary_config(&outer.to_string());
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("must declare dtype"));

        let root = temporary_config(&config_with_text_change(
            "dtype",
            serde_json::json!("float16"),
        ));
        let error = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap_err();
        fs::remove_dir_all(&root).unwrap();
        assert!(error.to_string().contains("checkpoint dtype"));
    }

    /// Verifies BF16 dense execution is rejected on Candle's CPU backend.
    #[test]
    fn bf16_rejects_candle_cpu_backend() {
        let error =
            validate_qwen3_5_device_dtype(&Device::Cpu, ModelDType::BF16).unwrap_err();
        assert!(matches!(
            error,
            ModelLoadError::UnsupportedDTypeOnDevice {
                requested: ModelDType::BF16,
                ..
            }
        ));
        validate_qwen3_5_device_dtype(&Device::Cpu, ModelDType::F32).unwrap();
    }
}
