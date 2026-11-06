//! Hugging Face-style model metadata loading and validation.

use crate::ModelLoadError;
use crate::dtype::{ModelDType, supported_dtypes_from_torch_dtype};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub architecture: String,
    pub model_type: String,
    pub vocab_size: Option<usize>,
    pub torch_dtype: Option<String>,
    pub supported_dtypes: Vec<ModelDType>,
}

/// Loads and validates `config.json` from a model directory.
///
/// The input is the model root path. The function returns selected config fields
/// needed to identify the model or `ModelLoadError` for missing and malformed
/// configuration.
pub fn load_model_config(root: &Path) -> Result<ModelConfig, ModelLoadError> {
    let path = root.join("config.json");
    let value = parse_json_file(&path)?;
    let architecture = value
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ModelLoadError::InvalidConfig(
                "config.json must include architectures[0]".to_owned(),
            )
        })?
        .to_owned();
    let model_type = required_json_string(&value, "model_type")?.to_owned();
    let text_config = value.get("text_config");
    let outer_vocab_size = optional_json_usize(&value, "vocab_size")?;
    let text_vocab_size = text_config
        .map(|text_config| optional_json_usize(text_config, "vocab_size"))
        .transpose()?
        .flatten();
    if let (Some(outer), Some(text)) = (outer_vocab_size, text_vocab_size)
        && outer != text
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "outer config vocab_size {outer} conflicts with text_config vocab_size \
             {text}"
        )));
    }
    let vocab_size = text_vocab_size.or(outer_vocab_size);
    let outer_dtype = dtype_metadata(&value)?;
    let text_dtype = text_config.map(dtype_metadata).transpose()?.flatten();
    if let (Some(outer), Some(text)) = (outer_dtype, text_dtype)
        && outer != text
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "outer config dtype {outer:?} conflicts with text_config dtype {text:?}"
        )));
    }
    // Transformers 5 writes `dtype`; older checkpoints use `torch_dtype`.
    // Multimodal wrappers may place both dtype and vocabulary metadata inside
    // `text_config`, so resolve that scope before family dispatch as well.
    let torch_dtype = text_dtype.or(outer_dtype).map(str::to_owned);
    let supported_dtypes = supported_dtypes_from_torch_dtype(torch_dtype.as_deref());

    Ok(ModelConfig {
        architecture,
        model_type,
        vocab_size,
        torch_dtype,
        supported_dtypes,
    })
}

/// Resolves `torch_dtype` and Transformers 5's `dtype` spelling in one object.
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

/// Validates that a model artifact supports the requested runtime dtype.
///
/// The generic model config supplies the declared architecture and its derived
/// dtype support. The function returns the requested dtype unchanged when it is
/// supported, or a structured startup error listing the available dtypes.
pub(crate) fn validate_model_dtype(
    config: &ModelConfig,
    requested: ModelDType,
) -> Result<(), ModelLoadError> {
    if config.supported_dtypes.contains(&requested) {
        return Ok(());
    }

    Err(ModelLoadError::UnsupportedDType {
        architecture: config.architecture.clone(),
        requested,
        supported: config.supported_dtypes.clone(),
    })
}

/// Validates config and tokenizer consistency.
///
/// The inputs come from config and tokenizer loading. The function returns
/// `Ok(())` when the pieces agree enough for future inference, or
/// `ModelLoadError::InvalidConfig` for inconsistent metadata.
pub(crate) fn validate_model_components(
    config: &ModelConfig,
    tokenizer_vocab_size: usize,
) -> Result<(), ModelLoadError> {
    if let Some(config_vocab_size) = config.vocab_size
        && tokenizer_vocab_size > config_vocab_size
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "tokenizer size {tokenizer_vocab_size} exceeds config vocab_size \
             {config_vocab_size}"
        )));
    }

    if let Some(config_vocab_size) = config.vocab_size
        && config_vocab_size == 0
    {
        return Err(ModelLoadError::InvalidConfig(format!(
            "config vocab_size {config_vocab_size} is not usable with tokenizer \
             {tokenizer_vocab_size}"
        )));
    }

    Ok(())
}

/// Parses a JSON file using serde_json.
///
/// The input path must exist and contain a complete JSON document. The function
/// returns the parsed value or `ModelLoadError` for missing files, I/O failures, or
/// malformed JSON.
pub(crate) fn parse_json_file(path: &Path) -> Result<Value, ModelLoadError> {
    require_file(path)?;
    let content = fs::read_to_string(path)
        .map_err(|source| ModelLoadError::Io(path.to_path_buf(), source))?;
    serde_json::from_str(&content)
        .map_err(|source| ModelLoadError::Json(path.to_path_buf(), source.to_string()))
}

/// Ensures a required model component exists as a file.
///
/// The input path is checked with filesystem metadata. The function returns
/// `Ok(())` for files and `ModelLoadError::MissingFile` for absent paths or
/// directories.
pub(crate) fn require_file(path: &Path) -> Result<(), ModelLoadError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(ModelLoadError::MissingFile(path.to_path_buf()))
    }
}

/// Extracts an optional non-negative integer field from a JSON object.
///
/// The input value must be an object when the field exists. The function returns
/// `None` for absent fields, `Some(usize)` for integer-valued numbers, or
/// `ModelLoadError::InvalidConfig` for unsupported values.
pub(crate) fn optional_json_usize(
    value: &Value,
    field: &str,
) -> Result<Option<usize>, ModelLoadError> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };

    if let Some(number) = value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
    {
        Ok(Some(number))
    } else {
        Err(ModelLoadError::InvalidConfig(format!(
            "config field {field:?} must be a non-negative integer"
        )))
    }
}

/// Extracts a required string field from a JSON object.
///
/// The input value must be an object with a string member named by `field`. The
/// function returns the borrowed string or `ModelLoadError::InvalidConfig` when the
/// field is absent or not a string.
fn required_json_string<'a>(
    value: &'a Value,
    field: &str,
) -> Result<&'a str, ModelLoadError> {
    value.get(field).and_then(Value::as_str).ok_or_else(|| {
        ModelLoadError::InvalidConfig(format!(
            "config.json must include string field {field:?}"
        ))
    })
}

/// Extracts an optional string field from a JSON object.
///
/// The function returns `None` when the field is absent, the borrowed string when
/// present, or `ModelLoadError::InvalidConfig` when a non-string value is supplied.
pub(crate) fn optional_json_string<'a>(
    value: &'a Value,
    field: &str,
) -> Result<Option<&'a str>, ModelLoadError> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };

    value.as_str().map(Some).ok_or_else(|| {
        ModelLoadError::InvalidConfig(format!(
            "config field {field:?} must be a string"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Writes one temporary model configuration for metadata parser tests.
    fn temporary_model_config(config: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-model-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("config.json"), config).unwrap();
        root
    }

    /// Verifies that a BF16 declaration adds BF16 to the supported dtype set.
    #[test]
    fn loads_bf16_as_a_supported_model_dtype() {
        let root = temporary_model_config(
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "torch_dtype": "bfloat16"
            }"#,
        );

        let config = load_model_config(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
        validate_model_dtype(&config, ModelDType::F32).unwrap();
        validate_model_dtype(&config, ModelDType::BF16).unwrap();
    }

    /// Verifies the Transformers 5 `dtype` spelling for checkpoint precision.
    #[test]
    fn loads_transformers_five_dtype_metadata() {
        let root = temporary_model_config(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "dtype": "bfloat16"
            }"#,
        );

        let config = load_model_config(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
    }

    /// Verifies dtype discovery inside conditional-generation text configuration.
    #[test]
    fn loads_nested_conditional_generation_metadata() {
        let root = temporary_model_config(
            r#"{
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "vocab_size": 248320,
                    "dtype": "bfloat16"
                }
            }"#,
        );

        let config = load_model_config(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.vocab_size, Some(248_320));
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
    }

    /// Verifies rejection when equivalent dtype fields disagree.
    #[test]
    fn rejects_conflicting_dtype_spellings() {
        let root = temporary_model_config(
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "dtype": "bfloat16",
                "torch_dtype": "float32"
            }"#,
        );

        let error = load_model_config(&root).unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains("conflicts with torch_dtype"));
    }

    /// Verifies that undeclared BF16 execution is rejected.
    #[test]
    fn rejects_bf16_when_the_model_does_not_declare_it() {
        let root = temporary_model_config(
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "torch_dtype": "float32"
            }"#,
        );

        let config = load_model_config(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();
        let error = validate_model_dtype(&config, ModelDType::BF16).unwrap_err();

        assert!(matches!(
            error,
            ModelLoadError::UnsupportedDType {
                requested: ModelDType::BF16,
                ..
            }
        ));
        assert!(error.to_string().contains("supported dtypes: f32"));
    }

    /// Verifies that non-string dtype metadata produces a configuration error.
    #[test]
    fn rejects_non_string_torch_dtype_metadata() {
        let root = temporary_model_config(
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "torch_dtype": 16
            }"#,
        );

        let error = load_model_config(&root).unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains("torch_dtype"));
        assert!(error.to_string().contains("must be a string"));
    }
}
