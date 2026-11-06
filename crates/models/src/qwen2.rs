//! Qwen2 model-family loading for the shared inference runtime.

use crate::dtype::{ModelDType, supported_dtypes_from_torch_dtype};
use crate::metadata::{optional_json_string, parse_json_file};
use crate::{ModelLoadError, ModelShard};
use candle::Device;
use candle_nn::VarBuilder;
pub use candle_transformers::models::qwen2::{
    Config as Qwen2Config, ModelForCausalLM as Qwen2ForCausalLM,
};
use std::path::Path;

/// Qwen2 construction config plus runtime dtypes supported by the artifact.
pub(crate) struct LoadedQwen2Config {
    pub(crate) model: Qwen2Config,
    pub(crate) supported_dtypes: Vec<ModelDType>,
}

/// Loads the Qwen2 config from `config.json`.
///
/// The inputs are the model root and already validated high-level architecture
/// metadata. The function returns the Qwen2 construction config plus the
/// supported runtime dtypes, or `ModelLoadError` when the model is not Qwen2 or
/// the config cannot deserialize.
pub(crate) fn load_qwen2_config(
    root: &Path,
    architecture: &str,
    model_type: &str,
) -> Result<LoadedQwen2Config, ModelLoadError> {
    if architecture != "Qwen2ForCausalLM" || model_type != "qwen2" {
        return Err(ModelLoadError::InvalidConfig(format!(
            "generation currently supports Qwen2ForCausalLM/qwen2, got \
             {architecture}/{model_type}"
        )));
    }

    let path = root.join("config.json");
    let value = parse_json_file(&path)?;
    let torch_dtype = optional_json_string(&value, "torch_dtype")?;
    let supported_dtypes = supported_dtypes_from_torch_dtype(torch_dtype);
    let model = serde_json::from_value(value)
        .map_err(|source| ModelLoadError::Json(path, source.to_string()))?;

    Ok(LoadedQwen2Config {
        model,
        supported_dtypes,
    })
}

/// Builds the Qwen2 causal language model on the selected device.
///
/// The inputs are parsed Qwen2 config, loaded safetensors shard descriptors, and
/// the target device and runtime dtype. The function returns a ready
/// `ModelForCausalLM` or `ModelLoadError` when Candle cannot mmap weights or
/// initialize model layers.
pub(crate) fn load_qwen2_model(
    config: &Qwen2Config,
    shards: &[ModelShard],
    device: &Device,
    dtype: ModelDType,
) -> Result<Qwen2ForCausalLM, ModelLoadError> {
    let paths = shards
        .iter()
        .map(|shard| shard.path().to_path_buf())
        .collect::<Vec<_>>();
    let var_builder = {
        // Safety: Candle mmaps read-only safetensors files. The files are not
        // mutated by this process, and the model owns the mapped weight storage it
        // needs.
        unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, dtype.candle_dtype(), device)
        }
    }
    .map_err(|source| ModelLoadError::Candle(source.to_string()))?;

    Qwen2ForCausalLM::new(config, var_builder)
        .map_err(|source| ModelLoadError::Candle(source.to_string()))
}

/// Validates the requested dtype against Qwen2-specific config metadata.
pub(crate) fn validate_qwen2_dtype(
    config: &LoadedQwen2Config,
    requested: ModelDType,
) -> Result<(), ModelLoadError> {
    if config.supported_dtypes.contains(&requested) {
        return Ok(());
    }

    Err(ModelLoadError::UnsupportedDType {
        architecture: "Qwen2ForCausalLM".to_owned(),
        requested,
        supported: config.supported_dtypes.clone(),
    })
}

/// Validates dtype support in Candle's selected Qwen2 execution backend.
///
/// Candle 0.11 cannot execute the BF16 matrix multiplication used while creating
/// Qwen2 RoPE state on CPU. CUDA remains the supported BF16 path; F32 works on
/// both current backends.
pub(crate) fn validate_qwen2_device_dtype(
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
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Writes a temporary Qwen2 configuration with the requested dtype declaration.
    fn temporary_qwen_config(torch_dtype: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{
                    "architectures": ["Qwen2ForCausalLM"],
                    "model_type": "qwen2",
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 64,
                    "sliding_window": 64,
                    "max_window_layers": 1,
                    "tie_word_embeddings": true,
                    "rope_theta": 10000.0,
                    "rms_norm_eps": 0.000001,
                    "use_sliding_window": false,
                    "hidden_act": "silu",
                    "torch_dtype": "{torch_dtype}"
                }}"#
            ),
        )
        .unwrap();
        root
    }

    /// Verifies Qwen2 parsing retains architecture fields and BF16 support.
    #[test]
    fn qwen_config_retains_supported_bf16_dtype() {
        let root = temporary_qwen_config("bfloat16");
        let config = load_qwen2_config(&root, "Qwen2ForCausalLM", "qwen2").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.model.hidden_size, 16);
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
        validate_qwen2_dtype(&config, ModelDType::BF16).unwrap();
    }

    /// Verifies Qwen2 rejects BF16 when the checkpoint does not declare it.
    #[test]
    fn qwen_config_rejects_undeclared_bf16_dtype() {
        let root = temporary_qwen_config("float32");
        let config = load_qwen2_config(&root, "Qwen2ForCausalLM", "qwen2").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert!(validate_qwen2_dtype(&config, ModelDType::F32).is_ok());
        assert!(matches!(
            validate_qwen2_dtype(&config, ModelDType::BF16),
            Err(ModelLoadError::UnsupportedDType { .. })
        ));
    }

    /// Verifies Qwen2 rejects BF16 execution on Candle's CPU backend.
    #[test]
    fn qwen_bf16_rejects_candle_cpu_backend() {
        let error =
            validate_qwen2_device_dtype(&Device::Cpu, ModelDType::BF16).unwrap_err();

        assert!(matches!(
            error,
            ModelLoadError::UnsupportedDTypeOnDevice {
                requested: ModelDType::BF16,
                ..
            }
        ));
        assert!(error.to_string().contains("Candle cpu backend"));
        validate_qwen2_device_dtype(&Device::Cpu, ModelDType::F32).unwrap();
    }
}
