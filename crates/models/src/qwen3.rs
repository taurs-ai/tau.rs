//! Qwen3 model-family loading for the shared inference runtime.

use crate::dtype::{ModelDType, supported_dtypes_from_torch_dtype};
use crate::metadata::{optional_json_string, parse_json_file};
use crate::{ModelLoadError, ModelShard};
use candle::Device;
use candle_nn::VarBuilder;
pub use candle_transformers::models::qwen3::{
    Config as Qwen3Config, ModelForCausalLM as Qwen3ForCausalLM,
};
use std::path::Path;

/// Qwen3 construction config plus runtime dtypes supported by the artifact.
#[derive(Debug)]
pub(crate) struct LoadedQwen3Config {
    pub(crate) model: Qwen3Config,
    pub(crate) supported_dtypes: Vec<ModelDType>,
}

/// Loads the dense Qwen3 config from `config.json`.
///
/// The inputs are the model root and already validated high-level architecture
/// metadata. The function returns the Qwen3 construction config plus the
/// supported runtime dtypes, or `ModelLoadError` when the model is not dense
/// Qwen3 or the config cannot deserialize.
pub(crate) fn load_qwen3_config(
    root: &Path,
    architecture: &str,
    model_type: &str,
) -> Result<LoadedQwen3Config, ModelLoadError> {
    if architecture != "Qwen3ForCausalLM" || model_type != "qwen3" {
        return Err(ModelLoadError::InvalidConfig(format!(
            "generation currently supports dense Qwen3ForCausalLM/qwen3, got \
             {architecture}/{model_type}"
        )));
    }

    let path = root.join("config.json");
    let value = parse_json_file(&path)?;
    let torch_dtype = optional_json_string(&value, "torch_dtype")?;
    let supported_dtypes = supported_dtypes_from_torch_dtype(torch_dtype);
    let model = serde_json::from_value(value)
        .map_err(|source| ModelLoadError::Json(path, source.to_string()))?;

    Ok(LoadedQwen3Config {
        model,
        supported_dtypes,
    })
}

/// Builds the dense Qwen3 causal language model on the selected device.
///
/// The inputs are parsed Qwen3 config, loaded safetensors shard descriptors, and
/// the target device and runtime dtype. The function returns a ready
/// `ModelForCausalLM` or `ModelLoadError` when Candle cannot mmap weights or
/// initialize model layers.
pub(crate) fn load_qwen3_model(
    config: &Qwen3Config,
    shards: &[ModelShard],
    device: &Device,
    dtype: ModelDType,
) -> Result<Qwen3ForCausalLM, ModelLoadError> {
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

    Qwen3ForCausalLM::new(config, var_builder)
        .map_err(|source| ModelLoadError::Candle(source.to_string()))
}

/// Validates the requested dtype against Qwen3-specific config metadata.
pub(crate) fn validate_qwen3_dtype(
    config: &LoadedQwen3Config,
    requested: ModelDType,
) -> Result<(), ModelLoadError> {
    if config.supported_dtypes.contains(&requested) {
        return Ok(());
    }

    Err(ModelLoadError::UnsupportedDType {
        architecture: "Qwen3ForCausalLM".to_owned(),
        requested,
        supported: config.supported_dtypes.clone(),
    })
}

/// Validates dtype support in Candle's selected dense-Qwen3 execution backend.
///
/// Candle 0.11 can construct Qwen3 BF16 state on CPU, but the dense projection
/// matrix multiplications do not support BF16 there. CUDA remains the supported
/// BF16 path; F32 works on both current backends.
pub(crate) fn validate_qwen3_device_dtype(
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

    /// Writes a temporary Qwen3 configuration with the requested dtype declaration.
    fn temporary_qwen3_config(torch_dtype: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen3-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{
                    "architectures": ["Qwen3ForCausalLM"],
                    "model_type": "qwen3",
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "head_dim": 8,
                    "attention_bias": false,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 128,
                    "sliding_window": null,
                    "max_window_layers": 1,
                    "tie_word_embeddings": true,
                    "rope_theta": 1000000.0,
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

    /// Verifies Qwen3 parsing retains dimensions and BF16 support.
    #[test]
    fn qwen3_config_retains_architecture_fields_and_bf16_dtype() {
        let root = temporary_qwen3_config("bfloat16");
        let config = load_qwen3_config(&root, "Qwen3ForCausalLM", "qwen3").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(config.model.hidden_size, 16);
        assert_eq!(config.model.head_dim, 8);
        assert_eq!(config.model.num_attention_heads, 2);
        assert_eq!(config.model.num_key_value_heads, 1);
        assert_eq!(config.model.max_position_embeddings, 128);
        assert_eq!(config.model.sliding_window, None);
        assert!(!config.model.attention_bias);
        assert!(!config.model.use_sliding_window);
        assert_eq!(
            config.supported_dtypes,
            vec![ModelDType::F32, ModelDType::BF16]
        );
        validate_qwen3_dtype(&config, ModelDType::BF16).unwrap();
    }

    /// Verifies Qwen3 rejects BF16 when the checkpoint does not declare it.
    #[test]
    fn qwen3_config_rejects_undeclared_bf16_dtype() {
        let root = temporary_qwen3_config("float32");
        let config = load_qwen3_config(&root, "Qwen3ForCausalLM", "qwen3").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert!(validate_qwen3_dtype(&config, ModelDType::F32).is_ok());
        assert!(matches!(
            validate_qwen3_dtype(&config, ModelDType::BF16),
            Err(ModelLoadError::UnsupportedDType { .. })
        ));
    }

    /// Verifies that other Qwen3-family dispatch keys do not enter the dense loader.
    #[test]
    fn qwen3_config_rejects_other_qwen3_families() {
        let root = temporary_qwen3_config("bfloat16");
        let error =
            load_qwen3_config(&root, "Qwen3MoeForCausalLM", "qwen3_moe").unwrap_err();
        fs::remove_dir_all(&root).unwrap();

        assert!(error.to_string().contains("dense Qwen3ForCausalLM/qwen3"));
        assert!(error.to_string().contains("Qwen3MoeForCausalLM/qwen3_moe"));
    }

    /// Verifies Qwen3 rejects BF16 execution on Candle's CPU backend.
    #[test]
    fn qwen3_bf16_rejects_candle_cpu_backend() {
        let error =
            validate_qwen3_device_dtype(&Device::Cpu, ModelDType::BF16).unwrap_err();

        assert!(matches!(
            error,
            ModelLoadError::UnsupportedDTypeOnDevice {
                requested: ModelDType::BF16,
                ..
            }
        ));
        validate_qwen3_device_dtype(&Device::Cpu, ModelDType::F32).unwrap();
    }
}
