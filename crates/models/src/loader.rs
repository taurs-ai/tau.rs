//! Model loader.

//! This module is the model-family dispatch point for Hugging Face-style model
//! directories. It reads `config.json`, decides which Rust model implementation
//! matches the declared architecture, and loads the companion runtime inputs that
//! every generated-text engine needs.
//!
//! The dispatch key is model behavior, not the weight container. For example,
//! Qwen2, Llama, and Gemma may all use safetensors files, but they need different
//! model structs, config parsing, and prompt handling.

use crate::dtype::ModelDType;
use crate::generation_config::{GenerationConfig, load_generation_config};
use crate::metadata::{
    ModelConfig, load_model_config, validate_model_components, validate_model_dtype,
};
use crate::qwen2::{
    Qwen2ForCausalLM, load_qwen2_config, load_qwen2_model, validate_qwen2_device_dtype,
    validate_qwen2_dtype,
};
use crate::qwen3::{
    Qwen3ForCausalLM, load_qwen3_config, load_qwen3_model, validate_qwen3_device_dtype,
    validate_qwen3_dtype,
};
use crate::qwen3_5::{
    load_qwen3_5_config, load_qwen3_5_model, validate_qwen3_5_device_dtype,
    validate_qwen3_5_dtype, validate_qwen3_5_tensor_contract,
};
use crate::qwen3_5_moe::{
    load_qwen3_5_moe_config, load_qwen3_5_moe_model, validate_qwen3_5_moe_device_dtype,
    validate_qwen3_5_moe_dtype, validate_qwen3_5_moe_tensor_contract,
};
use crate::safetensors::{load_safetensors_shards, validate_safetensors_shards};
use crate::tokenizer::{load_tokenizer, validate_tokenizer};
use crate::{ModelLoadError, ModelLoadOptions, ModelShard};
use candle::Device;
use std::path::Path;
use tokenizers::Tokenizer;

/// Fully loaded runtime inputs for a supported model family.
///
/// Each variant contains the initialized family-specific model plus the data needed
/// to run text generation. Keeping the family tag at this boundary lets the public
/// engine factory construct the matching chat engine without detecting the model a
/// second time.
pub enum LoadedModel {
    Qwen2(LoadedCausalModel<Qwen2ForCausalLM>),
    Qwen3(LoadedCausalModel<Qwen3ForCausalLM>),
    Qwen3_5(LoadedCausalModel<crate::Qwen3_5Model>),
    Qwen3_5Moe(LoadedCausalModel<crate::Qwen3_5MoeModel>),
    // Llama(LoadedLlamaModel),
    // Gemma(LoadedGemmaModel),
}

/// Inference-time state shared with a causal engine after model construction.
///
/// The initialized model remains family-specific while tokenizer, generation,
/// context, and dtype state have the same runtime contract for every current
/// causal model. Construction-only config and shard maps are deliberately absent.
/// Engine options stay at engine construction and do not travel through this bundle.
pub struct LoadedCausalModel<M> {
    pub model: M,
    pub tokenizer: Tokenizer,
    pub generation_config: GenerationConfig,
    pub context_window: usize,
    pub dtype: ModelDType,
}

/// Model-directory inputs shared by every current causal-language-model family.
///
/// The tokenizer and generation config move into the family-specific runtime
/// bundle. Safetensors shard descriptors remain construction-only and are dropped
/// after the Candle model has been initialized.
struct SharedModelComponents {
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    shards: Vec<ModelShard>,
}

impl SharedModelComponents {
    /// Finishes a family-specific model load with the shared runtime inputs.
    fn into_loaded<M>(
        self,
        model: M,
        context_window: usize,
        dtype: ModelDType,
    ) -> LoadedCausalModel<M> {
        LoadedCausalModel {
            model,
            tokenizer: self.tokenizer,
            generation_config: self.generation_config,
            context_window,
            dtype,
        }
    }
}

/// Loads and cross-validates model artifacts that are independent of architecture.
fn load_shared_model_components(
    root: &Path,
    config: &ModelConfig,
) -> Result<SharedModelComponents, ModelLoadError> {
    let generation_config = load_generation_config(root)?;
    let tokenizer = load_tokenizer(root)?;
    let tokenizer_vocab_size = validate_tokenizer(&tokenizer)?;
    let shards = load_safetensors_shards(root)?;
    validate_safetensors_shards(&shards)?;
    validate_model_components(config, tokenizer_vocab_size)?;

    Ok(SharedModelComponents {
        tokenizer,
        generation_config,
        shards,
    })
}

/// Loads a supported model directory into a typed runtime bundle.
///
/// The model root must contain a valid `config.json`, tokenizer, and at least
/// one safetensors shard. Generation settings come from `generation_config.json`
/// when present, otherwise from `config.json`. The function first reads generic
/// metadata, then dispatches to a model-family-specific loader for the actual
/// Candle model construction on the caller-selected device. File-backed shard maps
/// stay local to this function and are released before the loaded bundle is
/// returned.
pub fn load_model(
    root: &Path,
    device: &Device,
    dtype: ModelDType,
) -> Result<LoadedModel, ModelLoadError> {
    load_model_with_options(root, device, dtype, ModelLoadOptions::default())
}

/// Loads a model with explicit runtime kernel-selection options.
pub fn load_model_with_options(
    root: &Path,
    device: &Device,
    dtype: ModelDType,
    options: ModelLoadOptions,
) -> Result<LoadedModel, ModelLoadError> {
    if !root.is_dir() {
        return Err(ModelLoadError::InvalidModelDir(root.to_path_buf()));
    }

    let config = load_model_config(root)?;
    if (options.moe_prefill_config().is_some()
        || options.moe_decode_options().is_some())
        && (!device.is_cuda()
            || dtype != ModelDType::BF16
            || !options.optimize_kernel()
            || !matches!(
                config.model_type.as_str(),
                "qwen3_5_moe" | "qwen3_5_moe_text"
            ))
    {
        return Err(ModelLoadError::InvalidConfig(
            "explicit MoE configuration requires CUDA/BF16 Qwen3.5/3.6 MoE and optimized kernels".to_owned()));
    }
    validate_model_dtype(&config, dtype)?;
    match (config.architecture.as_str(), config.model_type.as_str()) {
        ("Qwen2ForCausalLM", "qwen2") => {
            // Qwen2 needs Candle's full Qwen config to build the layer graph. Once
            // the model exists, inference only needs the context limit derived
            // from that config.
            let qwen_config =
                load_qwen2_config(root, &config.architecture, &config.model_type)?;
            validate_qwen2_dtype(&qwen_config, dtype)?;
            validate_qwen2_device_dtype(device, dtype)?;
            let components = load_shared_model_components(root, &config)?;
            let model = load_qwen2_model(
                &qwen_config.model,
                &components.shards,
                device,
                dtype,
            )?;
            let context_window = qwen_config.model.max_position_embeddings;

            Ok(LoadedModel::Qwen2(components.into_loaded(
                model,
                context_window,
                dtype,
            )))
        }
        ("Qwen3ForCausalLM", "qwen3") => {
            // Qwen3 has a distinct construction config: notably explicit head
            // dimensions and per-head Q/K normalization used by Candle's dense
            // Qwen3 implementation.
            let qwen_config =
                load_qwen3_config(root, &config.architecture, &config.model_type)?;
            validate_qwen3_dtype(&qwen_config, dtype)?;
            validate_qwen3_device_dtype(device, dtype)?;
            let components = load_shared_model_components(root, &config)?;
            let model = load_qwen3_model(
                &qwen_config.model,
                &components.shards,
                device,
                dtype,
            )?;
            let context_window = qwen_config.model.max_position_embeddings;

            Ok(LoadedModel::Qwen3(components.into_loaded(
                model,
                context_window,
                dtype,
            )))
        }
        ("Qwen3_5ForConditionalGeneration", "qwen3_5") => {
            // Keep this branch explicit so near-matching Qwen3.5 MoE layouts
            // cannot enter the dense constructor.
            let qwen_config =
                load_qwen3_5_config(root, &config.architecture, &config.model_type)?;
            validate_qwen3_5_dtype(&qwen_config, dtype)?;
            validate_qwen3_5_device_dtype(device, dtype)?;
            let components = load_shared_model_components(root, &config)?;
            validate_qwen3_5_tensor_contract(&qwen_config, &components.shards)?;
            let model = load_qwen3_5_model(
                &qwen_config,
                &components.shards,
                device,
                dtype,
                options,
            )?;
            let context_window = qwen_config.context_window();

            Ok(LoadedModel::Qwen3_5(components.into_loaded(
                crate::Qwen3_5Model::from_inner(model),
                context_window,
                dtype,
            )))
        }
        ("Qwen3_5MoeForCausalLM", "qwen3_5_moe_text")
        | ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe") => {
            let qwen_config = load_qwen3_5_moe_config(
                root,
                &config.architecture,
                &config.model_type,
            )?;
            validate_qwen3_5_moe_dtype(&qwen_config, dtype)?;
            validate_qwen3_5_moe_device_dtype(device, dtype)?;
            let components = load_shared_model_components(root, &config)?;
            validate_qwen3_5_moe_tensor_contract(&qwen_config, &components.shards)?;
            let model = load_qwen3_5_moe_model(
                &qwen_config,
                &components.shards,
                device,
                dtype,
                options,
            )?;
            let context_window = qwen_config.model.hybrid.max_position_embeddings;

            Ok(LoadedModel::Qwen3_5Moe(components.into_loaded(
                crate::Qwen3_5MoeModel::from_inner(model),
                context_window,
                dtype,
            )))
        }
        (architecture, model_type) => Err(ModelLoadError::InvalidConfig(format!(
            "unsupported model architecture/type: {architecture}/{model_type}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const QWEN3_TEST_MODEL: &str = "/mnt/shared/models/Qwen3-0.6B";

    /// Resolves the local Qwen3 artifact used by ignored loader tests.
    fn qwen3_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(QWEN3_TEST_MODEL))
    }

    /// Resolves the generated Qwen3.5 MoE artifact used by loader tests.
    fn qwen3_5_moe_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_MOE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.5-MoE-Tiny-Dev")
            })
    }

    /// Resolves the generated dense Qwen3.5 artifact used by loader tests.
    fn qwen3_5_dense_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_DENSE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.8-Dense-Tiny-Dev")
            })
    }

    /// Verifies full dense-Qwen3 dispatch and model construction against the
    /// shared BF16 artifact without running generation.
    #[test]
    #[ignore = "loads a local 1.5 GB model from /mnt/shared"]
    fn loads_shared_qwen3_model() {
        let path = qwen3_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3 validation model is absent: {}",
            path.display()
        );

        let loaded =
            load_model(&path, &Device::Cpu, ModelDType::F32).unwrap_or_else(|error| {
                panic!(
                    "failed to load Qwen3 model from {}: {error}",
                    path.display()
                )
            });
        let LoadedModel::Qwen3(loaded) = loaded else {
            panic!("Qwen3 config dispatched to the wrong loaded-model variant")
        };

        assert_eq!(loaded.context_window, 40_960);
        assert_eq!(loaded.dtype, ModelDType::F32);
        assert_eq!(loaded.tokenizer.token_to_id("<|im_start|>"), Some(151_644));
        assert_eq!(loaded.tokenizer.token_to_id("<|im_end|>"), Some(151_645));
        assert!(loaded.generation_config.is_eos_token(151_643));
        assert!(loaded.generation_config.is_eos_token(151_645));
        drop(loaded.model);
    }

    /// Verifies exact Qwen3.5 MoE dispatch, shared components, and model creation.
    #[test]
    #[ignore = "loads the local generated Qwen3.5 MoE development model"]
    fn loads_generated_qwen3_5_moe_model() {
        let path = qwen3_5_moe_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3.5 MoE validation model is absent: {}",
            path.display()
        );

        let loaded =
            load_model(&path, &Device::Cpu, ModelDType::F32).unwrap_or_else(|error| {
                panic!(
                    "failed to load Qwen3.5 MoE model from {}: {error}",
                    path.display()
                )
            });
        let LoadedModel::Qwen3_5Moe(loaded) = loaded else {
            panic!("Qwen3.5 MoE config dispatched to the wrong model variant")
        };

        assert_eq!(loaded.context_window, 4096);
        assert_eq!(loaded.dtype, ModelDType::F32);
        assert_eq!(loaded.tokenizer.token_to_id("<|im_start|>"), Some(248_045));
        assert_eq!(loaded.tokenizer.token_to_id("<|im_end|>"), Some(248_046));
        assert!(loaded.generation_config.is_eos_token(248_044));
        drop(loaded.model);
    }

    /// Confirms the nested fixture declares BF16 before CPU backend rejection.
    #[test]
    #[ignore = "reads the local generated Qwen3.5 MoE development model config"]
    fn recognizes_generated_qwen3_5_moe_bf16_support() {
        let path = qwen3_5_moe_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3.5 MoE validation model is absent: {}",
            path.display()
        );

        let error = match load_model(&path, &Device::Cpu, ModelDType::BF16) {
            Ok(_) => panic!("Qwen3.5 MoE unexpectedly executed BF16 on CPU"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            ModelLoadError::UnsupportedDTypeOnDevice {
                requested: ModelDType::BF16,
                ..
            }
        ));
    }

    /// Loads the complete dense text model through exact shared-loader dispatch.
    #[test]
    #[ignore = "loads the local generated Qwen3.8 dense development model"]
    fn loads_generated_qwen3_5_dense_model() {
        let path = qwen3_5_dense_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3.5 dense validation model is absent: {}",
            path.display()
        );

        let loaded =
            load_model(&path, &Device::Cpu, ModelDType::F32).unwrap_or_else(|error| {
                panic!(
                    "failed to load dense Qwen3.5 model from {}: {error}",
                    path.display()
                )
            });
        let LoadedModel::Qwen3_5(loaded) = loaded else {
            panic!("dense Qwen3.5 config dispatched to the wrong model variant")
        };

        assert_eq!(loaded.context_window, 4096);
        assert_eq!(loaded.dtype, ModelDType::F32);
        assert_eq!(loaded.tokenizer.token_to_id("<|im_start|>"), Some(248_045));
        assert_eq!(loaded.tokenizer.token_to_id("<|im_end|>"), Some(248_046));
        assert!(loaded.generation_config.is_eos_token(248_044));
        drop(loaded.model);
    }

    /// Verifies that near-matching Qwen3 dispatch keys are rejected.
    #[test]
    fn rejects_non_dense_qwen3_dispatch_keys() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tau-models-qwen3-dispatch-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();

        for (architecture, model_type) in [
            ("Qwen3ForCausalLM", "qwen2"),
            ("Qwen3MoeForCausalLM", "qwen3_moe"),
        ] {
            fs::write(
                root.join("config.json"),
                format!(
                    r#"{{
                        "architectures": ["{architecture}"],
                        "model_type": "{model_type}"
                    }}"#
                ),
            )
            .unwrap();

            let error = match load_model(&root, &Device::Cpu, ModelDType::F32) {
                Ok(_) => panic!(
                    "unsupported Qwen3 dispatch key loaded: \
                     {architecture}/{model_type}"
                ),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains("unsupported model architecture/type")
            );
            assert!(error.to_string().contains(architecture));
            assert!(error.to_string().contains(model_type));
        }

        fs::remove_dir_all(&root).unwrap();
    }
}
