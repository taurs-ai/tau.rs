//! Qwen3.5 MoE safetensors-backed model loading and chat inference.
//!
//! CPU/F32 is the portable correctness path. CUDA/BF16 supports checkpoints that
//! declare BF16, including permitted FP32 Gated DeltaNet parameters. Uniform and
//! mixed storage are covered by artifact-backed model and API regressions;
//! CPU/BF16 is rejected before model allocation.

use crate::InferenceError;
use crate::chat::{ChatCompletion, ChatInferenceEngine, ChatMessage, ChatOptions};
use crate::chat::{ChatTemplateSelection, CheckpointChatTemplate};
use crate::engine::InferenceEngine;
pub use crate::model::{ModelDType, ModelLoadError, ModelShard, TensorInfo};
use crate::model_engine::{CausalModelEngine, EngineOptions};
use candle::Device;
use std::path::{Path, PathBuf};
use tau_models::Qwen3_5MoeModel;
use tau_models::{LoadedCausalModel, LoadedModel, load_model};
use tokenizers::Tokenizer;

/// Public text-chat engine for supported Qwen3.5 MoE checkpoint wrappers.
pub struct Qwen3_5MoeSafetensorsEngine {
    inner: CausalModelEngine<Qwen3_5MoeModel>,
    chat_template: CheckpointChatTemplate,
}

impl Qwen3_5MoeSafetensorsEngine {
    /// Loads a Qwen3.5 MoE model directory into a model-backed engine.
    ///
    /// CPU/F32 is validated. CPU/BF16 is rejected during loading, while CUDA/BF16
    /// is available when the checkpoint declares BF16 and CUDA is usable.
    pub fn load(
        path: impl AsRef<Path>,
        device: Device,
        dtype: ModelDType,
    ) -> Result<Self, ModelLoadError> {
        Self::load_with_chat_template(path, device, dtype, None)
    }

    /// Loads the engine with an optional Jinja-file override for chat formatting.
    /// Raw prompt generation and model execution keep their existing paths.
    pub fn load_with_chat_template(
        path: impl AsRef<Path>,
        device: Device,
        dtype: ModelDType,
        chat_template: Option<&Path>,
    ) -> Result<Self, ModelLoadError> {
        let selection = chat_template
            .map(|path| ChatTemplateSelection::new().with_file(path))
            .unwrap_or_default();
        Self::load_with_chat_template_selection(path, device, dtype, &selection)
    }

    /// Loads a model with construction-only named/file template preferences.
    /// The compiled selection is shared across requests; variables remain local.
    pub fn load_with_chat_template_selection(
        path: impl AsRef<Path>,
        device: Device,
        dtype: ModelDType,
        selection: &ChatTemplateSelection,
    ) -> Result<Self, ModelLoadError> {
        let root = path.as_ref().to_path_buf();
        let loaded = load_model(&root, &device, dtype)?;
        let loaded = match loaded {
            LoadedModel::Qwen3_5Moe(loaded) => loaded,
            _ => {
                return Err(ModelLoadError::InvalidConfig(
                    "Qwen3_5MoeSafetensorsEngine requires a Qwen3.5 MoE model"
                        .to_owned(),
                ));
            }
        };

        Self::from_loaded(root, device, loaded, EngineOptions::default(), selection)
    }

    /// Builds an engine from state produced by the shared model loader.
    pub(crate) fn from_loaded(
        root: PathBuf,
        device: Device,
        loaded: LoadedCausalModel<Qwen3_5MoeModel>,
        options: EngineOptions,
        selection: &ChatTemplateSelection,
    ) -> Result<Self, ModelLoadError> {
        let chat_template = CheckpointChatTemplate::load(&root, selection)
            .map_err(crate::chat::template_load_error)?;
        Ok(Self {
            inner: CausalModelEngine::from_loaded(root, device, loaded, options),
            chat_template,
        })
    }

    /// Returns the root directory used to load the model.
    pub fn root(&self) -> &Path {
        self.inner.root()
    }

    /// Returns whether generation settings were loaded from either config file.
    pub fn generation_config_loaded(&self) -> bool {
        self.inner.generation_config_loaded()
    }

    /// Returns the default maximum number of generated tokens.
    pub fn max_new_tokens(&self) -> usize {
        self.inner.max_new_tokens()
    }

    /// Returns the primary dtype for bulk weights and activations.
    /// Gated DeltaNet precision-sensitive parameters and recurrence use FP32.
    pub fn dtype(&self) -> ModelDType {
        self.inner.dtype()
    }

    /// Returns the loaded tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.inner.tokenizer()
    }
}

impl InferenceEngine for Qwen3_5MoeSafetensorsEngine {
    /// Returns the checkpoint identifier exposed by the shared model engine.
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    /// Returns the backend name reported in diagnostics.
    fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }

    /// Returns the backend version reported in diagnostics.
    fn backend_version(&self) -> &str {
        self.inner.backend_version()
    }

    /// Forwards numeric compatibility requests to the shared model engine.
    fn infer_f32(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError> {
        self.inner.infer_f32(input, output)
    }

    /// Generates text from a plain prompt without applying a chat template.
    fn infer_text(&mut self, input: &str) -> Result<String, InferenceError> {
        self.inner.infer_text(input)
    }
}

impl ChatInferenceEngine for Qwen3_5MoeSafetensorsEngine {
    /// Generates only the visible assistant text for a Qwen3.5 MoE conversation.
    fn chat_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<String, InferenceError> {
        self.chat_completion_with_options(messages, options)
            .map(ChatCompletion::into_content)
    }

    /// Formats a Qwen3.5 prompt and returns text plus generation metadata.
    fn chat_completion_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<ChatCompletion, InferenceError> {
        let prompt = self
            .chat_template
            .render(messages, options.template_options())?;
        let max_new_tokens =
            options.max_tokens().unwrap_or(self.inner.max_new_tokens());
        self.inner
            .generate_from_prompt_with_metadata(&prompt, max_new_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::DEFAULT_MAX_NEW_TOKENS;
    use std::path::PathBuf;

    /// Resolves the generated MoE-model fixture used by ignored integration tests.
    fn qwen3_5_moe_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_MOE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.5-MoE-Tiny-Dev")
            })
    }

    /// Loads the generated MoE fixture and checks its exposed metadata.
    #[test]
    #[ignore = "loads the local generated Qwen3.5 MoE development model"]
    fn loads_generated_qwen3_5_moe_engine() {
        let path = qwen3_5_moe_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());

        let engine =
            Qwen3_5MoeSafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32)
                .unwrap();

        assert_eq!(engine.model_id(), "Qwen3.5-MoE-Tiny-Dev");
        assert_eq!(engine.root(), path);
        assert_eq!(engine.dtype(), ModelDType::F32);
        assert!(engine.generation_config_loaded());
        assert_eq!(engine.max_new_tokens(), DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(
            engine.tokenizer().token_to_id("<|im_start|>"),
            Some(248_045)
        );
        assert_eq!(engine.tokenizer().token_to_id("<|im_end|>"), Some(248_046));
        assert_eq!(engine.tokenizer().token_to_id("<think>"), Some(248_068));
        assert_eq!(engine.tokenizer().token_to_id("</think>"), Some(248_069));
    }

    /// Verifies factory dispatch can generate chat text with the MoE fixture.
    #[test]
    #[ignore = "loads and runs the local generated Qwen3.5 MoE development model"]
    fn model_factory_generates_qwen3_5_moe_chat() {
        let path = qwen3_5_moe_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());

        let mut engine =
            crate::load_model_engine_with_dtype(&path, Device::Cpu, ModelDType::F32)
                .unwrap();
        assert_eq!(engine.model_id(), "Qwen3.5-MoE-Tiny-Dev");

        let messages = [ChatMessage::new("user", "Say hello.")];
        let output = engine
            .chat_with_options(&messages, ChatOptions::new().with_max_tokens(Some(2)))
            .unwrap();

        assert!(!output.trim().is_empty());
    }

    /// Runs identical chats twice to verify requests do not share stale KV state.
    #[test]
    #[ignore = "loads and runs the local generated Qwen3.5 MoE model twice"]
    fn clears_qwen3_5_moe_cache_between_chat_requests() {
        let path = qwen3_5_moe_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());

        let mut engine =
            Qwen3_5MoeSafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32)
                .unwrap();
        let messages = [ChatMessage::new("user", "Say hello.")];
        let options = ChatOptions::new().with_max_tokens(Some(2));

        let first = engine
            .chat_with_options(&messages, options.clone())
            .unwrap();
        let second = engine
            .chat_with_options(&messages, options.clone())
            .unwrap();

        assert_eq!(second, first);
    }
}
