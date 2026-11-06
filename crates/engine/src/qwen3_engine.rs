//! Qwen3 safetensors-backed model loading for model-backed core engines.

use crate::InferenceError;
use crate::chat::{ChatCompletion, ChatInferenceEngine, ChatMessage, ChatOptions};
use crate::engine::InferenceEngine;
pub use crate::model::{ModelDType, ModelLoadError, ModelShard, TensorInfo};
use crate::model_engine::{CausalModelEngine, EngineOptions};
use candle::Device;
use std::path::{Path, PathBuf};
use tau_models::Qwen3ForCausalLM;
use tau_models::{LoadedCausalModel, LoadedModel, load_model};
use tokenizers::Tokenizer;

pub struct Qwen3SafetensorsEngine {
    inner: CausalModelEngine<Qwen3ForCausalLM>,
}

impl Qwen3SafetensorsEngine {
    /// Loads a dense Qwen3 safetensors model directory into a model-backed engine.
    ///
    /// The input path must contain `config.json`, `tokenizer.json`, and at least
    /// one `.safetensors` file. Generation settings come from
    /// `generation_config.json` when present, otherwise from `config.json`.
    /// The caller chooses the device and dtype used for model tensors, and the
    /// engine reuses that placement and precision for inference state.
    pub fn load(
        path: impl AsRef<Path>,
        device: Device,
        dtype: ModelDType,
    ) -> Result<Self, ModelLoadError> {
        let root = path.as_ref().to_path_buf();
        let loaded = load_model(&root, &device, dtype)?;
        let loaded = match loaded {
            LoadedModel::Qwen3(loaded) => loaded,
            _ => {
                return Err(ModelLoadError::InvalidConfig(
                    "Qwen3SafetensorsEngine requires a Qwen3 model".to_owned(),
                ));
            }
        };

        Ok(Self::from_loaded(
            root,
            device,
            loaded,
            EngineOptions::default(),
        ))
    }

    /// Builds a Qwen3 engine from state produced by the shared model loader.
    ///
    /// The model factory passes an already selected and initialized Qwen3 bundle so
    /// config detection and weight loading happen only once.
    pub(crate) fn from_loaded(
        root: PathBuf,
        device: Device,
        loaded: LoadedCausalModel<Qwen3ForCausalLM>,
        options: EngineOptions,
    ) -> Self {
        Self {
            inner: CausalModelEngine::from_loaded(root, device, loaded, options),
        }
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

    /// Returns the dtype used by model weights and floating-point inference state.
    pub fn dtype(&self) -> ModelDType {
        self.inner.dtype()
    }

    /// Returns the loaded tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.inner.tokenizer()
    }
}

impl InferenceEngine for Qwen3SafetensorsEngine {
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

impl ChatInferenceEngine for Qwen3SafetensorsEngine {
    /// Generates only the visible assistant text for a Qwen3 conversation.
    fn chat_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<String, InferenceError> {
        self.chat_completion_with_options(messages, options)
            .map(ChatCompletion::into_content)
    }

    /// Formats a Qwen3 prompt and returns text plus generation metadata.
    fn chat_completion_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<ChatCompletion, InferenceError> {
        options.template_options().require_supported(false)?;
        let prompt = qwen3_chat_prompt(messages)?;
        let max_new_tokens =
            options.max_tokens().unwrap_or(self.inner.max_new_tokens());
        self.inner
            .generate_from_prompt_with_metadata(&prompt, max_new_tokens)
    }
}

/// Formats the text-only subset of Qwen3's chat template.
///
/// Qwen3 does not add Qwen2's implicit default system prompt. Historical assistant
/// reasoning is removed when it is wrapped in `<think>...</think>`, matching the
/// Qwen3 template's multi-turn behavior. Structured tool calls are outside the
/// current `ChatMessage` representation and are rejected rather than reformatted.
fn qwen3_chat_prompt(messages: &[ChatMessage]) -> Result<String, InferenceError> {
    if messages.is_empty() {
        return Err(InferenceError::Backend(
            "messages must not be empty".to_owned(),
        ));
    }

    let mut prompt = String::new();
    for message in messages {
        let role = qwen3_chat_role(&message.role)?;
        let content = if role == "assistant" {
            qwen3_assistant_history_content(&message.content)
        } else {
            &message.content
        };

        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

/// Maps API chat roles to the text-only roles supported by the Qwen3 formatter.
fn qwen3_chat_role(role: &str) -> Result<&'static str, InferenceError> {
    match role {
        "system" | "developer" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        _ => Err(InferenceError::Backend(format!(
            "unsupported Qwen3 chat role {role:?}"
        ))),
    }
}

/// Removes a prior Qwen3 reasoning block from assistant conversation history.
fn qwen3_assistant_history_content(content: &str) -> &str {
    content
        .rsplit_once("</think>")
        .map(|(_, answer)| answer.trim_start_matches('\n'))
        .unwrap_or(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::DEFAULT_MAX_NEW_TOKENS;
    use std::path::PathBuf;

    const QWEN3_TEST_MODEL: &str = "/mnt/shared/models/Qwen3-0.6B";

    /// Resolves the Qwen3 fixture path used by ignored integration tests.
    fn qwen3_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(QWEN3_TEST_MODEL))
    }

    /// Formats a user-only conversation without inventing a system message.
    #[test]
    fn formats_qwen3_chat_without_an_implicit_system_prompt() {
        let prompt = qwen3_chat_prompt(&[ChatMessage::new("user", "hello")]).unwrap();

        assert_eq!(
            prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Keeps explicit instructions while removing saved reasoning from history.
    #[test]
    fn retains_explicit_system_prompt_and_removes_historical_thinking() {
        let messages = [
            ChatMessage::new("system", "Be concise."),
            ChatMessage::new("user", "first"),
            ChatMessage::new(
                "assistant",
                "<think>\nprivate reasoning\n</think>\n\nanswer",
            ),
            ChatMessage::new("user", "next"),
        ];

        let prompt = qwen3_chat_prompt(&messages).unwrap();

        assert!(prompt.starts_with("<|im_start|>system\nBe concise.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>assistant\nanswer<|im_end|>\n"));
        assert!(!prompt.contains("private reasoning"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    /// Rejects tool messages until the public chat type can represent them safely.
    #[test]
    fn rejects_tool_roles_until_structured_tool_messages_are_supported() {
        let error =
            qwen3_chat_prompt(&[ChatMessage::new("tool", "result")]).unwrap_err();

        assert!(error.to_string().contains("unsupported Qwen3 chat role"));
    }

    /// Loads the shared Qwen3 fixture and checks its exposed metadata.
    #[test]
    #[ignore = "loads a local 1.5 GB model from /mnt/shared"]
    fn loads_shared_qwen3_engine() {
        let path = qwen3_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3 validation model is absent: {}",
            path.display()
        );

        let engine =
            Qwen3SafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32).unwrap();

        assert_eq!(engine.model_id(), "Qwen3-0.6B");
        assert_eq!(engine.root(), path);
        assert_eq!(engine.dtype(), ModelDType::F32);
        assert!(engine.generation_config_loaded());
        assert_eq!(engine.max_new_tokens(), DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(engine.tokenizer().token_to_id("<think>"), Some(151_667));
        assert_eq!(engine.tokenizer().token_to_id("</think>"), Some(151_668));
    }

    /// Generates a short response with the shared Qwen3 fixture.
    #[test]
    #[ignore = "loads and runs a local 1.5 GB model from /mnt/shared"]
    fn generates_shared_qwen3_chat() {
        let path = qwen3_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3 validation model is absent: {}",
            path.display()
        );

        let mut engine =
            Qwen3SafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32).unwrap();
        let messages = [ChatMessage::new("user", "Say hello in one word.")];
        let options = ChatOptions::new().with_max_tokens(Some(2));

        let output = engine
            .chat_with_options(&messages, options.clone())
            .unwrap();

        assert!(!output.trim().is_empty());
    }

    /// Runs identical chats twice to verify requests do not share stale KV state.
    #[test]
    #[ignore = "loads and runs a local 1.5 GB model twice from /mnt/shared"]
    fn clears_qwen3_kv_cache_between_chat_requests() {
        let path = qwen3_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3 validation model is absent: {}",
            path.display()
        );

        let mut engine =
            Qwen3SafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32).unwrap();
        let messages = [ChatMessage::new("user", "Say hello in one word.")];
        let options = ChatOptions::new().with_max_tokens(Some(2));

        let first = engine
            .chat_with_options(&messages, options.clone())
            .unwrap();
        let second = engine
            .chat_with_options(&messages, options.clone())
            .unwrap();

        assert_eq!(second, first);
    }

    /// Checks CUDA/BF16 factory dispatch and generation with the shared fixture.
    #[test]
    #[ignore = "loads and runs a local 1.5 GB model from /mnt/shared"]
    fn model_factory_generates_shared_qwen3_chat_in_bf16() {
        let path = qwen3_test_model_path();
        assert!(
            path.is_dir(),
            "Qwen3 validation model is absent: {}",
            path.display()
        );

        let device = Device::new_cuda(0).expect("BF16 Qwen3 test requires CUDA GPU 0");
        let mut engine =
            crate::load_model_engine_with_dtype(&path, device, ModelDType::BF16)
                .unwrap();
        assert_eq!(engine.model_id(), "Qwen3-0.6B");

        let messages = [ChatMessage::new("user", "Say hello in one word.")];
        let options = ChatOptions::new().with_max_tokens(Some(2));
        let output = engine
            .chat_with_options(&messages, options.clone())
            .unwrap();

        assert!(!output.trim().is_empty());
    }
}
