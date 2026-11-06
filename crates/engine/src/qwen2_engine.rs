//! Qwen2 safetensors-backed model loading for model-backed core engines.

use crate::InferenceError;
use crate::chat::{ChatCompletion, ChatInferenceEngine, ChatMessage, ChatOptions};
use crate::engine::InferenceEngine;
pub use crate::model::{ModelDType, ModelLoadError, ModelShard, TensorInfo};
use crate::model_engine::{CausalModelEngine, EngineOptions};
use candle::Device;
use std::path::{Path, PathBuf};
use tau_models::Qwen2ForCausalLM;
use tau_models::{LoadedCausalModel, LoadedModel, load_model};
use tokenizers::Tokenizer;

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

pub struct Qwen2SafetensorsEngine {
    inner: CausalModelEngine<Qwen2ForCausalLM>,
}

impl Qwen2SafetensorsEngine {
    /// Loads a Qwen2 safetensors model directory into a model-backed engine.
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
            LoadedModel::Qwen2(loaded) => loaded,
            _ => {
                return Err(ModelLoadError::InvalidConfig(
                    "Qwen2SafetensorsEngine requires a Qwen2 model".to_owned(),
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

    /// Builds a Qwen2 engine from state produced by the shared model loader.
    ///
    /// The model factory passes an already selected and initialized Qwen2 bundle so
    /// config detection and weight loading happen only once.
    pub(crate) fn from_loaded(
        root: PathBuf,
        device: Device,
        loaded: LoadedCausalModel<Qwen2ForCausalLM>,
        options: EngineOptions,
    ) -> Self {
        Self {
            inner: CausalModelEngine::from_loaded(root, device, loaded, options),
        }
    }

    /// Returns the root directory used to load the model.
    ///
    /// The method borrows the engine and returns the model path. It cannot fail
    /// because the path was validated during construction.
    pub fn root(&self) -> &Path {
        self.inner.root()
    }

    /// Returns whether generation settings were loaded from either config file.
    ///
    /// The method borrows the engine and returns `true` after successful model
    /// construction. It cannot fail.
    pub fn generation_config_loaded(&self) -> bool {
        self.inner.generation_config_loaded()
    }

    /// Returns the default maximum number of generated tokens.
    ///
    /// The method borrows model generation settings and returns the default used when
    /// an API request does not specify `max_tokens`. It cannot fail.
    pub fn max_new_tokens(&self) -> usize {
        self.inner.max_new_tokens()
    }

    /// Returns the dtype used by model weights and floating-point inference state.
    pub fn dtype(&self) -> ModelDType {
        self.inner.dtype()
    }

    /// Returns the loaded tokenizer.
    ///
    /// The method borrows tokenizer state for future inference code. It cannot fail
    /// because tokenizer loading and validation happen during construction.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.inner.tokenizer()
    }
}

impl InferenceEngine for Qwen2SafetensorsEngine {
    /// Returns the model id exposed by OpenAI-compatible model APIs.
    ///
    /// The id is derived from the model directory name. The method borrows engine
    /// state and cannot fail.
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    /// Returns the backend family name used by diagnostics.
    ///
    /// Safetensors engines load model files directly, so the backend name is the
    /// fixed string `safetensors`. The method cannot fail.
    fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }

    /// Returns the server crate version for the safetensors loader.
    ///
    /// The method borrows no mutable state and cannot fail.
    fn backend_version(&self) -> &str {
        self.inner.backend_version()
    }

    /// Rejects numeric smoke-test inference for model-backed engines.
    ///
    /// The input and output buffers are ignored because safetensors models require
    /// tokenized generation. The method returns `InferenceError::Unsupported`.
    fn infer_f32(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError> {
        self.inner.infer_f32(input, output)
    }

    /// Generates text from a plain text prompt.
    ///
    /// The input is tokenized directly without applying a chat template. The method
    /// returns generated text or `InferenceError` for tokenization, context, or
    /// model execution failures.
    fn infer_text(&mut self, input: &str) -> Result<String, InferenceError> {
        self.inner.infer_text(input)
    }
}

impl ChatInferenceEngine for Qwen2SafetensorsEngine {
    /// Generates an assistant response from chat messages.
    ///
    /// The input messages are formatted with Qwen's chat markers and generation
    /// options may override the default maximum token count. The method returns
    /// assistant text or `InferenceError` for bad message roles, tokenization,
    /// context overflow, or model execution failures.
    fn chat_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<String, InferenceError> {
        self.chat_completion_with_options(messages, options)
            .map(ChatCompletion::into_content)
    }

    /// Generates a Qwen2 chat completion with token counts and a finish reason.
    fn chat_completion_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<ChatCompletion, InferenceError> {
        options.template_options().require_supported(false)?;
        let prompt = qwen_chat_prompt(messages)?;
        let max_new_tokens =
            options.max_tokens().unwrap_or(self.inner.max_new_tokens());
        self.inner
            .generate_from_prompt_with_metadata(&prompt, max_new_tokens)
    }
}

/// Formats OpenAI-style chat messages as a Qwen chat prompt.
///
/// The input messages must contain supported roles. The function returns a prompt
/// ending with Qwen's assistant-generation marker or `InferenceError` when no
/// messages exist or a role cannot be represented in Qwen's text template.
fn qwen_chat_prompt(messages: &[ChatMessage]) -> Result<String, InferenceError> {
    if messages.is_empty() {
        return Err(InferenceError::Backend(
            "messages must not be empty".to_owned(),
        ));
    }

    let has_initial_system = messages
        .first()
        .is_some_and(|message| matches!(message.role.as_str(), "system" | "developer"));
    let mut prompt = String::new();
    if !has_initial_system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(DEFAULT_SYSTEM_PROMPT);
        prompt.push_str("<|im_end|>\n");
    }

    for message in messages {
        let role = qwen_chat_role(&message.role)?;
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

/// Maps API chat roles to Qwen chat-template roles.
///
/// The input is a role string from a chat message. The function returns a Qwen role
/// for supported system, developer, user, assistant, and tool messages, or
/// `InferenceError` for unsupported roles.
fn qwen_chat_role(role: &str) -> Result<&'static str, InferenceError> {
    match role {
        "system" | "developer" => Ok("system"),
        "user" | "tool" | "function" => Ok("user"),
        "assistant" => Ok("assistant"),
        _ => Err(InferenceError::Backend(format!(
            "unsupported chat role {role:?}"
        ))),
    }
}

#[cfg(test)]
mod tests;
