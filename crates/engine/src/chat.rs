//! Engine traits and adapters using the shared tau-chat conversation types.

use crate::InferenceError;
use crate::engine::InferenceEngine;
pub(crate) use tau_chat::CheckpointChatTemplate;
pub use tau_chat::{
    ChatCompletion, ChatFinishReason, ChatMessage, ChatOptions, ChatTemplateOptions,
    ChatTemplateSelection,
};

impl From<tau_chat::ChatTemplateError> for InferenceError {
    /// Adapts chat-local request failures at the engine boundary, preserving the
    /// existing public inference error category and message prefix.
    fn from(error: tau_chat::ChatTemplateError) -> Self {
        Self::Backend(error.to_string())
    }
}

/// Adapts startup errors where engines combine models and chat formatting.
/// The chat crate has no dependency on model error types or execution backends.
pub(crate) fn template_load_error(
    error: tau_chat::ChatTemplateError,
) -> crate::ModelLoadError {
    match error {
        tau_chat::ChatTemplateError::Io(path, source) => {
            crate::ModelLoadError::Io(path, source)
        }
        tau_chat::ChatTemplateError::Json(path, message) => {
            crate::ModelLoadError::Json(path, message)
        }
        error => crate::ModelLoadError::InvalidConfig(error.to_string()),
    }
}

pub trait ChatInferenceEngine: InferenceEngine {
    /// Runs one chat inference request through the engine.
    ///
    /// The input is an ordered list of chat messages. The default implementation uses
    /// default chat options and delegates to `chat_with_options`; engines may
    /// override that method when they need custom chat semantics.
    fn chat(&mut self, messages: &[ChatMessage]) -> Result<String, InferenceError> {
        self.chat_with_options(messages, ChatOptions::default())
    }

    /// Runs one chat request with generation options.
    ///
    /// The inputs are ordered chat messages and optional generation settings. The
    /// default implementation rejects template controls, uses the latest user message,
    /// falls back to the final message, and calls `infer_text`. Implementations
    /// return `InferenceError` when validation or generation fails.
    fn chat_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<String, InferenceError> {
        options.template_options().require_supported(false)?;
        let message = messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .or_else(|| messages.last())
            .ok_or_else(|| {
                InferenceError::Backend("messages must not be empty".to_owned())
            })?;

        self.infer_text(&message.content)
    }

    /// Runs chat inference and returns response metadata for API serialization.
    ///
    /// Text-only engines default to unknown token counts and a normal stop. Tokenizer-
    /// backed engines override this method with exact accounting and stop state.
    fn chat_completion_with_options(
        &mut self,
        messages: &[ChatMessage],
        options: ChatOptions,
    ) -> Result<ChatCompletion, InferenceError> {
        let content = self.chat_with_options(messages, options)?;
        Ok(ChatCompletion::new(content, 0, 0, ChatFinishReason::Stop))
    }
}

/// Define a ChatEngine, which implements chat methods built on the given
// InferenceEngine.
pub struct ChatEngine<E> {
    engine: E,
}

impl<E> ChatEngine<E>
where
    E: InferenceEngine,
{
    /// Wraps an inference engine with default chat behavior.
    ///
    /// The input engine is moved into the wrapper. The wrapper delegates base
    /// inference methods to the inner engine and implements chat by selecting a text
    /// prompt from the message list.
    pub fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Returns the wrapped inference engine.
    ///
    /// The method borrows the wrapper and returns immutable access to the inner
    /// engine. It performs no inference and cannot fail.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Returns mutable access to the wrapped inference engine.
    ///
    /// The method borrows the wrapper mutably and returns mutable access to the
    /// inner engine. It performs no inference and cannot fail.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Consumes the wrapper and returns the wrapped inference engine.
    ///
    /// The method takes ownership of the wrapper and moves out the inner engine. It
    /// performs no inference and cannot fail.
    pub fn into_inner(self) -> E {
        self.engine
    }
}

impl<E> InferenceEngine for ChatEngine<E>
where
    E: InferenceEngine,
{
    /// Returns the wrapped engine's public model id.
    ///
    /// The method delegates to the inner inference engine and cannot fail.
    fn model_id(&self) -> &str {
        self.engine.model_id()
    }

    /// Returns the wrapped engine's backend name.
    ///
    /// The method delegates to the inner inference engine and cannot fail.
    fn backend_name(&self) -> &str {
        self.engine.backend_name()
    }

    /// Returns the wrapped engine's backend version.
    ///
    /// The method delegates to the inner inference engine and cannot fail.
    fn backend_version(&self) -> &str {
        self.engine.backend_version()
    }

    /// Runs flat `f32` inference through the wrapped engine.
    ///
    /// The input and output buffers are forwarded unchanged. The method returns any
    /// inference error reported by the inner engine.
    fn infer_f32(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError> {
        self.engine.infer_f32(input, output)
    }

    /// Runs text inference through the wrapped engine.
    ///
    /// The input is forwarded unchanged. The method returns the inner engine's text
    /// output or inference error.
    fn infer_text(&mut self, input: &str) -> Result<String, InferenceError> {
        self.engine.infer_text(input)
    }
}

impl<E> ChatInferenceEngine for ChatEngine<E> where E: InferenceEngine {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-exported values stay identical to chat-owned types, while failures
    /// retain the public engine/load categories at their respective boundaries.
    #[test]
    fn preserves_chat_type_and_error_compatibility() {
        let _: tau_chat::ChatOptions = ChatOptions::new();
        let _: ChatMessage = tau_chat::ChatMessage::new("user", "hello");
        let error = InferenceError::from(tau_chat::ChatTemplateError::InvalidOptions(
            "invalid control".to_owned(),
        ));
        assert_eq!(
            error.to_string(),
            "backend error: chat template controls: invalid control"
        );
        let path = std::path::PathBuf::from("tokenizer_config.json");
        let error = template_load_error(tau_chat::ChatTemplateError::Io(
            path.clone(),
            std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        ));
        assert!(matches!(error, crate::ModelLoadError::Io(p, e)
            if p == path && e.kind() == std::io::ErrorKind::PermissionDenied));
        let error = template_load_error(tau_chat::ChatTemplateError::InvalidConfig(
            "unknown name".to_owned(),
        ));
        assert_eq!(
            error.to_string(),
            "invalid model config: chat template: unknown name"
        );
    }

    struct EchoTextEngine;

    impl InferenceEngine for EchoTextEngine {
        /// Identifies the test double as the echo model.
        fn model_id(&self) -> &str {
            "echo"
        }

        /// Identifies the test double's backend as the echo backend.
        fn backend_name(&self) -> &str {
            "echo"
        }

        /// Returns the fixed backend version used by this test double.
        fn backend_version(&self) -> &str {
            "test"
        }

        /// Returns the input unchanged so chat-message selection can be observed.
        fn infer_text(&mut self, input: &str) -> Result<String, InferenceError> {
            Ok(input.to_owned())
        }
    }

    /// Verifies that the default chat adapter uses the latest user message.
    ///
    /// The test sends system and user messages to a text-only echo engine wrapped by
    /// `ChatEngine`. It fails by panicking if chat selection or text delegation
    /// changes.
    #[test]
    fn chat_engine_echoes_latest_user_message() {
        let mut engine = ChatEngine::new(EchoTextEngine);
        let messages = [
            ChatMessage::new("system", "ignore"),
            ChatMessage::new("user", "hello"),
            ChatMessage::new("assistant", "old"),
            ChatMessage::new("user", "latest"),
        ];

        let output = engine.chat(&messages).unwrap();

        assert_eq!(output, "latest");
    }
}
