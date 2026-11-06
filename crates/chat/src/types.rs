//! Conversation values shared by chat formatting and inference adapters.

use crate::ChatTemplateOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Original HTTP text parts, retained for checkpoint-specific normalization.
    pub text_parts: Option<Vec<String>>,
}

impl ChatMessage {
    /// Creates a chat message from role and content text.
    ///
    /// The inputs are converted into owned UTF-8 strings. The function returns a
    /// `ChatMessage` and cannot fail.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Into::<String>::into(role),
            content: Into::<String>::into(content),
            text_parts: None,
        }
    }

    /// Preserves text-array boundaries for templates and concatenated text for
    /// legacy engines that consume the existing `content` field directly.
    pub fn from_text_parts(role: impl Into<String>, parts: Vec<String>) -> Self {
        Self {
            role: Into::<String>::into(role),
            content: parts.join(""),
            text_parts: Some(parts),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatOptions {
    max_tokens: Option<usize>,
    template: ChatTemplateOptions,
}

impl ChatOptions {
    /// Creates default chat generation options.
    ///
    /// The function takes no input and returns options with no caller-provided token
    /// limit. It cannot fail.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns options with a maximum generated-token limit.
    ///
    /// The input is an optional token count from an API request. The function returns
    /// updated options and cannot fail; engines decide how to apply absent limits.
    pub fn with_max_tokens(mut self, max_tokens: Option<usize>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Returns the caller-provided generated-token limit, when present.
    ///
    /// The method borrows options and returns `Some(limit)` for explicit API limits
    /// or `None` when the engine should use its default. It cannot fail.
    pub fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }

    /// Attaches request-local controls without changing generation settings.
    pub fn with_template_options(mut self, template: ChatTemplateOptions) -> Self {
        self.template = template;
        self
    }

    /// Borrows the immutable variables used by the checkpoint renderer.
    pub fn template_options(&self) -> &ChatTemplateOptions {
        &self.template
    }
}

/// OpenAI-compatible reason that chat generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFinishReason {
    /// The model emitted an end-of-sequence token.
    Stop,
    /// Generation reached the effective output limit (request or KV-cache budget).
    Length,
}

impl ChatFinishReason {
    /// Returns the stable string value for this finish reason.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// Generated assistant text plus token accounting and termination metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatCompletion {
    content: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: ChatFinishReason,
}

impl ChatCompletion {
    /// Creates a completed chat generation result.
    pub fn new(
        content: impl Into<String>,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: ChatFinishReason,
    ) -> Self {
        Self {
            content: Into::<String>::into(content),
            prompt_tokens,
            completion_tokens,
            finish_reason,
        }
    }

    /// Returns the generated assistant content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Consumes the result and returns the generated assistant content.
    pub fn into_content(self) -> String {
        self.content
    }

    /// Returns the number of tokens in the formatted model prompt.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Returns the number of non-EOS tokens generated for the assistant response.
    pub fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }

    /// Returns why generation stopped.
    pub fn finish_reason(&self) -> ChatFinishReason {
        self.finish_reason
    }
}
