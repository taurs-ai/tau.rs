//! Shared checkpoint-template compilation and request-local text rendering.

mod jinja;
mod source;

pub use source::ChatTemplateSelection;
#[cfg(test)]
mod tests;

use crate::ChatTemplateError;
use crate::{ChatMessage, ChatTemplateOptions};
use minijinja::{Environment, value::Serde};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use source::template_error;
use std::path::Path;

/// Compiled template and special-token context, immutable across requests.
/// Missing metadata permits raw inference but cannot silently guess a chat format.
pub struct CheckpointChatTemplate {
    environment: Option<Environment<'static>>,
    special_tokens: serde_json::Map<String, Value>,
    structured_content: bool,
}

impl CheckpointChatTemplate {
    /// Discovers a catalog, resolves its source-specific policy and compiles only
    /// the selected entry. No request can mutate selection or the environment.
    pub fn load(
        root: &Path,
        selection: &ChatTemplateSelection,
    ) -> Result<Self, ChatTemplateError> {
        let (catalog, tokens) = source::load(root, selection)?;
        let (environment, structured_content) = if let Some(catalog) = catalog {
            let selected = catalog.select(selection.name())?;
            let structured = jinja::structured_content(&selected.source)
                .map_err(|e| template_error(format!("{}: {e}", selected.origin)))?;
            let environment = jinja::compile(selected.source.clone())
                .map_err(|e| template_error(format!("{}: {e}", selected.origin)))?;
            eprintln!(
                "tau-chat template: convention={}, policy={}, available={:?}, \
                 requested_name={:?}, selected_name={:?}, source={}, \
                 effective_sha256={:x}",
                catalog.convention,
                catalog.policy.label(),
                catalog
                    .entries
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect::<Vec<_>>(),
                selection.name(),
                selected.name,
                selected.origin,
                Sha256::digest(selected.source.as_bytes()),
            );
            (Some(environment), structured)
        } else {
            (None, false)
        };
        Ok(Self {
            environment,
            special_tokens: tokens,
            structured_content,
        })
    }

    /// Formats a conversation with the selected template and request controls.
    /// Message normalization and variable precedence happen in a fresh context,
    /// leaving caller-owned history and the compiled environment unchanged.
    /// Returns prompt text; the caller owns tokenization and model execution.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<String, ChatTemplateError> {
        let error = |e: String| ChatTemplateError::Render(e);
        // Missing chat metadata is allowed for raw inference, but chat needs an
        // explicit format. Report the problem instead of guessing a template.
        let environment = self.environment.as_ref().ok_or_else(|| {
            error("no checkpoint chat template; supply --chat-template PATH".to_owned())
        })?;
        if messages.is_empty() {
            return Err(error("messages must not be empty".to_owned()));
        }
        // Build separate template messages so normalization never rewrites the
        // caller's conversation. Unsupported roles fail before any rendering.
        let mut values = Vec::with_capacity(messages.len());
        for message in messages {
            if !matches!(
                message.role.as_str(),
                "system" | "developer" | "user" | "assistant"
            ) {
                return Err(error(format!(
                    "unsupported text-chat role {:?}",
                    message.role
                )));
            }
            // For array input, preserve typed text parts when the template
            // expects structured content; otherwise join them with spaces.
            // Existing string content passes through unchanged.
            let content = match &message.text_parts {
                Some(parts) if self.structured_content => json!(
                    parts
                        .iter()
                        .map(|text| json!({"type":"text", "text":text}))
                        .collect::<Vec<_>>()
                ),
                Some(parts) => json!(parts.join(" ")),
                None => json!(message.content),
            };
            values.push(json!({"role":message.role, "content":content}));
        }
        // Start a new assistant response: a trailing assistant string becomes
        // a user turn. Structured content retains its original role.
        // This trailing-message behavior matches SGLang.
        if let Some(last) = values.last_mut()
            && last["role"] == "assistant"
            && last["content"].is_string()
        {
            last["role"] = json!("user");
        }
        // Start with checkpoint special tokens, then add this conversation.
        // Request values stay local instead of becoming environment globals
        // that could leak into later requests.
        let mut context = self.special_tokens.clone();
        context.insert("messages".to_owned(), Value::Array(values));
        // Ask the template to emit the prefix for a new assistant reply.
        context.insert("add_generation_prompt".to_owned(), json!(true));
        // Null explicitly means no tools/documents. Omitting the keys would
        // make them undefined, which templates can distinguish from null.
        context.insert("tools".to_owned(), Value::Null);
        context.insert("documents".to_owned(), Value::Null);
        // Normalized request variables take precedence where overrides are
        // permitted. Unspecified controls stay undefined so template defaults
        // still apply; reserved context arguments were validated earlier.
        context.extend(options.variables().clone());
        // Reuse the compiled template with this request's context. Return any
        // template exception to the caller before model execution begins.
        environment
            .get_template("chat")
            .and_then(|template| {
                template.render(minijinja::Value::from(Serde(&context)))
            })
            .map_err(|e| error(e.to_string()))
    }
}
