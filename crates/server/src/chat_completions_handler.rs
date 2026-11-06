//! OpenAI-compatible chat completion request handling.

use crate::http_response::{HttpResponse, json_escape};
mod request;
pub(super) use request::ChatCompletionError;
use request::parse_chat_completion_request;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tau_chat::{ChatCompletion, ChatOptions};
use tau_engine::ChatInferenceEngine;

static CHAT_COMPLETION_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Handles an OpenAI-compatible chat completion request.
///
/// The input body must be JSON with a known model and messages array. The function
/// returns an HTTP response for streaming or non-streaming loopback output, or
/// `ChatCompletionError` for malformed JSON, missing fields, empty messages, or
/// unknown model ids.
pub(super) fn chat_completions(
    engine: &mut dyn ChatInferenceEngine,
    body: &str,
) -> Result<HttpResponse, ChatCompletionError> {
    let request = parse_chat_completion_request(body)?;
    if request.model != engine.model_id() {
        return Err(ChatCompletionError::ModelNotFound(request.model));
    }

    if request.messages.is_empty() {
        return Err(ChatCompletionError::BadRequest(
            "messages must not be empty".to_owned(),
        ));
    }

    let options = ChatOptions::new()
        .with_max_tokens(request.max_tokens)
        .with_template_options(request.template);
    let completion = engine
        .chat_completion_with_options(&request.messages, options)
        .map_err(|error| ChatCompletionError::BadRequest(error.to_string()))?;
    let metadata = new_chat_response_metadata();

    if request.stream {
        Ok(HttpResponse::ok(
            "text/event-stream; charset=utf-8",
            openai_chat_completion_stream(engine.model_id(), &completion, &metadata),
        ))
    } else {
        Ok(HttpResponse::ok(
            "application/json",
            openai_chat_completion(engine.model_id(), &completion, &metadata),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatResponseMetadata {
    id: String,
    created: u64,
}

/// Creates one unique completion id and Unix creation timestamp for a response.
fn new_chat_response_metadata() -> ChatResponseMetadata {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let sequence = CHAT_COMPLETION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    ChatResponseMetadata {
        id: format!("chatcmpl-tau-{:x}-{sequence:x}", elapsed.as_nanos()),
        created: elapsed.as_secs(),
    }
}

/// Builds a non-streaming OpenAI-compatible chat completion response.
///
/// The inputs are the model id and generated assistant result. The output is JSON
/// text with exact usage and finish metadata.
fn openai_chat_completion(
    model: &str,
    completion: &ChatCompletion,
    metadata: &ChatResponseMetadata,
) -> String {
    let total_tokens = completion.prompt_tokens() + completion.completion_tokens();
    format!(
        concat!(
            "{{",
            "\"id\":\"{}\",",
            "\"object\":\"chat.completion\",",
            "\"created\":{},",
            "\"model\":\"{}\",",
            "\"choices\":[{{",
            "\"index\":0,",
            "\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},",
            "\"finish_reason\":\"{}\"",
            "}}],",
            "\"usage\":{{",
            "\"prompt_tokens\":{},",
            "\"completion_tokens\":{},",
            "\"total_tokens\":{}",
            "}}",
            "}}\n"
        ),
        json_escape(&metadata.id),
        metadata.created,
        json_escape(model),
        json_escape(completion.content()),
        completion.finish_reason().as_str(),
        completion.prompt_tokens(),
        completion.completion_tokens(),
        total_tokens,
    )
}

/// Builds a streaming OpenAI-compatible Server-Sent Events response.
///
/// The inputs are the model id and generated assistant result. The output is SSE
/// text ending with `[DONE]` and preserves the generation finish reason.
fn openai_chat_completion_stream(
    model: &str,
    completion: &ChatCompletion,
    metadata: &ChatResponseMetadata,
) -> String {
    format!(
        concat!(
            "data: {{",
            "\"id\":\"{}\",",
            "\"object\":\"chat.completion.chunk\",",
            "\"created\":{},",
            "\"model\":\"{}\",",
            "\"choices\":[{{",
            "\"index\":0,",
            "\"delta\":{{\"role\":\"assistant\"}},",
            "\"finish_reason\":null",
            "}}]",
            "}}\n\n",
            "data: {{",
            "\"id\":\"{}\",",
            "\"object\":\"chat.completion.chunk\",",
            "\"created\":{},",
            "\"model\":\"{}\",",
            "\"choices\":[{{",
            "\"index\":0,",
            "\"delta\":{{\"content\":\"{}\"}},",
            "\"finish_reason\":null",
            "}}]",
            "}}\n\n",
            "data: {{",
            "\"id\":\"{}\",",
            "\"object\":\"chat.completion.chunk\",",
            "\"created\":{},",
            "\"model\":\"{}\",",
            "\"choices\":[{{",
            "\"index\":0,",
            "\"delta\":{{}},",
            "\"finish_reason\":\"{}\"",
            "}}]",
            "}}\n\n",
            "data: [DONE]\n\n"
        ),
        json_escape(&metadata.id),
        metadata.created,
        json_escape(model),
        json_escape(&metadata.id),
        metadata.created,
        json_escape(model),
        json_escape(completion.content()),
        json_escape(&metadata.id),
        metadata.created,
        json_escape(model),
        completion.finish_reason().as_str(),
    )
}

#[cfg(test)]
mod tests;
