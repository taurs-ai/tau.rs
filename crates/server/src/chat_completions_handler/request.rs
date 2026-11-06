//! HTTP chat request decoding; template semantics belong to tau_engine.

use serde_json::Value as JsonValue;
use tau_chat::{ChatMessage, ChatTemplateOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ChatCompletionRequest {
    pub(super) model: String,
    pub(super) messages: Vec<ChatMessage>,
    pub(super) stream: bool,
    pub(super) max_tokens: Option<usize>,
    pub(super) template: ChatTemplateOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatCompletionError {
    BadRequest(String),
    ModelNotFound(String),
}

/// Parses the JSON body for `/v1/chat/completions`.
///
/// The input string is expected to contain `model`, `messages`, and optional
/// `stream`, `max_tokens`, and `max_completion_tokens`. The function returns a
/// structured request or
/// `ChatCompletionError::BadRequest` when JSON parsing or field validation fails.
pub(super) fn parse_chat_completion_request(
    body: &str,
) -> Result<ChatCompletionRequest, ChatCompletionError> {
    let value: JsonValue = serde_json::from_str(body)
        .map_err(|source| ChatCompletionError::BadRequest(source.to_string()))?;
    reject_deferred_template_options(&value)?;

    let model = value
        .get("model")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            ChatCompletionError::BadRequest("missing model field".to_owned())
        })?
        .to_owned();
    let stream = value
        .get("stream")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let messages = value
        .get("messages")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            ChatCompletionError::BadRequest("missing messages array".to_owned())
        })?
        .iter()
        .map(parse_chat_message)
        .collect::<Result<Vec<_>, _>>()?;
    let max_tokens = parse_chat_max_tokens(&value)?;
    let template = ChatTemplateOptions::from_request_fields(
        value.get("reasoning_effort"),
        value.get("reasoning"),
        value.get("chat_template_kwargs"),
    )
    .map_err(tau_engine::InferenceError::from)
    .map_err(|e| ChatCompletionError::BadRequest(e.to_string()))?;

    Ok(ChatCompletionRequest {
        model,
        messages,
        stream,
        max_tokens,
        template,
    })
}

/// Parses optional OpenAI chat token-limit fields.
///
/// The input JSON object may include `max_completion_tokens` or legacy
/// `max_tokens`. The function returns the first present positive limit, `None` when
/// neither field is present, or `ChatCompletionError::BadRequest` for invalid
/// field types or zero values.
fn parse_chat_max_tokens(
    value: &JsonValue,
) -> Result<Option<usize>, ChatCompletionError> {
    let max_completion_tokens =
        parse_optional_positive_usize(value, "max_completion_tokens")?;
    let max_tokens = parse_optional_positive_usize(value, "max_tokens")?;

    Ok(max_completion_tokens.or(max_tokens))
}

/// Parses one optional positive integer field from a JSON object.
///
/// The inputs are a JSON value and a field name. The function returns `None` when
/// the field is absent, `Some(limit)` for positive integers, or
/// `ChatCompletionError::BadRequest` for invalid token-limit values.
fn parse_optional_positive_usize(
    value: &JsonValue,
    field: &str,
) -> Result<Option<usize>, ChatCompletionError> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };

    let Some(limit) = value.as_u64().and_then(|v| usize::try_from(v).ok()) else {
        return Err(ChatCompletionError::BadRequest(format!(
            "{field} must be a positive integer"
        )));
    };

    if limit == 0 {
        Err(ChatCompletionError::BadRequest(format!(
            "{field} must be greater than zero"
        )))
    } else {
        Ok(Some(limit))
    }
}

/// Parses one chat message object from a JSON value.
///
/// The input must contain a string `role` and supported `content`. The function
/// returns a `ChatMessage` or `ChatCompletionError::BadRequest` when required
/// fields are missing or content has an unsupported shape.
fn parse_chat_message(value: &JsonValue) -> Result<ChatMessage, ChatCompletionError> {
    for field in ["reasoning_content", "tool_calls", "tool_call_id"] {
        if value
            .get(field)
            .is_some_and(|v| !matches!(v, JsonValue::Null))
        {
            return Err(ChatCompletionError::BadRequest(format!(
                "{field} is not supported by this text-chat milestone"
            )));
        }
    }
    let role = value
        .get("role")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            ChatCompletionError::BadRequest("message is missing role".to_owned())
        })?
        .to_owned();
    let source = value.get("content").ok_or_else(|| {
        ChatCompletionError::BadRequest("message is missing content".to_owned())
    })?;
    if let Some(parts) = source.as_array() {
        let mut texts = Vec::with_capacity(parts.len());
        for part in parts {
            // Legacy untyped text parts remain accepted, but an explicit type
            // must be a supported string rather than silently ignoring bad data.
            let invalid_kind = part.get("type").is_some_and(|kind| {
                !matches!(kind.as_str(), Some("text" | "input_text"))
            });
            if invalid_kind {
                return Err(ChatCompletionError::BadRequest(
                    "only text content parts are supported".to_owned(),
                ));
            }
            texts.push(
                part.get("text")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| {
                        ChatCompletionError::BadRequest(
                            "text content part must have string text".to_owned(),
                        )
                    })?
                    .to_owned(),
            );
        }
        Ok(ChatMessage::from_text_parts(role, texts))
    } else {
        let text = source.as_str().ok_or_else(|| {
            ChatCompletionError::BadRequest(
                "message content must be a string or text parts array".to_owned(),
            )
        })?;
        Ok(ChatMessage::new(role, text))
    }
}

/// Rejects unimplemented template controls before inference instead of silently
/// applying checkpoint defaults to a request that explicitly asked for otherwise.
fn reject_deferred_template_options(
    value: &JsonValue,
) -> Result<(), ChatCompletionError> {
    for field in ["tools", "tool_choice"] {
        // Empty tools and tool_choice=none are ordinary no-tools requests.
        let requested = match (field, value.get(field)) {
            (_, None | Some(JsonValue::Null)) => false,
            ("tools", Some(JsonValue::Array(items))) => !items.is_empty(),
            ("tool_choice", Some(value)) => value.as_str() != Some("none"),
            _ => true,
        };
        if requested {
            return Err(ChatCompletionError::BadRequest(format!(
                "{field} is not supported yet; checkpoint defaults are used"
            )));
        }
    }
    if value
        .get("continue_final_message")
        .is_some_and(|v| !matches!(v, JsonValue::Bool(false) | JsonValue::Null))
    {
        return Err(ChatCompletionError::BadRequest(
            "continue_final_message is not supported yet".to_owned(),
        ));
    }
    Ok(())
}
