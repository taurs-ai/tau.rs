use super::*;

/// HTTP decoding must retain custom JSON values before shared normalization;
/// sorted maps or all-f64 parsing would change the template's actual token IDs.
#[test]
fn preserves_template_controls_and_json_representation() {
    let request = parse_chat_completion_request(concat!(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"#,
        r#""reasoning_effort":"xhigh","reasoning":{"effort":"low"},"#,
        r#""chat_template_kwargs":{"enable_thinking":false,"#,
        r#""custom":{"z":9007199254740993,"a":1.0}}}"#,
    ))
    .unwrap();
    let variables = request.template.variables();
    assert_eq!(variables["reasoning_effort"], "low");
    assert_eq!(variables["enable_thinking"], false);
    assert_eq!(variables["custom"]["z"].as_u64(), Some(9007199254740993));
    assert!(variables["custom"]["a"].is_f64());
    assert_eq!(
        variables["custom"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
}

/// Default/empty controls remain usable; unimplemented requested semantics
/// fail at parsing rather than reaching any inference engine.
#[test]
fn rejects_deferred_template_features_and_nontext_parts() {
    let body = |extra: &str| {
        format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hello"}}]{extra}}}"#
        )
    };
    for extra in [
        r#", "chat_template_kwargs":{}"#,
        r#", "tools":[], "tool_choice":"none""#,
        r#", "continue_final_message":false"#,
    ] {
        assert!(parse_chat_completion_request(&body(extra)).is_ok());
    }
    for extra in [
        r#", "chat_template_kwargs":"bad""#,
        r#", "continue_final_message":true"#,
        r#", "tools":[{}]"#,
    ] {
        assert!(
            parse_chat_completion_request(&body(extra)).is_err(),
            "{extra}"
        );
    }
    for content in [
        r#"[{"type":"image_url","image_url":{"url":"x"}}]"#,
        r#"[{"type":"text","text":1}]"#,
        r#"[{"type":false,"text":"hello"}]"#,
    ] {
        let body = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":{content}}}]}}"#
        );
        assert!(parse_chat_completion_request(&body).is_err());
    }
}

/// Retaining text-part boundaries lets the selected template normalize them
/// like SGLang while preserving the legacy joined content for other engines.
#[test]
fn preserves_text_parts_for_checkpoint_rendering() {
    let r = parse_chat_completion_request(concat!(
        r#"{"model":"m","messages":[{"role":"user","content":["#,
        r#"{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
    ))
    .unwrap();
    assert_eq!(r.messages[0].content, "ab");
    assert_eq!(
        r.messages[0].text_parts.as_deref(),
        Some(["a".to_owned(), "b".to_owned()].as_slice())
    );
}

/// Verifies that `max_tokens` is accepted for chat requests.
///
/// The test parses a minimal OpenAI-compatible chat body and expects the legacy
/// token-limit field to become the request's generation limit. It fails if the
/// parser rejects or ignores the field.
#[test]
fn parses_max_tokens() {
    let request = parse_chat_completion_request(concat!(
        r#"{"model":"tau-loopback","max_tokens":3,"#,
        r#""messages":[{"role":"user","content":"hi"}]}"#,
    ))
    .unwrap();

    assert_eq!(request.max_tokens, Some(3));
}

/// Verifies that `max_completion_tokens` takes precedence over `max_tokens`.
///
/// The test supplies both OpenAI token-limit spellings and expects the newer
/// completion-specific field to win. It fails if precedence changes.
#[test]
fn prefers_max_completion_tokens() {
    let request = parse_chat_completion_request(concat!(
        r#"{"model":"tau-loopback","#,
        r#""max_tokens":3,"max_completion_tokens":5,"#,
        r#""messages":[{"role":"user","content":"hi"}]}"#,
    ))
    .unwrap();

    assert_eq!(request.max_tokens, Some(5));
}

/// Verifies that zero token limits are rejected.
///
/// The test supplies `max_tokens: 0` and expects a bad-request parser error. It
/// fails if the server accepts a request that cannot generate any tokens.
#[test]
fn rejects_zero_max_tokens() {
    let error = parse_chat_completion_request(concat!(
        r#"{"model":"tau-loopback","max_tokens":0,"#,
        r#""messages":[{"role":"user","content":"hi"}]}"#,
    ))
    .unwrap_err();

    assert!(matches!(error, ChatCompletionError::BadRequest(_)));
}

/// Verifies exact token usage and a length termination are serialized.
#[test]
fn serializes_chat_completion_metadata() {
    let completion =
        ChatCompletion::new("hello", 12, 3, tau_engine::ChatFinishReason::Length);

    let metadata = new_chat_response_metadata();
    let response = openai_chat_completion("test-model", &completion, &metadata);

    assert!(response.contains(r#""finish_reason":"length""#));
    assert!(response.contains(r#""prompt_tokens":12"#));
    assert!(response.contains(r#""completion_tokens":3"#));
    assert!(response.contains(r#""total_tokens":15"#));
}

/// Verifies streaming terminal chunks preserve a length termination.
#[test]
fn serializes_stream_finish_reason() {
    let completion =
        ChatCompletion::new("hello", 12, 3, tau_engine::ChatFinishReason::Length);

    let metadata = new_chat_response_metadata();
    let response = openai_chat_completion_stream("test-model", &completion, &metadata);

    assert!(response.contains(r#""finish_reason":"length""#));
    assert!(response.ends_with("data: [DONE]\n\n"));
}

/// Verifies response ids are unique and creation times are Unix timestamps.
#[test]
fn creates_openai_response_metadata() {
    let first = new_chat_response_metadata();
    let second = new_chat_response_metadata();

    assert_ne!(first.id, second.id);
    assert!(first.id.starts_with("chatcmpl-tau-"));
    assert!(first.created > 0);
    assert!(second.created >= first.created);
}
