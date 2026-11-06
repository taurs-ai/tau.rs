use super::super::{DEFAULT_SYSTEM_PROMPT, qwen_chat_prompt};
use super::qwen_test_model_path;
use crate::ChatMessage;
use serde_json::Value;
use tau_models::{load_tokenizer, validate_tokenizer};

const QWEN_TOKENIZATION_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/qwen2_5_0_5b_reference.json"
));

/// Verifies Qwen chat prompt formatting for a simple user request.
///
/// The test formats one user message and expects the default system prompt,
/// user marker, and assistant generation marker. It fails if prompt formatting
/// stops matching the Qwen chat template shape.
#[test]
fn formats_qwen_chat_prompt() {
    let prompt = qwen_chat_prompt(&[ChatMessage::new("user", "hello")]).unwrap();

    assert!(prompt.starts_with("<|im_start|>system\n"));
    assert!(prompt.contains(DEFAULT_SYSTEM_PROMPT));
    assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

/// Compares tau.rs tokenization with the pinned Hugging Face reference.
///
/// The ignored test loads only tokenizer assets from the Qwen2.5 model directory,
/// not model weights. It compares raw token ids, token strings, attention masks,
/// decoding, supported chat prompts, and chat tokenization exactly. It fails when
/// the model directory is absent because explicitly invoking this parity test must
/// not report a false success.
#[test]
#[ignore = "requires the pinned Qwen2.5 tokenizer under /mnt/shared"]
fn matches_pinned_hugging_face_tokenization_reference() {
    let model_path = qwen_test_model_path();
    assert!(
        model_path.is_dir(),
        "Qwen tokenizer validation model is absent: {}",
        model_path.display()
    );

    let reference: Value = serde_json::from_str(QWEN_TOKENIZATION_REFERENCE)
        .expect("generated Qwen tokenization reference must be valid JSON");
    assert_eq!(reference["schema_version"].as_u64(), Some(1));
    assert_eq!(
        reference["summary"]["all_reference_implementations_match"].as_bool(),
        Some(true),
        "Hugging Face slow and fast references must agree"
    );

    let tokenizer = load_tokenizer(&model_path).unwrap_or_else(|error| {
        panic!(
            "failed to load tokenizer from {}: {error}",
            model_path.display()
        )
    });
    let vocab_size = validate_tokenizer(&tokenizer).unwrap();
    let expected_tokenizer = &reference["tokenizers"]["huggingface_fast"];
    assert_eq!(
        vocab_size,
        json_usize(expected_tokenizer, "vocab_size_with_added_tokens")
    );

    let special_tokens = json_string_array(expected_tokenizer, "all_special_tokens");
    let special_ids = json_u32_array(expected_tokenizer, "all_special_ids");
    assert_eq!(special_tokens.len(), special_ids.len());
    for (token, expected_id) in special_tokens.iter().zip(special_ids) {
        assert_eq!(
            tokenizer.token_to_id(token),
            Some(expected_id),
            "special token {token:?}"
        );
    }

    for case in json_array(&reference, "raw_cases") {
        compare_raw_case(&tokenizer, case);
    }
    for case in json_array(&reference, "chat_cases") {
        if json_str(case, "scope") == "supported" {
            compare_supported_chat_case(&tokenizer, case);
        }
    }
}

/// Checks one raw-text case against every value in the pinned reference.
fn compare_raw_case(tokenizer: &tokenizers::Tokenizer, case: &Value) {
    let name = json_str(case, "name");
    let text = json_str(case, "text");
    let expected = &case["huggingface_fast"];
    let encoding = tokenizer
        .encode(text, false)
        .unwrap_or_else(|error| panic!("raw case {name:?} failed to encode: {error}"));

    assert_eq!(
        encoding.get_ids(),
        json_u32_array(expected, "input_ids"),
        "raw case {name:?} input ids"
    );
    assert_eq!(
        encoding.get_tokens(),
        json_string_array(expected, "tokens"),
        "raw case {name:?} token strings"
    );
    assert_eq!(
        encoding.get_attention_mask(),
        json_u32_array(expected, "attention_mask"),
        "raw case {name:?} attention mask"
    );

    let decoded_with_special = tokenizer.decode(encoding.get_ids(), false).unwrap();
    let decoded_without_special = tokenizer.decode(encoding.get_ids(), true).unwrap();
    assert_eq!(
        decoded_with_special,
        json_str(expected, "decoded_skip_special_false"),
        "raw case {name:?} decoding with special tokens"
    );
    assert_eq!(
        decoded_without_special,
        json_str(expected, "decoded_skip_special_true"),
        "raw case {name:?} decoding without special tokens"
    );
}

/// Checks prompt formatting and tokenization for one supported chat case.
fn compare_supported_chat_case(tokenizer: &tokenizers::Tokenizer, case: &Value) {
    let name = json_str(case, "name");
    assert_eq!(
        case["add_generation_prompt"].as_bool(),
        Some(true),
        "tau.rs always adds an assistant generation prompt"
    );

    let messages = json_array(case, "messages")
        .iter()
        .map(|message| {
            ChatMessage::new(json_str(message, "role"), json_str(message, "content"))
        })
        .collect::<Vec<_>>();
    let prompt = qwen_chat_prompt(&messages)
        .unwrap_or_else(|error| panic!("chat case {name:?} failed to format: {error}"));
    let expected = &case["huggingface_fast"];

    assert_eq!(
        prompt,
        json_str(expected, "prompt"),
        "chat case {name:?} prompt"
    );

    let encoding = tokenizer
        .encode(prompt, false)
        .unwrap_or_else(|error| panic!("chat case {name:?} failed to encode: {error}"));
    assert_eq!(
        encoding.get_ids(),
        json_u32_array(expected, "input_ids"),
        "chat case {name:?} input ids"
    );
    assert_eq!(
        encoding.get_tokens(),
        json_string_array(expected, "tokens"),
        "chat case {name:?} token strings"
    );
}

/// Reads a required JSON array field or reports a descriptive test failure.
fn json_array<'a>(value: &'a Value, field: &str) -> &'a [Value] {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("reference field {field:?} must be an array"))
}

/// Reads a required JSON string field or reports a descriptive test failure.
fn json_str<'a>(value: &'a Value, field: &str) -> &'a str {
    value[field]
        .as_str()
        .unwrap_or_else(|| panic!("reference field {field:?} must be a string"))
}

/// Converts a required JSON array to the token id representation used by Candle.
fn json_u32_array(value: &Value, field: &str) -> Vec<u32> {
    json_array(value, field)
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .unwrap_or_else(|| {
                    panic!("reference field {field:?} must contain u32 values")
                })
        })
        .collect()
}

/// Copies a required JSON string array into owned test values.
fn json_string_array(value: &Value, field: &str) -> Vec<String> {
    json_array(value, field)
        .iter()
        .map(|item| {
            item.as_str()
                .unwrap_or_else(|| {
                    panic!("reference field {field:?} must contain strings")
                })
                .to_owned()
        })
        .collect()
}

/// Reads a required non-negative JSON integer as a platform-sized value.
fn json_usize(value: &Value, field: &str) -> usize {
    value[field]
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .unwrap_or_else(|| panic!("reference field {field:?} must be a usize"))
}
