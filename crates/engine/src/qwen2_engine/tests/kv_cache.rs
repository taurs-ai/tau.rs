use super::super::Qwen2SafetensorsEngine;
use super::qwen_test_model_path;
use crate::{ChatInferenceEngine, ChatMessage, ChatOptions, Device, ModelDType};

/// Verifies that one completed request does not contaminate the next request.
///
/// The ignored test runs the same deterministic chat request twice through one
/// loaded engine and expects identical decoded output. It skips itself when the
/// shared model is not mounted and fails if KV-cache state survives the reset
/// around a completed generation.
#[test]
#[ignore = "loads and runs a local 943 MB model twice from /mnt/shared"]
fn clears_kv_cache_between_chat_requests() {
    let path = qwen_test_model_path();

    if !path.exists() {
        eprintln!("skipping Qwen KV-cache test; {} is absent", path.display());
        return;
    }

    let mut engine =
        Qwen2SafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32).unwrap();
    let messages = [ChatMessage::new("user", "Say hello in one word.")];
    let options = ChatOptions::new().with_max_tokens(Some(4));

    let first = engine
        .chat_with_options(&messages, options.clone())
        .unwrap();
    let second = engine.chat_with_options(&messages, options).unwrap();

    assert_eq!(second, first);
}
