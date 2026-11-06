use super::super::Qwen2SafetensorsEngine;
use super::qwen_test_model_path;
use crate::generation::DEFAULT_MAX_NEW_TOKENS;
use crate::{
    ChatInferenceEngine, ChatMessage, ChatOptions, Device, InferenceEngine, ModelDType,
};

/// Verifies that the shared Qwen2.5 model can be loaded when present.
///
/// The ignored test loads config, generation config, tokenizer, safetensors
/// metadata, and the model runtime from the shared model path. It skips itself
/// when the shared model is not mounted and fails if any component cannot be
/// loaded.
#[test]
#[ignore = "loads a local 943 MB model from /mnt/shared"]
fn loads_shared_qwen_model_when_available() {
    let path = qwen_test_model_path();

    if !path.exists() {
        eprintln!("skipping Qwen load test; {} is absent", path.display());
        return;
    }

    let engine =
        Qwen2SafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32).unwrap();

    assert_eq!(engine.model_id(), "Qwen2.5-0.5B-Instruct");
    assert_eq!(engine.tokenizer().token_to_id("<|im_start|>"), Some(151644));
    assert_eq!(engine.tokenizer().token_to_id("<|im_end|>"), Some(151645));
    assert!(engine.generation_config_loaded());
    assert_eq!(engine.max_new_tokens(), DEFAULT_MAX_NEW_TOKENS);
    assert_eq!(engine.dtype(), ModelDType::F32);
}

/// Verifies that the shared Qwen2.5 model can generate chat text on CPU.
///
/// The ignored test loads the shared model, sends one chat message, and limits
/// generation to a few tokens so it stays usable as a smoke test. It skips itself
/// when the shared model is not mounted and fails if generation returns an error
/// or empty output.
#[test]
#[ignore = "loads and runs a local 943 MB model from /mnt/shared"]
fn generates_shared_qwen_chat_when_available() {
    let path = qwen_test_model_path();

    if !path.exists() {
        eprintln!(
            "skipping Qwen generation test; {} is absent",
            path.display()
        );
        return;
    }

    let mut engine =
        Qwen2SafetensorsEngine::load(&path, Device::Cpu, ModelDType::F32).unwrap();
    let messages = [ChatMessage::new("user", "Say hello in one word.")];
    let options = ChatOptions::new().with_max_tokens(Some(4));

    let output = engine.chat_with_options(&messages, options).unwrap();

    assert!(!output.trim().is_empty());
}

/// Verifies that the pinned BF16 Qwen artifact loads and generates in BF16.
#[test]
#[ignore = "loads and runs a local 943 MB model from /mnt/shared"]
fn generates_shared_qwen_chat_in_bf16_when_available() {
    let path = qwen_test_model_path();

    if !path.exists() {
        eprintln!(
            "skipping Qwen BF16 generation test; {} is absent",
            path.display()
        );
        return;
    }

    let device = Device::new_cuda(0).expect("BF16 Qwen test requires CUDA GPU 0");
    let mut engine =
        Qwen2SafetensorsEngine::load(&path, device, ModelDType::BF16).unwrap();
    assert_eq!(engine.dtype(), ModelDType::BF16);

    let messages = [ChatMessage::new("user", "Say hello in one word.")];
    let options = ChatOptions::new().with_max_tokens(Some(4));
    let output = engine.chat_with_options(&messages, options).unwrap();

    assert!(!output.trim().is_empty());
}
