//! Factory contracts for the shared causal engine.

use super::*;

#[test]
fn graph_policy_allows_eager_cpu_without_initializing_cuda() {
    let error = |mode| {
        load_model_engine_with_runtime_options(
            "/absent/model",
            Device::Cpu,
            ModelDType::F32,
            EngineOptions::new().with_cuda_graph(mode),
        )
        .err()
        .unwrap()
        .to_string()
    };
    let reference = error(crate::CudaGraphMode::None);
    for mode in [crate::CudaGraphMode::Layer, crate::CudaGraphMode::Full] {
        assert_eq!(
            error(mode),
            reference,
            "CPU should reach ordinary model validation"
        );
    }
}

/// Unsupported graph families reach ordinary loader validation without changing
/// the caller device policy.
#[test]
fn graph_policy_retains_eager_paths_for_unsupported_families() {
    let device = Device::new_cuda(0).expect("graph factory test requires CUDA GPU 0");
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "tau-engine-unsupported-graph-model-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    for (architecture, model_type) in [
        ("Qwen2ForCausalLM", "qwen2"),
        ("Qwen3ForCausalLM", "qwen3"),
        ("Qwen3_5MoeForCausalLM", "qwen3_5_moe_text"),
        ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe"),
        ("Qwen3_5ForConditionalGeneration", "qwen3_5_moe"),
    ] {
        fs::write(
            root.join("config.json"),
            serde_json::json!({
                "architectures": [architecture],
                "model_type": model_type,
                "torch_dtype": "bfloat16",
            })
            .to_string(),
        )
        .unwrap();
        for dtype in [ModelDType::F32, ModelDType::BF16] {
            for optimize_kernel in [false, true] {
                let error = |mode| {
                    load_model_engine_with_runtime_options(
                        &root,
                        device.clone(),
                        dtype,
                        EngineOptions::new()
                            .with_cuda_graph(mode)
                            .with_optimize_kernel(optimize_kernel),
                    )
                    .err()
                    .expect("incomplete fixture must fail ordinary loader validation")
                    .to_string()
                };
                let reference = error(crate::CudaGraphMode::None);
                for mode in [crate::CudaGraphMode::Layer, crate::CudaGraphMode::Full] {
                    assert_eq!(error(mode), reference);
                }
            }
        }
    }
    assert!(device.as_cuda_device().unwrap().is_event_tracking());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn zero_prefill_chunk_size_is_rejected_before_loading() {
    let error = load_model_engine_with_runtime_options(
        "/absent",
        Device::Cpu,
        ModelDType::F32,
        EngineOptions::new().with_prefill_chunk_size(0),
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("prefill chunk size must be positive")
    );
}

#[test]
fn inactive_nondefault_attention_geometry_is_rejected_before_loading() {
    let error = load_model_engine_with_runtime_options(
        "/absent",
        Device::Cpu,
        ModelDType::F32,
        EngineOptions::new()
            .with_attention_block_size(crate::AttentionBlockSize::Tokens128),
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("non-default attention block size requires")
    );
}

/// Rejects model metadata that does not identify a supported architecture.
#[test]
fn rejects_unsupported_model_family() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "tau-engine-unsupported-model-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
            "architectures": ["ImaginaryForCausalLM"],
            "model_type": "imaginary"
        }"#,
    )
    .unwrap();

    let result = load_model_engine(&root, Device::Cpu);
    fs::remove_dir_all(&root).unwrap();

    match result {
        Ok(_) => panic!("unsupported model family unexpectedly loaded"),
        Err(error) => assert!(
            error
                .to_string()
                .contains("unsupported model architecture/type")
        ),
    }
}

/// Rejects BF16 when the checkpoint metadata declares only F32 support.
#[test]
fn rejects_bf16_for_a_model_that_does_not_declare_it() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "tau-engine-unsupported-dtype-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "torch_dtype": "float32"
        }"#,
    )
    .unwrap();

    let result = load_model_engine_with_dtype(&root, Device::Cpu, ModelDType::BF16);
    fs::remove_dir_all(&root).unwrap();

    match result {
        Ok(_) => panic!("unsupported dtype unexpectedly loaded"),
        Err(ModelLoadError::UnsupportedDType {
            requested: ModelDType::BF16,
            supported,
            ..
        }) => assert_eq!(supported, vec![ModelDType::F32]),
        Err(error) => panic!("unexpected model load error: {error}"),
    }
}
