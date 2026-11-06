//! Real-model exact CPU/GPU selection and full generation parity.
use super::*;
use crate::generation::{GpuGreedy, logits_values, next_token_from_values};
use tau_models::{LoadedModel, load_model_with_options};

fn real_model() -> (
    Device,
    tau_models::LoadedCausalModel<tau_models::Qwen3_5Model>,
) {
    let root = std::env::var("TAU_QWEN3_5_FULL_TEST_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join("work/models/Qwen3.5-4B")
        });
    // SAFETY: The model and sampler stay on one private stream for the whole test.
    let device = unsafe { tau_models::new_capture_device(0) }.unwrap();
    let LoadedModel::Qwen3_5(loaded) = load_model_with_options(
        &root,
        &device,
        ModelDType::BF16,
        ModelLoadOptions::default(),
    )
    .unwrap() else {
        panic!("expected dense Qwen3.5")
    };
    (device, loaded)
}

/// Uses real logits at the exact SGLang prefixes. GPU selection must equal the
/// retained CPU selector on every vector; cross-runtime rounding cannot excuse a
/// difference here because both selectors consume the same device tensor.
#[test]
#[ignore = "requires real 4B, CUDA GPU 0 and TAU_SGLANG_REFERENCE"]
fn real_model_gpu_greedy_matches_cpu_at_sglang_prefixes() {
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("TAU_SGLANG_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let (device, loaded) = real_model();
    let mut model = loaded.model;
    let mut sampler = GpuGreedy::default();
    let ids = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let mut count = 0;
    for mode in [
        crate::CudaGraphMode::None,
        crate::CudaGraphMode::Layer,
        crate::CudaGraphMode::Full,
    ] {
        model.set_cuda_graph(mode);
        for case in reference["cases"].as_array().unwrap() {
            let prompt = ids(&case["input_ids"]);
            let outputs = ids(&case["output_ids"]);
            model.clear_kv_cache();
            model.reserve_kv_cache(4096).unwrap();
            let mut logits = None;
            for offset in (0..prompt.len()).step_by(256) {
                let end = (offset + 256).min(prompt.len());
                let input = CandleTensor::new(&prompt[offset..end], &device)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap();
                logits = model
                    .forward_prefill_chunk(
                        &input,
                        offset,
                        end - offset == 256,
                        end == prompt.len(),
                    )
                    .unwrap();
            }
            for (step, &token) in outputs.iter().enumerate() {
                let output = logits.as_ref().unwrap();
                let expected =
                    next_token_from_values(&logits_values(output).unwrap()).unwrap();
                assert_eq!(
                    sampler.select(output).unwrap(),
                    Some(expected),
                    "case={} mode={mode:?} step={step}",
                    case["id"]
                );
                count += 1;
                if step + 1 < outputs.len() {
                    logits = Some(
                        model
                            .forward_decode_token(token, prompt.len() + step, &device)
                            .unwrap(),
                    );
                }
            }
        }
    }
    model.clear_kv_cache();
    println!("Exact real-model GPU/CPU selections: {count}");
}

/// Tracing deliberately retains full-logit CPU selection and its existing timing
/// scopes. Compare complete returned IDs against ordinary GPU serving, using the
/// same model sequentially so two 4B copies never compete for 16 GB of VRAM.
#[test]
#[ignore = "requires real Qwen3.5-4B and CUDA GPU 0"]
fn real_model_gpu_generation_matches_traced_cpu_reference_and_resets() {
    let (device, loaded) = real_model();
    let tau_models::LoadedCausalModel {
        mut model,
        generation_config,
        ..
    } = loaded;
    let mut sampler = GpuGreedy::default();
    let prompts = [
        vec![1u32, 2, 3],
        (0..1025).map(|i| i % 100 + 10).collect(),
        vec![1, 2, 3],
    ];
    for mode in [
        crate::CudaGraphMode::None,
        crate::CudaGraphMode::Layer,
        crate::CudaGraphMode::Full,
    ] {
        model.set_cuda_graph(mode);
        for prompt in &prompts {
            let options = EngineOptions::new()
                .with_cuda_graph(mode)
                .with_tracing(true);
            let mut reference_metrics = crate::instrumentation::start_request(
                "selection-test",
                ModelDType::BF16,
                64,
                options,
                &device,
                &generation_config.eos_token_ids,
            );
            let settings = || token_generation::GenerationSettings {
                device: &device,
                config: &generation_config,
                prefill_chunk_size: 256,
                cuda_graph: mode,
            };
            let expected = token_generation::generate(
                &mut model,
                prompt,
                64,
                settings(),
                &mut reference_metrics,
                &mut sampler,
            )
            .unwrap();
            let mut ordinary = InferenceMetrics::default();
            let actual = token_generation::generate(
                &mut model,
                prompt,
                64,
                settings(),
                &mut ordinary,
                &mut sampler,
            )
            .unwrap();
            assert_eq!(
                actual,
                expected,
                "mode={mode:?} prompt length={}",
                prompt.len()
            );
        }
    }
}

/// A tiny GPU model makes EOS and failure cleanup observable without relying on
/// a real model to choose a particular token. Its first maximum is tied at IDs
/// 1/2; the second step emits EOS. NaNs inject an actual native-selection error.
#[test]
fn gpu_generation_preserves_limits_eos_and_cleanup_after_selection_failure() {
    struct Model {
        device: Device,
        resets: usize,
        nan: bool,
    }
    impl CausalLanguageModel for Model {
        fn forward(
            &mut self,
            _: &CandleTensor,
            offset: usize,
        ) -> candle::Result<CandleTensor> {
            let values = if self.nan {
                [f32::NAN; 4]
            } else if offset == 0 {
                [0., 4., 4., -1.]
            } else {
                [0., 0., 0., 5.]
            };
            CandleTensor::from_vec(values.to_vec(), (1, 1, 4), &self.device)
        }
        fn clear_kv_cache(&mut self) {
            self.resets += 1;
        }
    }
    let device = Device::new_cuda(0).unwrap();
    let mut model = Model {
        device: device.clone(),
        resets: 0,
        nan: false,
    };
    let config = crate::generation::GenerationConfig {
        eos_token_ids: vec![3],
        max_new_tokens: 4,
    };
    let mut sampler = GpuGreedy::default();
    let settings = |config| token_generation::GenerationSettings {
        device: &device,
        config,
        prefill_chunk_size: 256,
        cuda_graph: crate::CudaGraphMode::None,
    };
    for (limit, expected) in [(0, vec![]), (1, vec![2]), (4, vec![2])] {
        let before = model.resets;
        let actual = token_generation::generate(
            &mut model,
            &[1],
            limit,
            settings(&config),
            &mut InferenceMetrics::default(),
            &mut sampler,
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(model.resets, before + 2);
    }
    let mut immediate_eos = config.clone();
    immediate_eos.eos_token_ids = vec![2];
    assert!(
        token_generation::generate(
            &mut model,
            &[1],
            4,
            settings(&immediate_eos),
            &mut InferenceMetrics::default(),
            &mut sampler
        )
        .unwrap()
        .is_empty()
    );
    model.nan = true;
    let before = model.resets;
    assert!(
        token_generation::generate(
            &mut model,
            &[1],
            4,
            settings(&config),
            &mut InferenceMetrics::default(),
            &mut sampler
        )
        .unwrap_err()
        .to_string()
        .contains("all NaN")
    );
    assert_eq!(model.resets, before + 2);
    model.nan = false;
    assert_eq!(
        token_generation::generate(
            &mut model,
            &[1],
            4,
            settings(&config),
            &mut InferenceMetrics::default(),
            &mut sampler
        )
        .unwrap(),
        [2]
    );
}
