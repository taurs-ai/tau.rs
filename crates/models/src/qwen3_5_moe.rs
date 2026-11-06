//! Qwen3.5 MoE model loading with portable and optimized execution paths.
//!
//! This module owns the text decoder used by both `Qwen3_5MoeForCausalLM` and
//! `Qwen3_5MoeForConditionalGeneration` checkpoints. The conditional-generation
//! wrapper exposes the same autoregressive text path while leaving vision input
//! execution as a separate API milestone. Reference execution uses the recurrent
//! delta rule for Gated DeltaNet, and unsupported MoE inputs dispatch experts
//! explicitly. Matching BF16 CUDA single-token inputs can
//! instead use the device-resident packed-expert runner.
//!
//! This top-level file is the architecture façade. It retains MoE feed-forward
//! construction, expert layouts, and checkpoint dispatch while sharing the hybrid
//! text backbone with dense Qwen3.5.

mod config;
mod tensors;

pub(crate) use self::config::{
    load_qwen3_5_moe_config, validate_qwen3_5_moe_device_dtype,
    validate_qwen3_5_moe_dtype,
};
pub(crate) use self::tensors::validate_qwen3_5_moe_tensor_contract;

use self::config::{
    LoadedQwen3_5MoeConfig, Qwen3_5MoeCheckpointKind, Qwen3_5MoeConfig,
};
use crate::components::{SparseMoeBlock, SparseMoeWeightLayout};
use crate::dtype::ModelDType;
use crate::qwen3_5_common::HybridCausalLm;
use crate::{ModelLoadError, ModelLoadOptions, ModelShard};
use candle::{Device, Result as CandleResult};
use candle_nn::VarBuilder;

/// Shared text model specialized with sparse MoE feed-forwards.
pub(crate) type Qwen3_5MoeModel = HybridCausalLm<SparseMoeBlock>;

/// Builds one MoE text model from wrapper-specific prefixes and expert layout.
fn build_qwen3_5_moe_model(
    cfg: &Qwen3_5MoeConfig,
    expert_weight_layout: SparseMoeWeightLayout,
    model_vb: VarBuilder,
    lm_head_vb: VarBuilder,
    options: ModelLoadOptions,
) -> CandleResult<Qwen3_5MoeModel> {
    let options = options.with_moe_reference_precision();
    let sparse_moe_config = cfg.sparse_moe_config(expert_weight_layout);
    HybridCausalLm::new(
        &cfg.hybrid,
        "Qwen3.5 MoE",
        options,
        model_vb,
        lm_head_vb,
        |_, mlp_vb| {
            SparseMoeBlock::new(&sparse_moe_config, options.optimize_kernel(), mlp_vb)?
                .with_prefill_config(options.moe_prefill_config())?
                .with_decode_options(options.moe_decode_options())
        },
    )
}

/// Loads Qwen3.5 MoE weights from one or more safetensors shards.
pub(crate) fn load_qwen3_5_moe_model(
    config: &LoadedQwen3_5MoeConfig,
    shards: &[ModelShard],
    device: &Device,
    dtype: ModelDType,
    options: ModelLoadOptions,
) -> Result<Qwen3_5MoeModel, ModelLoadError> {
    let paths = shards
        .iter()
        .map(|shard| shard.path().to_path_buf())
        .collect::<Vec<_>>();
    let var_builder = {
        // Safety: model shards are memory-mapped read-only and never mutated.
        unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, dtype.candle_dtype(), device)
        }
    }
    .map_err(|source| ModelLoadError::Candle(source.to_string()))?;

    // Both checkpoint kinds describe the same MoE text network and produce the
    // same `HybridCausalLm<SparseMoeBlock>` runtime type. This match only adapts
    // differences in how the two checkpoint wrappers store and name their weights;
    // it does not choose between dense and MoE execution.
    match config.checkpoint_kind {
        Qwen3_5MoeCheckpointKind::CausalLm => {
            // A standalone causal-LM checkpoint stores its text model directly
            // below `model` and stores each routed expert in separate tensors.
            build_qwen3_5_moe_model(
                &config.model,
                SparseMoeWeightLayout::Individual,
                var_builder.pp("model"),
                var_builder.pp("lm_head"),
                options,
            )
        }
        Qwen3_5MoeCheckpointKind::ConditionalGeneration => {
            // A conditional-generation checkpoint wraps the text model below
            // `model.language_model` and combines routed experts into packed tensors.
            // Vision and other wrapper weights are outside this text-only runtime.
            build_qwen3_5_moe_model(
                &config.model,
                SparseMoeWeightLayout::Packed,
                var_builder.pp("model").pp("language_model"),
                var_builder.pp("lm_head"),
                options,
            )
        }
    }
    .map_err(|source| ModelLoadError::Candle(source.to_string()))
}

#[cfg(test)]
mod tests {
    use super::config::{
        CONDITIONAL_ARCHITECTURE, CONDITIONAL_MODEL_TYPE, load_qwen3_5_moe_config,
    };
    use super::*;
    use crate::safetensors::{load_safetensors_shards, validate_safetensors_shards};
    use candle::DType;
    use candle::Tensor;
    use std::path::PathBuf;

    /// Resolves the generated fixture, with an environment override for CI.
    fn local_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_MOE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.5-MoE-Tiny-Dev")
            })
    }

    /// Loads every tensor from the generated model without evaluating a prompt.
    #[test]
    #[ignore = "loads the local generated Qwen3.5 MoE development model"]
    fn loads_generated_development_model() {
        let root = local_test_model_path();
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        validate_safetensors_shards(&shards).unwrap();
        let model = load_qwen3_5_moe_model(
            &config,
            &shards,
            &Device::Cpu,
            ModelDType::F32,
            ModelLoadOptions::default(),
        )
        .unwrap();

        assert_eq!(config.model.hybrid.num_hidden_layers, 4);
        assert_eq!(config.model.num_experts, 256);
        assert_eq!(config.model.hybrid.max_position_embeddings, 4096);
        drop(model);
    }

    /// Checks prefill, incremental decode, cache reset, and reference logits.
    #[test]
    #[ignore = "runs the local generated Qwen3.5 MoE development model"]
    fn runs_prefill_and_cached_decode() {
        let root = local_test_model_path();
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        let mut model = load_qwen3_5_moe_model(
            &config,
            &shards,
            &Device::Cpu,
            ModelDType::F32,
            ModelLoadOptions::default(),
        )
        .unwrap();

        let prefill = Tensor::new(&[[248_045u32, 248_046]], &Device::Cpu).unwrap();
        let logits = model.forward_eager(&prefill, 0).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
        // Reference values come from Transformers 5.15.1 running this exact
        // generated checkpoint. Checking a small slice catches errors in every
        // layer without committing a full-vocabulary golden tensor.
        let actual = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .narrow(0, 0, 8)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = [
            0.277_580_14,
            -0.442_775_2,
            -0.150_128_6,
            -0.196_054_95,
            0.415_801_55,
            0.306_672_04,
            -0.180_684_9,
            0.084_178_85,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }

        let decode = Tensor::new(&[[248_044u32]], &Device::Cpu).unwrap();
        let logits = model.forward_eager(&decode, 2).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
        let actual = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .narrow(0, 0, 8)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = [
            0.259_377_72,
            -0.322_923_9,
            0.244_929_43,
            -0.064_930_29,
            0.050_463_2,
            0.233_918_37,
            -0.520_859_66,
            -0.036_175_407,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }

        model.clear_cache();
        let logits = model.forward_eager(&prefill, 0).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
    }

    /// Constructs every text tensor in BF16 without invoking unsupported CPU ops.
    #[test]
    #[ignore = "loads BF16 weights from the local Qwen3.5 MoE development model"]
    fn loads_conditional_generation_bf16_weights() {
        let root = local_test_model_path();
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        validate_qwen3_5_moe_dtype(&config, ModelDType::BF16).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();

        // The public loader correctly rejects BF16 execution on Candle CPU. This
        // family-level construction test intentionally bypasses that device guard
        // to exercise nested prefixes and packed expert slicing at their native
        // checkpoint dtype without running a matrix multiplication.
        let model = load_qwen3_5_moe_model(
            &config,
            &shards,
            &Device::Cpu,
            ModelDType::BF16,
            ModelLoadOptions::default(),
        )
        .unwrap();

        drop(model);
    }

    /// Executes BF16 prefill, repeated cached decode, and cache reset on CUDA.
    #[test]
    #[ignore = "requires a CUDA GPU for Qwen3.5 MoE BF16 execution"]
    fn runs_conditional_generation_in_bf16_on_cuda() {
        let root = local_test_model_path();
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        validate_qwen3_5_moe_dtype(&config, ModelDType::BF16).unwrap();
        let device =
            Device::new_cuda(0).expect("BF16 Qwen3.5 test requires CUDA GPU 0");
        validate_qwen3_5_moe_device_dtype(&device, ModelDType::BF16).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        let mut model = load_qwen3_5_moe_model(
            &config,
            &shards,
            &device,
            ModelDType::BF16,
            ModelLoadOptions::default(),
        )
        .unwrap();

        let prefill = Tensor::new(&[[248_045u32, 248_046]], &device).unwrap();
        let logits = model.forward_eager(&prefill, 0).unwrap();

        assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
        assert_eq!(logits.dtype(), DType::BF16);
        assert!(!logits.device().is_cpu());

        for (step, token_id) in [248_044u32, 248_043, 248_042].into_iter().enumerate() {
            let decode = Tensor::new(&[[token_id]], &device).unwrap();
            let logits = model.forward_eager(&decode, 2 + step).unwrap();

            assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
            assert_eq!(logits.dtype(), DType::BF16);
            assert!(!logits.device().is_cpu());
        }

        model.clear_cache();
        let logits = model.forward_eager(&prefill, 0).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
        assert_eq!(logits.dtype(), DType::BF16);
    }

    /// Compares optimized MoE decode with the same-binary Candle reference path.
    #[test]
    #[ignore = "requires CUDA GPU 0 and the generated Qwen3.5 MoE model"]
    fn optimized_moe_cuda_matches_reference_prefill_and_decode() {
        assert_optimized_moe_reference(local_test_model_path(), 2e-1, true);
    }

    #[test]
    #[ignore = "requires the mixed-storage MoE fixture and a CUDA GPU"]
    fn mixed_storage_optimized_moe_matches_reference() {
        let root = std::env::var("TAU_QWEN3_5_MOE_MIXED_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/test-models/Qwen3.5-MoE-Mixed-Tiny-Dev")
            });
        // Direct FP32 initialization changes the fixture's weights. Apply the
        // established MoE error bound and require exact top-token agreement.
        assert_optimized_moe_reference(root, 2e-1, true);
    }

    fn assert_optimized_moe_reference(
        root: PathBuf,
        tolerance: f32,
        require_same_top: bool,
    ) {
        let assert_logits_match =
            |stage: &str, optimized: &Tensor, reference: &Tensor| {
                let optimized = optimized
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let reference = reference
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert!(
                    optimized
                        .iter()
                        .chain(&reference)
                        .all(|value| value.is_finite())
                );
                assert_eq!(optimized.len(), reference.len());
                let max_error = optimized
                    .iter()
                    .zip(&reference)
                    .map(|(&optimized, &reference)| (optimized - reference).abs())
                    .fold(0.0f32, f32::max);
                let optimized_top = optimized
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index)
                    .unwrap();
                let reference_top = reference
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index)
                    .unwrap();

                let reference_gap = reference[reference_top] - reference[optimized_top];
                let optimized_gap = optimized[optimized_top] - optimized[reference_top];
                if require_same_top {
                    assert_eq!(
                        optimized_top, reference_top,
                        "optimized/reference top token differs during {stage}"
                    );
                } else {
                    assert!(
                        reference_gap <= tolerance && optimized_gap <= tolerance,
                        "{stage} top-token disagreement exceeds tolerance: \
                         {reference_gap}, {optimized_gap}"
                    );
                    if optimized_top != reference_top {
                        eprintln!(
                            "{stage} near-tied argmax: reference_gap={reference_gap}, \
                             optimized_gap={optimized_gap}"
                        );
                    }
                }
                assert!(
                    max_error <= tolerance,
                    "maximum optimized/reference logit error {max_error} exceeds \
                     {tolerance} during {stage}"
                );
                println!(
                    "{stage} optimized/reference maximum logit error: {max_error}"
                );
            };

        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_moe_config(
            &root,
            CONDITIONAL_ARCHITECTURE,
            CONDITIONAL_MODEL_TYPE,
        )
        .unwrap();
        let device =
            Device::new_cuda(0).expect("BF16 Qwen3.5 comparison requires CUDA GPU 0");
        let shards = load_safetensors_shards(&root).unwrap();
        validate_qwen3_5_moe_tensor_contract(&config, &shards).unwrap();
        let mut optimized = load_qwen3_5_moe_model(
            &config,
            &shards,
            &device,
            ModelDType::BF16,
            ModelLoadOptions::default(),
        )
        .unwrap();
        let mut reference = load_qwen3_5_moe_model(
            &config,
            &shards,
            &device,
            ModelDType::BF16,
            ModelLoadOptions::new().with_optimize_kernel(false),
        )
        .unwrap();

        let prefill = Tensor::new(&[[248_045u32, 248_046]], &device).unwrap();
        let optimized_logits = optimized.forward_eager(&prefill, 0).unwrap();
        let reference_logits = reference.forward_eager(&prefill, 0).unwrap();
        assert_logits_match("prefill", &optimized_logits, &reference_logits);

        for (step, token_id) in [248_044u32, 248_043, 248_042].into_iter().enumerate() {
            let decode = Tensor::new(&[[token_id]], &device).unwrap();
            let optimized_logits = optimized.forward_eager(&decode, 2 + step).unwrap();
            let reference_logits = reference.forward_eager(&decode, 2 + step).unwrap();
            assert_logits_match(
                &format!("decode step {step}"),
                &optimized_logits,
                &reference_logits,
            );
        }

        optimized.clear_cache();
        reference.clear_cache();
        let optimized_logits = optimized.forward_eager(&prefill, 0).unwrap();
        let reference_logits = reference.forward_eager(&prefill, 0).unwrap();
        assert_logits_match("post-reset prefill", &optimized_logits, &reference_logits);
    }
}
