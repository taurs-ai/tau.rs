//! Dense Qwen3.5 model loading and reference execution.
//!
//! The supported checkpoint is the `Qwen3_5ForConditionalGeneration` wrapper
//! used by Qwen3.8 dense models. tau.rs constructs only its nested text decoder
//! and wrapper-level LM head; vision and MTP tensors remain memory-mapped and lazy.
//! The hybrid text backbone is shared with MoE; this façade retains the dense MLP,
//! checkpoint wrapper layout, and architecture-specific loading boundary.

mod config;
mod tensors;

pub(crate) use self::config::{
    load_qwen3_5_config, validate_qwen3_5_device_dtype, validate_qwen3_5_dtype,
};
pub(crate) use self::tensors::validate_qwen3_5_tensor_contract;

use self::config::{LoadedQwen3_5Config, Qwen3_5Config};
use crate::components::Mlp;
use crate::dtype::ModelDType;
use crate::qwen3_5_common::HybridCausalLm;
use crate::{ModelLoadError, ModelLoadOptions, ModelShard};
use candle::{Device, Result as CandleResult};
use candle_nn::VarBuilder;

/// Shared text model specialized with dense gated MLP feed-forwards.
pub(crate) type Qwen3_5Model = HybridCausalLm<Mlp>;

/// Builds the text-generation portion of the conditional-generation checkpoint.
fn build_qwen3_5_model(
    cfg: &Qwen3_5Config,
    vb: VarBuilder,
    options: ModelLoadOptions,
) -> CandleResult<Qwen3_5Model> {
    HybridCausalLm::new(
        &cfg.hybrid,
        "dense Qwen3.5",
        options,
        vb.pp("model").pp("language_model"),
        vb.pp("lm_head"),
        |_, mlp_vb| {
            Mlp::new(
                cfg.hybrid.hidden_size,
                cfg.intermediate_size,
                cfg.hybrid.hidden_act,
                options.optimize_kernel(),
                mlp_vb,
            )
        },
    )
}

/// Loads dense Qwen3.5 text weights from one or more validated shards.
pub(crate) fn load_qwen3_5_model(
    config: &LoadedQwen3_5Config,
    shards: &[ModelShard],
    device: &Device,
    dtype: ModelDType,
    options: ModelLoadOptions,
) -> Result<Qwen3_5Model, ModelLoadError> {
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

    build_qwen3_5_model(&config.model, var_builder, options)
        .map_err(|source| ModelLoadError::Candle(source.to_string()))
}

#[cfg(test)]
mod tests {
    use super::config::{ARCHITECTURE, MODEL_TYPE, load_qwen3_5_config};
    use super::*;
    use crate::safetensors::{load_safetensors_shards, validate_safetensors_shards};
    use candle::{DType, Tensor};
    use serde::Deserialize;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    const DENSE_NUMERICAL_REFERENCE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/qwen3_5_dense_reference.json"
    ));

    #[derive(Debug, Deserialize)]
    struct NumericalReference {
        schema_version: usize,
        generator: ReferenceGenerator,
        artifact: ReferenceArtifact,
        input: ReferenceInput,
        tolerance: ReferenceTolerance,
        outputs: ReferenceOutputs,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceGenerator {
        transformers_version: String,
        torch_version: String,
        python_version: String,
        device: String,
        dtype: String,
        attention_implementation: String,
        torch_threads: usize,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceArtifact {
        fixture: String,
        production_model_id: String,
        production_model_revision: String,
        random_seed: u64,
        files_sha256: BTreeMap<String, String>,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceInput {
        prefill_token_ids: Vec<u32>,
        decode_token_id: u32,
        decode_offset: usize,
        use_cache: bool,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceTolerance {
        absolute: f32,
        relative: f32,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceOutputs {
        prefill: ReferenceLogits,
        cached_decode: ReferenceLogits,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceLogits {
        slice: ReferenceLogitSlice,
        top_k: Vec<ReferenceTopLogit>,
        argmax_token_id: usize,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceLogitSlice {
        start: usize,
        values: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct ReferenceTopLogit {
        token_id: usize,
        value: f32,
    }

    /// Resolves the generated dense Qwen3.5 artifact used by ignored tests.
    fn local_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_DENSE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.8-Dense-Tiny-Dev")
            })
    }

    /// Converts final-position logits to FP32 and verifies numerical sanity.
    fn checked_logits_with_dtype(logits: &Tensor, expected_dtype: DType) -> Vec<f32> {
        assert_eq!(logits.dims3().unwrap(), (1, 1, 248_320));
        assert_eq!(logits.dtype(), expected_dtype);
        let values = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|value| value.is_finite()),
            "dense Qwen3.5 produced non-finite logits"
        );
        values
    }

    /// Converts one logits tensor to F32 and verifies that every value is finite.
    fn checked_logits(logits: &Tensor) -> Vec<f32> {
        checked_logits_with_dtype(logits, DType::F32)
    }

    fn numerical_reference() -> NumericalReference {
        parse_numerical_reference(DENSE_NUMERICAL_REFERENCE)
    }

    fn mixed_numerical_reference() -> NumericalReference {
        parse_numerical_reference(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/qwen3_5_dense_mixed_reference.json"
        )))
    }

    fn mixed_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_DENSE_MIXED_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/test-models/Qwen3.8-Dense-Mixed-Tiny-Dev")
            })
    }

    /// Validates the committed reference's dependency and artifact provenance.
    fn parse_numerical_reference(contents: &str) -> NumericalReference {
        let reference: NumericalReference = serde_json::from_str(contents)
            .expect("dense Qwen3.5 numerical reference must be valid JSON");
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.generator.transformers_version, "5.15.1");
        assert_eq!(reference.generator.torch_version, "2.13.0+cpu");
        assert_eq!(reference.generator.python_version, "3.12.13");
        assert_eq!(reference.generator.device, "cpu");
        assert_eq!(reference.generator.dtype, "float32");
        assert_eq!(reference.generator.attention_implementation, "eager");
        assert_eq!(reference.generator.torch_threads, 1);
        assert_eq!(
            reference.artifact.fixture,
            "Qwen3.8-27B miniature dense development checkpoint"
        );
        assert_eq!(reference.artifact.production_model_id, "Qwen/Qwen3.8-27B");
        assert_eq!(
            reference.artifact.production_model_revision,
            "72a217afab8029b39e4af1c7273a829995a3dbaf"
        );
        assert_eq!(reference.artifact.random_seed, 12_345);
        assert_eq!(reference.artifact.files_sha256.len(), 6);
        assert!(reference.artifact.files_sha256.values().all(|digest| {
            digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        }));
        assert!(reference.input.use_cache);
        assert_eq!(reference.tolerance.absolute, 2e-5);
        assert_eq!(reference.tolerance.relative, 0.0);
        for logits in [&reference.outputs.prefill, &reference.outputs.cached_decode] {
            assert_eq!(logits.slice.start, 0);
            assert_eq!(logits.slice.values.len(), 16);
            assert_eq!(logits.top_k.len(), 8);
            assert_eq!(logits.argmax_token_id, logits.top_k[0].token_id);
        }
        reference
    }

    /// Compares the committed logit slice with an explicit absolute tolerance.
    fn assert_slice_matches_reference(
        actual: &[f32],
        expected: &ReferenceLogits,
        tolerance: f32,
        stage: &str,
    ) -> f32 {
        let slice_end = expected.slice.start + expected.slice.values.len();
        assert!(slice_end <= actual.len());
        let mut max_absolute_error = 0.0_f32;
        for (index, (&actual, &expected_value)) in actual
            [expected.slice.start..slice_end]
            .iter()
            .zip(&expected.slice.values)
            .enumerate()
        {
            let absolute_error = (actual - expected_value).abs();
            max_absolute_error = max_absolute_error.max(absolute_error);
            assert!(
                absolute_error <= tolerance,
                "{stage} logit {} differs: {actual} != {expected_value} within \
                 {tolerance}",
                index + expected.slice.start
            );
        }
        max_absolute_error
    }

    /// Compares a logit slice and global top-k values with the pinned reference.
    fn assert_matches_reference(
        actual: &[f32],
        expected: &ReferenceLogits,
        tolerance: f32,
        stage: &str,
    ) {
        assert_slice_matches_reference(actual, expected, tolerance, stage);

        let mut ranked_indices = (0..actual.len()).collect::<Vec<_>>();
        ranked_indices
            .sort_unstable_by(|&left, &right| actual[right].total_cmp(&actual[left]));
        assert_eq!(ranked_indices[0], expected.argmax_token_id);
        for (&actual_token_id, expected) in ranked_indices.iter().zip(&expected.top_k) {
            assert_eq!(actual_token_id, expected.token_id, "{stage} top-k token");
            let actual_value = actual[actual_token_id];
            assert!(
                (actual_value - expected.value).abs() <= tolerance,
                "{stage} top-k logit for token {actual_token_id} differs: \
                 {actual_value} != {} within {tolerance}",
                expected.value
            );
        }
    }

    /// Requires two complete logit vectors to match without dumping them on failure.
    fn assert_exact_logits(actual: &[f32], expected: &[f32], stage: &str) {
        assert_eq!(actual.len(), expected.len());
        if let Some((index, (&actual, &expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            panic!("{stage} logit {index} differs: {actual} != {expected}");
        }
    }

    /// Compares complete logits from optimized and reference-kernel model instances.
    fn assert_optimized_logits_close(
        optimized: &[f32],
        reference: &[f32],
        tolerance: f32,
        stage: &str,
    ) -> f32 {
        assert_eq!(optimized.len(), reference.len());
        let mut max_absolute_error = 0.0_f32;
        for (token_id, (&optimized, &reference)) in
            optimized.iter().zip(reference).enumerate()
        {
            let absolute_error = (optimized - reference).abs();
            max_absolute_error = max_absolute_error.max(absolute_error);
            assert!(
                absolute_error <= tolerance,
                "{stage} token {token_id} differs: optimized={optimized}, \
                 reference={reference}, tolerance={tolerance}"
            );
        }
        let optimized_argmax = optimized
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("optimized logits cannot be empty");
        let reference_argmax = reference
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("reference logits cannot be empty");
        let reference_top_gap =
            reference[reference_argmax] - reference[optimized_argmax];
        let optimized_top_gap =
            optimized[optimized_argmax] - optimized[reference_argmax];
        assert!(
            reference_top_gap <= tolerance && optimized_top_gap <= tolerance,
            "{stage} top-token disagreement exceeds numerical tolerance: \
             optimized_token={optimized_argmax}, reference_token={reference_argmax}, \
             reference_gap={reference_top_gap}, optimized_gap={optimized_top_gap}"
        );
        if optimized_argmax != reference_argmax {
            eprintln!(
                "{stage} near-tied argmax: optimized_token={optimized_argmax}, \
                 reference_token={reference_argmax}, \
                 reference_gap={reference_top_gap:.8}, \
                 optimized_gap={optimized_top_gap:.8}"
            );
        }
        max_absolute_error
    }

    /// Executes dense cache behavior and checks pinned Transformers FP32 logits.
    #[test]
    #[ignore = "runs the local generated Qwen3.8 dense development model"]
    fn matches_dense_transformers_prefill_decode_and_cache_reset() {
        assert_cpu_transformers_reference(
            local_test_model_path(),
            numerical_reference(),
        );
    }

    #[test]
    #[ignore = "requires the generated mixed-storage dense fixture"]
    fn mixed_storage_matches_dense_transformers_prefill_decode_and_cache_reset() {
        assert_cpu_transformers_reference(
            mixed_test_model_path(),
            mixed_numerical_reference(),
        );
    }

    fn assert_cpu_transformers_reference(root: PathBuf, reference: NumericalReference) {
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        validate_qwen3_5_tensor_contract(&config, &shards).unwrap();
        let mut model = load_qwen3_5_model(
            &config,
            &shards,
            &Device::Cpu,
            ModelDType::F32,
            ModelLoadOptions::default(),
        )
        .unwrap();

        let prefill =
            Tensor::new(reference.input.prefill_token_ids.as_slice(), &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
        let first_prefill = checked_logits(&model.forward_eager(&prefill, 0).unwrap());
        assert_matches_reference(
            &first_prefill,
            &reference.outputs.prefill,
            reference.tolerance.absolute,
            "prefill",
        );

        let decode =
            Tensor::new(&[[reference.input.decode_token_id]], &Device::Cpu).unwrap();
        let decode = checked_logits(
            &model
                .forward_eager(&decode, reference.input.decode_offset)
                .unwrap(),
        );
        assert_matches_reference(
            &decode,
            &reference.outputs.cached_decode,
            reference.tolerance.absolute,
            "cached decode",
        );

        model.clear_cache();
        let reset_prefill = checked_logits(&model.forward_eager(&prefill, 0).unwrap());
        assert_exact_logits(&reset_prefill, &first_prefill, "cache-reset prefill");
    }

    /// Executes reduced-precision prefill, cached decode, and cache reset on CUDA.
    #[test]
    #[ignore = "requires a CUDA GPU for dense Qwen3.5 BF16 execution"]
    fn runs_dense_conditional_generation_in_bf16_on_cuda() {
        assert_cuda_transformers_reference(
            local_test_model_path(),
            numerical_reference(),
            true,
        );
    }

    #[test]
    #[ignore = "requires the mixed-storage dense fixture and a CUDA GPU"]
    fn mixed_storage_matches_dense_transformers_on_cuda() {
        for optimize in [false, true] {
            assert_cuda_transformers_reference(
                mixed_test_model_path(),
                mixed_numerical_reference(),
                optimize,
            );
        }
    }

    fn assert_cuda_transformers_reference(
        root: PathBuf,
        reference: NumericalReference,
        optimize: bool,
    ) {
        // This compares BF16 CUDA execution with the pinned CPU/F32 oracle. The
        // bound covers reduced-precision accumulation while remaining much tighter
        // than the fixture's typical unit-scale logits.
        const BF16_REFERENCE_ABSOLUTE_TOLERANCE: f32 = 5e-2;

        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        validate_qwen3_5_dtype(&config, ModelDType::BF16).unwrap();
        let device =
            Device::new_cuda(0).expect("dense Qwen3.5 BF16 test requires CUDA GPU 0");
        validate_qwen3_5_device_dtype(&device, ModelDType::BF16).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        validate_qwen3_5_tensor_contract(&config, &shards).unwrap();
        let mut model = load_qwen3_5_model(
            &config,
            &shards,
            &device,
            ModelDType::BF16,
            ModelLoadOptions::new().with_optimize_kernel(optimize),
        )
        .unwrap();

        let prefill =
            Tensor::new(reference.input.prefill_token_ids.as_slice(), &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
        let first_prefill = checked_logits_with_dtype(
            &model.forward_eager(&prefill, 0).unwrap(),
            DType::BF16,
        );
        let prefill_max_absolute_error = assert_slice_matches_reference(
            &first_prefill,
            &reference.outputs.prefill,
            BF16_REFERENCE_ABSOLUTE_TOLERANCE,
            "CUDA/BF16 prefill",
        );

        let mut cached_decode_max_absolute_error = None;
        for (step, token_id) in
            [reference.input.decode_token_id, 248_043u32, 248_042u32]
                .into_iter()
                .enumerate()
        {
            let decode = Tensor::new(&[[token_id]], &device).unwrap();
            let logits = checked_logits_with_dtype(
                &model
                    .forward_eager(&decode, reference.input.decode_offset + step)
                    .unwrap(),
                DType::BF16,
            );
            if step == 0 {
                cached_decode_max_absolute_error =
                    Some(assert_slice_matches_reference(
                        &logits,
                        &reference.outputs.cached_decode,
                        BF16_REFERENCE_ABSOLUTE_TOLERANCE,
                        "CUDA/BF16 cached decode",
                    ));
            }
        }

        eprintln!(
            "dense CUDA/BF16 reference errors: \
             prefill_max_abs={prefill_max_absolute_error:.8}, \
             cached_decode_max_abs={:.8}, \
             tolerance={BF16_REFERENCE_ABSOLUTE_TOLERANCE}",
            cached_decode_max_absolute_error.expect("first cached decode was checked")
        );

        model.clear_cache();
        let reset_prefill = checked_logits_with_dtype(
            &model.forward_eager(&prefill, 0).unwrap(),
            DType::BF16,
        );
        assert_exact_logits(
            &reset_prefill,
            &first_prefill,
            "CUDA/BF16 cache-reset prefill",
        );
    }

    /// Checks the public optimization switch against Candle across a cached sequence.
    #[test]
    #[ignore = "requires the local Qwen3.8 dense model and a CUDA GPU"]
    fn optimized_dense_bf16_matches_reference_kernels() {
        assert_optimized_matches_reference(local_test_model_path());
    }

    #[test]
    #[ignore = "requires the mixed-storage dense fixture and a CUDA GPU"]
    fn mixed_storage_optimized_dense_matches_reference_kernels() {
        assert_optimized_matches_reference(mixed_test_model_path());
    }

    fn assert_optimized_matches_reference(root: PathBuf) {
        const ABSOLUTE_TOLERANCE: f32 = 5e-2;

        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let reference_fixture = numerical_reference();
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        let device =
            Device::new_cuda(0).expect("dense optimization test requires CUDA GPU 0");
        let mut optimized = load_qwen3_5_model(
            &config,
            &shards,
            &device,
            ModelDType::BF16,
            ModelLoadOptions::new().with_optimize_kernel(true),
        )
        .unwrap();
        let mut reference = load_qwen3_5_model(
            &config,
            &shards,
            &device,
            ModelDType::BF16,
            ModelLoadOptions::new().with_optimize_kernel(false),
        )
        .unwrap();

        let prefill = Tensor::new(
            reference_fixture.input.prefill_token_ids.as_slice(),
            &device,
        )
        .unwrap()
        .unsqueeze(0)
        .unwrap();
        let optimized_prefill = checked_logits_with_dtype(
            &optimized.forward_eager(&prefill, 0).unwrap(),
            DType::BF16,
        );
        let reference_prefill = checked_logits_with_dtype(
            &reference.forward_eager(&prefill, 0).unwrap(),
            DType::BF16,
        );
        let prefill_max_absolute_error = assert_optimized_logits_close(
            &optimized_prefill,
            &reference_prefill,
            ABSOLUTE_TOLERANCE,
            "optimized prefill",
        );

        let mut decode_max_absolute_error = 0.0_f32;
        for (step, token_id) in [
            reference_fixture.input.decode_token_id,
            248_043u32,
            248_042u32,
            248_041u32,
        ]
        .into_iter()
        .enumerate()
        {
            let decode = Tensor::new(&[[token_id]], &device).unwrap();
            let optimized_logits = checked_logits_with_dtype(
                &optimized
                    .forward_eager(
                        &decode,
                        reference_fixture.input.decode_offset + step,
                    )
                    .unwrap(),
                DType::BF16,
            );
            let reference_logits = checked_logits_with_dtype(
                &reference
                    .forward_eager(
                        &decode,
                        reference_fixture.input.decode_offset + step,
                    )
                    .unwrap(),
                DType::BF16,
            );
            decode_max_absolute_error =
                decode_max_absolute_error.max(assert_optimized_logits_close(
                    &optimized_logits,
                    &reference_logits,
                    ABSOLUTE_TOLERANCE,
                    "optimized cached decode",
                ));
        }

        optimized.clear_cache();
        reference.clear_cache();
        let optimized_reset = checked_logits_with_dtype(
            &optimized.forward_eager(&prefill, 0).unwrap(),
            DType::BF16,
        );
        let reference_reset = checked_logits_with_dtype(
            &reference.forward_eager(&prefill, 0).unwrap(),
            DType::BF16,
        );
        let reset_max_absolute_error = assert_optimized_logits_close(
            &optimized_reset,
            &reference_reset,
            ABSOLUTE_TOLERANCE,
            "optimized cache-reset prefill",
        );
        eprintln!(
            "dense optimized/reference errors: \
             prefill_max_abs={prefill_max_absolute_error:.8}, \
             decode_max_abs={decode_max_absolute_error:.8}, \
             reset_max_abs={reset_max_absolute_error:.8}"
        );
    }

    /// Constructs BF16 bulk weights with FP32 Gated DeltaNet parameters.
    #[test]
    #[ignore = "loads BF16 weights from the local Qwen3.8 dense development model"]
    fn loads_dense_conditional_generation_bf16_weights() {
        let root = local_test_model_path();
        assert!(root.is_dir(), "test model is absent: {}", root.display());
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        validate_qwen3_5_dtype(&config, ModelDType::BF16).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        validate_safetensors_shards(&shards).unwrap();
        validate_qwen3_5_tensor_contract(&config, &shards).unwrap();

        // The public loader rejects BF16 execution on Candle CPU. This
        // construction-only test deliberately bypasses that device guard so all
        // production prefixes and dense weights are checked without a matmul.
        let model = load_qwen3_5_model(
            &config,
            &shards,
            &Device::Cpu,
            ModelDType::BF16,
            ModelLoadOptions::default(),
        )
        .unwrap();

        assert_eq!(model.dtype(), DType::BF16);
        assert!(model.device().is_cpu());
        drop(model);
    }
    fn assert_exact_sequence_state(actual: &Qwen3_5Model, expected: &Qwen3_5Model) {
        let actual = actual.snapshot_current_state().unwrap();
        let expected = expected.snapshot_current_state().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.shape(), expected.shape(), "state tensor {index}");
            assert_eq!(actual.dtype(), expected.dtype(), "state tensor {index}");
            let values = |t: &Tensor| {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let a = values(actual);
            let e = values(expected);
            assert!(
                a.iter().chain(&e).all(|v| v.is_finite()),
                "nonfinite state tensor {index}"
            );
            assert_eq!(a, e, "state tensor {index}");
        }
    }

    /// Full-vocabulary/state comparison using the matching eager arithmetic.
    #[test]
    #[ignore = "requires the dense tiny fixture and CUDA GPU 0"]
    fn cuda_graph_dense_decode_matches_eager_across_requests() {
        let root = local_test_model_path();
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        let mlp_layers = config.model.hybrid.num_hidden_layers;
        let gdn_layers = config
            .model
            .hybrid
            .layer_types
            .iter()
            .filter(|layer| {
                matches!(
                    layer,
                    crate::qwen3_5_common::HybridLayerType::LinearAttention
                )
            })
            .count();
        let shards = load_safetensors_shards(&root).unwrap();
        // SAFETY: Test owns every tensor; all model operations use this thread/stream.
        let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
        let load = || {
            load_qwen3_5_model(
                &config,
                &shards,
                &device,
                ModelDType::BF16,
                ModelLoadOptions::default(),
            )
            .unwrap()
        };
        let mut graph = load();
        let mut eager = load();
        use crate::CudaGraphStats;
        // Cover all three policies on the same model, then return to eager and
        // re-enable Full. The repeated Full checks that retained graphs still
        // produce correct logits/state without recapture after eager requests
        // and cache resets.
        for mode in [
            crate::CudaGraphMode::None,
            crate::CudaGraphMode::Layer,
            crate::CudaGraphMode::Full,
            crate::CudaGraphMode::None,
            crate::CudaGraphMode::Full,
        ] {
            graph.set_cuda_graph(mode);
            assert_eq!(graph.cuda_graph_mode(), mode);
            let cuda_graph = mode == crate::CudaGraphMode::Layer;
            eager.set_cuda_graph(if mode == crate::CudaGraphMode::Full {
                mode
            } else {
                crate::CudaGraphMode::None
            });
            for tokens in [vec![1u32], vec![1, 2, 3], vec![5, 4], vec![1]] {
                graph.clear_cache();
                eager.clear_cache();
                // Keep cache layout fixed while testing execution-policy changes.
                // Storage/layout invalidation has its own lifecycle regression.
                graph.set_cuda_graph(crate::CudaGraphMode::Full);
                eager.set_cuda_graph(crate::CudaGraphMode::Full);
                graph.reserve_kv_cache(2048).unwrap();
                eager.reserve_kv_cache(2048).unwrap();
                graph.set_cuda_graph(mode);
                eager.set_cuda_graph(if mode == crate::CudaGraphMode::Full {
                    mode
                } else {
                    crate::CudaGraphMode::None
                });
                let input = Tensor::new(tokens.as_slice(), &device)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap();
                let before = graph.cuda_graph_counts();
                let expected = eager.forward_eager(&input, 0).unwrap();
                let actual = graph.forward_eager(&input, 0).unwrap();
                assert_exact_logits(
                    &checked_logits_with_dtype(&actual, DType::BF16),
                    &checked_logits_with_dtype(&expected, DType::BF16),
                    "graph eager prefill",
                );
                assert_exact_sequence_state(&graph, &eager);
                assert_eq!(graph.cuda_graph_counts(), before);
                let steps = if mode == crate::CudaGraphMode::Full && tokens.len() == 3 {
                    512
                } else {
                    8
                };
                for step in 0..steps {
                    let input = Tensor::new(&[step as u32 + 7], &device)
                        .unwrap()
                        .unsqueeze(0)
                        .unwrap();
                    let before = graph.cuda_graph_counts();
                    let expected = if mode == crate::CudaGraphMode::Full {
                        eager
                            .forward_prepared_eager(
                                step as u32 + 7,
                                tokens.len() + step,
                            )
                            .unwrap()
                    } else {
                        eager.forward_eager(&input, tokens.len() + step).unwrap()
                    };
                    let actual =
                        graph.forward_decode(&input, tokens.len() + step).unwrap();
                    assert_exact_logits(
                        &checked_logits_with_dtype(&actual, DType::BF16),
                        &checked_logits_with_dtype(&expected, DType::BF16),
                        "graph decode",
                    );
                    assert_exact_sequence_state(&graph, &eager);
                    let delta = graph.cuda_graph_counts().since(before);
                    delta.validate_mode(mode).unwrap();
                    assert_eq!(
                        delta.forward_graph_replays,
                        usize::from(mode == crate::CudaGraphMode::Full)
                    );
                    assert_eq!(
                        delta.forward_graph_captures,
                        usize::from(
                            mode == crate::CudaGraphMode::Full
                                && before.forward_graph_captures == 0
                        )
                    );
                    assert_eq!(
                        delta.gdn_layer_graph_replays,
                        if cuda_graph { gdn_layers } else { 0 }
                    );
                    assert_eq!(
                        delta.mlp_graph_replays,
                        if cuda_graph {
                            mlp_layers - gdn_layers
                        } else {
                            0
                        }
                    );
                    assert_eq!(
                        delta.gdn_layer_graph_captures,
                        if cuda_graph && before.gdn_layer_graph_captures == 0 {
                            gdn_layers
                        } else {
                            0
                        }
                    );
                    assert_eq!(
                        delta.mlp_graph_captures,
                        if cuda_graph && before.mlp_graph_captures == 0 {
                            mlp_layers - gdn_layers
                        } else {
                            0
                        }
                    );
                    if mode == crate::CudaGraphMode::None {
                        assert_eq!(delta, CudaGraphStats::default());
                    }
                }
            }
        }
    }
    #[test]
    #[ignore = "requires the dense tiny fixture and CUDA GPU 0"]
    fn prefill_chunk_graph_dense_matches_eager_across_requests() {
        let root = local_test_model_path();
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        let gdn_layers = config
            .model
            .hybrid
            .layer_types
            .iter()
            .filter(|kind| {
                matches!(
                    kind,
                    crate::qwen3_5_common::HybridLayerType::LinearAttention
                )
            })
            .count();
        let attention_layers = config.model.hybrid.num_hidden_layers - gdn_layers;
        let shards = load_safetensors_shards(&root).unwrap();
        // SAFETY: Model use, state updates and teardown are serialized on this thread.
        let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
        let load = || {
            load_qwen3_5_model(
                &config,
                &shards,
                &device,
                ModelDType::BF16,
                ModelLoadOptions::default(),
            )
            .unwrap()
        };
        for chunk_size in [1usize, 2, 4, 8] {
            let mut captured = load();
            let mut eager = load();
            let mut prefill_captured = false;
            // Start with both graph policies, run eager requests, then reuse
            // Full on the same model. Check that prefill graphs survive those
            // transitions and resets without recapture or stale sequence state.
            for mode in [
                crate::CudaGraphMode::Layer,
                crate::CudaGraphMode::Full,
                crate::CudaGraphMode::None,
                crate::CudaGraphMode::Full,
            ] {
                captured.set_cuda_graph(mode);
                let enabled = mode.is_enabled();
                eager.set_cuda_graph(if mode == crate::CudaGraphMode::Full {
                    mode
                } else {
                    crate::CudaGraphMode::None
                });
                for length in [
                    chunk_size.saturating_sub(1).max(1),
                    chunk_size,
                    2 * chunk_size + 1,
                    2 * chunk_size,
                    1,
                ] {
                    captured.clear_cache();
                    eager.clear_cache();
                    captured.set_cuda_graph(crate::CudaGraphMode::Full);
                    eager.set_cuda_graph(crate::CudaGraphMode::Full);
                    captured.reserve_kv_cache(2048).unwrap();
                    eager.reserve_kv_cache(2048).unwrap();
                    captured.set_cuda_graph(mode);
                    eager.set_cuda_graph(if mode == crate::CudaGraphMode::Full {
                        mode
                    } else {
                        crate::CudaGraphMode::None
                    });
                    let tokens = (0..length).map(|i| i as u32 + 1).collect::<Vec<_>>();
                    for offset in (0..length).step_by(chunk_size) {
                        let count = chunk_size.min(length - offset);
                        let input =
                            Tensor::new(&tokens[offset..offset + count], &device)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap();
                        let before = captured.cuda_graph_counts();
                        // Project every chunk in this test so every boundary's full
                        // vocabulary is checked; serving only projects the final one.
                        let actual = captured
                            .forward_prefill_chunk(
                                &input,
                                offset,
                                count == chunk_size,
                                true,
                            )
                            .unwrap()
                            .unwrap();
                        let expected = eager
                            .forward_prefill_chunk(&input, offset, false, true)
                            .unwrap()
                            .unwrap();
                        assert_exact_logits(
                            &checked_logits_with_dtype(&actual, DType::BF16),
                            &checked_logits_with_dtype(&expected, DType::BF16),
                            "prefill chunk replay",
                        );
                        assert_exact_sequence_state(&captured, &eager);
                        let delta = captured.cuda_graph_counts().since(before);
                        delta.validate_mode(mode).unwrap();
                        let replay = enabled && count == chunk_size;
                        assert_eq!(
                            delta.gdn_layer_graph_replays,
                            if replay { gdn_layers } else { 0 }
                        );
                        assert_eq!(
                            delta.mlp_graph_replays,
                            if replay { attention_layers } else { 0 }
                        );
                        assert_eq!(
                            delta.gdn_layer_graph_captures,
                            if replay && !prefill_captured {
                                gdn_layers
                            } else {
                                0
                            }
                        );
                        assert_eq!(
                            delta.mlp_graph_captures,
                            if replay && !prefill_captured {
                                attention_layers
                            } else {
                                0
                            }
                        );
                        prefill_captured |= replay;
                    }
                    for step in 0..3 {
                        let input = Tensor::new(&[step as u32 + 31], &device)
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap();
                        let actual =
                            captured.forward_decode(&input, length + step).unwrap();
                        let expected = if mode == crate::CudaGraphMode::Full {
                            eager
                                .forward_prepared_eager(step as u32 + 31, length + step)
                                .unwrap()
                        } else {
                            eager.forward_decode(&input, length + step).unwrap()
                        };
                        assert_exact_logits(
                            &checked_logits_with_dtype(&actual, DType::BF16),
                            &checked_logits_with_dtype(&expected, DType::BF16),
                            "decode after chunked prefill",
                        );
                    }
                }
            }
        }
    }
    /// Chunking changes GEMM shapes, so compare against monolithic prefill using
    /// the existing numerical bounds as well as subsequent cache-dependent decode.
    #[test]
    #[ignore = "requires the dense tiny fixture and CUDA GPU 0"]
    fn chunked_prefill_matches_unchunked_dense_reference() {
        let root = local_test_model_path();
        let config = load_qwen3_5_config(&root, ARCHITECTURE, MODEL_TYPE).unwrap();
        let shards = load_safetensors_shards(&root).unwrap();
        for (device, dtype, tolerance) in [
            (Device::Cpu, ModelDType::F32, 2e-5),
            (Device::new_cuda(0).unwrap(), ModelDType::BF16, 0.05),
        ] {
            let load = || {
                load_qwen3_5_model(
                    &config,
                    &shards,
                    &device,
                    dtype,
                    ModelLoadOptions::default(),
                )
                .unwrap()
            };
            let mut chunked = load();
            let mut whole = load();
            for chunk_size in [1, 4, 8] {
                let length = 2 * chunk_size + 3;
                let tokens = (0..length).map(|i| i as u32 + 1).collect::<Vec<_>>();
                let input = Tensor::new(tokens.as_slice(), &device)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap();
                chunked.clear_cache();
                whole.clear_cache();
                let mut actual = None;
                for offset in (0..length).step_by(chunk_size) {
                    let count = chunk_size.min(length - offset);
                    actual = chunked
                        .forward_prefill_chunk(
                            &input.narrow(1, offset, count).unwrap(),
                            offset,
                            count == chunk_size,
                            offset + count == length,
                        )
                        .unwrap();
                }
                let expected = whole.forward_eager(&input, 0).unwrap();
                assert_optimized_logits_close(
                    &checked_logits_with_dtype(&actual.unwrap(), expected.dtype()),
                    &checked_logits_with_dtype(&expected, expected.dtype()),
                    tolerance,
                    "chunked versus monolithic prefill",
                );
                for offset in length..length + 3 {
                    let input = Tensor::new(&[31u32], &device)
                        .unwrap()
                        .unsqueeze(0)
                        .unwrap();
                    let actual = chunked.forward_decode(&input, offset).unwrap();
                    let expected = whole.forward_decode(&input, offset).unwrap();
                    assert_optimized_logits_close(
                        &checked_logits_with_dtype(&actual, expected.dtype()),
                        &checked_logits_with_dtype(&expected, expected.dtype()),
                        tolerance,
                        "decode after chunked versus monolithic prefill",
                    );
                }
            }
        }
    }
}
