use super::super::routing::select_experts;
use super::*;

/// Replays captured reference inputs through each stateless sparse block.
/// This separates feed-forward arithmetic from upstream cached model state.
#[test]
#[ignore = "requires CUDA, a production-width MoE checkpoint and captured inputs"]
fn diagnose_captured_moe_inputs() {
    use std::path::PathBuf;
    let model = PathBuf::from(std::env::var("TAU_MOE_DIAGNOSTIC_MODEL").unwrap());
    let input = PathBuf::from(std::env::var("TAU_MOE_DIAGNOSTIC_INPUT").unwrap());
    let output = PathBuf::from(std::env::var("TAU_MOE_DIAGNOSTIC_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let metadata = crate::load_model_config(&model).unwrap();
    let cfg = crate::qwen3_5_moe::load_qwen3_5_moe_config(
        &model,
        &metadata.architecture,
        &metadata.model_type,
    )
    .unwrap();
    let cfg = cfg.model.sparse_moe_config(SparseMoeWeightLayout::Packed);
    let shards = crate::safetensors::load_safetensors_shards(&model).unwrap();
    let paths: Vec<_> = shards.iter().map(|s| s.path().to_path_buf()).collect();
    // SAFETY: Diagnostic eager/prepared work and readback share one private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    // SAFETY: Checkpoint mappings are read-only throughout the diagnostic.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device) }
            .unwrap();
    let read = |name: &str| {
        std::fs::read(input.join(name))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let dump = |name: &str, tensor: &Tensor| {
        let values = tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(output.join(format!("{name}.f32")), bytes).unwrap();
    };
    for layer in 0..4 {
        let values = read(&format!("layer-{layer}-input.f32"));
        assert_eq!(values.len(), cfg.hidden_size);
        let xs = Tensor::from_vec(values, (1, cfg.hidden_size), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weights = vb.pp(format!("model.language_model.layers.{layer}.mlp"));
        for optimize in [false, true] {
            let block = SparseMoeBlock::new(&cfg, optimize, weights.clone()).unwrap();
            let prefix = format!(
                "layer-{layer}-{}",
                if optimize { "optimized" } else { "native" }
            );
            let router = block.router.logits(&xs).unwrap();
            dump(&format!("{prefix}-router"), &router);
            let routed = block.forward_routed(&xs, &router).unwrap();
            dump(&format!("{prefix}-routed"), &routed);
            if let Some(runner) = &block.fused_moe_runner {
                let (ids, scores) = runner.selected_routing().unwrap();
                dump(
                    &format!("{prefix}-ids"),
                    &Tensor::new(ids.as_slice(), &Device::Cpu).unwrap(),
                );
                dump(
                    &format!("{prefix}-scores"),
                    &Tensor::new(scores.as_slice(), &Device::Cpu).unwrap(),
                );
            } else {
                let scores = candle_nn::ops::softmax_last_dim(
                    &router.to_dtype(DType::F32).unwrap(),
                )
                .unwrap();
                let scores = scores.to_vec2::<f32>().unwrap();
                let ids: Vec<u32> = select_experts(&scores[0], cfg.experts_per_token)
                    .unwrap()
                    .iter()
                    .map(|id| *id as u32)
                    .collect();
                dump(
                    &format!("{prefix}-ids"),
                    &Tensor::new(ids.as_slice(), &Device::Cpu).unwrap(),
                );
            }
            let shared = block.shared_expert.as_ref().unwrap().forward(&xs).unwrap();
            dump(&format!("{prefix}-shared"), &shared);
            dump(&format!("{prefix}-output"), &(&routed + &shared).unwrap());
            if optimize {
                // Probe prepared stages on exactly the same captured input. This
                // isolates BLAS/rounding changes from upstream recurrent state.
                let mut program =
                    crate::components::DecodeProgram::new(&device).unwrap();
                let input = xs.reshape((1, 1, cfg.hidden_size)).unwrap();
                let router = block
                    .router
                    .prepare_decode_program(&mut program, &input)
                    .unwrap();
                let routed = block
                    .fused_moe_runner
                    .as_ref()
                    .unwrap()
                    .prepare_decode_program(&mut program, &input, &router)
                    .unwrap();
                let combined = block
                    .shared_expert
                    .as_ref()
                    .unwrap()
                    .prepare_decode_combine(&mut program, &input, &routed)
                    .unwrap();
                program.run().unwrap();
                for (name, value) in [
                    ("router", &router),
                    ("routed", &routed),
                    ("output", &combined),
                ] {
                    dump(&format!("layer-{layer}-prepared-{name}"), value);
                }
                let (ids, scores) = block
                    .fused_moe_runner
                    .as_ref()
                    .unwrap()
                    .selected_routing()
                    .unwrap();
                dump(
                    &format!("layer-{layer}-prepared-ids"),
                    &Tensor::new(ids.as_slice(), &Device::Cpu).unwrap(),
                );
                dump(
                    &format!("layer-{layer}-prepared-scores"),
                    &Tensor::new(scores.as_slice(), &Device::Cpu).unwrap(),
                );
            }
            // Tau-only layer captures can isolate preparation without an external
            // route dump; forced-reference arithmetic requires both route files.
            if !input.join(format!("layer-{layer}-topk-ids.f32")).exists() {
                continue;
            }
            // Force the reference routes and weights to isolate combination
            // rounding from possible router/selection differences.
            let ids = read(&format!("layer-{layer}-topk-ids.f32"));
            let scores = read(&format!("layer-{layer}-topk-weights.f32"));
            let mut sum =
                Tensor::zeros((1, cfg.hidden_size), DType::F32, &device).unwrap();
            let mut rounded_sum = sum.clone();
            for (id, score) in ids.iter().zip(&scores) {
                let expert = block
                    .experts
                    .expert_for_test(*id as usize)
                    .forward(&xs)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let weighted = (expert * f64::from(*score)).unwrap();
                sum = (&sum + &weighted).unwrap();
                let rounded = weighted
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                rounded_sum = (&rounded_sum + rounded).unwrap();
            }
            dump(
                &format!("{prefix}-forced-f32-sum"),
                &sum.to_dtype(DType::BF16).unwrap(),
            );
            dump(
                &format!("{prefix}-forced-rounded-sum"),
                &rounded_sum.to_dtype(DType::BF16).unwrap(),
            );
        }
    }
}
