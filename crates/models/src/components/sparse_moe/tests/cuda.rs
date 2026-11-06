use super::*;
use crate::components::GraphFeedForward;

/// Compares one-token device-resident routing with the Candle reference path.
#[test]
fn fused_bf16_cuda_decode_matches_reference() {
    const HIDDEN_SIZE: usize = 32;
    const INTERMEDIATE_SIZE: usize = 16;
    const NUM_EXPERTS: usize = 256;
    const TOP_K: usize = 8;

    let device =
        Device::new_cuda(0).expect("fused MoE comparison test requires CUDA GPU 0");
    let mut router_values = vec![0.0f32; NUM_EXPERTS * HIDDEN_SIZE];
    for expert in 0..NUM_EXPERTS {
        router_values[expert * HIDDEN_SIZE] = expert as f32 / 32.0;
    }
    let gate_up_values = (0..NUM_EXPERTS * 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE)
        .map(|index| (index % 29) as f32 * 0.002 - 0.028)
        .collect::<Vec<_>>();
    let down_values = (0..NUM_EXPERTS * HIDDEN_SIZE * INTERMEDIATE_SIZE)
        .map(|index| (index % 23) as f32 * 0.003 - 0.033)
        .collect::<Vec<_>>();
    let router = Tensor::from_vec(router_values, (NUM_EXPERTS, HIDDEN_SIZE), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let gate_up = Tensor::from_vec(
        gate_up_values,
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        &device,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let down = Tensor::from_vec(
        down_values,
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        &device,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let tensors = HashMap::from([
        ("gate.weight".to_owned(), router),
        ("experts.gate_up_proj".to_owned(), gate_up),
        ("experts.down_proj".to_owned(), down),
    ]);
    let config = SparseMoeConfig {
        hidden_size: HIDDEN_SIZE,
        expert_intermediate_size: INTERMEDIATE_SIZE,
        expert_activation: Activation::Silu,
        num_experts: NUM_EXPERTS,
        experts_per_token: TOP_K,
        router_bias: false,
        router_score_function: RouterScoreFunction::Softmax,
        normalize_selected_weights: true,
        shared_expert: None,
        weight_layout: SparseMoeWeightLayout::Packed,
    };
    let optimized = SparseMoeBlock::new(
        &config,
        true,
        VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
    )
    .unwrap();
    let reference = SparseMoeBlock::new(
        &config,
        false,
        VarBuilder::from_tensors(tensors, DType::BF16, &device),
    )
    .unwrap();
    assert!(optimized.fused_moe_runner.is_some());

    let input_values = (0..HIDDEN_SIZE)
        .map(|column| column as f32 * 0.01 - 0.12)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(input_values, (1, 1, HIDDEN_SIZE), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let optimized = optimized
        .forward_decode(&input)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let reference = reference
        .forward(&input)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for (&optimized, reference) in optimized.iter().zip(reference) {
        assert!(optimized.is_finite());
        assert!(
            (optimized - reference).abs() <= 2e-2,
            "optimized value {optimized} differs from reference {reference}"
        );
    }
}
