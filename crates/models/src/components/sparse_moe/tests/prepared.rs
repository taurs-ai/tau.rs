//! Exact block-level checks for dynamic routing and shared-expert graph execution.

use super::*;
use crate::components::{CapturedGraph, DecodeProgram, GraphFeedForward};
use candle::Var;

/// Reads every BF16 output as FP32 without discarding any stored precision.
fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Changes the selected expert set and shared gate across a single retained
/// graph. The zero input also forces an exact top-k tie across all 256 experts.
#[test]
fn prepared_moe_matches_eager_with_changing_routes_and_shared_gate() {
    const H: usize = 32;
    const I: usize = 16;
    const E: usize = 256;
    // SAFETY: The block, graph and all ordinary work use this private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let tensor = |data: Vec<f32>, shape: &[usize]| {
        Tensor::from_vec(data, shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    let mut router = vec![0.; E * H];
    for id in 0..E {
        router[id * H] = id as f32 / 32.;
    }
    let mut shared_gate = vec![0.; H];
    shared_gate[0] = 16.;
    let weights = |count, seed| {
        (0..count)
            .map(|i| (((i * 17 + seed) % 31) as f32 - 15.) / 128.)
            .collect::<Vec<_>>()
    };
    let tensors = HashMap::from([
        ("gate.weight".to_owned(), tensor(router, &[E, H])),
        (
            "experts.gate_up_proj".to_owned(),
            tensor(weights(E * 2 * I * H, 1), &[E, 2 * I, H]),
        ),
        (
            "experts.down_proj".to_owned(),
            tensor(weights(E * H * I, 2), &[E, H, I]),
        ),
        (
            "shared_expert.gate_proj.weight".to_owned(),
            tensor(weights(I * H, 3), &[I, H]),
        ),
        (
            "shared_expert.up_proj.weight".to_owned(),
            tensor(weights(I * H, 4), &[I, H]),
        ),
        (
            "shared_expert.down_proj.weight".to_owned(),
            tensor(weights(H * I, 5), &[H, I]),
        ),
        (
            "shared_expert_gate.weight".to_owned(),
            tensor(shared_gate, &[1, H]),
        ),
    ]);
    let cfg = SparseMoeConfig {
        hidden_size: H,
        expert_intermediate_size: I,
        expert_activation: Activation::Silu,
        num_experts: E,
        experts_per_token: 8,
        router_bias: false,
        router_score_function: RouterScoreFunction::Softmax,
        normalize_selected_weights: true,
        shared_expert: Some(SharedExpertConfig {
            intermediate_size: I,
            activation: Activation::Silu,
            gate: Some(SharedExpertGateConfig {
                activation: Activation::Sigmoid,
                bias: false,
            }),
        }),
        weight_layout: SparseMoeWeightLayout::Packed,
    };
    let block = SparseMoeBlock::new(
        &cfg,
        true,
        VarBuilder::from_tensors(tensors, DType::BF16, &device),
    )
    .unwrap();
    let input = Var::zeros((1, 1, H), DType::BF16, &device).unwrap();
    assert!(block.supports_prepared_decode(&input));
    assert!(!block.supports_layer_graph(&input));
    assert!(!block.supports_prepared_decode(&input.broadcast_as((1, 2, H)).unwrap()));
    let mut program = DecodeProgram::new(&device).unwrap();
    let output = block.prepare_decode_program(&mut program, &input).unwrap();
    program.run().unwrap();
    device.synchronize().unwrap();
    // SAFETY: The closure borrows owners retained until after graph destruction.
    let mut graph =
        unsafe { CapturedGraph::capture(&device, || program.run()) }.unwrap();
    assert_eq!(graph.allocation_nodes().unwrap(), 0);
    for first in [1., -1., 0., 4., -4., 1.] {
        let mut data = weights(H, 9);
        data[0] = first;
        input.set(&tensor(data, &[1, 1, H])).unwrap();
        graph.replay().unwrap();
        let replay = values(&output);
        let replay_routes = block
            .fused_moe_runner
            .as_ref()
            .unwrap()
            .selected_routing()
            .unwrap();
        program.run().unwrap();
        assert_eq!(values(&output), replay, "prepared eager/replay at {first}");
        let eager = block.forward_decode(&input).unwrap();
        assert_eq!(values(&eager), replay, "ordinary/prepared at {first}");
        assert_eq!(
            values(&block.forward(&input).unwrap()),
            replay,
            "one-token prefill/decode precision differs at {first}"
        );
        let eager_routes = block
            .fused_moe_runner
            .as_ref()
            .unwrap()
            .selected_routing()
            .unwrap();
        assert_eq!(replay_routes, eager_routes);
        let expected_ids = if first > 0. {
            (248..256).rev().collect::<Vec<i32>>()
        } else {
            (0..8).collect()
        };
        assert_eq!(replay_routes.0, expected_ids);
    }
}
