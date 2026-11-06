use super::*;

/// Verifies top-k dispatch, selected-weight normalization, and shared gating.
#[test]
fn routes_tokens_and_adds_gated_shared_expert() {
    let device = Device::Cpu;
    let identity = Tensor::new(&[[1f32, 0.0], [0.0, 1.0]], &device).unwrap();
    let doubled = Tensor::new(&[[2f32, 0.0], [0.0, 2.0]], &device).unwrap();
    let zero_gate = Tensor::zeros((1, 2), DType::F32, &device).unwrap();
    let tensors = HashMap::from([
        ("gate.weight".to_owned(), identity.clone()),
        ("experts.0.gate_proj.weight".to_owned(), identity.clone()),
        ("experts.0.up_proj.weight".to_owned(), identity.clone()),
        ("experts.0.down_proj.weight".to_owned(), identity.clone()),
        ("experts.1.gate_proj.weight".to_owned(), identity.clone()),
        ("experts.1.up_proj.weight".to_owned(), identity.clone()),
        ("experts.1.down_proj.weight".to_owned(), doubled),
        (
            "shared_expert.gate_proj.weight".to_owned(),
            identity.clone(),
        ),
        ("shared_expert.up_proj.weight".to_owned(), identity.clone()),
        ("shared_expert.down_proj.weight".to_owned(), identity),
        ("shared_expert_gate.weight".to_owned(), zero_gate),
    ]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let moe = SparseMoeBlock::new(&valid_config(), false, vb).unwrap();
    let input = Tensor::new(&[[[1f32, 0.0], [0.0, 1.0]]], &device).unwrap();

    let output = moe.forward(&input).unwrap().to_vec3::<f32>().unwrap();

    assert_eq!(output, vec![vec![vec![1.5, 0.0], vec![0.0, 2.5]]]);
}

/// Verifies packed expert weights use the same routing and MLP execution.
#[test]
fn routes_tokens_through_packed_experts() {
    let device = Device::Cpu;
    let identity = Tensor::new(&[[1f32, 0.0], [0.0, 1.0]], &device).unwrap();
    let doubled = Tensor::new(&[[2f32, 0.0], [0.0, 2.0]], &device).unwrap();
    let gate_up = Tensor::stack(
        &[
            &Tensor::cat(&[&identity, &identity], 0).unwrap(),
            &Tensor::cat(&[&identity, &identity], 0).unwrap(),
        ],
        0,
    )
    .unwrap();
    let down = Tensor::stack(&[&identity, &doubled], 0).unwrap();
    let zero_gate = Tensor::zeros((1, 2), DType::F32, &device).unwrap();
    let tensors = HashMap::from([
        ("gate.weight".to_owned(), identity.clone()),
        ("experts.gate_up_proj".to_owned(), gate_up),
        ("experts.down_proj".to_owned(), down),
        (
            "shared_expert.gate_proj.weight".to_owned(),
            identity.clone(),
        ),
        ("shared_expert.up_proj.weight".to_owned(), identity.clone()),
        ("shared_expert.down_proj.weight".to_owned(), identity),
        ("shared_expert_gate.weight".to_owned(), zero_gate),
    ]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let mut config = valid_config();
    config.weight_layout = SparseMoeWeightLayout::Packed;
    let moe = SparseMoeBlock::new(&config, false, vb).unwrap();
    let input = Tensor::new(&[[[1f32, 0.0], [0.0, 1.0]]], &device).unwrap();

    let output = moe.forward(&input).unwrap().to_vec3::<f32>().unwrap();

    assert_eq!(output, vec![vec![vec![1.5, 0.0], vec![0.0, 2.5]]]);
}
