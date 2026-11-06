//! Projection-layout, gating and optimized/reference arithmetic regressions.

use super::*;
use candle::{DType, Device};
use std::collections::HashMap;

/// The explicit weighted path must not change dense forward or its handle
/// precision. Check every stored bit before/after a routed-style invocation.
#[test]
fn weighted_projection_preserves_ordinary_dense_forward() {
    let device = Device::new_cuda(0).unwrap();
    let matrix = |rows, cols, seed| {
        Tensor::new(
            (0..rows * cols)
                .map(|i| (((i * 13 + seed) % 31) as f32 - 15.) / 64.)
                .collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .reshape((rows, cols))
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
    };
    let weights = HashMap::from([
        ("gate_proj.weight".to_owned(), matrix(16, 32, 1)),
        ("up_proj.weight".to_owned(), matrix(16, 32, 2)),
        ("down_proj.weight".to_owned(), matrix(32, 16, 3)),
    ]);
    let mlp = Mlp::new(
        32,
        16,
        Activation::Silu,
        true,
        VarBuilder::from_tensors(weights, DType::BF16, &device),
    )
    .unwrap();
    let input = matrix(17, 32, 4);
    let bits = |tensor: Tensor| {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let before = bits(mlp.forward(&input).unwrap());
    let scale = Tensor::new(vec![0.9f32; 17], &device)
        .unwrap()
        .reshape((17, 1))
        .unwrap();
    let weighted = mlp.forward_weighted(&input, &scale).unwrap();
    assert_eq!(weighted.dims(), input.dims());
    assert_eq!(weighted.dtype(), DType::BF16);
    assert_eq!(before, bits(mlp.forward(&input).unwrap()));
    let wrong = scale.to_dtype(DType::BF16).unwrap();
    assert!(mlp.forward_weighted(&input, &wrong).is_err());
}

/// Builds a small deterministic MLP and verifies gate/up/down composition.
#[test]
fn applies_gated_projection_sequence() {
    let device = Device::Cpu;
    let identity = Tensor::new(&[[1f32, 0.0], [0.0, 1.0]], &device).unwrap();
    let up = Tensor::new(&[[2f32, 0.0], [0.0, 3.0]], &device).unwrap();
    let tensors = HashMap::from([
        ("gate_proj.weight".to_owned(), identity.clone()),
        ("up_proj.weight".to_owned(), up),
        ("down_proj.weight".to_owned(), identity),
    ]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let mlp = Mlp::new(2, 2, Activation::Relu, true, vb).unwrap();
    let input = Tensor::new(&[[-1f32, 2.0]], &device).unwrap();

    let output = mlp.forward(&input).unwrap().to_vec2::<f32>().unwrap();

    assert_eq!(output, vec![vec![0.0, 12.0]]);
}

/// Verifies packed and separate CPU arithmetic produce identical SwiGLU output.
#[test]
fn packed_candle_fallback_matches_separate_projection() {
    let device = Device::Cpu;
    let gate = Tensor::new(&[[1f32, -0.5], [0.25, 2.0]], &device).unwrap();
    let up = Tensor::new(&[[2f32, 0.5], [-1.0, 1.5]], &device).unwrap();
    let down = Tensor::new(&[[1f32, 0.25], [-0.75, 1.0]], &device).unwrap();
    let input = Tensor::new(&[[0.5f32, -1.0], [2.0, 0.25]], &device).unwrap();

    let separate = Mlp::from_separate_weights(
        gate.clone(),
        up.clone(),
        down.clone(),
        Activation::Silu,
        true,
    )
    .unwrap();
    assert!(matches!(
        &separate.gate_up_projection,
        GateUpProjection::Separate { .. }
    ));
    let packed = Mlp {
        decode_graph: None,
        prefill_graph: None,
        hidden_size: 2,
        device: device.clone(),
        dtype: DType::F32,
        gate_up_projection: GateUpProjection::Packed {
            gate_up_proj: Linear::new(Tensor::cat(&[&gate, &up], 0).unwrap(), None),
        },
        down_proj: Linear::new(down, None),
        activation: Activation::Silu,
    };

    let separate = separate.forward(&input).unwrap().to_vec2::<f32>().unwrap();
    let packed = packed.forward(&input).unwrap().to_vec2::<f32>().unwrap();
    for (separate_row, packed_row) in separate.iter().zip(packed) {
        for (&separate, packed) in separate_row.iter().zip(packed_row) {
            assert!((separate - packed).abs() <= 1e-6);
        }
    }
}

/// Compares optimized and reference MLP execution on CUDA in FP32 and BF16.
#[test]
fn optimized_cuda_mlp_matches_reference() {
    let device =
        Device::new_cuda(0).expect("optimized MLP comparison test requires CUDA GPU 0");
    let gate_values = [[1f32, -0.5], [0.25, 2.0], [-1.5, 0.75], [0.5, 1.25]];
    let up_values = [[2f32, 0.5], [-1.0, 1.5], [0.75, -0.25], [1.0, 0.5]];
    let down_values = [[1f32, 0.25, -0.5, 0.75], [-0.75, 1.0, 0.5, -0.25]];
    let input_values = [[[0.5f32, -1.0], [2.0, 0.25]]];

    for (dtype, tolerance) in [(DType::F32, 1e-4), (DType::BF16, 5e-2)] {
        let gate = Tensor::new(&gate_values, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let up = Tensor::new(&up_values, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let down = Tensor::new(&down_values, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let input = Tensor::new(&input_values, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let reference = Mlp::from_separate_weights(
            gate.clone(),
            up.clone(),
            down.clone(),
            Activation::Silu,
            false,
        )
        .unwrap();
        let optimized =
            Mlp::from_separate_weights(gate, up, down, Activation::Silu, true).unwrap();
        assert!(matches!(
            &reference.gate_up_projection,
            GateUpProjection::Separate { .. }
        ));
        assert!(matches!(
            &optimized.gate_up_projection,
            GateUpProjection::Packed { .. }
        ));

        let reference = reference
            .forward(&input)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let optimized = optimized
            .forward(&input)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        for (reference_batch, optimized_batch) in reference.iter().zip(optimized) {
            for (reference_row, optimized_row) in
                reference_batch.iter().zip(optimized_batch)
            {
                for (&reference, optimized) in reference_row.iter().zip(optimized_row) {
                    assert!(
                        (reference - optimized).abs() <= tolerance,
                        "optimized value {optimized} differs from reference \
                         {reference} by more than {tolerance}"
                    );
                }
            }
        }
    }
}
/// Both checkpoint layouts must select the same execution policy and produce
/// identical outputs. Offset packed views cover the sparse-expert use case.
#[test]
fn packed_weight_constructor_preserves_policy_and_outputs() {
    let gpu = Device::new_cuda(0).unwrap();
    for (device, dtype, activation, optimize_kernel, packed) in [
        (Device::Cpu, DType::F32, Activation::Silu, true, false),
        (Device::Cpu, DType::F32, Activation::Relu, true, false),
        (gpu.clone(), DType::BF16, Activation::Silu, true, true),
        (gpu.clone(), DType::BF16, Activation::Silu, false, false),
        (gpu.clone(), DType::BF16, Activation::Relu, true, false),
        (gpu.clone(), DType::F32, Activation::Silu, true, true),
        (gpu, DType::F32, Activation::Silu, false, false),
    ] {
        let tensor = |v: &[[f32; 2]; 2]| {
            Tensor::new(v, &device).unwrap().to_dtype(dtype).unwrap()
        };
        let gate = tensor(&[[1.0, -0.5], [0.25, 2.0]]);
        let up = tensor(&[[2.0, 0.5], [-1.0, 1.5]]);
        let down = tensor(&[[1.0, 0.25], [-0.75, 1.0]]);
        let input = tensor(&[[0.5, -1.0], [2.0, 0.25]]).unsqueeze(0).unwrap();
        let padding = gate.zeros_like().unwrap();
        let packed_weight = Tensor::cat(&[&padding, &gate, &up], 0)
            .unwrap()
            .narrow(0, 2, 4)
            .unwrap();
        let separate = Mlp::from_separate_weights(
            gate,
            up,
            down.clone(),
            activation,
            optimize_kernel,
        )
        .unwrap();
        let from_packed = Mlp::from_packed_weights(
            packed_weight,
            down,
            2,
            activation,
            optimize_kernel,
        )
        .unwrap();
        assert_eq!(
            matches!(
                from_packed.gate_up_projection,
                GateUpProjection::Packed { .. }
            ),
            packed
        );
        let bits = |xs: Tensor| {
            xs.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(from_packed.forward(&input).unwrap()),
            bits(separate.forward(&input).unwrap())
        );
    }
}
