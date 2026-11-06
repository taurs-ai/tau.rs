//! Weight-free regression from actual first-GDN operands on the H200 node.
//! The 12.5 KiB fixture retains sixteen tokens of one head, before normalization;
//! its source identities and reduction scope live beside the safetensors data.

use super::*;

fn operands(device: &Device) -> HashMap<String, Tensor> {
    candle::safetensors::load_buffer(
        include_bytes!("fixtures/gdn_prefill_sm90_head0.safetensors"),
        device,
    )
    .unwrap()
}

fn reference(
    values: &HashMap<String, Tensor>,
    initial: &Tensor,
    round_normalized: bool,
) -> (Tensor, Tensor) {
    let forward = if round_normalized {
        super::super::golden::recurrence_with_bf16_normalized_qk
    } else {
        super::super::golden::recurrence
    };
    forward(
        &values["q"],
        &values["k"],
        &values["v"],
        &values["g"],
        &values["beta"],
        initial,
    )
    .unwrap()
}

fn flat(tensor: &Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn real_operands_expose_normalized_qk_storage_rounding() {
    let values = operands(&Device::Cpu);
    let initial = Tensor::zeros((1, 1, 128, 128), DType::F32, &Device::Cpu).unwrap();
    let rounded = flat(&reference(&values, &initial, true).0);
    let unrounded = flat(&reference(&values, &initial, false).0);
    let failures = rounded
        .iter()
        .zip(&unrounded)
        .filter(|(a, b)| (*a - *b).abs() > 2e-5 + 2e-4 * a.abs())
        .count();
    assert!(
        failures > 0,
        "fixture must reject the missing BF16 boundary under the native FP32 budget"
    );
}

/// Independent higher-precision math checks both the stored-Q/K prefill policy
/// and the FP32-normalized control on real operands, including nonzero memory.
#[test]
fn real_operands_match_independent_fp64_with_explicit_state() {
    let values = operands(&Device::Cpu);
    for seeded in [false, true] {
        let initial_values = (0..128 * 128)
            .map(|i| {
                if seeded {
                    ((i % 31) as f32 - 15.) * 0.0001
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();
        let initial =
            Tensor::from_vec(initial_values.clone(), (1, 1, 128, 128), &Device::Cpu)
                .unwrap();
        for rounded in [false, true] {
            let (output, state) = reference(&values, &initial, rounded);
            let expected = super::prefill_recurrence::oracle(
                &flat(&values["q"].to_dtype(DType::F32).unwrap()),
                &flat(&values["k"].to_dtype(DType::F32).unwrap()),
                &flat(&values["v"].to_dtype(DType::F32).unwrap()),
                &flat(&values["g"]),
                &flat(&values["beta"].to_dtype(DType::F32).unwrap()),
                &initial_values,
                super::prefill_recurrence::Fp64Config {
                    tokens: values["q"].dim(1).unwrap(),
                    heads: 1,
                    round_normalized_qk: rounded,
                },
            );
            for (actual, expected) in
                [(flat(&output), expected.0), (flat(&state), expected.1)]
            {
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        a.is_finite() && (a - e).abs() <= 2e-5 + 2e-4 * e.abs(),
                        "real-operand FP64 seeded={seeded} rounded={rounded}: {a} versus {e}"
                    );
                }
            }
        }
    }
}

#[test]
fn native_prefill_retains_real_operand_bf16_qk_boundary() {
    let device = Device::new_cuda(0).unwrap();
    let values = operands(&device);
    let initial = Tensor::zeros((1, 1, 128, 128), DType::F32, &device).unwrap();
    let expected = reference(&values, &initial, true);
    let tokens = values["q"].dim(1).unwrap();
    let mixed = Tensor::cat(&[&values["q"], &values["k"], &values["v"]], 3)
        .unwrap()
        .reshape((1, tokens, 384))
        .unwrap();
    let (output, state) = crate::components::cuda::mixed_prefill_cuda(
        &mixed,
        &values["g"].exp().unwrap(),
        &values["beta"].to_dtype(DType::BF16).unwrap(),
        &initial,
        1,
        1,
        true,
    )
    .unwrap();
    super::super::golden::assert_close(
        &output,
        &expected.0,
        2e-5,
        2e-4,
        "real-operand MoE prefill output",
    );
    super::super::golden::assert_close(
        &state,
        &expected.1,
        2e-5,
        2e-4,
        "real-operand MoE prefill state",
    );
    let (without_rounding, _) = crate::components::cuda::mixed_prefill_cuda(
        &mixed,
        &values["g"].exp().unwrap(),
        &values["beta"].to_dtype(DType::BF16).unwrap(),
        &initial,
        1,
        1,
        false,
    )
    .unwrap();
    assert!(
        flat(&without_rounding)
            .iter()
            .zip(flat(&expected.0))
            .any(|(a, b)| (a - b).abs() > 2e-5 + 2e-4 * b.abs()),
        "native negative control did not expose the missing BF16 boundary"
    );
}

/// Replays one complete trained GDN layer on the recorded H200 input. The
/// standalone artifact contains only this layer's weights and reference tensors,
/// so no full checkpoint or full-model GPU allocation is needed.
#[test]
#[ignore = "requires validation/artifacts/h200-gdn-layer0.safetensors; fits RTX 16GB"]
fn replay_same_input_real_gdn_layer() {
    use crate::test_support::tensor_capture::{self as capture, Value};
    let path = std::env::var("TAU_GDN_LAYER_FIXTURE").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/artifacts/h200-gdn-layer0.safetensors"
        )
        .to_owned()
    });
    let device = Device::new_cuda(0).unwrap();
    let mut tensors = candle::safetensors::load(path, &device).unwrap();
    let input = tensors.remove("fixture.input").unwrap();
    let expected = tensors.remove("fixture.cpu_fp32").unwrap();
    let sglang = tensors.remove("fixture.sglang").unwrap();
    let tau_h200 = tensors.remove("fixture.tau_h200").unwrap();
    let config = GatedDeltaNetConfig {
        hidden_size: 2048,
        conv_kernel_size: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        num_key_heads: 16,
        num_value_heads: 32,
        norm_eps: 1e-6,
        conv_activation: Activation::Silu,
        gate_activation: Activation::Silu,
    };
    let mut observations = HashMap::new();
    let mut outputs = HashMap::new();
    for (name, optimize, qk_rounding) in [
        ("native", true, None),
        ("portable", false, None),
        ("native_fp32_qk", true, Some(false)),
    ] {
        let mut layer = GatedDeltaNet::new(
            &config,
            optimize,
            VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
        )
        .unwrap()
        .with_reference_precision(true);
        layer.prefill_qk_rounding_override = qk_rounding;
        capture::begin(None);
        let actual = layer.forward(&input).unwrap();
        let stages = capture::finish();
        layer.clear_cache();
        let repeated = layer.forward(&input).unwrap();
        assert_tensor_exact(&actual, &repeated);
        // Retain the existing complete-path GDN absolute output budget. The
        // recurrence-only test above independently uses the tight FP32 budget.
        assert_bf16_close(&actual, &expected, 3e-2, name);
        assert_bf16_close(&actual, &sglang, 3e-2, "recorded SGLang layer");
        let error = |reference: &Tensor| {
            let a = actual
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = reference
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut square = 0f64;
            let mut maximum = 0f32;
            for (a, b) in a.iter().zip(b) {
                maximum = maximum.max((a - b).abs());
                square += f64::from(a - b).powi(2);
            }
            (maximum, (square / a.len() as f64).sqrt())
        };
        eprintln!(
            "SAME_INPUT_GDN {name} vs_cpu_fp32={:?} vs_sglang={:?} vs_tau_h200={:?}",
            error(&expected),
            error(&sglang),
            error(&tau_h200)
        );
        outputs.insert(name, actual);
        observations.insert(name, stages);
    }
    assert_tensor_exact(&outputs["native"], &outputs["portable"]);
    for (stage, a) in &observations["native"] {
        let Value::Tensor(a) = a else { continue };
        let Value::Tensor(b) = &observations["portable"][stage] else {
            continue;
        };
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let maximum = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let changed = a
            .iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "SAME_INPUT_GDN stage={stage} native_vs_portable_max={maximum:e} changed={changed}/{}",
            a.len()
        );
    }
}
