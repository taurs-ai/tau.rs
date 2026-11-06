//! Native FP32 golden accuracy and independent FP64 mathematical cross-check.

use super::*;

fn flat(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn data(count: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 37 + seed) % 101) as f32 - 50.) * scale)
        .collect()
}

/// Selects dimensions and normalized-Q/K storage for the independent FP64 math.
/// Synthetic and real-operand tests share the oracle while choosing their own
/// explicit initial memory and prefill storage policy.
pub(super) struct Fp64Config {
    pub tokens: usize,
    pub heads: usize,
    pub round_normalized_qk: bool,
}

/// Independent FP64 math; casts in the caller establish the BF16 input contract.
pub(super) fn oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    initial: &[f32],
    config: Fp64Config,
) -> (Vec<f32>, Vec<f32>) {
    let Fp64Config {
        tokens,
        heads,
        round_normalized_qk,
    } = config;
    let mut state: Vec<f64> = initial.iter().map(|&x| f64::from(x)).collect();
    let mut output = vec![0.; tokens * heads * 128];
    for t in 0..tokens {
        for h in 0..heads {
            let offset = (t * heads + h) * 128;
            let norm = |x: &[f32]| {
                (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() + 1e-6).sqrt()
            };
            let qs = norm(&q[offset..offset + 128]);
            let ks = norm(&k[offset..offset + 128]);
            let round = |x: f64| {
                if round_normalized_qk {
                    f64::from(half::bf16::from_f32(x as f32).to_f32())
                } else {
                    x
                }
            };
            let query: Vec<_> = q[offset..offset + 128]
                .iter()
                .map(|&x| round(f64::from(x) / qs) / 128f64.sqrt())
                .collect();
            let key: Vec<_> = k[offset..offset + 128]
                .iter()
                .map(|&x| round(f64::from(x) / ks))
                .collect();
            let alpha = f64::from(g[t * heads + h]).exp();
            for col in 0..128 {
                let mut prediction = 0.;
                for (row, &key_value) in key.iter().enumerate() {
                    let index = (h * 128 + row) * 128 + col;
                    state[index] *= alpha;
                    prediction += state[index] * key_value;
                }
                let delta = (f64::from(v[offset + col]) - prediction)
                    * f64::from(beta[t * heads + h]);
                let mut readout = 0.;
                for (row, (&key_value, &query_value)) in
                    key.iter().zip(&query).enumerate()
                {
                    let index = (h * 128 + row) * 128 + col;
                    state[index] += key_value * delta;
                    readout += state[index] * query_value;
                }
                output[offset + col] = readout as f32;
            }
        }
    }
    (output, state.iter().map(|&x| x as f32).collect())
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-5 + 2e-4 * e.abs(),
            "recurrence {i}: {a} versus {e}"
        );
    }
}

#[test]
fn prefill_recurrence_matches_native_fp32_and_chunk_handoff() {
    let device = Device::new_cuda(0).unwrap();
    let mut cfg = cuda_test_config();
    cfg.key_head_dim = 128;
    cfg.value_head_dim = 128;
    cfg.num_key_heads = 2;
    cfg.num_value_heads = 4;
    let mut layer = GatedDeltaNet::new(
        &cfg,
        true,
        VarBuilder::from_tensors(test_tensors(&cfg, &device), DType::BF16, &device),
    )
    .unwrap();
    for tokens in [2, 31, 32, 33, 63, 64, 65, 255, 256, 257] {
        let shape = (1, tokens, 4, 128);
        let input = |seed| {
            Tensor::from_vec(data(tokens * 4 * 128, seed, 0.02), shape, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let q = input(1);
        let k = input(3);
        let v = input(7);
        let g = Tensor::from_vec(
            (0..tokens * 4)
                .map(|i| -0.001 - (i % 13) as f32 * 0.1)
                .collect::<Vec<_>>(),
            (1, tokens, 4),
            &device,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            (0..tokens * 4)
                .map(|i| (i % 11) as f32 / 10.)
                .collect::<Vec<_>>(),
            (1, tokens, 4),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let initial = if tokens % 2 == 0 {
            vec![0.; 4 * 128 * 128]
        } else {
            data(4 * 128 * 128, 11, 0.001)
        };
        let state =
            Tensor::from_vec(initial.clone(), (1, 4, 128, 128), &device).unwrap();
        let expected = oracle(
            &flat(&q),
            &flat(&k),
            &flat(&v),
            &flat(&g),
            &flat(&beta),
            &initial,
            Fp64Config {
                tokens,
                heads: 4,
                round_normalized_qk: false,
            },
        );
        layer.recurrent_state = Some(state.clone());
        let actual = layer.recurrent_delta_rule(&q, &k, &v, &g, &beta).unwrap();
        let reference =
            super::super::golden::recurrence(&q, &k, &v, &g, &beta, &state).unwrap();
        super::super::golden::assert_close(
            &actual,
            &reference.0,
            2e-5,
            2e-4,
            "recurrence output",
        );
        super::super::golden::assert_close(
            layer.recurrent_state.as_ref().unwrap(),
            &reference.1,
            2e-5,
            2e-4,
            "recurrence state",
        );
        // The independent higher-precision oracle checks the FP32 golden itself.
        close(&flat(&reference.0), &expected.0);
        close(&flat(&reference.1), &expected.1);
        if tokens == 257 {
            layer.recurrent_state = Some(state);
            let mut parts = Vec::new();
            for (start, len) in [(0, 64), (64, 192), (256, 1)] {
                parts.push(
                    layer
                        .recurrent_delta_rule(
                            &q.narrow(1, start, len).unwrap(),
                            &k.narrow(1, start, len).unwrap(),
                            &v.narrow(1, start, len).unwrap(),
                            &g.narrow(1, start, len).unwrap(),
                            &beta.narrow(1, start, len).unwrap(),
                        )
                        .unwrap(),
                );
            }
            super::super::golden::assert_close(
                &Tensor::cat(&parts, 1).unwrap(),
                &reference.0,
                2e-5,
                2e-4,
                "chunked output",
            );
            super::super::golden::assert_close(
                layer.recurrent_state.as_ref().unwrap(),
                &reference.1,
                2e-5,
                2e-4,
                "chunked state",
            );
        }
    }
}

/// Complete BF16 prefill, including projection/convolution rounding, is compared
/// with FP32 execution of the same weights and teacher-forced activations.
#[test]
fn production_prefill_boundaries_match_native_fp32() {
    let device = Device::new_cuda(0).unwrap();
    let mut cfg = cuda_test_config();
    cfg.key_head_dim = 128;
    cfg.value_head_dim = 128;
    cfg.num_key_heads = 2;
    cfg.num_value_heads = 4;
    let mut tensors = test_tensors(&cfg, &device);
    add_fp32_parameters(&mut tensors, &cfg, &device);
    let mut optimized = GatedDeltaNet::new(
        &cfg,
        true,
        VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
    )
    .unwrap();
    let mut reference = super::super::golden::layer(&cfg, &tensors, &device).unwrap();
    for tokens in [2, 63, 64, 65, 255, 256, 257] {
        optimized.clear_cache();
        reference.clear_cache();
        let input = patterned_bf16((1, tokens, cfg.hidden_size), 17, 0.02, &device);
        for input in [
            input,
            patterned_bf16((1, 1, cfg.hidden_size), 19, 0.018, &device),
        ] {
            let actual = optimized.forward(&input).unwrap();
            let expected = reference
                .forward(&input.to_dtype(DType::F32).unwrap())
                .unwrap();
            super::super::golden::assert_close(
                &actual,
                &expected,
                3e-2,
                0.,
                "component output",
            );
            let (conv, state) = optimized.snapshot_current_state().unwrap();
            let (expected_conv, expected_state) =
                reference.snapshot_current_state().unwrap();
            super::super::golden::assert_close(
                &conv,
                &expected_conv,
                3e-4,
                0.,
                "component history",
            );
            super::super::golden::assert_close(
                &state,
                &expected_state,
                3e-4,
                0.,
                "component state",
            );
        }
    }
}
