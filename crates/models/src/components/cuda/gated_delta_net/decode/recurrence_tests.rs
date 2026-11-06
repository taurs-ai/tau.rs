//! Production-size recurrence checks, established before replacing the native kernel.
//!
//! The independent scalar oracle consumes the convolution kernel's BF16 output.
//! This isolates recurrence errors from GPU/CPU exp rounding in the unchanged
//! convolution. Component tests separately compare the complete path with Candle.

use super::*;
use candle::Device;

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// A deterministic, non-periodic pattern makes head, row and tile swaps visible.
fn pattern(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 20001) as f32 / 10000.0 - 1.0) * scale
        })
        .collect()
}

fn tensor(
    data: &[f32],
    shape: impl Into<Shape>,
    dtype: DType,
    device: &Device,
) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

/// Uses absolute plus relative bounds so near-zero and large values are both checked.
fn close(actual: &[f32], expected: &[f64], atol: f64, rtol: f64, label: &str) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0_f64;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - e).abs();
        worst = worst.max(error);
        assert!(
            a.is_finite() && e.is_finite() && error <= atol + rtol * e.abs(),
            "{label}[{index}]: {a} != {e}, error {error}"
        );
    }
    worst
}

fn rounded(value: f64) -> f64 {
    f64::from(bf16::from_f64(value).to_f32())
}

/// FP64 scalar math with explicit BF16 boundaries; deliberately independent of
/// the CUDA thread layout, reduction ordering, and register tiling.
fn oracle(
    runner: &GatedDeltaNetRunner,
    state: &mut [f64],
    convolved: &[bf16],
    qkvz: &[f32],
    ba: &[f32],
) -> Vec<f64> {
    let d = runner.dimensions;
    let dt = values(&runner.dt_bias);
    let negative_a = values(&runner.negative_a);
    let weights = values(&runner.norm_weight);
    let key_width = d.num_key_heads * d.key_head_dim;
    let mut output = vec![0.0; d.value_width];
    for head in 0..d.num_value_heads {
        let key_head = head / (d.num_value_heads / d.num_key_heads);
        let mut q = Vec::with_capacity(d.key_head_dim);
        let mut k = Vec::with_capacity(d.key_head_dim);
        for row in 0..d.key_head_dim {
            q.push(f64::from(
                convolved[key_head * d.key_head_dim + row].to_f32(),
            ));
            k.push(f64::from(
                convolved[key_width + key_head * d.key_head_dim + row].to_f32(),
            ));
        }
        let q_scale = (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt()
            * (d.key_head_dim as f64).sqrt();
        let k_scale = (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
        for row in 0..d.key_head_dim {
            q[row] /= q_scale;
            k[row] /= k_scale;
        }
        let beta = rounded(1.0 / (1.0 + (-f64::from(ba[head])).exp()));
        let step = f64::from(ba[d.num_value_heads + head]) + f64::from(dt[head]);
        let softplus = step.max(0.0) + (-step.abs()).exp().ln_1p();
        let alpha = (softplus * f64::from(negative_a[head])).exp();
        for column in 0..d.value_head_dim {
            let base = head * d.key_head_dim * d.value_head_dim + column;
            let mut prediction = 0.0;
            for (row, &key) in k.iter().enumerate() {
                let index = base + row * d.value_head_dim;
                state[index] *= alpha;
                prediction += state[index] * key;
            }
            let value = f64::from(
                convolved[2 * key_width + head * d.value_head_dim + column].to_f32(),
            );
            let delta = beta * (value - prediction);
            let mut result = 0.0;
            for row in 0..d.key_head_dim {
                let index = base + row * d.value_head_dim;
                state[index] += k[row] * delta;
                result += state[index] * q[row];
            }
            output[head * d.value_head_dim + column] = rounded(result);
        }
        let row = &mut output[head * d.value_head_dim..(head + 1) * d.value_head_dim];
        let rms = (row.iter().map(|x| x * x).sum::<f64>() / d.value_head_dim as f64
            + f64::from(runner.epsilon))
        .sqrt();
        for (column, result) in row.iter_mut().enumerate() {
            let gate = f64::from(qkvz[d.conv_width + head * d.value_head_dim + column]);
            *result = rounded(
                *result / rms * f64::from(weights[column]) * gate
                    / (1.0 + (-gate).exp()),
            );
        }
    }
    output
}

fn check_sequence(
    key_dim: usize,
    value_dim: usize,
    key_heads: usize,
    heads: usize,
    steps: usize,
    reference_reduction: bool,
) {
    let device = Device::new_cuda(0).expect("recurrence tests require CUDA");
    let width = 2 * key_heads * key_dim + heads * value_dim;
    // Different decay regimes include almost-retained memory and rapid forgetting.
    let negative_a = (0..heads)
        .map(|h| [-0.00013, -0.07123, -0.6123, -3.0123][h % 4])
        .collect::<Vec<_>>();
    let weights = (0..value_dim)
        .map(|i| 0.712345 + i as f32 * 0.001234)
        .collect::<Vec<_>>();
    let mut runner = GatedDeltaNetRunner::new(
        16,
        4,
        key_dim,
        value_dim,
        key_heads,
        heads,
        1e-6,
        tensor(
            &pattern(width * 4, 41, 0.3),
            (width, 1, 4),
            DType::BF16,
            &device,
        ),
        tensor(&pattern(heads, 43, 2.0), heads, DType::BF16, &device),
        tensor(&negative_a, heads, DType::F32, &device),
        tensor(&weights, value_dim, DType::F32, &device),
    )
    .unwrap()
    .unwrap();
    runner.set_reference_reduction(reference_reduction);
    let initial_conv = tensor(
        &pattern(width * 4, 47, 0.2),
        (1, width, 4),
        DType::BF16,
        &device,
    );
    let initial_state = tensor(
        &pattern(heads * key_dim * value_dim, 53, 0.04),
        (1, heads, key_dim, value_dim),
        DType::F32,
        &device,
    );
    let addresses = runner.state_addresses().unwrap();
    let mut worst_output = 0.0_f64;
    let mut worst_state = 0.0_f64;
    // Nonzero seed, zero reset, then the original seed expose state ownership bugs.
    let mut first_final = None;
    for request in 0..3 {
        if request == 1 {
            runner.seed_state(None, None).unwrap();
        } else {
            runner
                .seed_state(Some(&initial_conv), Some(&initial_state))
                .unwrap();
        }
        let mut state = if request == 1 {
            vec![0.0; heads * key_dim * value_dim]
        } else {
            values(&initial_state).into_iter().map(f64::from).collect()
        };
        for step in 0..steps {
            let mut input =
                pattern(width + heads * value_dim, 101 + step as u32 * 17, 1.5);
            // Near-zero Q/K exercises the epsilon guard without changing V or gates.
            if step % 17 == 0 {
                input[..2 * key_heads * key_dim].fill(0.0);
            }
            let qkvz = tensor(&input, (1, 1, input.len()), DType::BF16, &device);
            let ba = tensor(
                &pattern(2 * heads, 107 + step as u32 * 19, 5.0),
                (1, 1, 2 * heads),
                DType::BF16,
                &device,
            );
            let before_qkvz = values(&qkvz);
            let before_ba = values(&ba);
            let actual = values(&runner.forward(&qkvz, &ba).unwrap());
            let convolved = {
                let workspace = runner.lock_workspace().unwrap();
                device
                    .as_cuda_device()
                    .unwrap()
                    .cuda_stream()
                    .clone_dtoh(&workspace.convolved_qkv)
                    .unwrap()
            };
            let expected =
                oracle(&runner, &mut state, &convolved, &before_qkvz, &before_ba);
            // Bounds are fixed against the old kernel before introducing tiling.
            worst_output =
                worst_output.max(close(&actual, &expected, 0.002, 0.012, "output"));
            let (_, actual_state) = runner.export_state().unwrap();
            worst_state = worst_state.max(close(
                &values(&actual_state),
                &state,
                2e-5,
                2e-4,
                "state",
            ));
            assert_eq!(values(&qkvz), before_qkvz);
            assert_eq!(values(&ba), before_ba);
            assert_eq!(runner.state_addresses().unwrap(), addresses);
        }
        let (conv, recurrent) = runner.export_state().unwrap();
        let final_state = (values(&conv), values(&recurrent));
        if request == 0 {
            first_final = Some(final_state);
        } else if request == 2 {
            assert_eq!(Some(final_state), first_final);
        }
    }
    eprintln!(
        "GDN oracle K={key_dim} V={value_dim} heads={heads} steps={steps}: \
         output_max={worst_output:e}, state_max={worst_state:e}"
    );
}

#[test]
fn production_recurrence_matches_independent_oracle_over_long_sequences() {
    check_sequence(128, 128, 2, 4, 256, false);
}

#[test]
fn production_recurrence_covers_4b_and_27b_head_counts() {
    check_sequence(128, 128, 16, 32, 16, false);
    check_sequence(128, 128, 16, 48, 16, false);
}

#[test]
fn generic_recurrence_preserves_irregular_and_boundary_shapes() {
    for (key, value) in [(7, 13), (64, 64), (128, 96), (256, 256)] {
        check_sequence(key, value, 1, 2, 8, false);
    }
}

/// The MoE reduction preset must preserve independent FP64 accuracy and complete
/// state/reset/retained-address contracts across a long recurrent sequence.
#[test]
fn moe_reference_recurrence_matches_independent_oracle() {
    check_sequence(128, 128, 2, 4, 256, true);
}
