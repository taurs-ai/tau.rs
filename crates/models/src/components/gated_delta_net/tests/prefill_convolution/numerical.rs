//! BF16 convolution accuracy against the unoptimized native FP32 expression.

use super::*;
use crate::components::cuda::prefill_convolution_cuda;

/// Bound each rounding stage rather than comparing two BF16 reduction orders.
fn assert_matches_fp32(
    input: &Tensor,
    history: &Tensor,
    weights: &Tensor,
    output: &Tensor,
    next: &Tensor,
) {
    let (batch, channels, tokens) = input.dims3().unwrap();
    assert_eq!(batch, 1);
    let (golden, golden_next) = convolved(
        &input.to_dtype(DType::F32).unwrap(),
        &history.to_dtype(DType::F32).unwrap(),
        &weights.to_dtype(DType::F32).unwrap(),
    );
    assert_eq!(golden.dtype(), DType::F32);
    let native = prefill_convolution_cuda(input, history, weights).unwrap();
    // The activation and history handoff still obey exact invariants.
    assert_tensor_exact(output, &native.silu().unwrap());
    assert_eq!(next.dtype(), history.dtype());
    assert_tensor_exact(&next.to_dtype(DType::F32).unwrap(), &golden_next);
    let x = values(input);
    let h = values(history);
    let w = values(weights);
    let raw = values(&native);
    let expected_raw = values(&golden);
    let actual = values(output);
    let expected = values(&golden.silu().unwrap());
    let u = bf16::EPSILON.to_f32() / 2.;
    // Candle's BF16 SiLU rounds exp(-x), 1 + exp(-x), and the division.
    let activation_roundoff = 3. * u / (1. - 3. * u);
    let subnormal_roundoff = f32::from_bits(1 << 15); // Half of 2^-133.
    let mut max_raw_error = 0f32;
    let mut max_output_error = 0f32;
    for i in 0..channels * tokens {
        let channel = i / tokens;
        let token = i % tokens;
        let sum_abs_products: f32 = (0..4)
            .map(|tap| {
                let position = token + 1 + tap;
                let sample = if position < 4 {
                    h[channel * 4 + position]
                } else {
                    x[channel * tokens + position - 4]
                };
                (sample * w[channel * 4 + tap]).abs()
            })
            .sum();
        // BF16 products are exactly representable in FP32 for these fixtures.
        // Two four-term FP32 sums can use different trees. 4*epsilon*sum|p|
        // bounds both trees' accumulation error; u bounds the BF16 cast.
        let accumulation = 4. * f32::EPSILON * sum_abs_products;
        let raw_bound =
            u * expected_raw[i].abs() + (1. + u) * accumulation + subnormal_roundoff;
        let raw_error = (raw[i] - expected_raw[i]).abs();
        // |SiLU'| < 1.1; propagate the raw error and all three BF16 operations.
        let output_bound = (1. + activation_roundoff) * 1.1 * raw_bound
            + activation_roundoff * expected[i].abs()
            + subnormal_roundoff;
        let output_error = (actual[i] - expected[i]).abs();
        assert!(
            raw[i].is_finite() && expected_raw[i].is_finite() && raw_error <= raw_bound,
            "convolution[{i}]: {} versus FP32 {}, error={raw_error}, \
             bound={raw_bound}",
            raw[i],
            expected_raw[i]
        );
        assert!(
            actual[i].is_finite()
                && expected[i].is_finite()
                && output_error <= output_bound,
            "activated convolution[{i}]: {} versus FP32 {}, \
             error={output_error}, bound={output_bound}",
            actual[i],
            expected[i]
        );
        max_raw_error = max_raw_error.max(raw_error);
        max_output_error = max_output_error.max(output_error);
    }
    eprintln!(
        "FP32_CONVOLUTION tokens={tokens} max_raw_error={max_raw_error:e} \
         max_output_error={max_output_error:e}"
    );
}

#[test]
fn prefill_convolution_matches_native_fp32_with_mixed_magnitudes() {
    let device = Device::new_cuda(0).unwrap();
    let channels = 8192;
    let mut state = 42u64;
    let mut random = |count| -> Vec<bf16> {
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mantissa = (state >> 32) as i32 as f32 / i32::MAX as f32;
                bf16::from_f32(mantissa * 2f32.powi(((state >> 55) % 12) as i32 - 8))
            })
            .collect()
    };
    for tokens in [2, 3, 4, 27, 511, 512, 513] {
        let mut layer = layer(channels, 4, &device);
        layer.conv_weight =
            Tensor::from_vec(random(channels * 4), (channels, 1, 4), &device).unwrap();
        let input =
            Tensor::from_vec(random(channels * tokens), (1, tokens, channels), &device)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
        let history =
            Tensor::from_vec(random(channels * 4), (1, channels, 4), &device).unwrap();
        let saved_input = input.copy().unwrap();
        let saved_history = history.copy().unwrap();
        let saved_weights = layer.conv_weight.copy().unwrap();
        layer.conv_state = Some(history.clone());
        let actual = layer.causal_convolution(&input).unwrap();
        assert_tensor_exact(&input, &saved_input);
        assert_tensor_exact(&history, &saved_history);
        assert_tensor_exact(&layer.conv_weight, &saved_weights);
        assert_matches_fp32(
            &input,
            &history,
            &layer.conv_weight,
            &actual,
            layer.conv_state.as_ref().unwrap(),
        );
    }
}

#[test]
fn prefill_convolution_matches_fp32_at_bf16_midpoints() {
    let device = Device::new_cuda(0).unwrap();
    // Reduced from B200's 8192-channel, 511-token failure. Pairwise FP32 lands
    // above the BF16 midpoint; another FP32 tree lands exactly on the midpoint.
    let samples = [185f32 / 512., -75. / 65536., -45. / 1024., 1.2890625];
    let weights = [73f32 / 8388608., 45. / 16384., -167. / 128., -23. / 1024.];
    let products = std::array::from_fn::<_, 4, _>(|i| samples[i] * weights[i]);
    let pairwise = (products[0] + products[1]) + (products[2] + products[3]);
    let sequential = products.iter().sum::<f32>();
    assert_ne!(bf16::from_f32(pairwise), bf16::from_f32(sequential));
    let mut layer = layer(1, 4, &device);
    layer.conv_weight = Tensor::from_vec(weights.to_vec(), (1, 1, 4), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let history = Tensor::new(&[[[0., samples[0], samples[1], samples[2]]]], &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    for tokens in [2, 3, 4, 27, 511, 512, 513] {
        let mut data = vec![bf16::ZERO; tokens];
        data[0] = bf16::from_f32(samples[3]);
        let input = Tensor::from_vec(data, (1, 1, tokens), &device).unwrap();
        layer.conv_state = Some(history.clone());
        let actual = layer.causal_convolution(&input).unwrap();
        assert_matches_fp32(
            &input,
            &history,
            &layer.conv_weight,
            &actual,
            layer.conv_state.as_ref().unwrap(),
        );
    }
}
