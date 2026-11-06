//! Convolution layout/lifecycle checks and independent FP32 numerical coverage.
mod graph;
mod numerical;

use super::super::convolution;
use super::*;
use half::bf16;

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Original unoptimized expression, before activation or runtime dispatch.
fn convolved(input: &Tensor, history: &Tensor, weights: &Tensor) -> (Tensor, Tensor) {
    let (_, channels, tokens) = input.dims3().unwrap();
    let taps = history.dim(2).unwrap();
    let extended = Tensor::cat(&[history, input], 2).unwrap();
    let next = extended.narrow(2, tokens, taps).unwrap();
    let output = extended
        .conv1d(weights, 0, 1, 1, channels)
        .unwrap()
        .narrow(2, 1, tokens)
        .unwrap();
    (output, next)
}

/// Original activated expression, retained for deterministic control fixtures.
fn reference(input: &Tensor, history: &Tensor, weights: &Tensor) -> (Tensor, Tensor) {
    let (output, next) = convolved(input, history, weights);
    (output.silu().unwrap(), next)
}

/// Independent scalar window indexing and FP64 sum; BF16 rounds before SiLU.
fn oracle(input: &Tensor, history: &Tensor, weights: &Tensor) -> (Tensor, Tensor) {
    let (batch, channels, tokens) = input.dims3().unwrap();
    let taps = history.dim(2).unwrap();
    let x = values(input);
    let h = values(history);
    let w = values(weights);
    let mut output = Vec::new();
    let mut next = Vec::new();
    for b in 0..batch {
        for c in 0..channels {
            let sample = |t: usize| {
                if t < taps {
                    h[(b * channels + c) * taps + t]
                } else {
                    x[(b * channels + c) * tokens + t - taps]
                }
            };
            for t in 0..tokens {
                let sum: f64 = (0..taps)
                    .map(|j| f64::from(sample(t + 1 + j)) * f64::from(w[c * taps + j]))
                    .sum();
                output.push(bf16::from_f64(sum));
            }
            for j in 0..taps {
                next.push(bf16::from_f32(sample(tokens + j)));
            }
        }
    }
    (
        Tensor::from_vec(output, input.shape(), input.device())
            .unwrap()
            .silu()
            .unwrap(),
        Tensor::from_vec(next, history.shape(), history.device()).unwrap(),
    )
}

fn layer(channels: usize, taps: usize, device: &Device) -> GatedDeltaNet {
    let cfg = cuda_test_config();
    let mut layer = GatedDeltaNet::new(
        &cfg,
        true,
        VarBuilder::from_tensors(test_tensors(&cfg, device), DType::BF16, device),
    )
    .unwrap();
    // Only the standalone convolution is exercised; projections stay tiny.
    layer.conv_width = channels;
    layer.conv_kernel_size = taps;
    layer.conv_weight = patterned((channels, 1, taps), 3, device);
    layer
}

fn patterned(shape: (usize, usize, usize), seed: usize, device: &Device) -> Tensor {
    let count = shape.0 * shape.1 * shape.2;
    let data: Vec<_> = (0..count)
        .map(|i| bf16::from_f32(((i * 17 + seed * 7) % 97) as f32 / 32.0 - 1.5))
        .collect();
    Tensor::from_vec(data, shape, device).unwrap()
}

/// A projection-like transpose plus nonzero offset and excess backing storage.
fn strided(channels: usize, tokens: usize, seed: usize, device: &Device) -> Tensor {
    patterned((1, tokens + 3, channels + 2), seed, device)
        .narrow(1, 1, tokens)
        .unwrap()
        .narrow(2, 1, channels)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
}

fn check(layer: &mut GatedDeltaNet, input: &Tensor, history: &Tensor) {
    let old_input = input.copy().unwrap();
    let old_history = history.copy().unwrap();
    let old_weights = layer.conv_weight.copy().unwrap();
    let (expected, next) = reference(input, history, &layer.conv_weight);
    let (scalar, scalar_next) = oracle(input, history, &layer.conv_weight);
    assert_tensor_exact(&expected, &scalar);
    assert_tensor_exact(&next, &scalar_next);
    layer.conv_state = Some(history.clone());
    let actual = layer.causal_convolution(input).unwrap();
    assert_tensor_exact(&actual, &expected);
    assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &next);
    assert_tensor_exact(input, &old_input);
    assert_tensor_exact(history, &old_history);
    assert_tensor_exact(&layer.conv_weight, &old_weights);
}

/// The function can be called repeatedly without advancing a layer or changing
/// its borrowed tensors. Activation and dispatch policy come from its arguments.
#[test]
fn prefill_convolution_function_is_stateless_and_honors_arguments() {
    for (device, dtype) in [
        (Device::Cpu, DType::F32),
        (Device::new_cuda(0).unwrap(), DType::BF16),
    ] {
        let channels = 3;
        let weight = patterned((channels, 1, 4), 3, &device)
            .to_dtype(dtype)
            .unwrap();
        let history = strided(channels, 4, 5, &device).to_dtype(dtype).unwrap();
        let zero_history = history.zeros_like().unwrap();
        let saved_weight = weight.copy().unwrap();
        let saved_history = history.copy().unwrap();
        for tokens in [2, 27] {
            let input = strided(channels, tokens, 9, &device)
                .to_dtype(dtype)
                .unwrap();
            let saved_input = input.copy().unwrap();
            for activation in [Activation::Silu, Activation::Relu] {
                for optimize_kernel in [false, true] {
                    for previous in [None, Some(&history)] {
                        // Preserve the original expression independently of the
                        // function's history construction and native dispatch.
                        let extended = Tensor::cat(
                            &[previous.unwrap_or(&zero_history), &input],
                            2,
                        )
                        .unwrap();
                        let expected = extended
                            .conv1d(&weight, 0, 1, 1, channels)
                            .unwrap()
                            .narrow(2, 1, tokens)
                            .unwrap()
                            .apply(&activation)
                            .unwrap();
                        let expected_history = extended.narrow(2, tokens, 4).unwrap();
                        for _ in 0..2 {
                            let (output, next) = convolution::forward(
                                &input,
                                &weight,
                                previous,
                                activation,
                                optimize_kernel,
                            )
                            .unwrap();
                            assert_tensor_exact(&output, &expected);
                            assert_tensor_exact(&next, &expected_history);
                            assert_tensor_exact(&input, &saved_input);
                            assert_tensor_exact(&history, &saved_history);
                            assert_tensor_exact(&weight, &saved_weight);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn prefill_convolution_rejects_invalid_arguments_without_publishing_history() {
    let device = Device::Cpu;
    let mut layer = layer(3, 4, &device);
    layer.conv_weight = layer.conv_weight.to_dtype(DType::F32).unwrap();
    let input = strided(3, 2, 9, &device).to_dtype(DType::F32).unwrap();
    let history = strided(3, 4, 5, &device).to_dtype(DType::F32).unwrap();
    let saved_history = history.copy().unwrap();
    layer.conv_state = Some(history.clone());
    // The layer must publish returned history only after successful computation.
    let wrong_channels = input.narrow(1, 0, 2).unwrap();
    assert!(layer.causal_convolution(&wrong_channels).is_err());
    assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &saved_history);

    let bad_weight = Tensor::zeros((3, 2, 4), DType::F32, &device).unwrap();
    let wrong_dtype = layer.conv_weight.to_dtype(DType::BF16).unwrap();
    let bad_history = history.narrow(2, 0, 3).unwrap();
    for optimize_kernel in [false, true] {
        for (weight, previous) in [
            (&bad_weight, &history),
            (&wrong_dtype, &history),
            (&layer.conv_weight, &bad_history),
        ] {
            assert!(
                convolution::forward(
                    &input,
                    weight,
                    Some(previous),
                    Activation::Silu,
                    optimize_kernel,
                )
                .is_err()
            );
        }
    }
    assert_tensor_exact(&history, &saved_history);
    // A failed call must not disrupt a subsequent valid continuation.
    let (expected, next) = reference(&input, &history, &layer.conv_weight);
    assert_tensor_exact(&layer.causal_convolution(&input).unwrap(), &expected);
    assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &next);
}

#[test]
fn prefill_convolution_matches_scalar_and_frozen_reference() {
    let device = Device::new_cuda(0).unwrap();
    for channels in [1, 7, 33, 8192, 10240] {
        let mut layer = layer(channels, 4, &device);
        let lengths: &[usize] = if channels > 33 {
            &[2, 27, 512, 513]
        } else {
            &[1, 2, 3, 4, 27, 511, 512, 513]
        };
        for &tokens in lengths {
            let input = strided(channels, tokens, tokens, &device);
            let history = strided(channels, 4, 5, &device);
            check(&mut layer, &input, &history);
        }
    }
}

#[test]
fn prefill_convolution_preserves_tap_order_and_bf16_rounding() {
    let device = Device::new_cuda(0).unwrap();
    let mut layer = layer(1, 4, &device);
    let input = Tensor::zeros((1, 1, 5), DType::BF16, &device).unwrap();
    let history = Tensor::new(&[[[0f32, 1.0, 1.0 / 256.0, 0.0]]], &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    layer.conv_weight = Tensor::new(&[[[1f32, 1.0, 0.0, 0.0]]], &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    check(&mut layer, &input, &history);
    let (rounded, _) = reference(&input, &history, &layer.conv_weight);
    let unrounded = 1.0f32 + 1.0 / 256.0;
    let decode_rounding = bf16::from_f32(unrounded / (1.0 + (-unrounded).exp()));
    assert_ne!(values(&rounded)[0], decode_rounding.to_f32());
    for tap in 0..4 {
        let mut weights = vec![bf16::ZERO; 4];
        weights[tap] = bf16::ONE;
        layer.conv_weight = Tensor::from_vec(weights, (1, 1, 4), &device).unwrap();
        check(&mut layer, &strided(1, 7, 11, &device), &history);
    }
    layer.conv_weight =
        Tensor::new(&[[[16f32, -16.0, 1.0 / 128.0, -1.0 / 64.0]]], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
    check(
        &mut layer,
        &strided(1, 27, 9, &device),
        &strided(1, 4, 3, &device),
    );
    // A production-width failure reduced to one channel: pairwise FP32 addition
    // crosses a BF16 rounding boundary that sequential accumulation does not.
    // Exercise every relevant token-count regime, including two-token tails.
    let history = Tensor::new(&[[[0f32, 1.484375, -1.5859375, -0.65234375]]], &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    layer.conv_weight = Tensor::new(
        &[[[-3.265625f32, -1.3984375, -4.03125, 0.001663208]]],
        &device,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    for tokens in [2, 3, 4, 27, 511, 512, 513] {
        let mut data = vec![bf16::ZERO; tokens];
        data[0] = bf16::from_f32(0.003112793);
        let input = Tensor::from_vec(data, (1, 1, tokens), &device).unwrap();
        let (expected, next) = reference(&input, &history, &layer.conv_weight);
        layer.conv_state = Some(history.clone());
        assert_tensor_exact(&layer.causal_convolution(&input).unwrap(), &expected);
        assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &next);
    }
}

#[test]
fn prefill_convolution_chunks_and_resets_match_monolithic() {
    let device = Device::new_cuda(0).unwrap();
    let mut layer = layer(7, 4, &device);
    let mut first = None;
    for seed in [5, 9, 5] {
        layer.clear_cache();
        let input = strided(7, 547, seed, &device);
        let zeros = Tensor::zeros((1, 7, 4), DType::BF16, &device).unwrap();
        let (expected, next) = reference(&input, &zeros, &layer.conv_weight);
        let mut chunks = Vec::new();
        let mut offset = 0;
        for count in [2, 3, 512, 27, 1, 2] {
            chunks.push(
                layer
                    .causal_convolution(&input.narrow(2, offset, count).unwrap())
                    .unwrap(),
            );
            offset += count;
        }
        let actual = Tensor::cat(&chunks, 2).unwrap();
        assert_tensor_exact(&actual, &expected);
        assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &next);
        if seed == 5 {
            if let Some(first) = &first {
                assert_tensor_exact(&actual, first);
            } else {
                first = Some(actual);
            }
        }
    }
}

#[test]
fn prefill_convolution_other_window_sizes_keep_reference_behavior() {
    let device = Device::new_cuda(0).unwrap();
    for taps in [1, 2, 3, 5] {
        let mut layer = layer(7, taps, &device);
        check(
            &mut layer,
            &strided(7, 9, 2, &device),
            &strided(7, taps, 3, &device),
        );
    }
}

#[test]
fn prefill_convolution_dispatch_and_compact_history() {
    use crate::components::cuda::supports_prefill_convolution;
    let device = Device::new_cuda(0).unwrap();
    let mut layer = layer(7, 4, &device);
    let input = strided(7, 513, 2, &device);
    let history = strided(7, 4, 3, &device);
    // Contiguous weights with a nonzero storage offset also exercise the ABI.
    layer.conv_weight = patterned((8, 1, 4), 3, &device).narrow(0, 1, 7).unwrap();
    assert!(supports_prefill_convolution(
        &input,
        &history,
        &layer.conv_weight
    ));
    for tokens in [2, 3, 4, 513] {
        check(&mut layer, &input.narrow(2, 0, tokens).unwrap(), &history);
        let next = layer.conv_state.as_ref().unwrap();
        assert!(next.is_contiguous());
        let (storage, layout) = next.storage_and_layout();
        let candle::Storage::Cuda(storage) = &*storage else {
            unreachable!()
        };
        assert_eq!(layout.start_offset(), 0);
        assert_eq!(storage.as_cuda_slice::<bf16>().unwrap().len(), 7 * 4);
    }
    layer.optimize_prefill = false;
    check(&mut layer, &input, &history);
    layer.optimize_prefill = true;
    assert!(!supports_prefill_convolution(
        &input.narrow(2, 0, 1).unwrap(),
        &history,
        &layer.conv_weight
    ));
    assert!(!supports_prefill_convolution(
        &input.to_dtype(DType::F32).unwrap(),
        &history,
        &layer.conv_weight
    ));
    layer.conv_weight = patterned((7, 1, 5), 4, &device).narrow(2, 0, 4).unwrap();
    assert!(!supports_prefill_convolution(
        &input,
        &history,
        &layer.conv_weight
    ));
    // Reject strided weights before launch; exercise batch fallback separately.
    layer.conv_weight = layer.conv_weight.force_contiguous().unwrap();
    let input = Tensor::cat(&[&input, &input], 0).unwrap();
    let history = Tensor::cat(&[&history, &history], 0).unwrap();
    assert!(!supports_prefill_convolution(
        &input,
        &history,
        &layer.conv_weight
    ));
    let (expected, next) = reference(&input, &history, &layer.conv_weight);
    layer.conv_state = Some(history);
    assert_tensor_exact(&layer.causal_convolution(&input).unwrap(), &expected);
    assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &next);
}

#[test]
#[ignore = "release prefill convolution benchmark; serialize with other GPU work"]
fn measures_prefill_convolution() {
    // Match graph-mode eager tails: one stream and no per-allocation events.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let mut cases = Vec::new();
    for channels in [8192, 10240] {
        for tokens in [27, 512] {
            let mut layer = layer(channels, 4, &device);
            let input = strided(channels, tokens, 5, &device);
            let history = strided(channels, 4, 7, &device);
            check(&mut layer, &input, &history);
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..7 {
                let order = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for optimized in order {
                    layer.optimize_prefill = optimized;
                    layer.conv_state = Some(history.clone());
                    device.synchronize().unwrap();
                    let start = std::time::Instant::now();
                    for _ in 0..3 {
                        let _ = layer.causal_convolution(&input).unwrap();
                    }
                    device.synchronize().unwrap();
                    if round > 0 {
                        samples[usize::from(optimized)]
                            .push(start.elapsed().as_secs_f64() * 1e6 / 3.0);
                    }
                }
            }
            cases.push(serde_json::json!({"channels":channels,"tokens":tokens,
                "reference_us":samples[0],"native_us":samples[1]}));
        }
    }
    eprintln!(
        "PREFILL_CONV_BENCH {}",
        serde_json::json!({"cases":cases,
        "timing":"synchronized host wall time; convolution + activation + history \
         update; six alternating rounds of three calls",
        "scope":"BF16 batch-one operator, not full GDN or server throughput"})
    );
}
