//! GDN construction, numerical and state-lifecycle regressions.

mod graph;
mod lifecycle;
mod support;
use support::*;
mod prefill_bench;
mod prefill_convolution;
mod prefill_recurrence;
mod prefill_reference;

use super::*;
use candle::Device;
use candle::Shape;
use std::collections::HashMap;

/// Confirms a structurally valid model-neutral component config is accepted.
#[test]
fn validates_general_config() {
    valid_config().validate().unwrap();
}

/// Confirms key heads must repeat evenly across value heads.
#[test]
fn rejects_incompatible_head_counts() {
    let mut config = valid_config();
    config.num_key_heads = 2;
    config.num_value_heads = 3;

    let error = config.validate().unwrap_err();
    assert!(error.contains("value heads must be divisible"));
}

/// Confirms normalization epsilon must remain finite and positive.
#[test]
fn rejects_invalid_normalization_epsilon() {
    let mut config = valid_config();
    config.norm_eps = 0.0;

    let error = config.validate().unwrap_err();
    assert!(error.contains("epsilon must be positive"));
}

/// Constructs and executes GDN without any model-family configuration type.
#[test]
fn constructs_and_runs_from_general_config() {
    let device = Device::Cpu;
    let config = valid_config();
    let key_width = config.num_key_heads * config.key_head_dim;
    let value_width = config.num_value_heads * config.value_head_dim;
    let conv_width = key_width * 2 + value_width;
    let tensors = HashMap::from([
        (
            "in_proj_qkv.weight".to_owned(),
            Tensor::zeros((conv_width, config.hidden_size), DType::F32, &device)
                .unwrap(),
        ),
        (
            "in_proj_z.weight".to_owned(),
            Tensor::zeros((value_width, config.hidden_size), DType::F32, &device)
                .unwrap(),
        ),
        (
            "in_proj_b.weight".to_owned(),
            Tensor::zeros(
                (config.num_value_heads, config.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        ),
        (
            "in_proj_a.weight".to_owned(),
            Tensor::zeros(
                (config.num_value_heads, config.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        ),
        (
            "conv1d.weight".to_owned(),
            Tensor::zeros(
                (conv_width, 1, config.conv_kernel_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        ),
        (
            "dt_bias".to_owned(),
            Tensor::zeros(config.num_value_heads, DType::F32, &device).unwrap(),
        ),
        (
            "A_log".to_owned(),
            Tensor::zeros(config.num_value_heads, DType::F32, &device).unwrap(),
        ),
        (
            "norm.weight".to_owned(),
            Tensor::ones(config.value_head_dim, DType::F32, &device).unwrap(),
        ),
        (
            "out_proj.weight".to_owned(),
            Tensor::zeros((config.hidden_size, value_width), DType::F32, &device)
                .unwrap(),
        ),
    ]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let mut layer = GatedDeltaNet::new(&config, false, vb).unwrap();

    let prefill =
        Tensor::ones((1, 2, config.hidden_size), DType::F32, &device).unwrap();
    let output = layer.forward(&prefill).unwrap();
    assert_eq!(output.dims3().unwrap(), (1, 2, config.hidden_size));
    assert!(layer.conv_state.is_some());
    assert!(layer.recurrent_state.is_some());

    let decode = Tensor::ones((1, 1, config.hidden_size), DType::F32, &device).unwrap();
    let output = layer.forward(&decode).unwrap();
    assert_eq!(output.dims3().unwrap(), (1, 1, config.hidden_size));

    layer.clear_cache();
    assert!(layer.conv_state.is_none());
    assert!(layer.recurrent_state.is_none());
}

/// Reads actual mixed safetensors using a BF16 builder without rounding FP32
/// weights.
#[test]
fn preserves_fp32_parameters_when_loading_bf16_safetensors() {
    let device = Device::Cpu;
    let config = valid_config();
    let mut tensors = test_tensors(&config, &device);
    add_fp32_parameters(&mut tensors, &config, &device);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "tau-gdn-mixed-{}-{unique}.safetensors",
        std::process::id()
    ));
    candle::safetensors::save(&tensors, &path).unwrap();
    // SAFETY: This test owns the file and does not modify it while mapped.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::BF16, &device) }
            .unwrap();
    let layer = GatedDeltaNet::new(&config, false, vb).unwrap();
    for (name, actual) in [
        ("A_log", &layer.a_log),
        ("norm.weight", layer.norm.weight()),
    ] {
        assert_eq!(actual.dtype(), DType::F32);
        let expected = tensors[name].to_vec1::<f32>().unwrap();
        let rounded = tensors[name]
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(expected, rounded, "{name} must detect lossy loading");
        assert_eq!(actual.to_vec1::<f32>().unwrap(), expected, "{name}");
    }
    assert_eq!(layer.conv_weight.dtype(), DType::BF16);
    drop(layer);
    std::fs::remove_file(path).unwrap();
}

/// Compares native state transitions with Candle across prefill, decode, fallback,
/// and reset.
#[test]
fn optimized_cuda_decode_matches_stateful_reference() {
    assert_cuda_decode_matches_stateful_reference(false);
}

/// Exercises FP32 master parameters through native decode and reference fallback.
#[test]
fn optimized_cuda_decode_matches_mixed_storage_reference() {
    assert_cuda_decode_matches_stateful_reference(true);
}

/// Compare the entire production-head path, including convolution and both
/// state buffers, with Candle. Nontrivial FP32 weights test mixed storage.
#[test]
fn production_gdn_matches_native_fp32_outputs_and_state() {
    let device = Device::new_cuda(0).unwrap();
    let mut config = cuda_test_config();
    config.key_head_dim = 128;
    config.value_head_dim = 128;
    config.num_key_heads = 2;
    config.num_value_heads = 4;
    let mut tensors = test_tensors(&config, &device);
    add_fp32_parameters(&mut tensors, &config, &device);
    let mut optimized = GatedDeltaNet::new(
        &config,
        true,
        VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
    )
    .unwrap();
    let mut reference = super::golden::layer(&config, &tensors, &device).unwrap();
    let mut max_state_error = 0.0_f32;
    for step in 0..36 {
        if step == 18 {
            optimized.clear_cache();
            reference.clear_cache();
        }
        // Multi-token inputs exercise initial prefill and native -> reference
        // -> native transitions, including state seeding after a reset.
        let tokens = if step % 9 == 0 { 3 } else { 1 };
        let input =
            patterned_bf16((1, tokens, config.hidden_size), 17 + step, 0.02, &device);
        let actual = optimized.forward(&input).unwrap();
        let expected = reference
            .forward(&input.to_dtype(DType::F32).unwrap())
            .unwrap();
        assert_bf16_close(&actual, &expected, 3e-2, "production output");
        let (conv, state) = optimized.snapshot_current_state().unwrap();
        let (expected_conv, expected_state) =
            reference.snapshot_current_state().unwrap();
        assert_bf16_close(&conv, &expected_conv, 3e-4, "FP32 convolution history");
        // Complete-path comparison includes BF16 projection/convolution error.
        // The recurrence-only golden separately isolates algorithmic state error.
        max_state_error = max_state_error.max(assert_bf16_close(
            &state,
            &expected_state,
            3e-4,
            "production FP32 state",
        ));
    }
    eprintln!("production native FP32 state max error: {max_state_error:e}");
}

fn assert_cuda_decode_matches_stateful_reference(mixed_storage: bool) {
    const TOLERANCE: f32 = 3e-2;

    let device = Device::new_cuda(0).expect("Gated DeltaNet test requires CUDA GPU 0");
    let config = cuda_test_config();
    let mut tensors = test_tensors(&config, &device);
    if mixed_storage {
        add_fp32_parameters(&mut tensors, &config, &device);
    }
    let mut optimized = GatedDeltaNet::new(
        &config,
        true,
        VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
    )
    .unwrap();
    let mut reference = GatedDeltaNet::new(
        &config,
        false,
        VarBuilder::from_tensors(tensors, DType::BF16, &device),
    )
    .unwrap();
    assert!(optimized.optimized_decode.is_some());

    let prefill = patterned_bf16((1, 3, config.hidden_size), 17, 0.02, &device);
    let optimized_prefill = optimized.forward(&prefill).unwrap();
    let reference_prefill = reference.forward(&prefill).unwrap();
    assert_bf16_close(
        &optimized_prefill,
        &reference_prefill,
        0.0,
        "reference prefill",
    );

    let mut decode_max_absolute_error = 0.0_f32;
    for step in 0..4 {
        let decode =
            patterned_bf16((1, 1, config.hidden_size), 19 + step, 0.018, &device);
        let optimized_output = optimized.forward(&decode).unwrap();
        let reference_output = reference.forward(&decode).unwrap();
        decode_max_absolute_error = decode_max_absolute_error.max(assert_bf16_close(
            &optimized_output,
            &reference_output,
            TOLERANCE,
            "stateful decode",
        ));
    }

    let fallback = patterned_bf16((1, 2, config.hidden_size), 23, 0.017, &device);
    let optimized_fallback = optimized.forward(&fallback).unwrap();
    let reference_fallback = reference.forward(&fallback).unwrap();
    let fallback_max_absolute_error = assert_bf16_close(
        &optimized_fallback,
        &reference_fallback,
        TOLERANCE,
        "native-to-reference fallback",
    );

    optimized.clear_cache();
    reference.clear_cache();
    let reset_optimized = optimized.forward(&prefill).unwrap();
    let reset_reference = reference.forward(&prefill).unwrap();
    assert_bf16_close(
        &reset_optimized,
        &reset_reference,
        0.0,
        "cache-reset prefill",
    );
    eprintln!(
        "Gated DeltaNet CUDA errors: decode_max_abs={decode_max_absolute_error:.8}, \
         fallback_max_abs={fallback_max_absolute_error:.8}"
    );
}

/// Measures one-token GDN calls at the dense tiny model's production dimensions.
#[test]
#[ignore = "runs a release-mode CUDA microbenchmark on GPU 0"]
fn measures_tiny_dense_shape_decode_against_reference() {
    use std::time::{Duration, Instant};

    const WARMUP_ITERATIONS: usize = 8;
    const MEASURED_ITERATIONS: usize = 40;

    /// Times stateful decode calls and synchronizes around only the measured span.
    fn measure(
        layer: &mut GatedDeltaNet,
        input: &Tensor,
        device: &Device,
        iterations: usize,
    ) -> Duration {
        device.synchronize().unwrap();
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(layer.forward(input).unwrap());
        }
        device.synchronize().unwrap();
        started.elapsed()
    }

    let device =
        Device::new_cuda(0).expect("Gated DeltaNet microbenchmark requires CUDA GPU 0");
    let config = GatedDeltaNetConfig {
        hidden_size: 320,
        conv_kernel_size: 4,
        key_head_dim: 64,
        value_head_dim: 64,
        num_key_heads: 2,
        num_value_heads: 6,
        norm_eps: 1e-6,
        conv_activation: Activation::Silu,
        gate_activation: Activation::Silu,
    };
    let tensors = test_tensors(&config, &device);
    let mut optimized = GatedDeltaNet::new(
        &config,
        true,
        VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
    )
    .unwrap();
    let mut reference = GatedDeltaNet::new(
        &config,
        false,
        VarBuilder::from_tensors(tensors, DType::BF16, &device),
    )
    .unwrap();
    let input = patterned_bf16((1, 1, config.hidden_size), 29, 0.01, &device);

    measure(&mut reference, &input, &device, WARMUP_ITERATIONS);
    measure(&mut optimized, &input, &device, WARMUP_ITERATIONS);
    let reference_first = measure(&mut reference, &input, &device, MEASURED_ITERATIONS);
    let optimized_first = measure(&mut optimized, &input, &device, MEASURED_ITERATIONS);
    let optimized_second =
        measure(&mut optimized, &input, &device, MEASURED_ITERATIONS);
    let reference_second =
        measure(&mut reference, &input, &device, MEASURED_ITERATIONS);
    let reference_elapsed = reference_first + reference_second;
    let optimized_elapsed = optimized_first + optimized_second;
    let iterations = (2 * MEASURED_ITERATIONS) as f64;
    let reference_us = reference_elapsed.as_secs_f64() * 1e6 / iterations;
    let optimized_us = optimized_elapsed.as_secs_f64() * 1e6 / iterations;
    let speedup = reference_us / optimized_us;

    println!(
        "tiny_dense_gated_delta_net_decode reference_us={reference_us:.2} \
         optimized_us={optimized_us:.2} speedup={speedup:.2}x"
    );
    assert!(
        optimized_elapsed < reference_elapsed,
        "native Gated DeltaNet decode did not beat the reference path"
    );
}
