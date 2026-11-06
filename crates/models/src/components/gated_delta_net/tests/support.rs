//! Shared GDN fixtures and assertion helpers.

use super::*;

/// Returns a small valid GDN configuration independent of any model family.
pub(super) fn valid_config() -> GatedDeltaNetConfig {
    GatedDeltaNetConfig {
        hidden_size: 4,
        conv_kernel_size: 2,
        key_head_dim: 2,
        value_head_dim: 2,
        num_key_heads: 1,
        num_value_heads: 2,
        norm_eps: 1e-6,
        conv_activation: Activation::Silu,
        gate_activation: Activation::Silu,
    }
}

/// Creates a reproducible BF16 tensor whose entries exercise positive and negative
/// paths.
pub(super) fn patterned_bf16(
    shape: impl Into<Shape>,
    phase: usize,
    scale: f32,
    device: &Device,
) -> Tensor {
    let shape = Into::<Shape>::into(shape);
    let values = (0..shape.elem_count())
        .map(|index| {
            let centered = ((index * 17 + phase) % 31) as f32 - 15.0;
            centered * scale
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

/// Builds identical small checkpoint tensors for native/reference sequence tests.
pub(super) fn test_tensors(
    config: &GatedDeltaNetConfig,
    device: &Device,
) -> HashMap<String, Tensor> {
    let key_width = config.num_key_heads * config.key_head_dim;
    let value_width = config.num_value_heads * config.value_head_dim;
    let conv_width = key_width * 2 + value_width;
    HashMap::from([
        (
            "in_proj_qkv.weight".to_owned(),
            patterned_bf16((conv_width, config.hidden_size), 1, 0.003, device),
        ),
        (
            "in_proj_z.weight".to_owned(),
            patterned_bf16((value_width, config.hidden_size), 3, 0.004, device),
        ),
        (
            "in_proj_b.weight".to_owned(),
            patterned_bf16(
                (config.num_value_heads, config.hidden_size),
                5,
                0.005,
                device,
            ),
        ),
        (
            "in_proj_a.weight".to_owned(),
            patterned_bf16(
                (config.num_value_heads, config.hidden_size),
                7,
                0.004,
                device,
            ),
        ),
        (
            "conv1d.weight".to_owned(),
            patterned_bf16((conv_width, 1, config.conv_kernel_size), 9, 0.02, device),
        ),
        (
            "dt_bias".to_owned(),
            patterned_bf16(config.num_value_heads, 11, 0.02, device),
        ),
        (
            "A_log".to_owned(),
            Tensor::from_vec(
                (0..config.num_value_heads)
                    .map(|head| -1.0 + head as f32 * 0.1)
                    .collect::<Vec<_>>(),
                config.num_value_heads,
                device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
        ),
        (
            "norm.weight".to_owned(),
            Tensor::from_vec(
                (0..config.value_head_dim)
                    .map(|column| 0.9 + column as f32 * 0.03)
                    .collect::<Vec<_>>(),
                config.value_head_dim,
                device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
        ),
        (
            "out_proj.weight".to_owned(),
            patterned_bf16((config.hidden_size, value_width), 13, 0.006, device),
        ),
    ])
}

/// Supplies master parameters that cannot survive a BF16 round trip unchanged.
pub(super) fn add_fp32_parameters(
    tensors: &mut HashMap<String, Tensor>,
    config: &GatedDeltaNetConfig,
    device: &Device,
) {
    for (name, count, base) in [
        ("A_log", config.num_value_heads, -0.9731),
        ("norm.weight", config.value_head_dim, 0.9137),
    ] {
        let values = (0..count)
            .map(|i| base + i as f32 * 0.0317)
            .collect::<Vec<_>>();
        tensors.insert(
            name.to_owned(),
            Tensor::from_vec(values, count, device).unwrap(),
        );
    }
}

/// Checks two device tensors elementwise and returns their maximum absolute error.
pub(super) fn assert_bf16_close(
    optimized: &Tensor,
    reference: &Tensor,
    tolerance: f32,
    stage: &str,
) -> f32 {
    assert_eq!(optimized.dims(), reference.dims());
    let optimized = optimized
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let reference = reference
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_absolute_error = 0.0_f32;
    for (index, (&optimized, &reference)) in
        optimized.iter().zip(&reference).enumerate()
    {
        assert!(optimized.is_finite(), "{stage} value {index} is not finite");
        let absolute_error = (optimized - reference).abs();
        max_absolute_error = max_absolute_error.max(absolute_error);
        assert!(
            absolute_error <= tolerance,
            "{stage} value {index} differs: \
             optimized={optimized}, reference={reference}, \
             tolerance={tolerance}"
        );
    }
    max_absolute_error
}

pub(super) fn cuda_test_config() -> GatedDeltaNetConfig {
    GatedDeltaNetConfig {
        hidden_size: 16,
        conv_kernel_size: 4,
        key_head_dim: 4,
        value_head_dim: 4,
        num_key_heads: 1,
        num_value_heads: 2,
        norm_eps: 1e-6,
        conv_activation: Activation::Silu,
        gate_activation: Activation::Silu,
    }
}

/// Checks all bits, including FP32 recurrent state, without a BF16 error allowance.
pub(super) fn assert_tensor_exact(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.dims(), expected.dims());
    let values = |tensor: &Tensor| {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    for (index, (actual, expected)) in
        values(actual).into_iter().zip(values(expected)).enumerate()
    {
        assert!(actual.is_finite() && expected.is_finite());
        assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
    }
}

pub(super) fn native_state(layer: &GatedDeltaNet) -> (Tensor, Tensor) {
    layer
        .optimized_decode
        .as_ref()
        .unwrap()
        .runner
        .export_state()
        .unwrap()
}

pub(super) fn native_state_addresses(layer: &GatedDeltaNet) -> (u64, u64) {
    layer
        .optimized_decode
        .as_ref()
        .unwrap()
        .runner
        .state_addresses()
        .unwrap()
}

pub(super) fn assert_native_state_exact(
    actual: &GatedDeltaNet,
    expected: &GatedDeltaNet,
) {
    let (actual_conv, actual_recurrent) = native_state(actual);
    let (expected_conv, expected_recurrent) = native_state(expected);
    assert_tensor_exact(&actual_conv, &expected_conv);
    assert_tensor_exact(&actual_recurrent, &expected_recurrent);
}
