//! Test-only native FP32 golden. Never dispatches an optimized GDN operator.
//!
//! Promote the same checkpoint/boundary values to FP32 and use the existing
//! unoptimized Candle path. No frozen binary or optimized kernel defines truth.

use super::*;
use std::collections::HashMap;

pub(in crate::components) fn layer(
    config: &GatedDeltaNetConfig,
    tensors: &HashMap<String, Tensor>,
    device: &candle::Device,
) -> CandleResult<GatedDeltaNet> {
    let tensors = tensors
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor.to_dtype(DType::F32)?)))
        .collect::<CandleResult<HashMap<_, _>>>()?;
    GatedDeltaNet::new(
        config,
        false,
        VarBuilder::from_tensors(tensors, DType::F32, device),
    )
}

/// The recurrence has no weight dependency. A zero-weight native FP32 layer
/// supplies its dimension configuration; explicit inputs/state supply all math.
pub(in crate::components) fn recurrence(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    recurrence_with_storage(query, key, value, decay, beta, initial, false)
}

/// FP32 state arithmetic with the MoE prefill storage boundary: Q/K normalize
/// in FP32, round to BF16, then return to FP32 before query scaling. Operator
/// regressions use this composed path independently of native preparation.
pub(in crate::components) fn recurrence_with_bf16_normalized_qk(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    assert_eq!(query.dtype(), DType::BF16);
    assert_eq!(key.dtype(), DType::BF16);
    recurrence_with_storage(query, key, value, decay, beta, initial, true)
}

fn recurrence_with_storage(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
    round_normalized: bool,
) -> CandleResult<(Tensor, Tensor)> {
    let (_, _, heads, key_dim) = query.dims4()?;
    let config = GatedDeltaNetConfig {
        hidden_size: 1,
        conv_kernel_size: 4,
        key_head_dim: key_dim,
        value_head_dim: value.dim(3)?,
        num_key_heads: heads,
        num_value_heads: heads,
        norm_eps: 1e-6,
        conv_activation: Activation::Silu,
        gate_activation: Activation::Silu,
    };
    let mut layer = GatedDeltaNet::new(
        &config,
        false,
        VarBuilder::zeros(DType::F32, query.device()),
    )?
    .with_reference_precision(round_normalized);
    layer.recurrent_state = Some(initial.to_dtype(DType::F32)?);
    let operand = |tensor: &Tensor| {
        if round_normalized {
            Ok(tensor.clone())
        } else {
            tensor.to_dtype(DType::F32)
        }
    };
    let output = layer.recurrent_delta_rule(
        &operand(query)?,
        &operand(key)?,
        &value.to_dtype(DType::F32)?,
        &decay.to_dtype(DType::F32)?,
        &beta.to_dtype(DType::F32)?,
    )?;
    Ok((output, layer.recurrent_state.unwrap()))
}

/// Numerical accuracy relative to FP32, independent of its exact reduction tree.
/// Exact equality remains appropriate for state immutability and graph replay.
pub(in crate::components) fn assert_close(
    actual: &Tensor,
    expected: &Tensor,
    atol: f32,
    rtol: f32,
    label: &str,
) -> f32 {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(expected.dtype(), DType::F32, "golden must stay FP32");
    let flat = |tensor: &Tensor| {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let a = flat(actual);
    let e = flat(expected);
    let mut maximum = 0f32;
    let mut square_error = 0f64;
    for (i, (&actual, &expected)) in a.iter().zip(&e).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && error <= atol + rtol * expected.abs(),
            "{label}[{i}]: {actual} versus FP32 {expected}, error={error}"
        );
        maximum = maximum.max(error);
        square_error += f64::from(error).powi(2);
    }
    eprintln!(
        "FP32_GOLDEN {label} max_abs={maximum:e} rmse={:e}",
        (square_error / a.len() as f64).sqrt()
    );
    maximum
}
