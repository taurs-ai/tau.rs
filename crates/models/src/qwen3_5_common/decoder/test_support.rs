//! Shared decoder fixtures and independent arithmetic comparisons.

use super::*;
use crate::components::Mlp;
use candle::{DType, Device};
use candle_nn::Activation;
use std::collections::HashMap;

pub(super) fn patterned(
    shape: impl Into<candle::Shape>,
    phase: usize,
    dtype: DType,
    device: &Device,
) -> Tensor {
    let shape = Into::<candle::Shape>::into(shape);
    let values = (0..shape.elem_count())
        .map(|i| ((i + phase) as f32 * 0.07).sin() * 0.03)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

pub(super) fn synthetic_layer(device: &Device, mixed: bool) -> HybridDecoderLayer<Mlp> {
    synthetic_gdn_layer(device, mixed, 8)
}

pub(super) fn synthetic_gdn_layer(
    device: &Device,
    mixed: bool,
    head_dim: usize,
) -> HybridDecoderLayer<Mlp> {
    let primary_dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let config = GatedDeltaNetConfig {
        hidden_size: 32,
        conv_kernel_size: 4,
        key_head_dim: head_dim,
        value_head_dim: head_dim,
        num_key_heads: 2,
        num_value_heads: 4,
        norm_eps: 1e-6,
        conv_activation: Activation::Silu,
        gate_activation: Activation::Silu,
    };
    let mut tensors = HashMap::new();
    for (i, (name, dims)) in [
        ("gdn.in_proj_qkv.weight", vec![8 * head_dim, 32]),
        ("gdn.in_proj_z.weight", vec![4 * head_dim, 32]),
        ("gdn.in_proj_b.weight", vec![4, 32]),
        ("gdn.in_proj_a.weight", vec![4, 32]),
        ("gdn.conv1d.weight", vec![8 * head_dim, 1, 4]),
        ("gdn.dt_bias", vec![4]),
        ("gdn.A_log", vec![4]),
        ("gdn.norm.weight", vec![head_dim]),
        ("gdn.out_proj.weight", vec![32, 4 * head_dim]),
        ("mlp.gate_proj.weight", vec![64, 32]),
        ("mlp.up_proj.weight", vec![64, 32]),
        ("mlp.down_proj.weight", vec![32, 64]),
        ("input_norm.weight", vec![32]),
        ("post_norm.weight", vec![32]),
    ]
    .into_iter()
    .enumerate()
    {
        let dtype = if mixed && ["gdn.A_log", "gdn.norm.weight"].contains(&name) {
            DType::F32
        } else {
            primary_dtype
        };
        tensors.insert(name.to_owned(), patterned(dims, 1 + i * 13, dtype, device));
    }
    let vb = VarBuilder::from_tensors(tensors, primary_dtype, device);
    HybridDecoderLayer {
        decode_graph: None,
        prefill_graph: None,
        token_mixer: TokenMixer::Linear(Box::new(
            GatedDeltaNet::new(&config, true, vb.pp("gdn")).unwrap(),
        )),
        feed_forward: Mlp::new(32, 64, Activation::Silu, true, vb.pp("mlp")).unwrap(),
        input_layernorm: CenteredRmsNorm::new(32, 1e-6, true, vb.pp("input_norm"))
            .unwrap(),
        post_attention_layernorm: CenteredRmsNorm::new(
            32,
            1e-6,
            true,
            vb.pp("post_norm"),
        )
        .unwrap(),
    }
}

/// Reuse the same norms/MLP fixture with a full-attention token mixer.
pub(super) fn synthetic_attention_layer(device: &Device) -> HybridDecoderLayer<Mlp> {
    let mut layer = synthetic_layer(device, false);
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let tensors = [
        ("q_proj.weight", vec![64, 32]),
        ("k_proj.weight", vec![16, 32]),
        ("v_proj.weight", vec![16, 32]),
        ("o_proj.weight", vec![32, 32]),
        ("q_norm.weight", vec![8]),
        ("k_norm.weight", vec![8]),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (name, shape))| {
        (name.to_owned(), patterned(shape, 13 * i, dtype, device))
    })
    .collect();
    let rotary = PartialRotaryEmbedding::new(
        &crate::components::PartialRotaryConfig {
            head_dim: 8,
            rotary_dim: 4,
            max_position_embeddings: 128,
            theta: 10000.0,
        },
        dtype,
        device,
    )
    .unwrap();
    layer.token_mixer = TokenMixer::Full(Box::new(
        GatedAttention::new(
            &GatedAttentionConfig {
                hidden_size: 32,
                num_query_heads: 4,
                num_key_value_heads: 2,
                head_dim: 8,
                projection_bias: false,
                norm_eps: 1e-6,
            },
            Arc::new(rotary),
            crate::ModelLoadOptions::default(),
            VarBuilder::from_tensors(tensors, dtype, device),
        )
        .unwrap(),
    ));
    layer
}

/// Independent expression of the pre-refactor layer order. Do not use the
/// new shared helper here: eager/graph agreement alone could hide a common bug.
pub(super) fn original_layer_forward(
    layer: &mut HybridDecoderLayer<Mlp>,
    xs: &Tensor,
    mask: Option<&Tensor>,
    offset: usize,
) -> CandleResult<Tensor> {
    let residual = xs;
    let hidden = layer.input_layernorm.forward(xs)?;
    let hidden = layer.token_mixer.forward(&hidden, mask, offset)?;
    let hidden = (residual + hidden)?;
    let residual = &hidden;
    let hidden = layer.post_attention_layernorm.forward(&hidden)?;
    let hidden = layer.feed_forward.forward(&hidden)?;
    residual + hidden
}

pub(super) fn gdn(layer: &mut HybridDecoderLayer<Mlp>) -> &mut GatedDeltaNet {
    match &mut layer.token_mixer {
        TokenMixer::Linear(gdn) => gdn,
        _ => unreachable!(),
    }
}

pub(super) fn assert_exact(actual: &Tensor, expected: &Tensor) {
    let values = |xs: &Tensor| {
        xs.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(actual.dims(), expected.dims());
    for (a, b) in values(actual).iter().zip(values(expected)) {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
