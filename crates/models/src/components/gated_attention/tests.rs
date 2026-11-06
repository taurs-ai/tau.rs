use super::kv_cache::tests::{exact, patterned, values};
use super::*;
use crate::components::PartialRotaryConfig;
use candle::Device;
use std::collections::HashMap;

#[path = "decode_tests.rs"]
mod decode_tests;

#[test]
fn workspace_estimate_includes_expanded_kv_and_prefill_scores() {
    let layer = layer(&Device::Cpu, DType::F32, 4);
    let (kv, workspace) = layer.kv_memory_per_token(512).unwrap();
    assert_eq!(kv, 2 * 2 * 8 * 4);
    assert_eq!(workspace, 6 * 8 * 8 * 4 + 8 * 512 * (16 + 16) + 512 * 8);
    assert!(layer.kv_memory_per_token(usize::MAX).is_err());
}

fn layer(device: &Device, dtype: DType, groups: usize) -> GatedAttention {
    layer_shape(device, dtype, groups, 2, 8)
}

fn layer_shape(
    device: &Device,
    dtype: DType,
    groups: usize,
    heads: usize,
    dim: usize,
) -> GatedAttention {
    let width = heads * groups * dim;
    let mut tensors = HashMap::new();
    for (i, (name, shape)) in [
        ("q_proj.weight", vec![width * 2, 16]),
        ("k_proj.weight", vec![heads * dim, 16]),
        ("v_proj.weight", vec![heads * dim, 16]),
        ("o_proj.weight", vec![16, width]),
        ("q_norm.weight", vec![dim]),
        ("k_norm.weight", vec![dim]),
    ]
    .into_iter()
    .enumerate()
    {
        tensors.insert(name.to_owned(), patterned(&shape, i * 19, dtype, device));
    }
    let rotary = PartialRotaryEmbedding::new(
        &PartialRotaryConfig {
            head_dim: dim,
            rotary_dim: 4,
            max_position_embeddings: 2048,
            theta: 10000.0,
        },
        dtype,
        device,
    )
    .unwrap();
    GatedAttention::new(
        &GatedAttentionConfig {
            hidden_size: 16,
            num_query_heads: heads * groups,
            num_key_value_heads: heads,
            head_dim: dim,
            projection_bias: false,
            norm_eps: 1e-6,
        },
        Arc::new(rotary),
        crate::ModelLoadOptions::default(),
        VarBuilder::from_tensors(tensors, dtype, device),
    )
    .unwrap()
}

fn mask(tokens: usize, offset: usize, dtype: DType, device: &Device) -> Tensor {
    let data = (0..tokens)
        .flat_map(|q| {
            (0..offset + tokens).map(move |k| {
                if k <= offset + q {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(data, (1, 1, tokens, offset + tokens), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn check_attention(device: &Device, dtype: DType) {
    for groups in [1, 2, 4] {
        let mut layer = layer(device, dtype, groups);
        layer.reserve_kv_cache(24).unwrap();
        let ids = {
            let (k, v) = layer.kv_cache.buffers.as_ref().unwrap();
            (k.id(), v.id())
        };
        // A/B/A changes prompt lengths and content while reusing the same storage.
        for (phase, chunks) in [
            (7, vec![3, 2, 1, 1]),
            (29, vec![1, 1, 1]),
            (7, vec![3, 2, 1, 1]),
        ] {
            layer.clear_cache();
            layer.reserve_kv_cache(16).unwrap();
            for t in [
                &layer.kv_cache.buffers.as_ref().unwrap().0,
                &layer.kv_cache.buffers.as_ref().unwrap().1,
            ] {
                let poison = Tensor::full(f32::NAN, t.shape(), device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                t.slice_set(&poison, 2, 0).unwrap();
            }
            let mut reference = None;
            let mut offset = 0;
            for tokens in chunks {
                let xs = patterned(&[1, tokens, 16], phase + offset, dtype, device);
                let mask = mask(tokens, offset, dtype, device);
                let expected =
                    reference_forward(&layer, &xs, Some(&mask), offset, &mut reference)
                        .unwrap();
                let actual = layer.forward(&xs, Some(&mask), offset).unwrap();
                exact(&actual, &expected);
                let (k, v) = layer.kv_cache.prefix().unwrap();
                let (rk, rv) = reference.as_ref().unwrap();
                exact(&k, rk);
                exact(&v, rv);
                offset += tokens;
            }
            let (k, v) = layer.kv_cache.buffers.as_ref().unwrap();
            assert_eq!((k.id(), v.id()), ids);
        }
    }
}

#[test]
fn attention_matches_original_with_poisoned_tail_cpu() {
    check_attention(&Device::Cpu, DType::F32);
}

#[test]
fn attention_matches_original_with_poisoned_tail_cuda_bf16() {
    check_attention(&Device::new_cuda(0).unwrap(), DType::BF16);
}

/// Frozen preallocation-independent reference: original attention with growing cats.
fn reference_forward(
    layer: &GatedAttention,
    xs: &Tensor,
    attention_mask: Option<&Tensor>,
    offset: usize,
    cache: &mut Option<(Tensor, Tensor)>,
) -> CandleResult<Tensor> {
    let (batch_size, sequence_length, _) = xs.dims3()?;
    let query_and_gate = layer.q_proj.forward(xs)?.reshape((
        batch_size,
        sequence_length,
        layer.num_heads,
        layer.head_dim * 2,
    ))?;
    let query_and_gate = query_and_gate.chunk(2, D::Minus1)?;
    let query = layer.q_norm.forward(&query_and_gate[0])?.transpose(1, 2)?;
    let output_gate =
        query_and_gate[1].reshape((batch_size, sequence_length, layer.query_width))?;
    let key = layer
        .k_norm
        .forward(&layer.k_proj.forward(xs)?.reshape((
            batch_size,
            sequence_length,
            layer.num_kv_heads,
            layer.head_dim,
        ))?)?
        .transpose(1, 2)?;
    let value = layer
        .v_proj
        .forward(xs)?
        .reshape((
            batch_size,
            sequence_length,
            layer.num_kv_heads,
            layer.head_dim,
        ))?
        .transpose(1, 2)?;
    let (query, key) = layer.rotary.apply(&query, &key, offset)?;

    let (key, value) = match &*cache {
        Some((cached_key, cached_value)) => (
            Tensor::cat(&[cached_key, &key], 2)?,
            Tensor::cat(&[cached_value, &value], 2)?,
        ),
        None => (key, value),
    };
    *cache = Some((key.clone(), value.clone()));

    let key = candle_transformers::utils::repeat_kv(key, layer.num_kv_groups)?
        .contiguous()?;
    let value = candle_transformers::utils::repeat_kv(value, layer.num_kv_groups)?
        .contiguous()?;
    let scale = 1.0 / (layer.head_dim as f64).sqrt();
    // RoPE can return a non-contiguous view for single-token cached decode
    // even when its input was contiguous. Enforce Candle's CUDA matmul
    // contract at the operator boundary, after every layout-changing step.
    let query = query.contiguous()?;
    let key_transposed = key.transpose(2, 3)?.contiguous()?;
    let mut scores = (query.matmul(&key_transposed)? * scale)?;
    if let Some(mask) = attention_mask {
        scores = scores.broadcast_add(mask)?;
    }
    let probabilities =
        candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(value.dtype())?;
    let context = probabilities
        .matmul(&value)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch_size, sequence_length, layer.query_width))?;
    let context = context.broadcast_mul(&candle_nn::ops::sigmoid(&output_gate)?)?;
    layer.o_proj.forward(&context)
}

/// Supplies a compact valid grouped-query configuration.
fn valid_config() -> GatedAttentionConfig {
    GatedAttentionConfig {
        hidden_size: 16,
        num_query_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        projection_bias: false,
        norm_eps: 1e-6,
    }
}

/// Rejects incompatible head grouping and invalid normalization epsilon.
#[test]
fn validates_general_config() {
    valid_config().validate().unwrap();

    let mut invalid = valid_config();
    invalid.num_query_heads = 3;
    assert!(invalid.validate().unwrap_err().contains("divisible"));

    let mut invalid = valid_config();
    invalid.norm_eps = 0.0;
    assert!(invalid.validate().unwrap_err().contains("epsilon"));
}
