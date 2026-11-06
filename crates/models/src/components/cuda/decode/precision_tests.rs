//! Independent checks for the MoE attention precision preset. Dense defaults
//! continue to use the existing preparation and FP32 attention oracle tests.

use super::*;
use crate::components::write_metadata;
use half::bf16;

/// Converts actual BF16 outputs to FP32 without changing their stored values.
fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Constant heads make normalization exact, so this isolates coefficient dtype,
/// FP32 rotation, the final BF16 round, metadata updates and both cache layouts.
#[test]
fn fp32_rotary_preparation_matches_scalar_rotation() {
    // SAFETY: Every operation and retained buffer uses this test's private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let (heads, dim, capacity) = (2, 64, 65);
    let make = |shape| Tensor::ones(shape, DType::BF16, &device).unwrap();
    let q = make(vec![2 * heads * dim]);
    let k = make(vec![dim]);
    let v = make(vec![dim]);
    let norm = Tensor::zeros(dim, DType::BF16, &device).unwrap();
    let rotary = crate::components::PartialRotaryEmbedding::new(
        &crate::components::PartialRotaryConfig {
            head_dim: dim,
            rotary_dim: 32,
            max_position_embeddings: capacity,
            theta: 10_000_000.,
        },
        DType::F32,
        &device,
    )
    .unwrap();
    let (cos, sin, _) = rotary.prepared_parameters();
    let (cos, sin) = (values(cos), values(sin));
    for transposed in [false, true] {
        let key = Tensor::zeros((1, capacity, dim), DType::BF16, &device).unwrap();
        let value = key.zeros_like().unwrap();
        let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
        let mut program = DecodeProgram::new(&device).unwrap();
        let output = program
            .attention_prepare(
                &q,
                &k,
                &v,
                (&norm, 1e-6),
                (&norm, 1e-6),
                rotary.prepared_parameters(),
                (&key, &value),
                &metadata,
                heads,
                1,
                dim,
                capacity,
                transposed,
            )
            .unwrap();
        for position in [0, 33, 64, 0] {
            write_metadata(&metadata, [0, position as u32]).unwrap();
            program.run().unwrap();
            let actual = values(&output);
            let keys = values(&key);
            for feature in 0..dim {
                let expected = if feature < 32 {
                    let index = position * 16 + feature % 16;
                    bf16::from_f32(if feature < 16 {
                        cos[index] - sin[index]
                    } else {
                        cos[index] + sin[index]
                    })
                    .to_f32()
                } else {
                    1.
                };
                for head in 0..heads {
                    assert_eq!(actual[head * dim + feature], expected);
                }
                let index = if transposed {
                    feature * capacity + position
                } else {
                    position * dim + feature
                };
                assert_eq!(keys[index], expected);
            }
        }
    }
}

/// Checks the reference tensor-core probability storage boundary with an
/// independent FP64 scalar definition, including a partial tile and stale tail.
#[test]
fn moe_attention_rounds_unnormalized_products_before_pv() {
    // SAFETY: This test owns all work on one private CUDA stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let (heads, dim, length, capacity, tile) = (4, 64, 65, 96, 32);
    let pattern = |n, phase| {
        (0..n)
            .map(|i| {
                bf16::from_f32(((i * 17 + phase) % 97) as f32 / 64. - 0.75).to_f32()
            })
            .collect::<Vec<_>>()
    };
    let q = pattern(heads * dim, 3);
    let k = pattern(capacity * dim, 11);
    let mut v = pattern(capacity * dim, 19);
    v[length * dim..].fill(f32::NAN);
    let tensor = |data: Vec<f32>, shape| {
        Tensor::from_vec(data, shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    let query = tensor(q.clone(), vec![heads, dim]);
    let key = tensor(k.clone(), vec![1, capacity, dim]);
    let value = tensor(v.clone(), vec![1, capacity, dim]);
    let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
    write_metadata(&metadata, [0, length as u32 - 1]).unwrap();
    let mut program = DecodeProgram::new(&device).unwrap();
    let output = program
        .attention(
            &query,
            &key,
            &value,
            &metadata,
            heads,
            1,
            dim,
            capacity,
            false,
            crate::AttentionBlockSize::Tokens32,
            true,
        )
        .unwrap();
    program.run().unwrap();
    let actual = values(&output);
    for head in 0..heads {
        let scores = (0..length)
            .map(|t| {
                (0..dim)
                    .map(|d| f64::from(q[head * dim + d]) * f64::from(k[t * dim + d]))
                    .sum::<f64>()
                    / (dim as f64).sqrt()
            })
            .collect::<Vec<_>>();
        let global_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut numerator = vec![0f64; dim];
        let mut denominator = 0.;
        for start in (0..length).step_by(tile) {
            let end = (start + tile).min(length);
            let maximum = scores[start..end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let rescale = (maximum - global_max).exp();
            for token in start..end {
                let p = (scores[token] - maximum).exp();
                denominator += p * rescale;
                let stored = f64::from(bf16::from_f64(p).to_f32());
                for d in 0..dim {
                    numerator[d] += stored * f64::from(v[token * dim + d]) * rescale;
                }
            }
        }
        for d in 0..dim {
            let expected = bf16::from_f64(numerator[d] / denominator).to_f32();
            assert!((actual[head * dim + d] - expected).abs() <= 0.001);
        }
    }
    let first = values(&output);
    program.run().unwrap();
    assert_eq!(first, values(&output));
}
