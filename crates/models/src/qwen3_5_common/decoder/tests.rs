use super::test_support::*;
use super::*;
use crate::CudaGraphStats;
use candle::{DType, Device};

/// Production-width coverage complements the small stateful fixtures. The
/// independent expression retains separate add/norm calls as the baseline.
#[test]
fn fused_post_norm_preserves_production_width_decoder_math() {
    let device = Device::new_cuda(0).unwrap();
    for dtype in [DType::F32, DType::BF16] {
        for columns in [2560, 5120] {
            let input_norm = CenteredRmsNorm::from_weight_for_test(
                patterned(columns, 3, dtype, &device),
                1e-6,
                true,
            );
            let post_norm = CenteredRmsNorm::from_weight_for_test(
                patterned(columns, 7, dtype, &device),
                1e-6,
                true,
            );
            for rows in [1, 3] {
                let input = patterned((1, rows, columns), 11, dtype, &device);
                let actual = forward_block(
                    &input,
                    &input_norm,
                    &post_norm,
                    |xs| xs * 0.25,
                    |xs| xs * 0.5,
                )
                .unwrap();
                let branch = (input_norm.forward(&input).unwrap() * 0.25).unwrap();
                let residual = (&input + branch).unwrap();
                let normalized = post_norm.forward(&residual).unwrap();
                let expected = (&residual + (normalized * 0.5).unwrap()).unwrap();
                assert_exact(&actual, &expected);
            }
        }
    }
}

#[test]
fn graph_dispatch_preserves_eager_cpu_fallback_and_layer_math() {
    for attention in [false, true] {
        let build = || {
            if attention {
                synthetic_attention_layer(&Device::Cpu)
            } else {
                synthetic_layer(&Device::Cpu, true)
            }
        };
        let mut dispatched = build();
        let mut eager = build();
        let mut original = build();
        for _ in 0..2 {
            dispatched.clear_cache();
            eager.clear_cache();
            original.clear_cache();
            let mut offset = 0;
            for length in [3, 1, 2, 1] {
                let xs = patterned((1, length, 32), offset, DType::F32, &Device::Cpu);
                let expected =
                    original_layer_forward(&mut original, &xs, None, offset).unwrap();
                assert_exact(
                    &eager.forward_eager(&xs, None, offset).unwrap(),
                    &expected,
                );
                assert_exact(
                    &dispatched
                        .forward_graph_with_eager_fallback(
                            &xs,
                            None,
                            offset,
                            GraphPhase::Prefill,
                        )
                        .unwrap(),
                    &expected,
                );
                assert_eq!(dispatched.cuda_graph_counts(), CudaGraphStats::default());
                offset += length;
            }
        }
    }
}
