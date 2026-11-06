use super::*;
use crate::components::cuda::rms_norm::test_support::expected;

/// Covers all vectorized widths at attention-head and full prefill row counts.
#[test]
fn vectorized_norm_matches_reference_for_heads_and_prefill() {
    for columns in [256, 2560, 5120] {
        for rows in [4, 16, 512] {
            let input = (0..rows)
                .map(|row| {
                    (0..columns)
                        .map(|column| {
                            ((column * 17 + row * 31) % 251) as f32 / 71.0 - 1.5
                        })
                        .collect()
                })
                .collect();
            let weight = (0..columns)
                .map(|column| (column % 23) as f32 / 13.0 - 1.0)
                .collect();
            check_reduction(input, weight, 1e-6);
        }
    }
}

/// Input and weight alignment are independent. Offset eight also checks the
/// vectorized path on aligned slices rather than only fresh allocations.
#[test]
fn vectorized_norm_handles_aligned_and_unaligned_storage_offsets() {
    for columns in [256, 2560, 5120] {
        let input: Vec<Vec<f32>> = (0..3)
            .map(|row| {
                (0..columns)
                    .map(|column| ((row * 31 + column * 13) % 127) as f32 / 31.0 - 2.0)
                    .collect()
            })
            .collect();
        let weight: Vec<f32> = (0..columns)
            .map(|column| (column % 11) as f32 / 7.0 - 0.5)
            .collect();
        for (input_offset, weight_offset) in
            [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (4, 4), (8, 8)]
        {
            check_reduction_at_offsets(
                input.clone(),
                weight.clone(),
                1e-6,
                input_offset,
                weight_offset,
            );
        }
    }
}

/// Verifies FP32/BF16 parity between custom, reference, and host implementations.
#[test]
fn custom_and_reference_centered_rms_norm_match() {
    let device = Device::new_cuda(0)
        .expect("centered RMSNorm comparison test requires CUDA GPU 0");
    let input = vec![vec![1.0f32, -2.0, 3.0, -4.0], vec![0.5, 1.5, -2.5, 3.5]];
    let weight = vec![0.0f32, 0.25, -0.5, 1.0];
    let expected = expected(&input, &weight, 1e-6);

    for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::BF16, 2e-2)] {
        let input = Tensor::new(input.clone(), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let weight = Tensor::new(weight.as_slice(), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        assert!(supports_centered_rms_norm(&input, &weight, 1e-6));

        let custom = CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, true)
            .forward(&input)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let reference = CenteredRmsNorm::from_weight_for_test(weight, 1e-6, false)
            .forward(&input)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_close(&custom, &reference, tolerance);
        assert_close(&custom, &expected, tolerance);
    }
}

/// Verifies that unsupported non-contiguous input uses the Candle fallback.
#[test]
fn unsupported_native_layout_falls_back_to_reference() {
    let device = Device::new_cuda(0)
        .expect("centered RMSNorm fallback test requires CUDA GPU 0");
    let input = Tensor::new(&[[1.0f32, -2.0, 3.0], [0.5, 1.5, -2.5]], &device)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let weight = Tensor::new(&[0.25f32, -0.5], &device).unwrap();
    assert!(!input.is_contiguous());
    assert!(!supports_centered_rms_norm(&input, &weight, 1e-6));

    let automatic = CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, true)
        .forward(&input)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let reference = CenteredRmsNorm::from_weight_for_test(weight, 1e-6, false)
        .forward(&input)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    assert_eq!(automatic, reference);
}
