use super::*;

#[test]
fn fused_matches_reference_for_decode_prefill_and_irregular_widths() {
    for dtype in [DType::F32, DType::BF16] {
        for columns in [1, 31, 32, 33, 255, 256, 257, 2560, 5120] {
            for rows in [1, 5] {
                check(rows, columns, [0; 3], dtype, RmsNormVectorBytes::Bytes16);
            }
        }
        for columns in [2560, 5120] {
            check(512, columns, [0; 3], dtype, RmsNormVectorBytes::Bytes16);
        }
    }
}

#[test]
fn fused_handles_vector_policies_offsets_and_output_lifetimes() {
    for dtype in [DType::F32, DType::BF16] {
        for columns in [256, 2560, 5120] {
            for policy in [
                RmsNormVectorBytes::Scalar,
                RmsNormVectorBytes::Bytes4,
                RmsNormVectorBytes::Bytes8,
                RmsNormVectorBytes::Bytes16,
            ] {
                check(3, columns, [0; 3], dtype, policy);
            }
            for offsets in [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 2, 2],
                [4, 4, 4],
                [8, 8, 8],
            ] {
                check(3, columns, offsets, dtype, RmsNormVectorBytes::Bytes16);
            }
        }
    }
}

#[test]
fn fused_rounds_bf16_sum_before_normalization_and_preserves_aliases() {
    let device = Device::new_cuda(0).unwrap();
    for epsilon in [1e-6, 1e-3] {
        let input_values = [1.0, 1.0, -1.0, 0.0, 1024.0, 0.0001, -0.0001, 1e4];
        let branch_values = [
            0.00390625,
            0.01171875,
            -0.00390625,
            0.0,
            -1024.0,
            -0.00005,
            0.00005,
            -9990.0,
        ];
        let input = Tensor::new(
            (0..2560).map(|i| input_values[i % 8]).collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let branch = Tensor::new(
            (0..2560).map(|i| branch_values[i % 8]).collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let weight = Tensor::new(
            (0..2560)
                .map(|i| [0.0, -1.0, -1.5, 0.25][i % 4])
                .collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let norm = CenteredRmsNorm::from_weight_for_test(weight, epsilon, true);
        for branch in [&branch, &input] {
            let (sum, normalized) =
                norm.add_residual_and_normalize(&input, branch).unwrap();
            let (expected_sum, expected_norm) = norm
                .add_residual_and_normalize_unfused(&input, branch)
                .unwrap();
            exact(&sum, &expected_sum);
            exact(&normalized, &expected_norm);
        }
    }
}
