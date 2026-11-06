use super::*;

/// Every compiled vector-width ceiling must preserve the numerical contract,
/// including an explicit scalar control and multi-row execution.
#[test]
fn vector_width_policies_match_reference() {
    for columns in [256, 2560, 5120] {
        let input: Vec<Vec<f32>> = (0..5)
            .map(|row| {
                (0..columns)
                    .map(|column| ((row * 31 + column * 13) % 127) as f32 / 31.0 - 2.0)
                    .collect()
            })
            .collect();
        let weight: Vec<f32> = (0..columns)
            .map(|column| (column % 11) as f32 / 7.0 - 0.5)
            .collect();
        for policy in [
            RmsNormVectorBytes::Scalar,
            RmsNormVectorBytes::Bytes4,
            RmsNormVectorBytes::Bytes8,
            RmsNormVectorBytes::Bytes16,
        ] {
            check_reduction_with_policy(
                input.clone(),
                weight.clone(),
                1e-6,
                0,
                0,
                policy,
            );
        }
    }
}

/// Exercises unused lanes, warp/block boundaries, strided tails, and the
/// Qwen3.5-4B/27B widths in both single-row decode and multi-row prefill.
#[test]
fn reduction_matches_reference_at_warp_boundaries_and_model_widths() {
    for columns in [
        1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 513, 2559, 2560, 2561, 3072,
        5119, 5120, 5121,
    ] {
        for rows in [1, 5] {
            let input = (0..rows)
                .map(|row| {
                    (0..columns)
                        .map(|column| {
                            ((column * 17 + row * 13) % 127) as f32 / 37.0 - 1.7
                        })
                        .collect()
                })
                .collect();
            let weight = (0..columns)
                .map(|column| (column % 19) as f32 / 11.0 - 1.25)
                .collect();
            check_reduction(input, weight, 1e-6);
        }
    }
}

/// Each row isolates a different warp's final lane or a strided tail. Missing
/// a warp contribution cannot be hidden by otherwise similar row sums.
#[test]
fn reduction_includes_every_warp_and_the_strided_tail() {
    let columns = 513;
    let mut input = vec![vec![0.0; columns]; 9];
    for (warp, row) in input[..8].iter_mut().enumerate() {
        row[warp * 32 + 31] = if warp % 2 == 0 { 1.0 } else { -1.0 };
    }
    input[8][columns - 1] = 2.0;
    check_reduction(input, vec![0.25; columns], 1e-6);
}

/// Zero and tiny rows exercise epsilon; large rows exercise FP32 accumulation
/// with BF16 storage. Independent rows must not share their reduction totals.
#[test]
fn reduction_handles_zero_and_mixed_scale_rows() {
    for columns in [257, 2560, 5120] {
        let input: Vec<Vec<f32>> = [0.0, 1e-8, 1e-4, 1.0, 1e4]
            .into_iter()
            .map(|scale| {
                (0..columns)
                    .map(|column| scale * ((column % 23) as f32 - 11.0))
                    .collect()
            })
            .collect();
        let weight: Vec<f32> = (0..columns)
            .map(|column| [-1.0, -1.5, 0.0, 0.25][column % 4])
            .collect();
        for epsilon in [1e-6, 1e-3] {
            check_reduction(input.clone(), weight.clone(), epsilon);
        }
    }
}
