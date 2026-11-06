//! Independent FP64 norm oracle shared by both operator test suites.

/// Uses an independent FP64 reduction so wide-row checks do not inherit the
/// CUDA kernel's FP32 summation order. Centered scaling still rounds in FP32.
pub(super) fn expected(
    input: &[Vec<f32>],
    weight: &[f32],
    epsilon: f32,
) -> Vec<Vec<f32>> {
    input
        .iter()
        .map(|row| {
            let mean_square = row
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                / row.len() as f64;
            let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip();
            row.iter()
                .zip(weight)
                .map(|(&value, &weight)| {
                    (f64::from(value) * inverse_rms * f64::from(1.0 + weight)) as f32
                })
                .collect()
        })
        .collect()
}
