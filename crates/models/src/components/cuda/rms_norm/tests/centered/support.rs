use super::*;
use crate::components::cuda::rms_norm::test_support::expected;

/// Compares two matrices element by element within an absolute tolerance.
pub(super) fn assert_close(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(expected) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (&actual, &expected) in actual_row.iter().zip(expected_row) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "actual {actual} differs from expected {expected} by more than \
                 {tolerance}"
            );
        }
    }
}

/// Checks every element against Candle and an independent oracle, using the
/// actual stored inputs/weights (including BF16 quantization). Reduction-order
/// differences may move a BF16 result by one representable value.
pub(super) fn check_reduction(input: Vec<Vec<f32>>, weight: Vec<f32>, epsilon: f64) {
    check_reduction_at_offsets(input, weight, epsilon, 0, 0);
}

/// Slices after dtype conversion so nonzero element offsets really reach
/// native dispatch; converting an existing slice could silently realign it.
pub(super) fn check_reduction_at_offsets(
    input: Vec<Vec<f32>>,
    weight: Vec<f32>,
    epsilon: f64,
    input_offset: usize,
    weight_offset: usize,
) {
    check_reduction_with_policy(
        input,
        weight,
        epsilon,
        input_offset,
        weight_offset,
        RmsNormVectorBytes::Bytes16,
    );
}

pub(super) fn check_reduction_with_policy(
    input: Vec<Vec<f32>>,
    weight: Vec<f32>,
    epsilon: f64,
    input_offset: usize,
    weight_offset: usize,
    preferred_vector_bytes: RmsNormVectorBytes,
) {
    let device = Device::new_cuda(0).expect("RMSNorm tests require CUDA GPU 0");
    let rows = input.len();
    let columns = weight.len();
    let mut input_storage = vec![123.0; input_offset];
    input_storage.extend(input.into_iter().flatten());
    input_storage.push(123.0);
    let mut weight_storage = vec![123.0; weight_offset];
    weight_storage.extend(weight);
    weight_storage.push(123.0);
    let input_storage = Tensor::new(input_storage, &device).unwrap();
    let weight_storage = Tensor::new(weight_storage, &device).unwrap();
    for dtype in [DType::F32, DType::BF16] {
        let input_storage = input_storage.to_dtype(dtype).unwrap();
        let weight_storage = weight_storage.to_dtype(dtype).unwrap();
        let input = input_storage
            .narrow(0, input_offset, rows * columns)
            .unwrap()
            .reshape((rows, columns))
            .unwrap();
        let weight = weight_storage.narrow(0, weight_offset, columns).unwrap();
        assert_eq!(input.layout().start_offset(), input_offset);
        assert_eq!(weight.layout().start_offset(), weight_offset);
        let original_input = input_storage
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let original_weight = weight_storage
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(supports_centered_rms_norm(&input, &weight, epsilon));
        let custom =
            centered_rms_norm_cuda(&input, &weight, epsilon, preferred_vector_bytes)
                .unwrap();
        assert_eq!(custom.dims(), &[rows, columns]);
        assert_eq!(custom.dtype(), dtype);
        let custom = custom
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let reference =
            CenteredRmsNorm::from_weight_for_test(weight.clone(), epsilon, false)
                .forward(&input)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
        let oracle = expected(
            &input
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            &weight
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            epsilon as f32,
        );
        for (name, expected) in [("Candle", reference), ("FP64 oracle", oracle)] {
            for row in 0..rows {
                for column in 0..columns {
                    let actual = custom[row][column];
                    let expected = expected[row][column];
                    let close = match dtype {
                        DType::F32 => {
                            (actual - expected).abs() <= 1e-6 + 2e-6 * expected.abs()
                        }
                        DType::BF16 => {
                            let rounded = bf16::from_f32(expected);
                            actual == rounded.to_f32()
                                || bf16::from_f32(actual)
                                    .to_bits()
                                    .abs_diff(rounded.to_bits())
                                    <= 1
                        }
                        _ => unreachable!(),
                    };
                    assert!(
                        actual.is_finite() && expected.is_finite() && close,
                        "{dtype:?} [{rows}, {columns}], epsilon {epsilon}, \
                         [{row}, {column}]: {actual} differs from {name} {expected}"
                    );
                }
            }
        }
        assert_eq!(
            input_storage
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            original_input
        );
        assert_eq!(
            weight_storage
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            original_weight
        );
    }
}
