use super::*;
use crate::components::cuda::rms_norm::test_support::expected;

pub(super) fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

pub(super) fn exact(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    for (a, b) in values(actual).iter().zip(values(expected)) {
        assert_eq!(a.to_bits(), b.to_bits(), "residual differs: {a} != {b}");
    }
}

pub(super) fn close(actual: &Tensor, expected: &[f32]) {
    let actual_values = values(actual);
    assert_eq!(actual_values.len(), expected.len());
    for (index, (&a, &b)) in actual_values.iter().zip(expected).enumerate() {
        let matches = match actual.dtype() {
            DType::BF16 => {
                a == bf16::from_f32(b).to_f32()
                    || bf16::from_f32(a)
                        .to_bits()
                        .abs_diff(bf16::from_f32(b).to_bits())
                        <= 1
            }
            DType::F32 => (a - b).abs() <= 1e-6 + 2e-6 * b.abs(),
            _ => unreachable!(),
        };
        assert!(
            a.is_finite() && b.is_finite() && matches,
            "element {index}: {a} != {b}"
        );
    }
}

pub(super) fn patterned(
    count: usize,
    offset: usize,
    phase: usize,
    dtype: DType,
    device: &Device,
) -> Tensor {
    let data = (0..count + offset + 1)
        .map(|i| ((i * 17 + phase) % 127) as f32 / 37.0 - 1.7)
        .collect::<Vec<_>>();
    Tensor::new(data, device).unwrap().to_dtype(dtype).unwrap()
}

/// Compare both outputs with the unchanged optimized pair, Candle and an FP64
/// norm oracle. Inputs/weights, including padding outside sliced views, are immutable.
pub(super) fn check(
    rows: usize,
    columns: usize,
    offsets: [usize; 3],
    dtype: DType,
    policy: RmsNormVectorBytes,
) {
    let device = Device::new_cuda(0).unwrap();
    let count = rows * columns;
    let input_storage = patterned(count, offsets[0], 3, dtype, &device);
    let branch_storage = patterned(count, offsets[1], 19, dtype, &device);
    let weight_storage = patterned(columns, offsets[2], 47, dtype, &device);
    let input = input_storage
        .narrow(0, offsets[0], count)
        .unwrap()
        .reshape((1, rows, columns))
        .unwrap();
    let branch = branch_storage
        .narrow(0, offsets[1], count)
        .unwrap()
        .reshape(input.shape())
        .unwrap();
    let weight = weight_storage.narrow(0, offsets[2], columns).unwrap();
    let snapshots = [
        values(&input_storage),
        values(&branch_storage),
        values(&weight_storage),
    ];
    assert!(supports_fused_residual_rms_norm(
        &input, &branch, &weight, 1e-6
    ));
    let norm = CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, true);
    let (baseline_sum, baseline_norm) = norm
        .add_residual_and_normalize_unfused(&input, &branch)
        .unwrap();
    let (sum, normalized) =
        fused_residual_rms_norm_cuda(&input, &branch, &weight, 1e-6, policy).unwrap();
    exact(&sum, &baseline_sum);
    close(&normalized, &values(&baseline_norm));
    let reference = CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, false)
        .forward(&baseline_sum)
        .unwrap();
    close(&normalized, &values(&reference));
    let rows_f32 = values(&baseline_sum)
        .chunks(columns)
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let oracle = expected(&rows_f32, &values(&weight), 1e-6);
    close(
        &normalized,
        &oracle.into_iter().flatten().collect::<Vec<_>>(),
    );
    assert_eq!(sum.shape(), input.shape());
    assert_eq!(normalized.shape(), input.shape());
    assert!(sum.is_contiguous() && normalized.is_contiguous());
    assert_eq!(
        normalized.layout().start_offset() - sum.layout().start_offset(),
        count
    );
    assert_eq!(
        snapshots,
        [
            values(&input_storage),
            values(&branch_storage),
            values(&weight_storage)
        ]
    );
    // Either view must keep the shared allocation alive independently.
    drop(sum);
    close(&normalized, &values(&baseline_norm));
    let (sum, second) = norm.add_residual_and_normalize(&input, &branch).unwrap();
    drop(second);
    exact(&sum, &baseline_sum);
}
