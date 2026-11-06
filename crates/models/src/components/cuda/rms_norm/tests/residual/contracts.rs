use super::*;

#[test]
fn fused_component_keeps_cpu_reference_and_noncontiguous_fallbacks() {
    for device in [Device::Cpu, Device::new_cuda(0).unwrap()] {
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0], [-1.0, -2.0, -3.0]], &device)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let branch = (&input * 0.25).unwrap();
        let weight = Tensor::new(&[0.25f32, -0.5], &device).unwrap();
        assert!(!supports_fused_residual_rms_norm(
            &input, &branch, &weight, 1e-6
        ));
        for optimized in [false, true] {
            let norm =
                CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, optimized);
            let (sum, normalized) =
                norm.add_residual_and_normalize(&input, &branch).unwrap();
            let (expected_sum, expected_norm) = norm
                .add_residual_and_normalize_unfused(&input, &branch)
                .unwrap();
            exact(&sum, &expected_sum);
            exact(&normalized, &expected_norm);
        }
    }
    let device = Device::new_cuda(0).unwrap();
    let input = Tensor::ones((1, 2560), DType::BF16, &device).unwrap();
    let weight = Tensor::zeros(2560, DType::F32, &device).unwrap();
    assert!(!supports_fused_residual_rms_norm(
        &input, &input, &weight, 1e-6
    ));
    let norm = CenteredRmsNorm::from_weight_for_test(weight, 1e-6, true);
    let (sum, normalized) = norm.add_residual_and_normalize(&input, &input).unwrap();
    let (expected_sum, expected_norm) = norm
        .add_residual_and_normalize_unfused(&input, &input)
        .unwrap();
    exact(&sum, &expected_sum);
    exact(&normalized, &expected_norm);
}

#[test]
fn fused_rejects_invalid_contracts_before_native_launch() {
    let device = Device::new_cuda(0).unwrap();
    let input = Tensor::zeros((1, 256), DType::BF16, &device).unwrap();
    let weight = Tensor::zeros(256, DType::BF16, &device).unwrap();
    let wrong = Tensor::zeros((2, 256), DType::BF16, &device).unwrap();
    assert!(!supports_fused_residual_rms_norm(
        &input, &wrong, &weight, 1e-6
    ));
    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            fused_residual_rms_norm_cuda(
                &input,
                &input,
                &weight,
                epsilon,
                RmsNormVectorBytes::Bytes16
            )
            .is_err()
        );
    }
}

#[test]
fn optimize_kernel_false_preserves_unfused_execution_for_eligible_inputs() {
    let device = Device::new_cuda(0).unwrap();
    let input = Tensor::ones((1, 2560), DType::BF16, &device).unwrap();
    let weight = Tensor::zeros(2560, DType::BF16, &device).unwrap();
    assert!(supports_fused_residual_rms_norm(
        &input, &input, &weight, 1e-6
    ));
    let reference = CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, false);
    let (sum, normalized) = reference
        .add_residual_and_normalize(&input, &input)
        .unwrap();
    let (expected_sum, expected_norm) = reference
        .add_residual_and_normalize_unfused(&input, &input)
        .unwrap();
    exact(&sum, &expected_sum);
    exact(&normalized, &expected_norm);
    // Reference outputs retain independent allocations; the optimized result is
    // the second view of a packed allocation. This also checks dispatch, not only math.
    assert_eq!(normalized.layout().start_offset(), 0);
    let optimized = CenteredRmsNorm::from_weight_for_test(weight, 1e-6, true);
    let (_, normalized) = optimized
        .add_residual_and_normalize(&input, &input)
        .unwrap();
    assert_eq!(normalized.layout().start_offset(), input.elem_count());
}
