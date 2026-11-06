//! Compare strided/grouped preparation and recurrence with native FP32 math.

use super::preparation::*;
use crate::components::gated_delta_net::golden;
use candle::{DType, Device, Tensor};

#[test]
fn grouped_prefill_preparation_matches_native_fp32() {
    let device = Device::new_cuda(0).unwrap();
    for (tokens, kh, vh) in [(2, 2, 4), (65, 16, 32), (256, 16, 48)] {
        let channels = (2 * kh + vh) * 128;
        let values = (0..(channels + 3) * (tokens + 2))
            .map(|i| ((i * 17 % 97) as f32 - 48.) * 0.015)
            .collect::<Vec<_>>();
        let mixed = Tensor::from_vec(values, (1, channels + 3, tokens + 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .narrow(1, 1, channels)
            .unwrap()
            .narrow(2, 1, tokens)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        assert!(!mixed.is_contiguous());
        let decay = Tensor::full(-0.1f32, (1, tokens, vh), &device).unwrap();
        let alpha = decay.exp().unwrap();
        let beta = Tensor::full(0.47f32, (1, tokens, vh), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let initial = Tensor::full(0.003f32, (1, vh, 128, 128), &device).unwrap();
        let qk = |offset| {
            mixed
                .narrow(2, offset, kh * 128)
                .unwrap()
                .reshape((1, tokens, kh, 128))
                .unwrap()
                .unsqueeze(3)
                .unwrap()
                .broadcast_as((1, tokens, kh, vh / kh, 128))
                .unwrap()
                .reshape((1, tokens, vh, 128))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };
        let q = qk(0);
        let k = qk(kh * 128);
        let v = mixed
            .narrow(2, kh * 256, vh * 128)
            .unwrap()
            .reshape((1, tokens, vh, 128))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = golden::recurrence(
            &q,
            &k,
            &v,
            &decay,
            &beta.to_dtype(DType::F32).unwrap(),
            &initial,
        )
        .unwrap();
        let actual =
            mixed_prefill_cuda(&mixed, &alpha, &beta, &initial, kh, vh, false).unwrap();
        golden::assert_close(&actual.0, &expected.0, 2e-5, 2e-4, "grouped output");
        golden::assert_close(&actual.1, &expected.1, 2e-5, 2e-4, "grouped state");
        assert_eq!(
            initial.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0],
            0.003
        );
    }
}

#[test]
fn mixed_prefill_rejects_unsupported_shapes_and_cpu() {
    let device = Device::Cpu;
    let input = Tensor::zeros((1, 256, 1024), DType::BF16, &device).unwrap();
    assert!(!supports_mixed_prefill(&input, 2, 4));
    let device = Device::new_cuda(0).unwrap();
    let input = input.to_device(&device).unwrap();
    assert!(supports_mixed_prefill(&input, 2, 4));
    assert!(!supports_mixed_prefill(&input, 0, 4));
    assert!(!supports_mixed_prefill(&input, 3, 4));
    assert!(!supports_mixed_prefill(
        &input.narrow(1, 0, 1).unwrap(),
        2,
        4
    ));
    assert!(!supports_mixed_prefill(
        &input.to_dtype(DType::F32).unwrap(),
        2,
        4
    ));
}
