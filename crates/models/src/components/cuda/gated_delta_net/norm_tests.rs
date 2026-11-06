//! Measure optimized BF16 readout/norm accuracy against native FP32.

use super::norm::*;
use crate::components::gated_delta_net::golden;
use crate::components::rms_norm::GatedRmsNorm;
use candle::{DType, Device, Tensor};
use candle_nn::Activation;

#[test]
fn prefill_norm_matches_native_fp32_golden() {
    let device = Device::new_cuda(0).unwrap();
    for tokens in [2, 27, 256] {
        let shape = (1, tokens, 4, 128);
        let input = Tensor::from_vec(
            (0..tokens * 4 * 128)
                .map(|i| ((i * 37 % 103) as f32 - 51.) * 0.00317)
                .collect::<Vec<_>>(),
            shape,
            &device,
        )
        .unwrap();
        let gate = (&input * 5.1).unwrap().to_dtype(DType::BF16).unwrap();
        let weight = Tensor::from_vec(
            (0..128)
                .map(|i| 0.9137 + i as f32 * 0.00317)
                .collect::<Vec<_>>(),
            128,
            &device,
        )
        .unwrap();
        let expected = GatedRmsNorm::from_weight(weight.clone(), 1e-6)
            .forward(
                &input,
                &gate.to_dtype(DType::F32).unwrap(),
                &Activation::Silu,
            )
            .unwrap();
        let actual = prefill_norm_cuda(&input, &gate, &weight, 1e-6).unwrap();
        golden::assert_close(&actual, &expected, 0.002, 0.012, "gated norm");
    }
}
