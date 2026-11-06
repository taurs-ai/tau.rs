use super::*;

/// Replay must read updated retained inputs, including the multi-row path.
#[test]
fn vectorized_norm_graph_replay_matches_reference_after_input_changes() {
    use crate::components::cuda::graph::{CapturedGraph, new_capture_device};
    use candle::Var;

    // SAFETY: This fresh device is used only on this test thread; every tensor
    // is allocated afterward and all operations use its one serialized stream.
    let device = unsafe { new_capture_device(0) }.unwrap();
    for policy in [
        RmsNormVectorBytes::Bytes4,
        RmsNormVectorBytes::Bytes8,
        RmsNormVectorBytes::Bytes16,
    ] {
        for (rows, columns) in [(1, 2560), (1, 5120), (5, 2560), (5, 5120)] {
            let input = Var::zeros((1, rows, columns), DType::BF16, &device).unwrap();
            let output = Var::zeros((1, rows, columns), DType::BF16, &device).unwrap();
            let weight = Tensor::new(
                (0..columns)
                    .map(|i| (i % 19) as f32 / 11.0 - 1.25)
                    .collect::<Vec<_>>(),
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            let operation = {
                let input = input.as_detached_tensor();
                let output = output.clone();
                let weight = weight.clone();
                move || {
                    output.set(&centered_rms_norm_cuda(&input, &weight, 1e-6, policy)?)
                }
            };
            {
                let _cache = device
                    .as_cuda_device()
                    .unwrap()
                    .enable_cuda_graph_htod_cache();
                operation().unwrap();
            }
            device.synchronize().unwrap();
            // SAFETY: The closure owns all captured tensors;
            // writes/replays/readback
            // are serialized on this untracked device's stream, without
            // reallocation.
            let mut graph =
                unsafe { CapturedGraph::capture(&device, operation) }.unwrap();
            for phase in [0, 7, 19, 0] {
                let values = (0..rows * columns)
                    .map(|i| ((i * 17 + phase) % 127) as f32 / 37.0 - 1.7)
                    .collect::<Vec<_>>();
                input
                    .set(
                        &Tensor::from_vec(values, (1, rows, columns), &device)
                            .unwrap()
                            .to_dtype(DType::BF16)
                            .unwrap(),
                    )
                    .unwrap();
                graph.replay().unwrap();
                graph.synchronize().unwrap();
                let eager =
                    centered_rms_norm_cuda(&input, &weight, 1e-6, policy).unwrap();
                let reference =
                    CenteredRmsNorm::from_weight_for_test(weight.clone(), 1e-6, false)
                        .forward(&input)
                        .unwrap();
                let values = |tensor: &Tensor| {
                    tensor
                        .reshape((rows, columns))
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap()
                };
                assert_eq!(values(&output), values(&eager));
                assert_close(&values(&output), &values(&reference), 2e-2);
            }
        }
    }
}
