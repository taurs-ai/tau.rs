//! Small CUDA regressions for every compiled decode profile, including shapes
//! that defeat aligned vector copies. Real trained operand/FP64 gates remain
//! separate; these tests protect catalog coverage, masking and launch evidence.

use super::*;
use crate::components::{CapturedGraph, DecodeProgram};
use tau_cuda_kernels::{MoeDecodeConfig, MoeDecodeProfile};

#[test]
fn every_decode_profile_preserves_tail_shapes_and_actual_expert_receipts() {
    // SAFETY: This test retains all graph resources on one serialized stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    // The aligned case spans at least two staged-K chunks in both projections
    // for every profile. Odd and sub-vector shapes exercise masked/scalar loads.
    for (hidden, intermediate) in [(256, 256), (33, 29), (8, 7)] {
        let values = |count| {
            (0..count)
                .map(|i| bf16::from_f32(((i * 17 % 97) as f32 - 48.) / 64.))
                .collect::<Vec<_>>()
        };
        let gu = Tensor::from_vec(
            values(256 * 2 * intermediate * hidden),
            (256, 2 * intermediate, hidden),
            &device,
        )
        .unwrap();
        let down = Tensor::from_vec(
            values(256 * hidden * intermediate),
            (256, hidden, intermediate),
            &device,
        )
        .unwrap();
        let input = Tensor::from_vec(values(hidden), (1, 1, hidden), &device).unwrap();
        let logits = Tensor::zeros((1, 256), DType::BF16, &device).unwrap();
        let original = (
            tensor_hash(&gu),
            tensor_hash(&down),
            tensor_hash(&input),
            tensor_hash(&logits),
        );
        let mut baseline = None;
        for profile in MoeDecodeProfile::supported() {
            let mut runner = FusedMoeRunner::new(gu.clone(), down.clone(), 8)
                .unwrap()
                .unwrap();
            runner
                .configure_decode(MoeDecodeConfig {
                    gate_up: profile,
                    down: profile,
                })
                .unwrap();
            let mut program = DecodeProgram::new(&device).unwrap();
            let output = runner
                .prepare_decode_program(&mut program, &input, &logits)
                .unwrap();
            program.run().unwrap();
            device.synchronize().unwrap();
            // SAFETY: Graph drops before the retained program/runner/tensors.
            let mut graph =
                unsafe { CapturedGraph::capture(&device, || program.run()) }.unwrap();
            assert_eq!(graph.allocation_nodes().unwrap(), 0);
            graph.replay().unwrap();
            device.synchronize().unwrap();
            let actual = host_tensors(snapshot(&runner, &output));
            if let Some(expected) = &baseline {
                exact(&actual, expected);
            } else {
                baseline = Some(actual);
            }
            let workspace = runner.workspace.lock().unwrap();
            let words = device
                .as_cuda_device()
                .unwrap()
                .cuda_stream()
                .clone_dtoh(&workspace.receipts)
                .unwrap();
            assert!(
                words.chunks_exact(5).all(|w| w == profile.words()),
                "{profile}"
            );
            drop(workspace);
            drop(graph);
        }
        assert_eq!(
            original,
            (
                tensor_hash(&gu),
                tensor_hash(&down),
                tensor_hash(&input),
                tensor_hash(&logits)
            )
        );
    }
}
