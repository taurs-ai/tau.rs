use super::*;
use crate::components::cuda::graph::new_capture_device;
use crate::components::{CapturedGraph, graph_node_types};
use candle::Var;

#[test]
fn fused_graph_replays_both_outputs_and_removes_one_kernel() {
    use candle::cuda_backend::cudarc::driver::sys;
    // SAFETY: Fresh tensors and all work remain on this thread/device stream.
    let device = unsafe { new_capture_device(0) }.unwrap();
    for (rows, columns) in [(1, 32), (1, 2560), (3, 5120)] {
        let input = Var::zeros((1, rows, columns), DType::BF16, &device).unwrap();
        let branch = Var::zeros(input.shape(), DType::BF16, &device).unwrap();
        let weight = Tensor::ones(columns, DType::BF16, &device).unwrap();
        let norm = CenteredRmsNorm::from_weight_for_test(weight, 1e-6, true);
        let mut kernel_counts = Vec::new();
        for fused in [false, true] {
            let sum_slot = Var::zeros(input.shape(), input.dtype(), &device).unwrap();
            let norm_slot = Var::zeros(input.shape(), input.dtype(), &device).unwrap();
            let operation = {
                let input = input.as_detached_tensor();
                let branch = branch.as_detached_tensor();
                let sum_slot = sum_slot.clone();
                let norm_slot = norm_slot.clone();
                let norm = norm.clone();
                move || {
                    let (sum, normalized) = if fused {
                        norm.add_residual_and_normalize(&input, &branch)?
                    } else {
                        norm.add_residual_and_normalize_unfused(&input, &branch)?
                    };
                    sum_slot.set(&sum)?;
                    norm_slot.set(&normalized)
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
            // SAFETY: Closure owns every captured tensor. Replays and buffer writes
            // use the same serialized stream and retained addresses.
            let mut graph =
                unsafe { CapturedGraph::capture(&device, operation) }.unwrap();
            kernel_counts.push(
                graph_node_types(&graph)
                    .get(&sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL)
                    .copied()
                    .unwrap_or(0),
            );
            for phase in [3, 11, 3] {
                let x = patterned(rows * columns, 0, phase, DType::BF16, &device)
                    .narrow(0, 0, rows * columns)
                    .unwrap()
                    .reshape(input.shape())
                    .unwrap();
                let b = patterned(rows * columns, 0, phase + 41, DType::BF16, &device)
                    .narrow(0, 0, rows * columns)
                    .unwrap()
                    .reshape(input.shape())
                    .unwrap();
                input.set(&x).unwrap();
                branch.set(&b).unwrap();
                graph.replay().unwrap();
                graph.synchronize().unwrap();
                let (expected_sum, expected_norm) =
                    norm.add_residual_and_normalize_unfused(&x, &b).unwrap();
                exact(&sum_slot, &expected_sum);
                close(&norm_slot, &values(&expected_norm));
            }
        }
        assert_eq!(kernel_counts[0], kernel_counts[1] + 1);
    }
}
