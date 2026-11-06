//! Convolution graph policy and non-mutating warm-up contracts.

use super::*;

#[test]
fn prefill_convolution_prepared_warmup_keeps_inputs_and_policy() {
    // Prepared computation must leave input state intact through repeated warm-up.
    let device = Device::new_cuda(0).unwrap();
    let cfg = cuda_test_config();
    let mut layer = GatedDeltaNet::new(
        &cfg,
        true,
        VarBuilder::from_tensors(test_tensors(&cfg, &device), DType::BF16, &device),
    )
    .unwrap();
    let input = patterned((1, 8, cfg.hidden_size), 3, &device);
    let state = layer.new_prefill_state().unwrap();
    layer.seed_prefill_state(&state).unwrap();
    let conv_input = state.conv_input.copy().unwrap();
    let recurrent_input = state.recurrent_input.copy().unwrap();
    let mut compute = layer.prepared_prefill_operation(&state);
    let first = compute(&input).unwrap();
    let first_conv = state.conv_output.copy().unwrap();
    let first_recurrent = state.recurrent_output.copy().unwrap();
    assert_tensor_exact(&compute(&input).unwrap(), &first);
    assert_tensor_exact(&state.conv_input, &conv_input);
    assert_tensor_exact(&state.recurrent_input, &recurrent_input);
    assert_tensor_exact(&state.conv_output, &first_conv);
    assert_tensor_exact(&state.recurrent_output, &first_recurrent);
    assert_tensor_exact(&layer.forward(&input).unwrap(), &first);
    assert_tensor_exact(layer.conv_state.as_ref().unwrap(), &first_conv);
    assert_tensor_exact(layer.recurrent_state.as_ref().unwrap(), &first_recurrent);
}

#[test]
fn prefill_convolution_graph_records_native_policy_and_replays() {
    use crate::components::cuda::graph::{CapturedGraph, tests::node_types};
    use candle::cuda_backend::cudarc::driver::sys::CUgraphNodeType as Node;
    // SAFETY: All capture, state updates and readbacks are serialized on one stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let cfg = cuda_test_config();
    let mut kernel_counts = Vec::new();
    let mut outputs = Vec::new();
    for optimized in [false, true] {
        let mut layer = GatedDeltaNet::new(
            &cfg,
            true,
            VarBuilder::from_tensors(test_tensors(&cfg, &device), DType::BF16, &device),
        )
        .unwrap();
        layer.optimize_prefill = optimized;
        let state = layer.new_prefill_state().unwrap();
        let input =
            Var::from_tensor(&patterned((1, 8, cfg.hidden_size), 1, &device)).unwrap();
        let output = Var::zeros(input.shape(), input.dtype(), &device).unwrap();
        let mut compute = layer.prepared_prefill_operation(&state);
        let operation = {
            let input = input.as_detached_tensor();
            let output = output.clone();
            move || output.set(&compute(&input)?)
        };
        let mut operation = operation;
        {
            let _cache = device
                .as_cuda_device()
                .unwrap()
                .enable_cuda_graph_htod_cache();
            operation().unwrap();
            device.synchronize().unwrap();
        }
        // SAFETY: Closure owns all tensors; its temporary buffers drop during
        // capture. State input/output Vars were allocated before capture.
        let mut graph = unsafe { CapturedGraph::capture(&device, operation) }.unwrap();
        let nodes = node_types(&graph);
        kernel_counts.push(nodes[&Node::CU_GRAPH_NODE_TYPE_KERNEL]);
        assert_eq!(
            nodes.get(&Node::CU_GRAPH_NODE_TYPE_MEM_ALLOC),
            nodes.get(&Node::CU_GRAPH_NODE_TYPE_MEM_FREE)
        );
        let addresses = (
            crate::components::cuda::graph::tests::address(&state.conv_input),
            crate::components::cuda::graph::tests::address(&state.conv_output),
        );
        let mut results = Vec::new();
        for seed in [3, 9, 3] {
            layer.clear_cache();
            for chunk in 0..3 {
                layer.seed_prefill_state(&state).unwrap();
                let before_conv = state.conv_input.copy().unwrap();
                let before_recurrent = state.recurrent_input.copy().unwrap();
                input
                    .set(&patterned((1, 8, cfg.hidden_size), seed + chunk, &device))
                    .unwrap();
                graph.replay().unwrap();
                assert_tensor_exact(&state.conv_input, &before_conv);
                assert_tensor_exact(&state.recurrent_input, &before_recurrent);
                layer.publish_prefill_state(&state);
                results.push((
                    output.copy().unwrap(),
                    state.conv_output.copy().unwrap(),
                    state.recurrent_output.copy().unwrap(),
                ));
                assert_eq!(
                    addresses,
                    (
                        crate::components::cuda::graph::tests::address(
                            &state.conv_input
                        ),
                        crate::components::cuda::graph::tests::address(
                            &state.conv_output
                        )
                    )
                );
            }
        }
        graph.replay().unwrap();
        drop(graph); // queued replay teardown retains its tensors until completion
        outputs.push(results);
    }
    assert!(
        kernel_counts[1] < kernel_counts[0],
        "prefill policy failed to reach captured convolution: {kernel_counts:?}"
    );
    for ((a, ac, ar), (b, bc, br)) in outputs[0].iter().zip(&outputs[1]) {
        assert_tensor_exact(a, b);
        assert_tensor_exact(ac, bc);
        assert_tensor_exact(ar, br);
    }
}
