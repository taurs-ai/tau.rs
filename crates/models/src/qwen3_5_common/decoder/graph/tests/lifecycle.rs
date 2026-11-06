use super::*;

fn check_layer_lifecycle(device: &Device, mixed: bool) {
    let mut captured = synthetic_layer(device, mixed);
    let mut eager = synthetic_layer(device, mixed);
    let state_addresses = gdn(&mut captured).optimized_state_addresses().unwrap();
    assert_ne!(
        state_addresses,
        gdn(&mut eager).optimized_state_addresses().unwrap()
    );
    let cuda_graph = GraphPhase::Decode;
    let mut buffers = None;
    let mut replays = 0;
    let mut first_outputs = Vec::new();
    for (request, length) in [1, 3, 1].into_iter().enumerate() {
        captured.clear_cache();
        eager.clear_cache();
        let before = captured.cuda_graph_counts();
        let prefill = patterned((1, length, 32), 7, DType::BF16, device);
        assert_exact(
            &captured.forward_eager(&prefill, None, 0).unwrap(),
            &eager.forward_eager(&prefill, None, 0).unwrap(),
        );
        assert_eq!(
            captured.cuda_graph_counts(),
            before,
            "prefill must not capture/replay"
        );
        for step in 0..128 {
            let input = patterned((1, 1, 32), 19 + step, DType::BF16, device);
            let actual = captured
                .forward_graph_with_eager_fallback(
                    &input,
                    None,
                    length + step,
                    cuda_graph,
                )
                .unwrap();
            let expected = eager.forward_eager(&input, None, length + step).unwrap();
            assert_exact(&actual, &expected);
            let (ac, ar) = gdn(&mut captured).snapshot_optimized_state().unwrap();
            let (ec, er) = gdn(&mut eager).snapshot_optimized_state().unwrap();
            assert_exact(&ac, &ec);
            assert_exact(&ar, &er);
            assert_eq!(
                gdn(&mut captured).optimized_state_addresses().unwrap(),
                state_addresses
            );
            replays += 1;
            assert_eq!(
                captured.cuda_graph_counts(),
                CudaGraphStats {
                    gdn_layer_graph_captures: 1,
                    gdn_layer_graph_replays: replays,
                    ..Default::default()
                }
            );
            let graph = captured.decode_graph.as_ref().unwrap();
            let addresses = (
                graph_tensor_address(&graph.input),
                graph_tensor_address(&graph.output),
            );
            assert_eq!(*buffers.get_or_insert(addresses), addresses);
            if replays == 1 {
                use candle::cuda_backend::cudarc::driver::sys::CUgraphNodeType::*;
                let nodes = graph_node_types(&graph.graph);
                // Fusion removes one kernel from the previous minimum of
                // twelve; the final MLP residual still executes separately.
                assert!(
                    nodes.get(&CU_GRAPH_NODE_TYPE_KERNEL).copied().unwrap_or(0) >= 11
                );
                assert_eq!(
                    nodes.get(&CU_GRAPH_NODE_TYPE_MEM_ALLOC),
                    nodes.get(&CU_GRAPH_NODE_TYPE_MEM_FREE)
                );
                assert_eq!(
                    nodes.get(&CU_GRAPH_NODE_TYPE_MEMCPY).copied().unwrap_or(0),
                    1,
                    "only final output copy belongs inside capture"
                );
                assert_eq!(
                    nodes.get(&CU_GRAPH_NODE_TYPE_MEMSET).copied().unwrap_or(0),
                    0
                );
            }
            if request == 0 {
                first_outputs.push(actual.copy().unwrap());
            }
            if request == 2 {
                assert_exact(&actual, &first_outputs[step]);
            }
        }
        // Multi-token eager fallback transfers the same state out of the native
        // runner.
        let chunk = patterned((1, 2, 32), 37, DType::BF16, device);
        assert_exact(
            &captured
                .forward_graph_with_eager_fallback(&chunk, None, 129, cuda_graph)
                .unwrap(),
            &eager.forward_eager(&chunk, None, 129).unwrap(),
        );
        assert_eq!(
            captured.cuda_graph_counts().gdn_layer_graph_replays,
            replays
        );
        let input = patterned((1, 1, 32), 41, DType::BF16, device);
        assert_exact(
            &captured
                .forward_graph_with_eager_fallback(&input, None, 131, cuda_graph)
                .unwrap(),
            &eager.forward_eager(&input, None, 131).unwrap(),
        );
        replays += 1;
    }
    // Explicit zero reseed and reuse of the same executable.
    gdn(&mut captured).seed_optimized_state(None, None).unwrap();
    eager.clear_cache();
    let input = patterned((1, 1, 32), 47, DType::BF16, device);
    assert_exact(
        &captured
            .forward_graph_with_eager_fallback(&input, None, 0, cuda_graph)
            .unwrap(),
        &eager.forward_eager(&input, None, 0).unwrap(),
    );
    captured
        .forward_graph_with_eager_fallback(&input, None, 1, cuda_graph)
        .unwrap();
    drop(captured); // outstanding replay must finish before resources are released
    device.synchronize().unwrap();
}

#[test]
fn gdn_layer_graph_preserves_outputs_states_and_lifecycle() {
    // SAFETY: Each model and all tensors stay on this thread and active stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    for mixed in [false, true] {
        check_layer_lifecycle(&device, mixed);
    }
}
