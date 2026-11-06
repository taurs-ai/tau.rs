use super::*;

#[test]
fn attention_dispatch_replays_only_mlp_and_keeps_eager_calls_eager() {
    // SAFETY: All layers, buffers and operations stay on this thread/stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let mut dispatched = synthetic_attention_layer(&device);
    let mut eager = synthetic_attention_layer(&device);
    let mut prefill_captured = false;
    let mut decode_captured = false;
    for _ in 0..2 {
        dispatched.clear_cache();
        eager.clear_cache();
        let mut offset = 0;
        for (length, graph, phase) in [
            (2, true, GraphPhase::Prefill),
            (1, false, GraphPhase::Prefill), // partial tail after graph capture
            (1, true, GraphPhase::Decode),
            (2, true, GraphPhase::Prefill),
            (1, false, GraphPhase::Decode), // disabled mode with retained graphs
            (1, true, GraphPhase::Decode),
        ] {
            // Multi-token input is intentionally noncontiguous: MLP eligibility
            // must be checked after normalization, not on this layer input.
            let xs = patterned((1, 32, length), offset, DType::BF16, &device)
                .transpose(1, 2)
                .unwrap();
            let mask = Tensor::from_vec(
                (0..length)
                    .flat_map(|q| {
                        (0..offset + length).map(move |k| {
                            if k <= offset + q {
                                0f32
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
                (1, 1, length, offset + length),
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
            let before = dispatched.cuda_graph_counts();
            let actual = if graph {
                dispatched.forward_graph_with_eager_fallback(
                    &xs,
                    Some(&mask),
                    offset,
                    phase,
                )
            } else {
                dispatched.forward_eager(&xs, Some(&mask), offset)
            }
            .unwrap();
            let expected =
                original_layer_forward(&mut eager, &xs, Some(&mask), offset).unwrap();
            assert_exact(&actual, &expected);
            let captured = match phase {
                GraphPhase::Prefill => &mut prefill_captured,
                GraphPhase::Decode => &mut decode_captured,
            };
            assert_eq!(
                dispatched.cuda_graph_counts().since(before),
                CudaGraphStats {
                    mlp_graph_captures: usize::from(graph && !*captured),
                    mlp_graph_replays: usize::from(graph),
                    ..Default::default()
                }
            );
            *captured |= graph;
            offset += length;
        }
    }
}

#[test]
fn strict_gdn_graph_errors_do_not_fall_back_or_advance_state() {
    // SAFETY: All layers, buffers and operations stay on this thread/stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let mut captured = synthetic_layer(&device, true);
    let mut eager = synthetic_layer(&device, true);
    let input = |length| patterned((1, length, 32), length, DType::BF16, &device);
    // Multi-token decode is an explicit dispatcher fallback, but calling the
    // strict graph method directly must fail before mutating GDN state.
    assert!(
        captured
            .forward_gdn_graph(&input(2), GraphPhase::Decode)
            .unwrap_err()
            .to_string()
            .contains("does not support")
    );
    assert_eq!(captured.cuda_graph_counts(), CudaGraphStats::default());
    for (length, offset) in [(2, 0), (2, 2)] {
        assert_exact(
            &captured
                .forward_graph_with_eager_fallback(
                    &input(length),
                    None,
                    offset,
                    GraphPhase::Prefill,
                )
                .unwrap(),
            &original_layer_forward(&mut eager, &input(length), None, offset).unwrap(),
        );
    }
    let before = captured.cuda_graph_counts();
    assert!(
        captured
            .forward_graph_with_eager_fallback(&input(3), None, 4, GraphPhase::Prefill)
            .unwrap_err()
            .to_string()
            .contains("shape changed")
    );
    assert_eq!(captured.cuda_graph_counts(), before);
    assert_exact(
        &captured.forward_eager(&input(3), None, 4).unwrap(),
        &original_layer_forward(&mut eager, &input(3), None, 4).unwrap(),
    );
    assert_eq!(captured.cuda_graph_counts(), before);
    assert_exact(
        &captured
            .forward_graph_with_eager_fallback(&input(2), None, 7, GraphPhase::Prefill)
            .unwrap(),
        &original_layer_forward(&mut eager, &input(2), None, 7).unwrap(),
    );
    let (ac, ar) = gdn(&mut captured).snapshot_current_state().unwrap();
    let (ec, er) = gdn(&mut eager).snapshot_current_state().unwrap();
    assert_exact(&ac, &ec);
    assert_exact(&ar, &er);
}

#[test]
fn gdn_layer_graph_rejects_tracked_devices_before_warmup() {
    let device = Device::new_cuda(0).unwrap();
    let mut layer = synthetic_layer(&device, true);
    let input = patterned((1, 1, 32), 5, DType::BF16, &device);
    let cuda_graph = GraphPhase::Decode;
    let error = layer
        .forward_graph_with_eager_fallback(&input, None, 0, cuda_graph)
        .unwrap_err();
    assert!(error.to_string().contains("capture device"));
    assert_eq!(layer.cuda_graph_counts(), CudaGraphStats::default());
}
