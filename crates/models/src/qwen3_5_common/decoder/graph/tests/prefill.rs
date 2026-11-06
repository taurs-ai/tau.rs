use super::*;

/// Full-chunk replay must carry both states across an eager tail, decode,
/// another prefill chunk, and a new request without replacing graph storage.
#[test]
fn prefill_graph_preserves_chunk_state_tail_and_decode() {
    check_prefill_graph(8, &[1, 2, 4, 8]);
}

#[test]
fn native_prefill_graph_preserves_production_state_and_boundaries() {
    check_prefill_graph(128, &[64, 256]);
}

fn check_prefill_graph(head_dim: usize, chunk_sizes: &[usize]) {
    // SAFETY: All resources and operations stay on this thread/stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    for &chunk_size in chunk_sizes {
        for mixed in [false, true] {
            let mut captured = synthetic_gdn_layer(&device, mixed, head_dim);
            let mut eager = synthetic_gdn_layer(&device, mixed, head_dim);
            let mut prefill_addresses = None;
            let mut prefill_captured = false;
            let mut decode_captured = false;
            for length in [
                chunk_size.saturating_sub(1).max(1),
                chunk_size,
                2 * chunk_size + 1,
                2 * chunk_size,
                1,
                chunk_size + 1,
            ] {
                captured.clear_cache();
                eager.clear_cache();
                let prompt = patterned((1, length, 32), 7, DType::BF16, &device);
                for offset in (0..length).step_by(chunk_size) {
                    let count = chunk_size.min(length - offset);
                    let input = prompt
                        .narrow(1, offset, count)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let full_chunk = count == chunk_size;
                    let before = captured.cuda_graph_counts();
                    let actual = if full_chunk {
                        captured.forward_graph_with_eager_fallback(
                            &input,
                            None,
                            offset,
                            GraphPhase::Prefill,
                        )
                    } else {
                        captured.forward_eager(&input, None, offset)
                    }
                    .unwrap();
                    let expected = eager.forward_eager(&input, None, offset).unwrap();
                    assert_exact(&actual, &expected);
                    let (ac, ar) = gdn(&mut captured).snapshot_current_state().unwrap();
                    let (ec, er) = gdn(&mut eager).snapshot_current_state().unwrap();
                    assert_exact(&ac, &ec);
                    assert_exact(&ar, &er);
                    let delta = captured.cuda_graph_counts().since(before);
                    assert_eq!(
                        delta.mlp_graph_replays, 0,
                        "MLP is inside the GDN graph"
                    );
                    assert_eq!(delta.gdn_layer_graph_replays, usize::from(full_chunk));
                    assert_eq!(
                        delta.gdn_layer_graph_captures,
                        usize::from(full_chunk && !prefill_captured)
                    );
                    if full_chunk {
                        prefill_captured = true;
                        let graph = captured.prefill_graph.as_ref().unwrap();
                        let addresses = (
                            graph_tensor_address(&graph.input),
                            graph_tensor_address(&graph.output),
                        );
                        assert_eq!(
                            *prefill_addresses.get_or_insert(addresses),
                            addresses
                        );
                    }
                }
                for step in 0..3 {
                    let input = patterned((1, 1, 32), 19 + step, DType::BF16, &device);
                    let before = captured.cuda_graph_counts();
                    assert_exact(
                        &captured
                            .forward_graph_with_eager_fallback(
                                &input,
                                None,
                                length + step,
                                GraphPhase::Decode,
                            )
                            .unwrap(),
                        &eager.forward_eager(&input, None, length + step).unwrap(),
                    );
                    let (ac, ar) = gdn(&mut captured).snapshot_current_state().unwrap();
                    let (ec, er) = gdn(&mut eager).snapshot_current_state().unwrap();
                    assert_exact(&ac, &ec);
                    assert_exact(&ar, &er);
                    let delta = captured.cuda_graph_counts().since(before);
                    assert_eq!(
                        delta.gdn_layer_graph_captures,
                        usize::from(!decode_captured)
                    );
                    assert_eq!(delta.gdn_layer_graph_replays, 1);
                    decode_captured = true;
                }
                // Native decode -> captured multi-token prefill state handoff.
                let input = patterned((1, chunk_size, 32), 41, DType::BF16, &device);
                assert_exact(
                    &captured
                        .forward_graph_with_eager_fallback(
                            &input,
                            None,
                            length + 3,
                            GraphPhase::Prefill,
                        )
                        .unwrap(),
                    &eager.forward_eager(&input, None, length + 3).unwrap(),
                );
                prefill_captured = true;
            }
            let input = patterned((1, chunk_size, 32), 47, DType::BF16, &device);
            captured
                .forward_graph_with_eager_fallback(&input, None, 0, GraphPhase::Prefill)
                .unwrap();
            drop(captured); // teardown with pending prefill replay
            device.synchronize().unwrap();
        }
    }
}
