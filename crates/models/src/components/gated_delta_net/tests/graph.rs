//! Stateful capture primitive and production component lifecycle regressions.

use super::*;
use crate::components::cuda::graph::CapturedGraph;
use crate::components::cuda::graph::tests::{
    address, cuda_device, node_types, used_graph_memory,
};
use candle::Var;
use candle::cuda_backend::cudarc::driver::sys;
use std::cell::{Cell, RefCell};
use std::path::PathBuf;

/// Runs eager prefill and explicitly transfers its state into the retained runner.
fn prefill_pair(
    captured: &RefCell<GatedDeltaNet>,
    eager: &mut GatedDeltaNet,
    input: &Tensor,
) {
    let mut layer = captured.borrow_mut();
    layer.clear_cache();
    eager.clear_cache();
    assert_tensor_exact(
        &layer.forward(input).unwrap(),
        &eager.forward(input).unwrap(),
    );
    layer.prepare_optimized_decode().unwrap();
    eager.prepare_optimized_decode().unwrap();
    assert_native_state_exact(&layer, eager);
}

/// Compares a real captured GDN operation with an independently allocated runner.
///
/// RefCell limits each Rust borrow to preparation/capture/reset on this one thread.
/// The closure retains a borrow of the cell, not a long-lived borrow of the layer,
/// so reseeding uses the component's ordinary mutable methods while the graph lives.
/// GPU replay bypasses RefCell and the runner's mutex: stream serialization remains
/// mandatory, and this test harness is not a public replay/ownership API.
fn check_stateful_replay(
    config: &GatedDeltaNetConfig,
    vb: VarBuilder,
    iterations: usize,
) {
    let device = vb.device().clone();
    let make_layer = || GatedDeltaNet::new(config, true, vb.clone()).unwrap();
    // A cloned layer would share native workspace and invalidate the comparison.
    let captured = RefCell::new(make_layer());
    let mut eager = make_layer();
    let state_addresses = native_state_addresses(&captured.borrow());
    let eager_addresses = native_state_addresses(&eager);
    assert_ne!(state_addresses.0, eager_addresses.0);
    assert_ne!(state_addresses.1, eager_addresses.1);
    let shape = (1, 1, config.hidden_size);
    let input = Var::from_tensor(&patterned_bf16(shape, 19, 0.018, &device)).unwrap();
    let output = Var::zeros(shape, DType::BF16, &device).unwrap();
    let buffer_addresses = (address(&input), address(&output));
    let prefill = patterned_bf16((1, 3, config.hidden_size), 17, 0.02, &device);
    prefill_pair(&captured, &mut eager, &prefill);
    let (initial_conv, initial_recurrent) = native_state(&captured.borrow());
    assert_eq!(initial_conv.dtype(), DType::BF16);
    assert_eq!(initial_recurrent.dtype(), DType::F32);

    let calls = Cell::new(0);
    let operation = {
        let layer = &captured;
        let calls = &calls;
        let input = input.as_detached_tensor();
        let output = output.clone();
        move || {
            calls.set(calls.get() + 1);
            let result = layer.borrow().forward_prepared_decode(&input)?;
            // Only persistent output storage escapes. Projection/core/output
            // temporaries must all drop and record their frees during capture.
            output.set(&result)
        }
    };
    {
        // Cache only the operation to be captured. Leaving this enabled during
        // reference prefill/fallback retains their strided-layout parameters in
        // Candle's process-global cache, whose entries have no eviction API.
        let _cache = device
            .as_cuda_device()
            .unwrap()
            .enable_cuda_graph_htod_cache();
        operation().unwrap();
        device.synchronize().unwrap();
    }
    captured
        .borrow_mut()
        .seed_optimized_state(Some(&initial_conv), Some(&initial_recurrent))
        .unwrap();
    let sentinel = patterned_bf16(shape, 7, 0.01, &device);
    output.set(&sentinel).unwrap();
    // SAFETY: All retained tensors predate capture on an untracked device. The cell
    // retains weights and native workspace; the closure owns input/output handles.
    // All accesses, resets and replays are serialized on this device's stream/thread.
    let mut graph = unsafe { CapturedGraph::capture(&device, operation) }.unwrap();
    assert_eq!(calls.get(), 2);
    assert_tensor_exact(&output, &sentinel);
    assert_native_state_exact(&captured.borrow(), &eager);
    let types = node_types(&graph);
    let count = |kind| types.get(&kind).copied().unwrap_or(0);
    // Two input projections, the native core, and the output projection. A
    // production head uses separate tiled recurrence and gated normalization.
    let native_kernels = if config.key_head_dim == 128 && config.value_head_dim == 128 {
        3
    } else {
        2
    };
    assert!(
        count(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL) >= 3 + native_kernels,
        "{types:?}"
    );
    assert!(count(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC) > 0);
    assert_eq!(
        count(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC),
        count(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE),
        "every transient allocation needs a captured free: {types:?}"
    );
    // Seeding/zeroing must not accidentally become a replayed state reset.
    assert_eq!(count(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET), 0);
    // The only copy is the final result into the retained output buffer.
    assert_eq!(count(sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY), 1);
    eprintln!("GDN capture hidden={}: {types:?}", config.hidden_size);

    let mut first_outputs = Vec::new();
    let mut first_final_state = None;
    // Reuse this exact graph through A/B/A, including different prompt lengths.
    for (request, (length, phase)) in [(3, 17), (5, 5), (3, 17)].into_iter().enumerate()
    {
        if request > 0 {
            let prefill =
                patterned_bf16((1, length, config.hidden_size), phase, 0.02, &device);
            prefill_pair(&captured, &mut eager, &prefill);
        }
        for step in 0..iterations {
            input
                .set(&patterned_bf16(shape, 19 + step, 0.018, &device))
                .unwrap();
            graph.replay().unwrap();
            graph.synchronize().unwrap();
            let expected = eager.forward(&input.as_detached_tensor()).unwrap();
            assert_tensor_exact(&output, &expected);
            assert_native_state_exact(&captured.borrow(), &eager);
            assert_eq!(native_state_addresses(&captured.borrow()), state_addresses);
            assert_eq!((address(&input), address(&output)), buffer_addresses);
            assert_eq!(calls.get(), 2, "replay must bypass Rust and initialization");
            match request {
                0 => first_outputs.push(output.as_tensor().copy().unwrap()),
                2 => assert_tensor_exact(&output, &first_outputs[step]),
                _ => {}
            }
        }
        let (conv, recurrent) = native_state(&captured.borrow());
        match request {
            0 => first_final_state = Some((conv, recurrent)),
            2 => {
                let (first_conv, first_recurrent) = first_final_state.as_ref().unwrap();
                assert_tensor_exact(&conv, first_conv);
                assert_tensor_exact(&recurrent, first_recurrent);
            }
            _ => {}
        }

        // Graph -> reference fallback -> prepared graph -> ordinary native eager
        // -> graph exercises both device state and host ownership transitions.
        let fallback = patterned_bf16((1, 2, config.hidden_size), 23, 0.017, &device);
        {
            let mut layer = captured.borrow_mut();
            assert_tensor_exact(
                &layer.forward(&fallback).unwrap(),
                &eager.forward(&fallback).unwrap(),
            );
            assert!(!layer.optimized_state_active);
            assert_tensor_exact(
                layer.conv_state.as_ref().unwrap(),
                eager.conv_state.as_ref().unwrap(),
            );
            assert_tensor_exact(
                layer.recurrent_state.as_ref().unwrap(),
                eager.recurrent_state.as_ref().unwrap(),
            );
            assert!(layer.forward_prepared_decode(&input).is_err());
            layer.prepare_optimized_decode().unwrap();
        }
        graph.replay().unwrap();
        graph.synchronize().unwrap();
        assert_tensor_exact(
            &output,
            &eager.forward(&input.as_detached_tensor()).unwrap(),
        );
        assert_native_state_exact(&captured.borrow(), &eager);
        assert_tensor_exact(
            &captured
                .borrow_mut()
                .forward(&input.as_detached_tensor())
                .unwrap(),
            &eager.forward(&input.as_detached_tensor()).unwrap(),
        );
        graph.replay().unwrap();
        graph.synchronize().unwrap();
        assert_tensor_exact(
            &output,
            &eager.forward(&input.as_detached_tensor()).unwrap(),
        );
        assert_native_state_exact(&captured.borrow(), &eager);
    }

    // Explicit zero reset changes device buffers in place even with a live graph.
    captured
        .borrow_mut()
        .seed_optimized_state(None, None)
        .unwrap();
    eager.clear_cache();
    eager.prepare_optimized_decode().unwrap();
    assert_native_state_exact(&captured.borrow(), &eager);
    for state in [
        native_state(&captured.borrow()).0,
        native_state(&captured.borrow()).1,
    ] {
        assert_tensor_exact(&state, &state.zeros_like().unwrap());
    }
    // Queue several changing inputs without a per-token synchronization/readback.
    let mut expected = None;
    for step in 0..8 {
        input
            .set(&patterned_bf16(shape, 11 + step, 0.018, &device))
            .unwrap();
        graph.replay().unwrap();
        expected = Some(eager.forward(&input.as_detached_tensor()).unwrap());
    }
    graph.synchronize().unwrap();
    assert_tensor_exact(&output, &expected.unwrap());
    assert_native_state_exact(&captured.borrow(), &eager);

    // Rejected reseeds must preserve state and leave this same graph usable.
    let (conv, recurrent) = native_state(&captured.borrow());
    assert!(
        captured
            .borrow_mut()
            .seed_optimized_state(Some(&conv), None)
            .is_err()
    );
    let wrong_dtype = recurrent.to_dtype(DType::BF16).unwrap();
    assert!(
        captured
            .borrow_mut()
            .seed_optimized_state(Some(&conv), Some(&wrong_dtype))
            .is_err()
    );
    assert_native_state_exact(&captured.borrow(), &eager);
    graph.replay().unwrap();
    graph.synchronize().unwrap();
    assert_tensor_exact(
        &output,
        &eager.forward(&input.as_detached_tensor()).unwrap(),
    );
    assert_native_state_exact(&captured.borrow(), &eager);
    assert_eq!(native_state_addresses(&captured.borrow()), state_addresses);
    assert_eq!((address(&input), address(&output)), buffer_addresses);
    assert_eq!(calls.get(), 2);

    // Drop waits for an outstanding replay before releasing captured resources.
    graph.replay().unwrap();
    drop(graph);
    assert_tensor_exact(
        &output,
        &eager.forward(&input.as_detached_tensor()).unwrap(),
    );
    assert_native_state_exact(&captured.borrow(), &eager);
    drop((captured, eager, input, output));
    device.synchronize().unwrap();
    device
        .as_cuda_device()
        .unwrap()
        .cuda_stream()
        .context()
        .check_err()
        .unwrap();
}

fn check_synthetic(device: &Device, mixed_storage: bool, iterations: usize) {
    let config = cuda_test_config();
    let mut tensors = test_tensors(&config, device);
    if mixed_storage {
        add_fp32_parameters(&mut tensors, &config, device);
    }
    check_stateful_replay(
        &config,
        VarBuilder::from_tensors(tensors, DType::BF16, device),
        iterations,
    );
}

#[test]
fn capture_gdn_preserves_state_and_request_lifecycle() {
    check_synthetic(&cuda_device(), false, 128);
}

#[test]
fn capture_gdn_preserves_mixed_storage_state_and_request_lifecycle() {
    check_synthetic(&cuda_device(), true, 128);
}

/// The tiny default fixture does not enter a 128x128 specialization. Exercise
/// every value tile with grouped key heads, changed inputs, A/B/A reset/reseed,
/// reference fallback, exact full-state replay parity and queued graph teardown.
#[test]
fn capture_production_gdn_preserves_state_and_request_lifecycle() {
    let device = cuda_device();
    let mut config = cuda_test_config();
    config.key_head_dim = 128;
    config.value_head_dim = 128;
    config.num_key_heads = 2;
    config.num_value_heads = 4;
    let mut tensors = test_tensors(&config, &device);
    add_fp32_parameters(&mut tensors, &config, &device);
    check_stateful_replay(
        &config,
        VarBuilder::from_tensors(tensors, DType::BF16, &device),
        128,
    );
}

#[test]
#[ignore = "loads a GDN layer from local Qwen3.5-4B and requires NVIDIA CUDA GPU 0"]
fn capture_gdn_replays_qwen3_5_4b_layer() {
    use crate::qwen3_5_common::{HybridLayerType, HybridTextConfig};

    let device = cuda_device();
    let root = std::env::var_os("TAU_QWEN3_5_CAPTURE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap())
                .join("work/models/Qwen3.5-4B")
        });
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap())
            .unwrap();
    let hybrid: HybridTextConfig =
        serde_json::from_value(config["text_config"].clone()).unwrap();
    assert_eq!(hybrid.layer_types[0], HybridLayerType::LinearAttention);
    let shards = crate::safetensors::load_safetensors_shards(&root).unwrap();
    let paths = shards.iter().map(|shard| shard.path()).collect::<Vec<_>>();
    // SAFETY: The local checkpoint is mapped read-only and never modified.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device) }
            .unwrap();
    check_stateful_replay(
        &hybrid.gated_delta_net_config(),
        vb.pp("model.language_model.layers.0.linear_attn"),
        128,
    );
}

#[test]
#[ignore = "measures device-wide graph memory; run alone with --test-threads=1"]
fn capture_gdn_repeated_lifecycle_keeps_graph_memory_stable() {
    let device = cuda_device();
    check_synthetic(&device, true, 8);
    let initial = used_graph_memory(&device);
    for cycle in 0..16 {
        check_synthetic(&device, cycle % 2 == 0, 8);
        assert_eq!(
            used_graph_memory(&device),
            initial,
            "graph memory grew in cycle {cycle}"
        );
    }
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    // SAFETY: All graphs have been destroyed and their outstanding work completed.
    unsafe { sys::cuDeviceGraphMemTrim(stream.context().cu_device()) }
        .result()
        .unwrap();
    let trimmed = used_graph_memory(&device);
    assert_eq!(
        trimmed, 0,
        "destroyed GDN graphs must release all allocations"
    );
    eprintln!(
        "16 GDN graph lifecycles: stable used graph memory {initial} bytes; after trim \
         {trimmed} bytes"
    );
}
