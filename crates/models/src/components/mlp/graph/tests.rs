//! Capture/replay, fixed-buffer lifetime, clone and real-checkpoint regressions.

use super::*;
use candle::Device;
use candle_nn::{Activation, VarBuilder};

/// Compare complete test outputs, including nonfinite values and signed zeros.
fn check_graph_output(eager: &Tensor, graph: &Tensor) -> CandleResult<()> {
    if eager.dims() != graph.dims() {
        candle::bail!("MLP graph comparison shape mismatch")
    }
    let values = |xs: &Tensor| xs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
    let eager = values(eager)?;
    let graph = values(graph)?;
    let mut mismatches = 0;
    let mut max_abs_error = 0f32;
    let mut first = None;
    for (index, (&expected, &actual)) in eager.iter().zip(&graph).enumerate() {
        if !expected.is_finite()
            || !actual.is_finite()
            || expected.to_bits() != actual.to_bits()
        {
            mismatches += 1;
            first.get_or_insert((index, expected, actual));
            max_abs_error = if expected.is_finite() && actual.is_finite() {
                max_abs_error.max((expected - actual).abs())
            } else {
                f32::INFINITY
            };
        }
    }
    if mismatches > 0 {
        candle::bail!(
            "MLP graph comparison failed: {mismatches}/{} differing or nonfinite \
             values, max_abs_error={max_abs_error}, first={first:?}",
            eager.len()
        )
    }
    Ok(())
}

fn graph_fixture(device: &Device) -> Mlp {
    let weight = |rows, cols, phase| {
        let values = (0..rows * cols)
            .map(|i| ((i + phase) as f32 * 0.07).sin() * 0.03)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, cols), device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    Mlp::from_separate_weights(
        weight(64, 32, 1),
        weight(64, 32, 7),
        weight(32, 64, 11),
        Activation::Silu,
        true,
    )
    .unwrap()
}

/// Complete-vector probes reject numerical differences and nonfinite outputs.
#[test]
fn graph_comparison_detects_corruption() {
    let tensor = |xs: &[f32]| Tensor::new(xs, &Device::Cpu).unwrap();
    let expected = tensor(&[1., 2., 3.]);
    check_graph_output(&expected, &expected).unwrap();
    for bad in [
        vec![1., 2.25, 3.],
        vec![1., f32::NAN, 3.],
        vec![1., f32::INFINITY, 3.],
    ] {
        let error = check_graph_output(&expected, &tensor(&bad))
            .unwrap_err()
            .to_string();
        assert!(error.contains("MLP graph comparison failed"));
        assert!(error.contains("1/3"));
    }
    assert!(
        check_graph_output(&expected, &tensor(&[1.]))
            .unwrap_err()
            .to_string()
            .contains("shape mismatch")
    );
}

fn check_mlp_graph_lifecycle(mut mlp: Mlp) {
    use crate::components::cuda::graph::tests::address;
    let device = mlp.device.clone();
    let hidden = mlp.hidden_size;
    let input = |length, phase| {
        let values = (0..length * hidden)
            .map(|i| ((i + phase) as f32 * 0.03).cos() * 0.2)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, length, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    let cuda_graph = GraphPhase::Decode;
    let mut addresses = None;
    let mut calls = 0;
    let mut first = Vec::new();
    for (request, phase) in [17, 3, 17].into_iter().enumerate() {
        let before = mlp.cuda_graph_counts();
        // Ordinary forward remains eager, even for a one-token prompt.
        mlp.forward(&input(1, phase)).unwrap();
        mlp.forward(&input(2, phase)).unwrap();
        assert_eq!(mlp.cuda_graph_counts(), before);
        for step in 0..128 {
            let xs = input(1, phase + step);
            let actual = mlp.forward_graph(&xs, cuda_graph).unwrap();
            check_graph_output(&mlp.forward(&xs).unwrap(), &actual).unwrap();
            let graph = mlp.decode_graph.as_ref().unwrap();
            let current = (address(&graph.input), address(&graph.output));
            assert_eq!(*addresses.get_or_insert(current), current);
            calls += 1;
            assert_eq!(
                mlp.cuda_graph_counts(),
                CudaGraphStats {
                    mlp_graph_captures: 1,
                    mlp_graph_replays: calls,
                    ..Default::default()
                }
            );
            let values = actual
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            if request == 0 {
                first.push(values);
            } else if request == 2 {
                assert_eq!(values, first[step]);
            }
        }
    }
    // A clone must not alias live graph input/output buffers or inherit counters.
    let clone = mlp.clone();
    assert_eq!(clone.cuda_graph_counts(), CudaGraphStats::default());
    let before = mlp.cuda_graph_counts();
    mlp.forward(&input(1, 7)).unwrap();
    assert_eq!(mlp.cuda_graph_counts(), before);
    // Queue a replay without readback, then exercise teardown ordering.
    mlp.forward_graph(&input(1, 5), GraphPhase::Decode).unwrap();
    drop(mlp);
    device.synchronize().unwrap();
}

#[test]
fn mlp_graph_reuses_buffers_and_matches_eager() {
    // SAFETY: All tensors and operations stay on this test's thread/stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    for _ in 0..3 {
        check_mlp_graph_lifecycle(graph_fixture(&device));
    }
}

/// Alternating phases must retain two independent sets of buffers and reject
/// shape changes without replacing either captured graph or advancing counters.
#[test]
fn mlp_graph_keeps_prefill_and_decode_slots_independent() {
    use crate::components::cuda::graph::tests::address;
    // SAFETY: Captures, replay, output reads and teardown use one thread/stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let mut mlp = graph_fixture(&device);
    let prefill = Tensor::ones((1, 4, 32), DType::BF16, &device).unwrap();
    let decode = Tensor::zeros((1, 1, 32), DType::BF16, &device).unwrap();
    let slots = |graph: &MlpGraph| (address(&graph.input), address(&graph.output));
    let actual = mlp.forward_graph(&prefill, GraphPhase::Prefill).unwrap();
    check_graph_output(&mlp.forward(&prefill).unwrap(), &actual).unwrap();
    let prefill_addresses = slots(mlp.prefill_graph.as_ref().unwrap());
    let actual = mlp.forward_graph(&decode, GraphPhase::Decode).unwrap();
    check_graph_output(&mlp.forward(&decode).unwrap(), &actual).unwrap();
    let decode_addresses = slots(mlp.decode_graph.as_ref().unwrap());
    assert_ne!(prefill_addresses.0, decode_addresses.0);
    assert_ne!(prefill_addresses.1, decode_addresses.1);
    let changed_prefill = prefill.zeros_like().unwrap();
    let changed_decode = decode.ones_like().unwrap();
    for (xs, phase) in [
        (&changed_prefill, GraphPhase::Prefill),
        (&changed_decode, GraphPhase::Decode),
        (&prefill, GraphPhase::Prefill),
        (&decode, GraphPhase::Decode),
    ] {
        let actual = mlp.forward_graph(xs, phase).unwrap();
        check_graph_output(&mlp.forward(xs).unwrap(), &actual).unwrap();
        assert_eq!(
            slots(mlp.prefill_graph.as_ref().unwrap()),
            prefill_addresses
        );
        assert_eq!(slots(mlp.decode_graph.as_ref().unwrap()), decode_addresses);
    }
    let counts = mlp.cuda_graph_counts();
    assert_eq!(counts.mlp_graph_captures, 2);
    assert_eq!(counts.mlp_graph_replays, 6);
    let wrong_shape = Tensor::zeros((1, 3, 32), DType::BF16, &device).unwrap();
    assert!(
        mlp.forward_graph(&wrong_shape, GraphPhase::Prefill)
            .is_err()
    );
    assert_eq!(mlp.cuda_graph_counts(), counts);
    assert_eq!(
        slots(mlp.prefill_graph.as_ref().unwrap()),
        prefill_addresses
    );
    assert_eq!(slots(mlp.decode_graph.as_ref().unwrap()), decode_addresses);
    let actual = mlp.forward_graph(&prefill, GraphPhase::Prefill).unwrap();
    check_graph_output(&mlp.forward(&prefill).unwrap(), &actual).unwrap();
    let cloned = mlp.clone();
    assert!(cloned.prefill_graph.is_none() && cloned.decode_graph.is_none());
    assert_eq!(cloned.cuda_graph_counts(), CudaGraphStats::default());
}

#[test]
fn mlp_graph_rejects_tracked_device_and_unsupported_input() {
    let tracked = Device::new_cuda(0).unwrap();
    let mut mlp = graph_fixture(&tracked);
    let xs = Tensor::zeros((1, 1, 32), DType::BF16, &tracked).unwrap();
    assert!(
        mlp.forward_graph(&xs, GraphPhase::Decode)
            .unwrap_err()
            .to_string()
            .contains("capture device")
    );
    assert_eq!(mlp.cuda_graph_counts(), CudaGraphStats::default());
    let xs = Tensor::zeros((1, 1, 32), DType::BF16, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    // Use F32 weights because Candle's portable BF16 matmul is unsupported.
    let weight = Tensor::ones((32, 32), DType::F32, &Device::Cpu).unwrap();
    let mut cpu = Mlp::from_separate_weights(
        weight.clone(),
        weight.clone(),
        weight,
        Activation::Silu,
        true,
    )
    .unwrap();
    assert!(!cpu.supports_layer_graph(&xs));
    assert!(
        cpu.forward_graph(&xs, GraphPhase::Decode)
            .unwrap_err()
            .to_string()
            .contains("does not support")
    );
    // Only the dispatcher chooses eager; strict graph execution cannot hide it.
    cpu.forward(&xs).unwrap();
    assert_eq!(cpu.cuda_graph_counts(), CudaGraphStats::default());
}

#[test]
#[ignore = "loads local Qwen3.5-4B MLP weights and requires CUDA GPU 0"]
fn mlp_graph_replays_qwen3_5_4b_layer() {
    use std::path::PathBuf;
    let root = std::env::var_os("TAU_QWEN3_5_CAPTURE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap())
                .join("work/models/Qwen3.5-4B")
        });
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap())
            .unwrap();
    let text = &config["text_config"];
    // SAFETY: All operations and retained resources stay on this test thread.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let shards = crate::safetensors::load_safetensors_shards(&root).unwrap();
    let paths = shards.iter().map(|s| s.path()).collect::<Vec<_>>();
    // SAFETY: The checkpoint is mapped read-only and never modified.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device) }
            .unwrap();
    let mlp = Mlp::new(
        text["hidden_size"].as_u64().unwrap() as usize,
        text["intermediate_size"].as_u64().unwrap() as usize,
        Activation::Silu,
        true,
        vb.pp("model.language_model.layers.0.mlp"),
    )
    .unwrap();
    check_mlp_graph_lifecycle(mlp);
}
