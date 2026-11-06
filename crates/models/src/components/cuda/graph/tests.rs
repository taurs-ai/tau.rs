use super::*;
use crate::components::cuda::centered_rms_norm_cuda;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Module, Storage, Tensor, Var};
use candle_nn::{Linear, VarBuilder};
use std::cell::Cell;
use std::collections::HashMap;
use std::path::PathBuf;

pub(in crate::components) fn cuda_device() -> Device {
    // SAFETY: Tests keep all tensors and operations on this device's active stream.
    unsafe { new_capture_device(0) }
        .expect("CUDA capture tests require a usable NVIDIA CUDA GPU")
}

fn patterned(rows: usize, columns: usize, phase: usize, device: &Device) -> Tensor {
    let values = (0..rows * columns)
        .map(|i| (((i * 17 + phase) % 61) as f32 - 30.0) * 0.007)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, columns), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn assert_exact(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.dims(), expected.dims());
    for (index, (actual, expected)) in
        values(actual).into_iter().zip(values(expected)).enumerate()
    {
        assert!(actual.is_finite() && expected.is_finite());
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {index}: {actual} != {expected}"
        );
    }
}

pub(crate) fn address(tensor: &Tensor) -> u64 {
    let (storage, _) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        panic!("expected CUDA storage")
    };
    let stream = storage.device.cuda_stream();
    let (pointer, _guard) = storage
        .as_cuda_slice::<half::bf16>()
        .unwrap()
        .device_ptr(&stream);
    pointer
}

/// Inspects the actual graph, rather than inferring replay from output alone.
pub(crate) fn node_types<F>(
    graph: &CapturedGraph<F>,
) -> HashMap<sys::CUgraphNodeType, usize>
where
    F: FnMut() -> Result<()>,
{
    let mut count = graph.graph.node_count().unwrap();
    let mut nodes = vec![ptr::null_mut(); count];
    // SAFETY: The graph is live on this thread and both output buffers are sized.
    unsafe {
        sys::cuGraphGetNodes(graph.graph.graph, nodes.as_mut_ptr(), &mut count)
            .result()
            .unwrap();
    }
    let mut types = HashMap::new();
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        // SAFETY: Node belongs to the retained graph.
        unsafe { sys::cuGraphNodeGetType(node, &mut kind) }
            .result()
            .unwrap();
        *types.entry(kind).or_default() += 1;
    }
    types
}

fn project(
    input: &Tensor,
    linear: &Linear,
    norm: &Tensor,
    residual: &Tensor,
    composed: bool,
) -> Result<Tensor> {
    let projected = linear.forward(input)?;
    if composed {
        centered_rms_norm_cuda(
            &projected,
            norm,
            1e-6,
            tau_cuda_kernels::RmsNormVectorBytes::Bytes16,
        )? + residual
    } else {
        Ok(projected)
    }
}

/// Exercises projection alone and projection/native-normalization/residual together.
fn check_replay(device: &Device, weight: Tensor, norm: Tensor, iterations: usize) {
    let (output_width, input_width) = weight.dims2().unwrap();
    let linear = Linear::new(weight, None);
    for composed in [false, true] {
        let input = Var::from_tensor(&patterned(1, input_width, 1, device)).unwrap();
        let residual =
            Var::from_tensor(&patterned(1, output_width, 3, device)).unwrap();
        let output = Var::zeros((1, output_width), DType::BF16, device).unwrap();
        let original_addresses =
            (address(&input), address(&residual), address(&output));
        let calls = Cell::new(0);
        let operation = {
            // Move retained handles into the closure; detached inputs avoid retaining
            // an autograd tree of temporaries beyond their captured free operations.
            let input = input.as_detached_tensor();
            let residual = residual.as_detached_tensor();
            let output = output.clone();
            let linear = linear.clone();
            let norm = norm.clone();
            let calls = &calls;
            move || {
                calls.set(calls.get() + 1);
                let result = project(&input, &linear, &norm, &residual, composed)?;
                output.set(&result)
            }
        };
        let _cache = device
            .as_cuda_device()
            .unwrap()
            .enable_cuda_graph_htod_cache();
        operation().unwrap();
        device.synchronize().unwrap();
        let sentinel = patterned(1, output_width, 7, device);
        output.set(&sentinel).unwrap();
        // SAFETY: All persistent tensors are retained in the closure. Temporaries
        // drop during capture, and all writes/replays/readbacks use Candle's stream.
        let mut graph = unsafe { CapturedGraph::capture(device, operation) }.unwrap();
        assert_eq!(calls.get(), 2);
        assert_exact(&output, &sentinel); // Capture did not execute its GPU work.
        let types = node_types(&graph);
        assert!(
            types
                .get(&sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL)
                .copied()
                .unwrap_or(0)
                >= if composed { 3 } else { 1 }
        );
        assert_eq!(
            types.get(&sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC),
            types.get(&sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE),
            "all captured temporary allocations must have a captured free: {types:?}"
        );
        eprintln!(
            "capture composed={composed}, shape=1x{input_width}->{output_width}: \
             {types:?}"
        );
        let mut previous = None;
        for step in 0..iterations {
            input
                .set(&patterned(1, input_width, 11 + step, device))
                .unwrap();
            residual
                .set(&patterned(1, output_width, 23 + step, device))
                .unwrap();
            graph.replay().unwrap();
            graph.synchronize().unwrap();
            let expected = project(
                &input.as_detached_tensor(),
                &linear,
                &norm,
                &residual.as_detached_tensor(),
                composed,
            )
            .unwrap();
            assert_exact(&output, &expected);
            let current = values(&output);
            if let Some(previous) = previous {
                assert_ne!(current, previous, "changing input must change output");
            }
            previous = Some(current);
            assert_eq!(calls.get(), 2, "replay must bypass the Rust closure");
            assert_eq!(
                (address(&input), address(&residual), address(&output)),
                original_addresses
            );
        }
        // Drop must also complete an outstanding launch before releasing resources.
        graph.replay().unwrap();
        drop(graph);
        device
            .as_cuda_device()
            .unwrap()
            .cuda_stream()
            .context()
            .check_err()
            .unwrap();
        drop((input, residual, output));
        device
            .as_cuda_device()
            .unwrap()
            .cuda_stream()
            .context()
            .check_err()
            .expect("retained buffer cleanup after graph destruction");
    }
}

#[test]
fn capture_rejects_cpu_before_running_operation() {
    let calls = Cell::new(0);
    // SAFETY: No GPU work or pointers; a CPU device must be rejected before the
    // closure.
    let result = unsafe {
        CapturedGraph::capture(&Device::Cpu, || {
            calls.set(1);
            Ok(())
        })
    };
    let error = result.err().expect("CPU capture must fail");
    assert!(error.to_string().contains("requires an NVIDIA CUDA device"));
    assert_eq!(calls.get(), 0);
}

#[test]
fn capture_rejects_tracked_device_before_running_operation() {
    let device = Device::new_cuda(0).expect("test requires NVIDIA CUDA GPU 0");
    let calls = Cell::new(0);
    // SAFETY: This invalid configuration must be rejected before the closure runs.
    let error = unsafe {
        CapturedGraph::capture(&device, || {
            calls.set(1);
            Ok(())
        })
    }
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("event tracking disabled before tensor allocation")
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn capture_device_errors_without_visible_cuda_gpu() {
    const CHILD: &str = "TAU_CAPTURE_NO_CUDA_CHILD";
    if std::env::var_os(CHILD).is_some() {
        // SAFETY: No resources are created when initialization fails.
        let error =
            unsafe { new_capture_device(0) }.expect_err("must not fall back to CPU");
        assert!(
            error
                .to_string()
                .contains("requires a usable NVIDIA CUDA GPU")
        );
        return;
    }
    // CUDA caches enumeration, so test unavailable hardware in a fresh process.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "model::components::cuda::graph::tests::\
             capture_device_errors_without_visible_cuda_gpu",
        ])
        .env(CHILD, "1")
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn capture_replays_bf16_projection_and_native_sequence() {
    let device = cuda_device();
    check_replay(
        &device,
        patterned(64, 128, 3, &device),
        patterned(1, 64, 5, &device).flatten_all().unwrap(),
        128,
    );
}

#[test]
fn capture_cleans_up_errors_panics_and_invalidated_streams() {
    let device = cuda_device();
    let cuda = device.as_cuda_device().unwrap();
    let stream = cuda.cuda_stream();
    let input = Var::from_tensor(&patterned(1, 64, 1, &device)).unwrap();
    let output = Var::zeros((1, 64), DType::BF16, &device).unwrap();
    let _cache = cuda.enable_cuda_graph_htod_cache();
    let copy = || output.set(&input.as_detached_tensor());
    copy().unwrap();
    stream.synchronize().unwrap();
    // SAFETY: Retained buffers, one stream, and a copy with no escaping allocations.
    let error = unsafe {
        CapturedGraph::capture(&device, || {
            copy()?;
            candle::bail!("injected capture failure")
        })
    }
    .err()
    .unwrap();
    assert!(error.to_string().contains("injected capture failure"));
    require_idle(&stream).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Same retained copy resources; the guard must also handle unwinding.
        let _ = unsafe {
            CapturedGraph::capture(&device, || {
                copy()?;
                panic!("injected capture panic")
            })
        };
    }));
    assert!(panic.is_err());
    require_idle(&stream).unwrap();
    // A forbidden CUDA synchronization genuinely invalidates the capture.
    let error = unsafe {
        CapturedGraph::capture(&device, || {
            copy()?;
            stream.synchronize().map_err(cuda_error)
        })
    }
    .err()
    .unwrap();
    assert!(error.to_string().contains("CUDA graph operation failed"));
    require_idle(&stream).unwrap();
    // A subsequent ordinary operation and fresh capture must still work.
    stream.context().check_err().unwrap();
    copy().unwrap();
    let mut graph = unsafe { CapturedGraph::capture(&device, copy) }.unwrap();
    input.set(&patterned(1, 64, 7, &device)).unwrap();
    graph.replay().unwrap();
    graph.synchronize().unwrap();
    assert_exact(&output, &input);
}

#[test]
fn capture_rejects_empty_and_nested_capture() {
    let device = cuda_device();
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    // SAFETY: No resources or work are involved in either rejected operation.
    assert!(
        unsafe { CapturedGraph::capture(&device, || Ok(())) }
            .err()
            .unwrap()
            .to_string()
            .contains("no GPU work")
    );
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let outer = StreamCapture {
        stream: stream.clone(),
        active: true,
    };
    let error = unsafe { CapturedGraph::capture(&device, || Ok(())) }
        .err()
        .unwrap();
    assert!(error.to_string().contains("outside capture"));
    assert_eq!(
        stream.capture_status().unwrap(),
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    );
    drop(outer);
    require_idle(&stream).unwrap();
    stream.context().check_err().unwrap();
}

#[test]
#[ignore = "measures device-wide graph memory; run alone with --test-threads=1"]
fn capture_repeated_lifecycle_keeps_graph_memory_stable() {
    let device = cuda_device();
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    let weight = patterned(64, 128, 3, &device);
    let norm = patterned(1, 64, 5, &device).flatten_all().unwrap();
    // CUDA retains graph-pool backing pages after freeing graph allocations. Warm
    // that pool before checking for growth across subsequent complete lifecycles.
    check_replay(&device, weight.clone(), norm.clone(), 8);
    let initial = used_graph_memory(&device);
    for _ in 0..16 {
        check_replay(&device, weight.clone(), norm.clone(), 8);
        assert_eq!(
            used_graph_memory(&device),
            initial,
            "graph memory grew after warmed capture/replay/teardown"
        );
    }
    // SAFETY: No graph is executing, so unused backing pages may be returned.
    unsafe { sys::cuDeviceGraphMemTrim(stream.context().cu_device()) }
        .result()
        .unwrap();
    let trimmed = used_graph_memory(&device);
    assert!(trimmed <= initial);
    eprintln!(
        "32 measured graph lifecycles: stable used graph memory {initial} bytes; after \
         trim {trimmed} bytes"
    );
}

/// CUDA graph-pool backing pages can remain in use after graph destruction.
pub(in crate::components) fn used_graph_memory(device: &Device) -> u64 {
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    stream.synchronize().unwrap();
    let mut bytes = 0_u64;
    // SAFETY: The context is active and the attribute writes one u64 value.
    unsafe {
        sys::cuDeviceGetGraphMemAttribute(
            stream.context().cu_device(),
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
            (&mut bytes as *mut u64).cast(),
        )
    }
    .result()
    .unwrap();
    bytes
}

#[test]
#[ignore = "loads a projection from local Qwen3.5-4B and requires NVIDIA CUDA GPU 0"]
fn capture_replays_qwen3_5_4b_projection_and_native_sequence() {
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
    let hidden = config["text_config"]["hidden_size"].as_u64().unwrap() as usize;
    let intermediate =
        config["text_config"]["intermediate_size"].as_u64().unwrap() as usize;
    let shards = crate::safetensors::load_safetensors_shards(&root).unwrap();
    let paths = shards.iter().map(|shard| shard.path()).collect::<Vec<_>>();
    // SAFETY: The local checkpoint is memory-mapped read-only and never modified.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device) }
            .unwrap();
    let layer = vb.pp("model.language_model.layers.0");
    let weight = layer
        .get((hidden, intermediate), "mlp.down_proj.weight")
        .unwrap();
    let norm = layer.get(hidden, "input_layernorm.weight").unwrap();
    check_replay(&device, weight, norm, 128);
}
