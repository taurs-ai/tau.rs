//! Actual Candle add + current RMSNorm versus the fused component. No replica of
//! Candle's addition kernel or historical scalar norm is substituted as baseline.
use crate::components::rms_norm::CenteredRmsNorm;
use crate::components::{CapturedGraph, graph_node_types};
use candle::cuda_backend::cudarc::driver::sys;
use candle::{DType, Tensor};
use serde_json::json;
use std::time::Instant;

#[test]
#[ignore = "runs a release-mode CUDA residual/RMSNorm benchmark on GPU 0"]
fn measures_fused_residual_against_current_unfused() {
    if cfg!(debug_assertions) {
        panic!("use cargo test --release");
    }
    const GROUP: usize = 128;
    const ITERATIONS: usize = 512;
    const ROUNDS: usize = 10;
    // SAFETY: No tensors escape this thread; all operations share one stream.
    let device = unsafe { super::super::graph::new_capture_device(0) }.unwrap();
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    let start = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let mut cases = Vec::new();
    for dtype in [DType::BF16, DType::F32] {
        for (rows, columns) in [(1, 2560), (1, 5120), (512, 2560), (512, 5120)] {
            let make = |count: usize, phase: usize| {
                Tensor::new(
                    (0..count)
                        .map(|i| ((i * 17 + phase) % 127) as f32 / 37.0 - 1.7)
                        .collect::<Vec<_>>(),
                    &device,
                )
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
            };
            let input = make(rows * columns, 3).reshape((1, rows, columns)).unwrap();
            let branch = make(rows * columns, 19).reshape(input.shape()).unwrap();
            let norm =
                CenteredRmsNorm::from_weight_for_test(make(columns, 47), 1e-6, true);
            let pair = |fused: bool| {
                if fused {
                    norm.add_residual_and_normalize(&input, &branch)
                } else {
                    norm.add_residual_and_normalize_unfused(&input, &branch)
                }
            };
            let (_, expected) = pair(false).unwrap();
            let (_, actual) = pair(true).unwrap();
            assert_eq!(actual.dims(), expected.dims());
            // Operator correctness tests cover numerical accuracy;
            // assert finite work here.
            assert!(
                actual
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite())
            );
            let mut graphs = Vec::new();
            let mut kernels = Vec::new();
            for fused in [false, true] {
                {
                    let _cache = device
                        .as_cuda_device()
                        .unwrap()
                        .enable_cuda_graph_htod_cache();
                    for _ in 0..32 {
                        std::hint::black_box(pair(fused).unwrap());
                    }
                }
                device.synchronize().unwrap();
                let pair_ref = &pair;
                // SAFETY: Captured temporaries are freed in the same stream; the
                // closure borrows inputs/weights retained until graphs are dropped.
                let mut graph = unsafe {
                    CapturedGraph::capture(&device, move || {
                        for _ in 0..GROUP {
                            std::hint::black_box(pair_ref(fused)?);
                        }
                        Ok(())
                    })
                }
                .unwrap();
                let count = graph_node_types(&graph)
                    .get(&sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(count, GROUP * if fused { 1 } else { 2 });
                kernels.push(count);
                for _ in 0..4 {
                    graph.replay().unwrap();
                }
                graph.synchronize().unwrap();
                graphs.push(graph);
            }
            for graph_mode in [false, true] {
                let mut samples = Vec::new();
                for round in 0..ROUNDS {
                    let mut event_us = [0.0; 2];
                    let mut wall_us = [0.0; 2];
                    for variant in [round % 2, 1 - round % 2] {
                        device.synchronize().unwrap();
                        let wall = Instant::now();
                        start.record(&stream).unwrap();
                        if graph_mode {
                            for _ in 0..ITERATIONS / GROUP {
                                graphs[variant].replay().unwrap();
                            }
                        } else {
                            for _ in 0..ITERATIONS {
                                std::hint::black_box(pair(variant == 1).unwrap());
                            }
                        }
                        end.record(&stream).unwrap();
                        end.synchronize().unwrap();
                        wall_us[variant] =
                            wall.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64;
                        event_us[variant] = f64::from(start.elapsed_ms(&end).unwrap())
                            * 1000.0
                            / ITERATIONS as f64;
                    }
                    samples.push(
                        json!({"first": if round % 2 == 0 {"unfused"} else {"fused"},
                        "unfused_us": event_us[0], "fused_us": event_us[1],
                        "unfused_wall_us": wall_us[0], "fused_wall_us": wall_us[1]}),
                    );
                }
                cases.push(json!({
                    "dtype": format!("{dtype:?}"),
                    "rows": rows,
                    "columns": columns,
                    "mode": if graph_mode {"graph"} else {"eager_stream"},
                    "samples": samples,
                    "graph_kernel_counts": kernels
                }));
            }
        }
    }
    println!(
        "residual_rmsnorm_benchmark {}",
        json!({"schema_version":1,
        "gpu": stream.context().name().unwrap(), "iterations":ITERATIONS,
        "pairs_per_graph":GROUP,"rounds":ROUNDS,"cases":cases,
        "scope":"Actual Candle operator boundary, including output allocation/free. \
         Capture device has allocation event tracking disabled. Fixed inputs and \
         warmed caches; no server throughput claim."})
    );
}
