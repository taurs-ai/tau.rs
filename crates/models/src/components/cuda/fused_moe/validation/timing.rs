//! GPU event timing after independent correctness approval. The caller retains
//! all routing inputs, weight banks and program storage through graph teardown;
//! hot-layer and rotating-bank results deliberately remain separate.
use super::*;
use crate::components::{CapturedGraph, DecodeProgram};
use candle::cuda_backend::cudarc::driver::sys;

pub(super) fn measure<F: FnMut() -> CandleResult<()>>(
    device: &candle::Device,
    runner: &FusedMoeRunner,
    input: &Tensor,
    logits: &Tensor,
    graph: &mut CapturedGraph<F>,
    blocks_in_cycle: usize,
    enabled: bool,
) -> Vec<serde_json::Value> {
    if !enabled {
        return Vec::new();
    }
    let mut hot = DecodeProgram::new(device).unwrap();
    let _hot_output = runner
        .prepare_decode_program(&mut hot, input, logits)
        .unwrap();
    hot.run().unwrap();
    device.synchronize().unwrap();
    // SAFETY: Same retained owners/stream as the rotating program above.
    let mut hot_graph =
        unsafe { CapturedGraph::capture(device, || hot.run()) }.unwrap();
    assert_eq!(hot_graph.allocation_nodes().unwrap(), 0);
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    let start = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let mut samples = Vec::new();
    for round in 0..if enabled { 7 } else { 0 } {
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for cycle in order {
            let blocks = if cycle { blocks_in_cycle } else { 1 };
            for _ in 0..4 {
                if cycle {
                    graph.replay()
                } else {
                    hot_graph.replay()
                }
                .unwrap();
            }
            device.synchronize().unwrap();
            let wall = std::time::Instant::now();
            start.record(&stream).unwrap();
            for _ in 0..32 {
                if cycle {
                    graph.replay()
                } else {
                    hot_graph.replay()
                }
                .unwrap();
            }
            end.record(&stream).unwrap();
            end.synchronize().unwrap();
            if round > 0 {
                samples.push(serde_json::json!({"cycle_banks":cycle,
                "gpu_us_per_block":f64::from(start.elapsed_ms(&end).unwrap())*1000./(32*blocks) as f64,
                "wall_us_per_block":wall.elapsed().as_secs_f64()*1e6/(32*blocks) as f64}));
            }
        }
    }
    drop(hot_graph);
    samples
}

/// Measure the same seeded expert projection across hot or rotating banks, after
/// CPU approval of the full pipeline. Each closure retains its workspace/weights;
/// no pointer preparation, activation, casting or routing enters these graphs.
pub(super) fn projections(
    device: &candle::Device,
    runners: &[FusedMoeRunner],
    enabled: bool,
) -> Vec<serde_json::Value> {
    if !enabled {
        return Vec::new();
    }
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    let mut graphs = Vec::new();
    let mut programs = Vec::new();
    for down in [false, true] {
        for cycle in [false, true] {
            let mut program = DecodeProgram::new(device).unwrap();
            for runner in runners.iter().take(if cycle { runners.len() } else { 1 }) {
                let runner = runner.clone();
                program.push(Box::new(move || runner.validation_project(down)));
            }
            program.run().unwrap();
            device.synchronize().unwrap();
            programs.push((down, cycle, program));
        }
    }
    // Every program remains at a stable address until all graph owners drop.
    for (down, cycle, program) in &programs {
        let graph =
            unsafe { CapturedGraph::capture(device, || program.run()) }.unwrap();
        assert_eq!(graph.allocation_nodes().unwrap(), 0);
        graphs.push((*down, *cycle, graph));
    }
    let mut samples = Vec::new();
    for round in 0..7 {
        let order: Vec<_> = if round % 2 == 0 {
            (0..4).collect()
        } else {
            (0..4).rev().collect()
        };
        for index in order {
            let (down, cycle, graph) = &mut graphs[index];
            for _ in 0..4 {
                graph.replay().unwrap();
            }
            device.synchronize().unwrap();
            let start = stream
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let end = stream
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let wall = std::time::Instant::now();
            start.record(&stream).unwrap();
            for _ in 0..32 {
                graph.replay().unwrap();
            }
            end.record(&stream).unwrap();
            end.synchronize().unwrap();
            let count = 32 * if *cycle { runners.len() } else { 1 };
            if round > 0 {
                samples.push(serde_json::json!({"projection":if *down {"down"} else {"gate_up"},"cycle_banks":cycle,
                "gpu_us":f64::from(start.elapsed_ms(&end).unwrap())*1000./count as f64,
                "wall_us":wall.elapsed().as_secs_f64()*1e6/count as f64}));
            }
        }
    }
    drop(graphs);
    samples
}
