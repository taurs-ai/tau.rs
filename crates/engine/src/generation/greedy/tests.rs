//! Exact CPU-oracle checks for selection, layout fallback and workspace lifetime.
use super::*;
use crate::generation::{logits_values, next_token_from_values};

fn compare(sampler: &mut GpuGreedy, logits: &Tensor) {
    let expected = next_token_from_values(&logits_values(logits).unwrap());
    let actual = sampler.select(logits).map(|id| id.unwrap());
    assert_eq!(actual, expected);
}

/// Bit patterns cover subnormals and both signs of NaN independently of any
/// floating-point random generator. Maxima crossing 1024-element blocks exercise
/// the final reduction and last-ID tie policy, including partially filled blocks.
#[test]
fn cuda_greedy_matches_cpu_ordering_for_bf16_f32_and_offset_views() {
    let device = Device::new_cuda(0).unwrap();
    let mut sampler = GpuGreedy::default();
    let mut cases = vec![
        vec![f32::NAN],
        vec![f32::NAN, f32::NEG_INFINITY, f32::NAN],
        vec![f32::NEG_INFINITY; 1025],
        vec![-0., 0., -0.],
        vec![0., -0., 0.],
        vec![f32::INFINITY, f32::NAN, f32::INFINITY],
        vec![
            f32::from_bits(1),
            0.,
            f32::from_bits(2),
            f32::from_bits(0x80000001),
        ],
        vec![f32::from_bits(0xff800001), f32::from_bits(0x7f800001)],
    ];
    for length in [1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 248320] {
        let mut state = 0x12345678u32;
        let random = (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                f32::from_bits(state)
            })
            .collect::<Vec<_>>();
        cases.push(random);
        let mut ties = vec![-2.; length];
        ties[0] = f32::INFINITY;
        ties[length - 1] = f32::INFINITY;
        cases.push(ties);
    }
    for dtype in [DType::F32, DType::BF16] {
        for values in &cases {
            for offset in [0, 1, 3] {
                let mut storage = vec![f32::INFINITY; offset];
                storage.extend(values);
                storage.push(f32::INFINITY);
                let logits = Tensor::new(storage, &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
                    .narrow(0, offset, values.len())
                    .unwrap()
                    .reshape((1, 1, values.len()))
                    .unwrap();
                compare(&mut sampler, &logits);
            }
        }
    }
    // Construct BF16 storage directly: an earlier F32 conversion could erase
    // subnormal bits before the selector/reference comparison even begins.
    for bits in [
        vec![0x0002, 0x0000, 0x0001, 0x8001, 0x0000],
        vec![0x8001, 0x8002, 0x8000],
        vec![0x7fc1, 0xff81],
    ] {
        let stored = bits.into_iter().map(bf16::from_bits).collect::<Vec<_>>();
        let logits = Tensor::new(stored.as_slice(), &device)
            .unwrap()
            .reshape((1, 1, stored.len()))
            .unwrap();
        compare(&mut sampler, &logits);
    }
}

/// Alternating valid/all-NaN inputs at the same shape must overwrite both the
/// partial candidates and the result. Buffers survive dtype/request changes;
/// changing the vocabulary rebuilds only the sampler's own bounded workspace.
#[test]
fn cuda_greedy_reuses_storage_without_stale_ids_and_respects_private_streams() {
    for private in [false, true] {
        let device = Device::new_cuda(0).unwrap();
        if private {
            // SAFETY: No tensors exist yet; all work stays on this device stream.
            unsafe { device.as_cuda_device().unwrap().disable_event_tracking() };
        }
        let mut sampler = GpuGreedy::default();
        let length = 1025;
        let mut address = None;
        for (index, winner) in
            [Some(3), None, Some(1024), Some(3)].into_iter().enumerate()
        {
            let mut values = vec![f32::NAN; length];
            if let Some(winner) = winner {
                values[winner] = -1.;
            }
            let logits = Tensor::new(values, &device)
                .unwrap()
                .to_dtype(if index % 2 == 0 {
                    DType::F32
                } else {
                    DType::BF16
                })
                .unwrap()
                .reshape((1, 1, length))
                .unwrap();
            compare(&mut sampler, &logits);
            let workspace = sampler.workspace.as_ref().unwrap();
            let pointer = workspace.scratch.device_ptr(&workspace.stream).0;
            assert_eq!(*address.get_or_insert(pointer), pointer);
        }
        let different = Tensor::new(&[[[1f32, 2., 2.]]], &device).unwrap();
        assert_eq!(sampler.select(&different).unwrap(), Some(2));
        assert_eq!(sampler.workspace.as_ref().unwrap().vocabulary, 3);
    }
}

#[test]
fn greedy_fallback_preserves_cpu_empty_and_unsupported_tensor_contracts() {
    let mut sampler = GpuGreedy::default();
    let cpu = Tensor::new(&[[[1f32, 2., 2.]]], &Device::Cpu).unwrap();
    assert_eq!(sampler.select(&cpu).unwrap(), None);
    assert!(sampler.workspace.is_none());
    let device = Device::new_cuda(0).unwrap();
    let empty = Tensor::zeros((1, 1, 0), DType::F32, &device).unwrap();
    assert_eq!(sampler.select(&empty).unwrap(), None);
    assert!(next_token_from_values(&logits_values(&empty).unwrap()).is_err());
    let unsupported = cpu
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap();
    assert_eq!(sampler.select(&unsupported).unwrap(), None);
    let strided = Tensor::new(&[[1f32, 9.], [2., 9.], [3., 9.]], &device)
        .unwrap()
        .narrow(1, 0, 1)
        .unwrap()
        .transpose(0, 1)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    assert!(!strided.is_contiguous());
    assert_eq!(sampler.select(&strided).unwrap(), None);
    assert_eq!(
        next_token_from_values(&logits_values(&strided).unwrap()).unwrap(),
        2
    );
    assert!(sampler.workspace.is_none());
}

/// Isolates selection/readback from model forward. Event intervals cover each
/// host-driven loop, including submission gaps; they are not pure kernel times.
#[test]
#[ignore = "local GPU selection benchmark; run outside serving measurements"]
fn measures_cuda_greedy_against_cpu_readback() {
    use candle::cuda_backend::cudarc::driver::sys;
    use std::time::Instant;
    let device = Device::new_cuda(0).unwrap();
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    let event = || {
        stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap()
    };
    let (start, end) = (event(), event());
    let mut results = Vec::new();
    for dtype in [DType::BF16, DType::F32] {
        let logits = Tensor::new(
            (0..248320)
                .map(|i| (i % 1023) as f32 / 128.)
                .collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape((1, 1, 248320))
        .unwrap();
        let mut sampler = GpuGreedy::default();
        for _ in 0..10 {
            compare(&mut sampler, &logits);
        }
        let mut rows = Vec::new();
        for gpu in [false, true, true, false] {
            device.synchronize().unwrap();
            let clock = Instant::now();
            start.record(&stream).unwrap();
            for _ in 0..200 {
                let token = if gpu {
                    sampler.select(&logits).unwrap().unwrap()
                } else {
                    next_token_from_values(&logits_values(&logits).unwrap()).unwrap()
                };
                std::hint::black_box(token);
            }
            end.record(&stream).unwrap();
            end.synchronize().unwrap();
            rows.push(serde_json::json!({"gpu":gpu,"host_us":clock.elapsed().as_secs_f64()*1e6/200.,
                "stream_us":f64::from(start.elapsed_ms(&end).unwrap())*1000./200.,
                "readback_bytes":if gpu {8} else {248320*4}}));
        }
        results.push(serde_json::json!({"dtype":format!("{dtype:?}"),"samples":rows}));
    }
    println!(
        "greedy_benchmark {}",
        serde_json::json!({"vocabulary":248320,"results":results,
        "scope":"Selection and readback on ready logits; stream interval includes gaps between host-driven iterations"})
    );
}

/// Readback cannot become a graph node. Reject it before allocating sampler
/// storage or enqueueing work, leaving the caller's capture valid and empty.
#[test]
fn cuda_greedy_rejects_host_readback_inside_capture() {
    // SAFETY: This fresh device and every tensor stay on the test's one stream.
    let device = unsafe { tau_models::new_capture_device(0) }.unwrap();
    let logits = Tensor::new(&[[[1f32, 2.]]], &device).unwrap();
    device.synchronize().unwrap();
    let stream = device.as_cuda_device().unwrap().cuda_stream();
    let mut sampler = GpuGreedy::default();
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let result = sampler.select(&logits);
    let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH).unwrap();
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("outside CUDA capture")
    );
    assert!(sampler.workspace.is_none());
    drop(graph);
}
