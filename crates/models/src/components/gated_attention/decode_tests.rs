//! Decode contracts established against the expanded-KV implementation before
//! changing attention dispatch. The scalar oracle does not call Candle matmul.
use super::*;
use half::bf16;

fn close(actual: &Tensor, expected: &[f32], absolute: f32, relative: f32) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "nonfinite element {index}");
        assert!(
            (a - b).abs() <= absolute + relative * b.abs(),
            "element {index}: {a} != {b}, error {}",
            (a - b).abs()
        );
    }
}

fn random_values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) % 2001) as f32 / 2000.0 - 0.5
        })
        .collect()
}

/// Native unoptimized FP32 attention remains the arithmetic golden. The
/// prepared operator is also tested exactly against itself under graph replay.
#[test]
fn prepared_attention_matches_native_fp32_with_dynamic_length_and_poisoned_tail() {
    use crate::components::{CapturedGraph, DecodeProgram, write_metadata};
    // SAFETY: This test serializes every tensor and operation on one stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    for (kv_heads, groups, dim) in [(1, 6, 64), (4, 4, 128), (4, 4, 256), (4, 6, 256)] {
        let length = 1025;
        let capacity = length + 17;
        for scale in [1., 4., 16.] {
            let (q, k, v) = inputs(&device, DType::BF16, kv_heads, groups, dim, length);
            let q = (q.contiguous().unwrap() * scale).unwrap();
            let k = (k.contiguous().unwrap() * scale).unwrap();
            let v = v.contiguous().unwrap();
            let before = (values(&q), values(&k), values(&v));
            for block_size in [
                crate::AttentionBlockSize::Tokens32,
                crate::AttentionBlockSize::Tokens64,
                crate::AttentionBlockSize::Tokens128,
                crate::AttentionBlockSize::Tokens256,
            ] {
                for transposed in [false, true] {
                    let key =
                        Tensor::full(f32::NAN, (1, kv_heads, capacity, dim), &device)
                            .unwrap()
                            .to_dtype(DType::BF16)
                            .unwrap();
                    let value =
                        Tensor::full(f32::NAN, (1, kv_heads, capacity, dim), &device)
                            .unwrap()
                            .to_dtype(DType::BF16)
                            .unwrap();
                    if transposed {
                        key.reshape((1, kv_heads, dim, capacity))
                            .unwrap()
                            .slice_set(
                                &k.transpose(2, 3).unwrap().contiguous().unwrap(),
                                3,
                                0,
                            )
                            .unwrap();
                    } else {
                        key.slice_set(&k, 2, 0).unwrap();
                    }
                    value.slice_set(&v, 2, 0).unwrap();
                    let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
                    let mut program = DecodeProgram::new(&device).unwrap();
                    let output = program
                        .attention(
                            &q,
                            &key,
                            &value,
                            &metadata,
                            kv_heads * groups,
                            kv_heads,
                            dim,
                            capacity,
                            transposed,
                            block_size,
                            false,
                        )
                        .unwrap();
                    write_metadata(&metadata, [0, length as u32 - 1]).unwrap();
                    program.run().unwrap();
                    device.synchronize().unwrap();
                    // SAFETY: Program owns all retained input/output slots; no allocation
                    // or host length is captured. Metadata updates occur outside capture.
                    let mut graph =
                        unsafe { CapturedGraph::capture(&device, || program.run()) }
                            .unwrap();
                    assert_eq!(graph.allocation_nodes().unwrap(), 0);
                    for n in [
                        1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023,
                        1025, 1,
                    ] {
                        write_metadata(&metadata, [0, n - 1]).unwrap();
                        program.run().unwrap();
                        let eager = values(&output);
                        graph.replay().unwrap();
                        assert_eq!(values(&output), eager);
                        let fp32 = |t: &Tensor| {
                            t.to_device(&Device::Cpu)
                                .unwrap()
                                .to_dtype(DType::F32)
                                .unwrap()
                        };
                        let reference = layer_shape(
                            &Device::Cpu,
                            DType::F32,
                            groups,
                            kv_heads,
                            dim,
                        )
                        .attention_context(
                            &fp32(&q),
                            &fp32(&k).narrow(2, 0, n as usize).unwrap(),
                            &fp32(&v).narrow(2, 0, n as usize).unwrap(),
                            None,
                        )
                        .unwrap();
                        close(&output, &values(&reference), 0.008, 0.0);
                    }
                }
            }
            assert_eq!(before, (values(&q), values(&k), values(&v)));
        }
    }
}

/// Capacity 8192 needs up to 256 splits. Poison KV tails and change value markers
/// between replays so empty tiles must overwrite stale scratch after a short reset.
#[test]
fn prepared_attention_maximum_splits_preserve_grouping_and_poisoned_resets() {
    use crate::components::{CapturedGraph, DecodeProgram, write_metadata};
    // SAFETY: One private stream owns every buffer and all mutations in this test.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    // Racecheck retains per-thread access histories. Bound its diagnostic run to
    // one head and short live prefixes while preserving the 256-split combine
    // boundary. Ordinary tests and memcheck still cover the full live capacity,
    // production grouping and width.
    let bounded = std::env::var_os("TAU_TEST_ATTENTION_SANITIZER_SMALL").is_some();
    let (heads, kv_heads, dim) = if bounded { (1, 1, 64) } else { (24, 4, 256) };
    let query = Tensor::zeros((1, heads, 1, dim), DType::BF16, &device).unwrap();
    for (block_size, capacity) in [
        crate::AttentionBlockSize::Tokens32,
        crate::AttentionBlockSize::Tokens64,
        crate::AttentionBlockSize::Tokens128,
        crate::AttentionBlockSize::Tokens256,
    ]
    .into_iter()
    .filter(|tile| !bounded || *tile == crate::AttentionBlockSize::Tokens32)
    .flat_map(|tile| [4097, 8192].map(|capacity| (tile, capacity)))
    {
        let key =
            Tensor::zeros((1, kv_heads, capacity, dim), DType::BF16, &device).unwrap();
        let value = key.copy().unwrap();
        let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
        let mut program = DecodeProgram::new(&device).unwrap();
        let output = program
            .attention(
                &query, &key, &value, &metadata, heads, kv_heads, dim, capacity, false,
                block_size, false,
            )
            .unwrap();
        write_metadata(&metadata, [0, 0]).unwrap();
        program.run().unwrap();
        device.synchronize().unwrap();
        // SAFETY: The program retains fixed allocations on its original stream.
        let mut graph =
            unsafe { CapturedGraph::capture(&device, || program.run()) }.unwrap();
        assert_eq!(graph.allocation_nodes().unwrap(), 0);
        let lengths = if bounded {
            [65, 33, 1, 65, 63, 33, 1]
        } else {
            [capacity, 65, 1, 4097, 63, capacity, 1]
        };
        for (request, length) in lengths.into_iter().enumerate() {
            let poison = Tensor::full(f32::NAN, key.shape(), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            key.slice_set(&poison, 2, 0).unwrap();
            value.slice_set(&poison, 2, 0).unwrap();
            let keys = Tensor::zeros((1, kv_heads, length, dim), DType::BF16, &device)
                .unwrap();
            let markers: Vec<f32> = (0..kv_heads)
                .flat_map(|head| vec![(head + request + 1) as f32 / 32.; length * dim])
                .collect();
            let live = Tensor::from_vec(markers, (1, kv_heads, length, dim), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            key.slice_set(&keys, 2, 0).unwrap();
            value.slice_set(&live, 2, 0).unwrap();
            write_metadata(&metadata, [0, length as u32 - 1]).unwrap();
            graph.replay().unwrap();
            let expected: Vec<f32> = (0..kv_heads)
                .flat_map(|head| {
                    vec![(head + request + 1) as f32 / 32.; heads / kv_heads * dim]
                })
                .collect();
            close(&output, &expected, 0.008, 0.0);
            let replay = values(&output);
            program.run().unwrap();
            assert_eq!(values(&output), replay);
        }
    }
}

/// Independent FP64 dot products/softmax, with the production dtype boundaries:
/// rounded QK, rounded scaling, FP32 probabilities cast to BF16, rounded PV.
fn scalar_attention(query: &Tensor, key: &Tensor, value: &Tensor) -> Vec<f32> {
    let (_, qheads, _, dim) = query.dims4().unwrap();
    let (_, kvheads, length, _) = key.dims4().unwrap();
    let groups = qheads / kvheads;
    let round = |v: f64| {
        if query.dtype() == DType::BF16 {
            bf16::from_f32(v as f32).to_f32() as f64
        } else {
            (v as f32) as f64
        }
    };
    let (q, k, v) = (values(query), values(key), values(value));
    let mut output = vec![0f32; qheads * dim];
    for head in 0..qheads {
        let kvhead = head / groups;
        let mut scores = vec![0f64; length];
        for (token, score) in scores.iter_mut().enumerate() {
            let dot: f64 = (0..dim)
                .map(|d| {
                    f64::from(q[head * dim + d])
                        * f64::from(k[(kvhead * length + token) * dim + d])
                })
                .sum();
            *score = round(round(dot) / (dim as f64).sqrt());
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = scores.iter().map(|s| (s - maximum).exp()).sum();
        let probabilities: Vec<_> = scores
            .iter()
            .map(|s| round(f64::from(((s - maximum).exp() / denominator) as f32)))
            .collect();
        for d in 0..dim {
            let sum: f64 = (0..length)
                .map(|token| {
                    probabilities[token]
                        * f64::from(v[(kvhead * length + token) * dim + d])
                })
                .sum();
            output[head * dim + d] = round(sum) as f32;
        }
    }
    output
}

/// Nonzero storage offsets and excess capacity must not expose poisoned padding.
fn inputs(
    device: &Device,
    dtype: DType,
    heads: usize,
    groups: usize,
    dim: usize,
    length: usize,
) -> (Tensor, Tensor, Tensor) {
    let query = Tensor::from_vec(
        random_values(heads * groups * dim, 17),
        (1, heads * groups, 1, dim),
        device,
    )
    .unwrap()
    .to_dtype(dtype)
    .unwrap();
    let make_cache = |seed| {
        let data = Tensor::from_vec(
            random_values(heads * length * dim, seed),
            (1, heads, length, dim),
            device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let capacity = (length + 1).next_power_of_two() + 7;
        let storage = Tensor::full(f32::NAN, (1, heads, capacity, dim), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        storage.slice_set(&data, 2, 1).unwrap();
        storage.narrow(2, 1, length).unwrap()
    };
    let capacity = (length + 1).next_power_of_two() + 7;
    if capacity < GROUPED_DECODE_MIN_CONTEXT
        || !device.is_cuda()
        || dtype != DType::BF16
    {
        return (query, make_cache(43), make_cache(79));
    }
    let key_data = make_cache(43).contiguous().unwrap();
    let key_storage = Tensor::full(f32::NAN, (1, heads, dim, capacity), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    key_storage
        .slice_set(
            &key_data.transpose(2, 3).unwrap().contiguous().unwrap(),
            3,
            1,
        )
        .unwrap();
    let key = key_storage
        .narrow(3, 1, length)
        .unwrap()
        .transpose(2, 3)
        .unwrap();
    (query, key, make_cache(79))
}

fn check_scalar_oracle(device: &Device, dtype: DType) {
    // Includes production head counts/dimensions and odd/cache-boundary lengths.
    for (heads, groups, dim, length) in [
        (2, 1, 8, 1),
        (2, 2, 8, 3),
        (2, 2, 64, 257),
        (4, 4, 256, 1),
        (4, 4, 256, 31),
        (4, 4, 256, 32),
        (4, 4, 256, 33),
        (4, 4, 256, 127),
        (4, 4, 256, 129),
        (4, 4, 256, 255),
        (4, 4, 256, 256),
        (4, 4, 256, 257),
        (4, 4, 256, 513),
        (4, 4, 256, 1024),
        (4, 4, 256, 4096),
        (4, 6, 256, 129),
        (4, 6, 256, 1023),
    ] {
        let layer = layer_shape(device, dtype, groups, heads, dim);
        let (q, k, v) = inputs(device, dtype, heads, groups, dim, length);
        let expected = scalar_attention(&q, &k, &v);
        let before = (values(&q), values(&k), values(&v));
        let actual = layer.attention_context(&q, &k, &v, None).unwrap();
        assert_eq!(actual.dims(), [1, 1, heads * groups * dim]);
        assert_eq!(actual.dtype(), dtype);
        let tolerance = if dtype == DType::BF16 { 0.008 } else { 2e-5 };
        close(&actual, &expected, tolerance, 0.0);
        assert_eq!(before, (values(&q), values(&k), values(&v)));
    }
}

#[test]
fn decode_matches_independent_oracle_cpu() {
    check_scalar_oracle(&Device::Cpu, DType::F32);
}

#[test]
fn decode_matches_independent_oracle_cuda_bf16() {
    check_scalar_oracle(&Device::new_cuda(0).unwrap(), DType::BF16);
}

#[test]
fn decode_maps_each_query_group_to_the_correct_kv_head() {
    let device = Device::new_cuda(0).unwrap();
    for groups in [1, 4, 6] {
        let layer = layer_shape(&device, DType::BF16, groups, 4, 8);
        let q = Tensor::zeros((1, 4 * groups, 1, 8), DType::BF16, &device).unwrap();
        let k = Tensor::zeros((1, 4, 8, 512), DType::BF16, &device)
            .unwrap()
            .transpose(2, 3)
            .unwrap();
        let markers: Vec<f32> = (0..4)
            .flat_map(|h| vec![(h + 1) as f32 / 8.0; 512 * 8])
            .collect();
        let v = Tensor::from_vec(markers, (1, 4, 512, 8), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let expected: Vec<f32> = (0..4)
            .flat_map(|h| vec![(h + 1) as f32 / 8.0; groups * 8])
            .collect();
        close(
            &layer.attention_context(&q, &k, &v, None).unwrap(),
            &expected,
            0.,
            0.,
        );
    }
}

#[test]
fn unmasked_decode_preserves_prefill_tail_reset_and_cache_contents() {
    let device = Device::new_cuda(0).unwrap();
    for groups in [1, 4, 6] {
        let mut layer = layer_shape(&device, DType::BF16, groups, 4, 256);
        layer.reserve_kv_cache(512).unwrap();
        let ids = {
            let (k, v) = layer.kv_cache.buffers.as_ref().unwrap();
            (k.id(), v.id())
        };
        let mut first_outputs = Vec::new();
        for (request, (phase, chunks)) in [
            (7, vec![257, 3, 1, 1, 1]),
            (29, vec![1, 1, 1]),
            (7, vec![257, 3, 1, 1, 1]),
        ]
        .into_iter()
        .enumerate()
        {
            layer.clear_cache();
            layer.reserve_kv_cache(384).unwrap();
            for buffer in [
                &layer.kv_cache.buffers.as_ref().unwrap().0,
                &layer.kv_cache.buffers.as_ref().unwrap().1,
            ] {
                let poison = Tensor::full(f32::NAN, buffer.shape(), &device)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                buffer.slice_set(&poison, 2, 0).unwrap();
            }
            let (mut reference, mut offset) = (None, 0);
            let mut outputs = Vec::new();
            for tokens in chunks {
                let xs =
                    patterned(&[1, tokens, 16], phase + offset, DType::BF16, &device);
                let mask =
                    (tokens > 1).then(|| mask(tokens, offset, DType::BF16, &device));
                let expected = reference_forward(
                    &layer,
                    &xs,
                    mask.as_ref(),
                    offset,
                    &mut reference,
                )
                .unwrap();
                let actual = layer.forward(&xs, mask.as_ref(), offset).unwrap();
                // Existing dense BF16 model oracle has a 0.05 absolute bound.
                close(&actual, &values(&expected), 0.05, 0.0);
                outputs.push(values(&actual));
                let (k, v) = layer.kv_cache.prefix().unwrap();
                let (rk, rv) = reference.as_ref().unwrap();
                exact(&k, rk);
                exact(&v, rv);
                offset += tokens;
            }
            if request == 0 {
                first_outputs = outputs;
            } else if request == 2 {
                assert_eq!(first_outputs, outputs);
            }
            let (k, v) = layer.kv_cache.buffers.as_ref().unwrap();
            assert_eq!((k.id(), v.id()), ids);
        }
    }
}

#[test]
fn grouped_decode_dispatch_preserves_reference_fallbacks() {
    let device = Device::new_cuda(0).unwrap();
    let mut layer = layer_shape(&device, DType::BF16, 4, 4, 256);
    let (q, k, v) = inputs(&device, DType::BF16, 4, 4, 256, 257);
    assert!(layer.supports_grouped_decode(&q, &k, &v, None));
    layer.optimize_kernel = false;
    assert!(!layer.supports_grouped_decode(&q, &k, &v, None));
    let expected = layer.attention_context(&q, &k, &v, None).unwrap();
    layer.optimize_kernel = true;
    let actual = layer.attention_context(&q, &k, &v, None).unwrap();
    close(&actual, &values(&expected), 0.008, 0.0);

    let mask = Tensor::zeros((1, 1, 1, 257), DType::BF16, &device).unwrap();
    assert!(!layer.supports_grouped_decode(&q, &k, &v, Some(&mask)));
    exact(
        &layer.attention_context(&q, &k, &v, Some(&mask)).unwrap(),
        &expected,
    );
    let strided_k = k.contiguous().unwrap();
    assert!(!layer.supports_grouped_decode(&q, &strided_k, &v, None));
    exact(
        &layer.attention_context(&q, &strided_k, &v, None).unwrap(),
        &expected,
    );

    // Packing only the small query is permitted; the retained K/V stay strided.
    let strided_q = Tensor::cat(&[&q, &q], 3)
        .unwrap()
        .narrow(3, 0, 256)
        .unwrap();
    assert!(!strided_q.is_contiguous());
    assert!(layer.supports_grouped_decode(&strided_q, &k, &v, None));
    exact(
        &layer.attention_context(&strided_q, &k, &v, None).unwrap(),
        &actual,
    );
    let (qf, kf, vf) = (
        q.to_dtype(DType::F32).unwrap(),
        k.to_dtype(DType::F32).unwrap(),
        v.to_dtype(DType::F32).unwrap(),
    );
    assert!(!layer.supports_grouped_decode(&qf, &kf, &vf, None));
    close(
        &layer.attention_context(&qf, &kf, &vf, None).unwrap(),
        &scalar_attention(&qf, &kf, &vf),
        2e-5,
        0.0,
    );
    let qb = Tensor::cat(&[&q, &q], 0).unwrap();
    let kb = Tensor::cat(&[&k, &k], 0).unwrap();
    let vb = Tensor::cat(&[&v, &v], 0).unwrap();
    assert!(!layer.supports_grouped_decode(&qb, &kb, &vb, None));
    let expected_batch = Tensor::cat(&[&expected, &expected], 0).unwrap();
    close(
        &layer.attention_context(&qb, &kb, &vb, None).unwrap(),
        &values(&expected_batch),
        0.008,
        0.0,
    );
}

#[test]
#[ignore = "release CUDA attention operator benchmark; run separately from GPU tests"]
fn measures_decode_attention_context() {
    use crate::components::{CapturedGraph, DecodeProgram, write_metadata};
    use candle::cuda_backend::cudarc::driver::sys;
    use std::time::Instant;
    if cfg!(debug_assertions) {
        panic!("use cargo test --release");
    }
    // SAFETY: All resources remain on this thread and the single Candle stream.
    let device =
        unsafe { crate::components::cuda::graph::new_capture_device(0) }.unwrap();
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
    for groups in [4, 6] {
        let layer = layer_shape(&device, DType::BF16, groups, 4, 256);
        for length in [64, 1024, 4096] {
            let (q, k, v) = inputs(&device, DType::BF16, 4, groups, 256, length);
            for _ in 0..16 {
                std::hint::black_box(
                    layer.attention_context(&q, &k, &v, None).unwrap(),
                );
            }
            let mut samples = Vec::new();
            for _ in 0..6 {
                device.synchronize().unwrap();
                let wall = Instant::now();
                start.record(&stream).unwrap();
                for _ in 0..64 {
                    std::hint::black_box(
                        layer.attention_context(&q, &k, &v, None).unwrap(),
                    );
                }
                end.record(&stream).unwrap();
                end.synchronize().unwrap();
                samples.push(serde_json::json!({
                    "gpu_us":f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0 / 64.0,
                    "wall_us":wall.elapsed().as_secs_f64() * 1e6 / 64.0,
                }));
            }
            cases.push(
                serde_json::json!({"groups":groups,"context":length,"samples":samples}),
            );
        }
    }
    let mut prepared_cases = Vec::new();
    for groups in [4, 6] {
        for (length, capacity) in [(192, 512), (1152, 2048)] {
            let (q, k, v) = inputs(&device, DType::BF16, 4, groups, 256, length);
            let q = q.contiguous().unwrap();
            let key =
                Tensor::zeros((1, 4, capacity, 256), DType::BF16, &device).unwrap();
            let value = key.copy().unwrap();
            key.slice_set(&k.contiguous().unwrap(), 2, 0).unwrap();
            value.slice_set(&v.contiguous().unwrap(), 2, 0).unwrap();
            for block_size in [
                crate::AttentionBlockSize::Tokens256,
                crate::AttentionBlockSize::Tokens128,
                crate::AttentionBlockSize::Tokens64,
                crate::AttentionBlockSize::Tokens32,
            ] {
                let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
                write_metadata(&metadata, [0, length as u32 - 1]).unwrap();
                let mut program = DecodeProgram::new(&device).unwrap();
                let output = program
                    .attention(
                        &q,
                        &key,
                        &value,
                        &metadata,
                        groups * 4,
                        4,
                        256,
                        capacity,
                        false,
                        block_size,
                        false,
                    )
                    .unwrap();
                program.run().unwrap();
                device.synchronize().unwrap();
                // SAFETY: The program owns fixed slots on this private stream.
                let mut graph =
                    unsafe { CapturedGraph::capture(&device, || program.run()) }
                        .unwrap();
                for _ in 0..16 {
                    graph.replay().unwrap();
                }
                assert!(values(&output).iter().all(|v| v.is_finite()));
                let mut samples = Vec::new();
                for _ in 0..6 {
                    device.synchronize().unwrap();
                    start.record(&stream).unwrap();
                    for _ in 0..256 {
                        graph.replay().unwrap();
                    }
                    end.record(&stream).unwrap();
                    end.synchronize().unwrap();
                    samples.push(
                        f64::from(start.elapsed_ms(&end).unwrap()) * 1000. / 256.,
                    );
                }
                prepared_cases.push(serde_json::json!({
                    "query_heads":groups * 4,"kv_heads":4,"head_dim":256,
                    "live_length":length,"capacity":capacity,
                    "attention_block_size":block_size.tokens(),
                    "threads_per_block":256,
                    "launched_blocks":groups * 4 * capacity.div_ceil(block_size.tokens()),
                    "useful_blocks":groups * 4 * length.div_ceil(block_size.tokens()),
                    "gpu_us_per_replay":samples,
                    "iterations":256,"rounds":6,
                    "allocation_nodes":graph.allocation_nodes().unwrap(),
                }));
            }
        }
    }
    println!(
        "attention_context_benchmark {}",
        serde_json::json!({
            "scope":"Single-layer warmed eager attention math; includes temporaries, \
             excludes projections/norm/RoPE/cache append/gating; no server throughput \
             claim",
            "dtype": "BF16",
            "kv_heads": 4,
            "head_dim": 256,
            "iterations": 64,
            "rounds": 6,
            "cases": cases,
            "prepared_scope":"Warmed captured attention plus combine only; \
                excludes projections, Q/K norm, RoPE, KV append, gating and \
                serving. Block counts are geometry, not measured utilization.",
            "prepared_cases":prepared_cases
        })
    );
}
