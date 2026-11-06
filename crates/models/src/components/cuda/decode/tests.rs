use super::*;
use crate::components::{CapturedGraph, PartialRotaryConfig, PartialRotaryEmbedding};
use candle::Module;

/// Reads a flattened FP32 view so tests can compare every output element exactly.
fn values(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Covers negative, positive, near-zero and saturated scalar gates, including a
/// partial final CUDA block. Replay must preserve the ordinary BF16 sigmoid
/// intermediate instead of multiplying by an unrounded FP32 activation.
#[test]
fn prepared_sigmoid_gates_preserve_bf16_rounding_across_replay() {
    // SAFETY: Every allocation and graph operation uses this private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let input = Tensor::new(
        (0..257).map(|i| (i as f32 - 128.) / 7.).collect::<Vec<_>>(),
        &device,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let gate = candle::Var::zeros(1, DType::BF16, &device).unwrap();
    let packed = candle::Var::zeros(514, DType::BF16, &device).unwrap();
    let mut program = DecodeProgram::new(&device).unwrap();
    let output = program.scalar_gate(&input, &gate).unwrap();
    let vector_output = program.buffer(input.shape().clone()).unwrap();
    let vector_output = program
        .elementwise(3, &[&input, &packed], vector_output, 257, 257, 0, false)
        .unwrap();
    program.run().unwrap();
    device.synchronize().unwrap();
    // SAFETY: Input, gate, output and program outlive this borrowed graph.
    let mut graph =
        unsafe { CapturedGraph::capture(&device, || program.run()) }.unwrap();
    assert_eq!(graph.allocation_nodes().unwrap(), 0);
    for value in [
        -100.,
        -20.,
        -9.,
        -std::f32::consts::PI,
        -1.,
        0.,
        0.0001,
        0.1,
        0.5,
        1.7,
        8.,
        20.,
        100.,
    ] {
        gate.set(
            &Tensor::new(&[value], &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        )
        .unwrap();
        packed
            .set(&gate.broadcast_as(514).unwrap().contiguous().unwrap())
            .unwrap();
        graph.replay().unwrap();
        let expected = input
            .broadcast_mul(&candle_nn::ops::sigmoid(&gate).unwrap())
            .unwrap();
        assert_eq!(values(&output), values(&expected), "gate {value}");
        assert_eq!(
            values(&vector_output),
            values(&expected),
            "vector gate {value}"
        );
    }
}

/// Checks prepared projections/norm/SiLU against their existing operators, then
/// changes RoPE positions across boundaries in one allocation-free graph. Exact
/// output checks catch altered arithmetic or a position frozen during capture.
#[test]
fn prepared_projection_norm_silu_and_dynamic_rope_preserve_operator_contracts() {
    // SAFETY: All resources stay on this thread and private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let tensor = |count, seed| {
        Tensor::new(
            (0..count)
                .map(|i| ((i * 17 + seed) % 101) as f32 / 101. - 0.5)
                .collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
    };
    let input = tensor(256, 9).reshape((1, 1, 256)).unwrap();
    let layer = candle_nn::Linear::new(
        tensor(128 * 256, 31).reshape((128, 256)).unwrap(),
        None,
    );
    let mut program = DecodeProgram::new(&device).unwrap();
    let projected = program.linear(&input, &layer).unwrap();
    let weight = tensor(128, 21);
    let norm = program.norm(&projected, &weight, 1e-6).unwrap();
    let silu = program.silu(&norm).unwrap();
    program.run().unwrap();
    assert_eq!(values(&projected), values(&layer.forward(&input).unwrap()));
    let expected =
        crate::components::CenteredRmsNorm::from_weight_for_test(weight, 1e-6, true)
            .forward(&projected)
            .unwrap();
    assert_eq!(values(&norm), values(&expected));
    let expected = super::super::silu_and_mul_cuda(&norm).unwrap();
    assert_eq!(values(&silu), values(&expected));
    let rope = PartialRotaryEmbedding::new(
        &PartialRotaryConfig {
            head_dim: 64,
            rotary_dim: 16,
            max_position_embeddings: 1024,
            theta: 10000.,
        },
        DType::BF16,
        &device,
    )
    .unwrap();
    let query = input.reshape((1, 4, 1, 64)).unwrap();
    let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
    let mut rotation = DecodeProgram::new(&device).unwrap();
    let rotated = rope
        .prepare_decode_program(&mut rotation, &query, &metadata, 64)
        .unwrap();
    write_metadata(&metadata, [0, 0]).unwrap();
    rotation.run().unwrap();
    device.synchronize().unwrap();
    // SAFETY: Every output allocation predates capture and remains owned here.
    let mut graph =
        unsafe { CapturedGraph::capture(&device, || rotation.run()) }.unwrap();
    assert_eq!(graph.allocation_nodes().unwrap(), 0);
    for position in [0, 1, 255, 256, 257, 1023, 0] {
        write_metadata(&metadata, [0, position]).unwrap();
        graph.replay().unwrap();
        let expected = rope.apply(&query, &query, position as usize).unwrap().0;
        assert_eq!(values(&rotated), values(&expected));
    }
}

/// Production down projections can need cuBLAS scratch even after warm-up.
/// Keep their capture allocation-free across input changes, ordinary eager work
/// and queued teardown; the two projections share one serialized workspace.
#[test]
fn prepared_production_projection_replays_without_allocations() {
    const INPUT: usize = 9216;
    const OUTPUT: usize = 2560;
    // SAFETY: The program, graph and ordinary eager work use one private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    // Dyadic values keep these dot products exact in FP32 so this lifetime and
    // allocation regression does not depend on vendor reduction-tree choices.
    let tensor = |count, seed| {
        Tensor::new(
            (0..count)
                .map(|i| (((i * 17 + seed) % 31) as f32 - 15.) / 64.)
                .collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
    };
    let tokens = tensor(2 * INPUT, 9).reshape((2, INPUT)).unwrap();
    let layer = candle_nn::Linear::new(
        tensor(OUTPUT * INPUT, 31).reshape((OUTPUT, INPUT)).unwrap(),
        None,
    );
    let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
    let expected = |token| {
        let input = tokens
            .narrow(0, token, 1)
            .unwrap()
            .reshape((1, 1, INPUT))
            .unwrap();
        values(&layer.forward(&input).unwrap())
    };
    let references = [expected(0), expected(1)];
    let mut program = DecodeProgram::new(&device).unwrap();
    let input = program.buffer((1, 1, INPUT)).unwrap();
    let input = program
        .elementwise(0, &[&tokens, &metadata], input, INPUT, INPUT, 0, false)
        .unwrap();
    let first = program.linear(&input, &layer).unwrap();
    let second = program.linear(&input, &layer).unwrap();
    for token in [0u32, 1, 0] {
        write_metadata(&metadata, [token, 0]).unwrap();
        program.run().unwrap();
        assert_eq!(values(&first), references[token as usize]);
        assert_eq!(values(&second), references[token as usize]);
    }
    device.synchronize().unwrap();
    // SAFETY: The closure retains the program, handle and scratch until graph
    // teardown synchronizes outstanding replay. All tensor addresses are stable.
    let mut graph =
        unsafe { CapturedGraph::capture(&device, move || program.run()) }.unwrap();
    assert_eq!(graph.allocation_nodes().unwrap(), 0);
    for token in [1u32, 0, 1, 0] {
        write_metadata(&metadata, [token, 0]).unwrap();
        graph.replay().unwrap();
        assert_eq!(values(&first), references[token as usize]);
        assert_eq!(values(&second), references[token as usize]);
        // Ordinary inference uses its own handle before another replay.
        assert_eq!(expected(token as usize), references[token as usize]);
    }
    graph.replay().unwrap();
    drop(graph);
    assert_eq!(values(&first), references[0]);
    assert_eq!(values(&second), references[0]);
    assert_eq!(expected(1), references[1]);
}

/// Timing events surround replay on the private stream without becoming graph
/// nodes. Metadata changes and exact logits remain correct across event reuse.
#[test]
fn cuda_events_measure_replays_without_changing_graph_or_outputs() {
    use crate::components::{CapturedGraph, DecodeProgram, write_metadata};
    use tau_metrics::timing::DecodeTimer;
    // SAFETY: This test owns every operation and buffer on one serialized stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let weights = candle::Tensor::arange(0f32, 128., &device)
        .unwrap()
        .reshape((2, 64))
        .unwrap()
        .to_dtype(candle::DType::BF16)
        .unwrap();
    let metadata = candle::Tensor::zeros(2, candle::DType::U32, &device).unwrap();
    let mut program = DecodeProgram::new(&device).unwrap();
    let output = program.buffer((1, 1, 64)).unwrap();
    let output = program
        .elementwise(0, &[&weights, &metadata], output, 64, 64, 0, false)
        .unwrap();
    let output = program.add(&output, &output).unwrap();
    program.run().unwrap();
    device.synchronize().unwrap();
    // SAFETY: Program retains every slot and uses only this stream during capture.
    let mut graph =
        unsafe { CapturedGraph::capture(&device, || program.run()) }.unwrap();
    let mut timer = DecodeTimer::new(&device).unwrap();
    for token in [0, 1, 0, 1] {
        timer.start_step();
        write_metadata(&metadata, [token, 0]).unwrap();
        timer.begin_forward("complete_forward_replay").unwrap();
        graph.replay().unwrap();
        timer.end_forward().unwrap();
        let values = crate::test_support::timed_logits(&output, &mut timer);
        assert_eq!(
            values,
            (token * 64..(token + 1) * 64)
                .map(|i| (i * 2) as f32)
                .collect::<Vec<_>>()
        );
        timer.end_selection();
        let sample = timer.finish(false).unwrap();
        assert!(sample.totals.gpu_forward_us.unwrap().is_finite());
        assert!(sample.totals.gpu_forward_us.unwrap() > 0.0);
        assert!(sample.totals.gpu_logits_to_f32_us.unwrap() >= 0.0);
        assert_eq!(sample.totals.logits_readback_bytes, 256);
        assert_eq!(graph.allocation_nodes().unwrap(), 0);
    }
    timer.start_step();
    // SAFETY: Capture's error cleanup ends recording even when timing rejects it.
    let rejected =
        unsafe { CapturedGraph::capture(&device, || timer.begin_forward("invalid")) };
    assert!(
        rejected
            .err()
            .unwrap()
            .to_string()
            .contains("outside CUDA capture")
    );
    drop(timer);
    graph.replay().unwrap();
    graph.synchronize().unwrap();
}

/// The independent seven-operation program is the same-arithmetic reference.
/// Include sliced inputs/scales to exercise each norm reduction grouping, and
/// poisoned Q gate halves/cache tails to catch incorrect extraction or writes.
#[test]
fn fused_attention_preparation_matches_unfused_with_offsets_and_resets() {
    // SAFETY: All tensors, graphs and writes stay on one private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let bits = |t: &Tensor| values(t).into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let selected_dim = std::env::var("TAU_TEST_ATTENTION_PREP_DIM")
        .ok()
        .map(|s| s.parse::<usize>().unwrap());
    assert!(selected_dim.is_none_or(|d| matches!(d, 64 | 128 | 256)));
    for (heads, kv_heads, dim, rotary_dim) in [
        (6, 1, 64, 16),
        (16, 4, 128, 128),
        (16, 4, 256, 64),
        (24, 4, 256, 256),
    ]
    .into_iter()
    .filter(|(_, _, d, _)| selected_dim.is_none_or(|v| v == *d))
    {
        let capacity = 513;
        let rotary = PartialRotaryEmbedding::new(
            &PartialRotaryConfig {
                head_dim: dim,
                rotary_dim,
                max_position_embeddings: 1024,
                theta: 10_000_000.,
            },
            DType::BF16,
            &device,
        )
        .unwrap();
        for offset in [0, 1, 2, 4] {
            let tensor = |count, phase| {
                Tensor::new(
                    (0..count + offset)
                        .map(|i| (((i * 17 + phase) % 257) as f32 - 128.) / 64.)
                        .collect::<Vec<_>>(),
                    &device,
                )
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .narrow(0, offset, count)
                .unwrap()
            };
            let mut q_data = vec![f32::NAN; heads * 2 * dim + offset];
            for head in 0..heads {
                for d in 0..dim {
                    q_data[offset + head * 2 * dim + d] =
                        (((head * 31 + d * 7) % 257) as f32 - 128.) / 64.;
                }
            }
            let query_gate = Tensor::new(q_data, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .narrow(0, offset, heads * 2 * dim)
                .unwrap();
            let key = tensor(kv_heads * dim, 43);
            let value = tensor(kv_heads * dim, 79);
            let qw = tensor(dim, 11);
            let kw = tensor(dim, 29);
            let immutable = [&query_gate, &key, &value, &qw, &kw]
                .iter()
                .map(|t| bits(t))
                .collect::<Vec<_>>();
            for transposed in [false, true] {
                let cache = || {
                    Tensor::zeros((kv_heads, capacity, dim), DType::BF16, &device)
                        .unwrap()
                };
                let (rk, rv, fk, fv) = (cache(), cache(), cache(), cache());
                let metadata = Tensor::zeros(2, DType::U32, &device).unwrap();
                write_metadata(&metadata, [0, 0]).unwrap();
                let (q_epsilon, k_epsilon) = (1e-6, 3e-5);
                let mut reference = DecodeProgram::new(&device).unwrap();
                let q = reference.buffer((1, 1, heads * dim)).unwrap();
                let q = reference
                    .elementwise(2, &[&query_gate], q, heads * dim, dim, 0, false)
                    .unwrap();
                let q = reference.norm(&q, &qw, q_epsilon).unwrap();
                let k = reference.norm(&key, &kw, k_epsilon).unwrap();
                let q = rotary
                    .prepare_decode_program(&mut reference, &q, &metadata, dim)
                    .unwrap();
                let k = rotary
                    .prepare_decode_program(&mut reference, &k, &metadata, dim)
                    .unwrap();
                reference
                    .elementwise(
                        4,
                        &[&k, &metadata],
                        rk.clone(),
                        kv_heads * dim,
                        dim,
                        capacity,
                        transposed,
                    )
                    .unwrap();
                reference
                    .elementwise(
                        4,
                        &[&value, &metadata],
                        rv.clone(),
                        kv_heads * dim,
                        dim,
                        capacity,
                        false,
                    )
                    .unwrap();
                let mut fused = DecodeProgram::new(&device).unwrap();
                let output = fused
                    .attention_prepare(
                        &query_gate,
                        &key,
                        &value,
                        (&qw, q_epsilon),
                        (&kw, k_epsilon),
                        rotary.prepared_parameters(),
                        (&fk, &fv),
                        &metadata,
                        heads,
                        kv_heads,
                        dim,
                        capacity,
                        transposed,
                    )
                    .unwrap();
                fused.run().unwrap();
                device.synchronize().unwrap();
                // SAFETY: Fused program retains every fixed buffer on this stream.
                let mut graph =
                    unsafe { CapturedGraph::capture(&device, || fused.run()) }.unwrap();
                assert_eq!(graph.allocation_nodes().unwrap(), 0);
                let poison = Tensor::full(f32::NAN, rk.shape(), &device)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                for position in [0, 63, 64, 255, 512, 0] {
                    for buffer in [&rk, &rv, &fk, &fv] {
                        buffer.slice_set(&poison, 1, 0).unwrap();
                    }
                    write_metadata(&metadata, [0, position]).unwrap();
                    reference.run().unwrap();
                    graph.replay().unwrap();
                    assert_eq!(bits(&output), bits(&q), "Q: dim={dim} offset={offset}");
                    assert_eq!(bits(&fk), bits(&rk), "K: dim={dim} offset={offset}");
                    assert_eq!(bits(&fv), bits(&rv), "V: dim={dim} offset={offset}");
                    let replay = bits(&output);
                    fused.run().unwrap();
                    assert_eq!(bits(&output), replay);
                }
            }
            assert_eq!(
                immutable,
                [&query_gate, &key, &value, &qw, &kw]
                    .iter()
                    .map(|t| bits(t))
                    .collect::<Vec<_>>()
            );
        }
    }
}
