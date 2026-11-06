//! Generation contracts for the shared causal engine.

use super::*;

#[test]
fn graph_policy_violations_abort_and_reset_without_tracing() {
    use crate::CudaGraphMode::{Full, Layer, None};
    use crate::model::CudaGraphStats;
    for (mode, activity, allowed) in [
        (
            None,
            CudaGraphStats {
                mlp_graph_replays: 1,
                ..Default::default()
            },
            false,
        ),
        (
            Layer,
            CudaGraphStats {
                forward_graph_captures: 1,
                ..Default::default()
            },
            false,
        ),
        (
            Full,
            CudaGraphStats {
                mlp_graph_replays: 1,
                ..Default::default()
            },
            true,
        ),
        (Layer, CudaGraphStats::default(), true),
        (Full, CudaGraphStats::default(), true),
    ] {
        let mut engine = sequence_engine(EngineOptions::new().with_cuda_graph(mode));
        engine.model.graph_activity = activity;
        let result = engine.generate_from_prompt("hello", 1);
        assert_eq!(result.is_ok(), allowed);
        assert_eq!(engine.model.resets, 2);
        assert_eq!(engine.model.calls.len(), 1, "must not retry eagerly");
    }
}

/// Token execution has no engine/tokenizer dependency and owns cleanup even for
/// invalid settings, allocation failures and failures midway through prefill.
#[test]
fn token_generation_owns_cleanup_without_an_engine() {
    let CausalModelEngine {
        mut model,
        device,
        generation_config,
        ..
    } = sequence_engine(EngineOptions::new());
    let settings = |prefill_chunk_size| token_generation::GenerationSettings {
        device: &device,
        config: &generation_config,
        prefill_chunk_size,
        cuda_graph: crate::CudaGraphMode::None,
    };
    let mut metrics = InferenceMetrics::default();
    let mut sampler = crate::generation::GpuGreedy::default();
    assert!(
        token_generation::generate(
            &mut model,
            &[1; 5],
            4,
            settings(0),
            &mut metrics,
            &mut sampler
        )
        .is_err()
    );
    assert_eq!(model.resets, 2);
    assert!(model.reservations.is_empty());
    model.fail_reservation = true;
    assert!(
        token_generation::generate(
            &mut model,
            &[1; 5],
            4,
            settings(2),
            &mut metrics,
            &mut sampler
        )
        .is_err()
    );
    assert_eq!(model.resets, 4);
    assert_eq!(model.reservations, [(9, 3)]);
    assert!(model.calls.is_empty());
    model.fail_reservation = false;
    model.fail_decode = true;
    assert!(
        token_generation::generate(
            &mut model,
            &[1; 5],
            4,
            settings(2),
            &mut metrics,
            &mut sampler
        )
        .is_err()
    );
    assert_eq!(model.resets, 6);
    assert_eq!(model.calls, [0, 2]);
    assert_eq!(model.decode_calls, 0);
    model.fail_decode = false;
    model.eos_offset = usize::MAX;
    model.calls.clear();
    assert_eq!(
        token_generation::generate(
            &mut model,
            &[1; 5],
            4,
            settings(2),
            &mut metrics,
            &mut sampler
        )
        .unwrap(),
        [2; 4]
    );
    assert_eq!(model.calls, [0, 2, 4, 5, 6, 7]);
    assert_eq!(model.resets, 8);
}

/// Even a one-token prompt is prefill. Stop conditions must precede decode work.
#[test]
fn explicit_decode_phase_respects_prefill_eos_and_token_limit() {
    for tracing in [false, true] {
        let options = EngineOptions::new()
            .with_cuda_graph(crate::CudaGraphMode::Layer)
            .with_tracing(tracing);
        let mut engine = sequence_engine(options);
        engine.generate_from_prompt("hello", 1).unwrap();
        assert_eq!(engine.model.calls, [0]);
        assert_eq!(engine.model.decode_calls, 0);
        engine.generation_config.eos_token_ids = vec![2];
        engine.generate_from_prompt("hello", 4).unwrap();
        assert_eq!(engine.model.decode_calls, 0);
        engine.generation_config.eos_token_ids = vec![3];
        engine.generate_from_prompt("hello", 4).unwrap();
        assert_eq!(engine.model.decode_calls, 1);
        assert_eq!(engine.model.calls, [0, 0, 0, 1]);
    }
}

/// All instrumentation modes preserve token selection, EOS and cache reset.
#[test]
fn tracing_preserves_generation_and_cache_reset() {
    for tracing in [false, true] {
        for metrics in [false, true] {
            for vram in [false, true] {
                let mut plain = sequence_engine(EngineOptions::new());
                let mut traced = sequence_engine(
                    EngineOptions::new()
                        .with_tracing(tracing)
                        .with_metrics(metrics)
                        .with_vram_metrics(vram),
                );
                for _ in 0..2 {
                    let expected = plain
                        .generate_from_prompt_with_metadata("hello", 4)
                        .unwrap();
                    let actual = traced
                        .generate_from_prompt_with_metadata("hello", 4)
                        .unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(actual.content(), "answer");
                    assert_eq!(actual.completion_tokens(), 1);
                    assert_eq!(actual.finish_reason(), ChatFinishReason::Stop);
                }
                assert_eq!(plain.model.calls, [0, 1, 0, 1]);
                assert_eq!(traced.model.calls, plain.model.calls);
                assert_eq!(traced.model.resets, 4);
            }
        }
    }
}

#[test]
fn prefill_chunks_preserve_offsets_and_sample_only_after_the_tail() {
    let mut engine = sequence_engine(EngineOptions::new().with_prefill_chunk_size(4));
    engine
        .generate_tokens(&[1; 10], 1, &mut InferenceMetrics::default())
        .unwrap();
    assert_eq!(engine.model.calls, [0, 4, 8]);
    assert_eq!(
        engine.model.prefill_calls,
        [
            (0, 4, true, false),
            (4, 4, true, false),
            (8, 2, false, true)
        ]
    );
    assert_eq!(engine.model.decode_calls, 0);
    assert_eq!(engine.model.resets, 2);

    engine.model.calls.clear();
    engine.model.prefill_calls.clear();
    engine
        .generate_tokens(&[1; 8], 1, &mut InferenceMetrics::default())
        .unwrap();
    assert_eq!(
        engine.model.prefill_calls,
        [(0, 4, true, false), (4, 4, true, true)]
    );
}

/// Prefill's first output counts toward the limit; decode starts after the full
/// prompt.
#[test]
fn prefill_and_decode_preserve_token_limits_eos_and_cache_offsets() {
    for (limit, eos_offset, expected_tokens, expected_offsets) in [
        (0, 12, vec![], vec![]),
        (1, 12, vec![2], vec![0, 4, 8]),
        (2, 12, vec![2, 2], vec![0, 4, 8, 10]),
        (3, 12, vec![2, 2, 2], vec![0, 4, 8, 10, 11]),
        (4, 12, vec![2, 2, 2], vec![0, 4, 8, 10, 11, 12]),
        (4, 8, vec![], vec![0, 4, 8]),
    ] {
        let mut engine =
            sequence_engine(EngineOptions::new().with_prefill_chunk_size(4));
        engine.model.eos_offset = eos_offset;
        let generated = engine
            .generate_tokens(&[1; 10], limit, &mut InferenceMetrics::default())
            .unwrap();
        assert_eq!(
            generated, expected_tokens,
            "limit={limit}, eos={eos_offset}"
        );
        assert_eq!(engine.model.calls, expected_offsets);
        assert_eq!(
            engine.model.prefill_calls.len(),
            if limit == 0 { 0 } else { 3 }
        );
        assert_eq!(
            engine.model.decode_calls,
            expected_offsets.len().saturating_sub(3)
        );
        assert_eq!(engine.model.resets, 2);
    }
}

/// Failed decode still clears caches, allowing the next request to recover.
#[test]
fn traced_failure_resets_state_before_the_next_request() {
    let mut engine = sequence_engine(EngineOptions::new().with_tracing(true));
    engine.model.fail_decode = true;
    assert!(
        engine
            .generate_from_prompt_with_metadata("hello", 4)
            .is_err()
    );
    assert_eq!(engine.model.resets, 2);
    engine.model.fail_decode = false;
    assert_eq!(engine.generate_from_prompt("hello", 4).unwrap(), "answer");
    assert_eq!(engine.model.calls, [0, 1, 0, 1]);
    assert_eq!(engine.model.resets, 4);
}

/// Reserve the whole budget after reset, even if EOS will stop the request early.
#[test]
fn reserves_request_budget_before_prefill_and_cleans_up_failure() {
    let mut engine = sequence_engine(EngineOptions::new());
    engine.model.fail_reservation = true;
    assert!(engine.generate_from_prompt("hello", 4).is_err());
    assert_eq!(engine.model.reservations, [(5, 1)]);
    assert_eq!(engine.model.resets, 2);
    assert!(engine.model.calls.is_empty());
    engine.model.fail_reservation = false;
    assert_eq!(engine.generate_from_prompt("hello", 4).unwrap(), "answer");
    assert_eq!(engine.model.reservations, [(5, 1), (5, 3)]);
    assert_eq!(engine.model.resets, 4);
}
