use super::super::InferenceMetrics;
use super::*;
use crate::MetricsOptions;

fn metrics(enabled: bool) -> InferenceMetrics {
    InferenceMetrics::start(
        crate::test_support::request("fixture", "bf16", 3, &[9]),
        MetricsOptions::new().with_tracing(enabled),
        &candle::Device::Cpu,
    )
}

#[test]
fn trace_preserves_requested_and_effective_output_limits() {
    let mut metrics = InferenceMetrics::start(
        crate::test_support::request("fixture", "f32", 100, &[3]),
        MetricsOptions::new().with_tracing(true),
        &candle::Device::Cpu,
    );
    metrics.configure_output_limit(3);
    let trace = metrics.finish(None).unwrap();
    assert_eq!(trace.max_new_tokens, 100);
    assert_eq!(trace.effective_max_new_tokens, Some(3));
}

/// Detailed decode fields attach only to decode, preserve EOS, and keep capture
/// setup out of the request's warmed aggregate without changing the host timer.
#[test]
fn decode_phase_samples_preserve_legacy_spans_and_cold_totals() {
    let mut m = metrics(true);
    m.record_model_step(m.start_stage(), 0, 2, 2, false, &[1., 2., 3.]);
    m.record_model_step(m.start_stage(), 2, 1, 9, true, &[1., 2., 3.]);
    m.record_decode_phases(DecodePhases {
        forward_scope: "complete_forward_replay",
        cold_graph_capture: true,
        totals: super::super::timing::DecodePhaseTotals {
            input_setup_host_us: 100.,
            logits_readback_bytes: 12,
            gpu_forward_us: Some(10.),
            ..Default::default()
        },
    });
    let record = m.finish(None).unwrap();
    assert!(record.steps[0].decode_timing.is_none());
    assert!(record.steps[1].is_eos);
    assert!(record.steps[1].duration_us >= 0.0);
    assert_eq!(record.summary.decode_timing.steps, 1);
    assert_eq!(record.summary.decode_timing.cold_graph_steps, 1);
    assert_eq!(record.summary.decode_timing.total.gpu_forward_us, Some(10.));
    assert_eq!(record.summary.decode_timing.steady.gpu_forward_us, None);
}

/// Trace enablement must not overwrite the selected model kernel policy.
#[test]
fn tracing_preserves_reference_and_optimized_policy_labels() {
    for optimize_kernel in [false, true] {
        let mut metrics = InferenceMetrics::start(
            {
                let mut request =
                    crate::test_support::request("fixture", "bf16", 3, &[9]);
                request.optimize_kernel = optimize_kernel;
                request
            },
            MetricsOptions::new().with_tracing(true),
            &candle::Device::Cpu,
        );
        assert_eq!(
            metrics.finish(None).unwrap().optimize_kernel,
            optimize_kernel
        );
    }
}

/// Records the final EOS evaluation without counting it as an emitted token.
#[test]
fn traces_prefill_decode_eos_and_failures() {
    let mut metrics = metrics(true);
    metrics.record_tokenization(metrics.start_stage(), &[1, 2]);
    metrics.record_model_step(metrics.start_stage(), 0, 2, 2, false, &[1., 2., 3.]);
    metrics.record_model_step(metrics.start_stage(), 2, 1, 9, true, &[1., 2., 3.]);
    metrics.record_eos();
    let record = metrics.finish(None).unwrap();
    assert_eq!(record.prompt_token_ids, [1, 2]);
    assert_eq!(record.generated_token_ids, [2]);
    assert_eq!(record.steps.len(), 2);
    assert_eq!(record.steps[0].phase, "prefill");
    assert_eq!(record.steps[1].cache_offset, 2);
    assert_eq!(record.summary.decode_steps, 1);
    assert_eq!(record.summary.decode_tokens, 0);
    assert_eq!(record.summary.decode_tokens_per_second, None);
    assert_eq!(record.stop_reason, "eos");
    assert!(metrics.finish(None).is_none());
    let error = self::metrics(true)
        .finish(Some("generation_failed"))
        .unwrap();
    assert_eq!(error.status, "error");
    assert_eq!(error.error.as_deref(), Some("generation_failed"));
}

#[test]
fn graph_labels_require_actual_decode_replay() {
    let mut m = metrics(true);
    m.configure_cuda_graph(crate::GraphPolicy::Layer, None);
    m.record_model_step(m.start_stage(), 0, 1, 2, false, &[1., 2., 3.]);
    m.record_cuda_graph_step(GraphActivity::default());
    for offset in 1..=2 {
        m.record_model_step(m.start_stage(), offset, 1, 2, false, &[1., 2., 3.]);
        m.record_cuda_graph_step(GraphActivity {
            gdn_layer_graph_captures: if offset == 1 { 3 } else { 0 },
            gdn_layer_graph_replays: 3,
            mlp_graph_captures: usize::from(offset == 1),
            mlp_graph_replays: 1,
            ..GraphActivity::default()
        });
    }
    let record = m.finish(None).unwrap();
    assert_eq!(record.schema_version, 5);
    assert_eq!(record.execution, "cuda_graph");
    assert_eq!(record.steps[0].execution, "eager");
    assert_eq!(record.steps[0].graphs, GraphActivity::default());
    assert_eq!(record.steps[1].graphs.gdn_layer_graph_captures, 3);
    assert_eq!(record.steps[2].graphs.gdn_layer_graph_captures, 0);
    assert_eq!(record.steps[2].graphs.mlp_graph_captures, 0);
    assert_eq!(record.summary.graphs.gdn_layer_graph_replays, 6);
    assert_eq!(record.summary.graphs.mlp_graph_replays, 2);
    assert_eq!(record.cuda_graph_fallback_reason, None);
}

#[test]
fn full_policy_reports_layer_or_eager_coverage_without_claiming_full_replay() {
    for replay in [false, true] {
        let mut m = metrics(true);
        m.configure_cuda_graph(crate::GraphPolicy::Full, None);
        m.record_model_step(m.start_stage(), 0, 1, 2, false, &[1., 2., 3.]);
        m.record_cuda_graph_step(GraphActivity {
            mlp_graph_replays: usize::from(replay),
            ..Default::default()
        });
        let record = m.finish(None).unwrap();
        let scope = if replay { "layer" } else { "none" };
        assert_eq!(record.cuda_graph_requested, crate::GraphPolicy::Full);
        assert_eq!(record.cuda_graph_scope, scope);
        assert_eq!(record.steps[0].cuda_graph_scope, scope);
        assert_eq!(record.summary.graphs.forward_graph_replays, 0);
        assert_eq!(
            record.cuda_graph_coverage_reason,
            Some("maximum_supported_graph_policy")
        );
        let wire = serde_json::to_value(&record).unwrap();
        assert_eq!(wire["cuda_graph_requested"], "full");
        assert_eq!(wire["cuda_graph_scope"], scope);
    }
}

#[test]
fn prefill_trace_aggregates_full_chunk_graphs_and_records_chunk_size() {
    let mut m = metrics(true);
    m.configure_prefill(4);
    m.configure_cuda_graph(crate::GraphPolicy::Layer, None);
    m.record_model_step(m.start_stage(), 0, 17, 2, false, &[1., 2., 3.]);
    m.record_cuda_graph_step(GraphActivity {
        gdn_layer_graph_captures: 3,
        gdn_layer_graph_replays: 12,
        mlp_graph_captures: 1,
        mlp_graph_replays: 4,
        ..GraphActivity::default()
    });
    let record = m.finish(None).unwrap();
    assert_eq!(record.prefill_chunk_size, 4);
    assert_eq!(record.steps[0].phase, "prefill");
    assert_eq!(record.steps[0].input_tokens, 17);
    assert_eq!(record.steps[0].execution, "cuda_graph");
    assert_eq!(record.summary.graphs.gdn_layer_graph_replays, 12);
    assert_eq!(record.summary.decode_steps, 0);
    assert_eq!(record.cuda_graph_fallback_reason, None);
}

#[test]
fn missing_replay_and_ineligible_graph_modes_remain_visible() {
    for reason in [None, Some("unsupported_dtype")] {
        let mut m = metrics(true);
        m.configure_cuda_graph(crate::GraphPolicy::Layer, reason);
        m.record_model_step(m.start_stage(), 0, 1, 2, false, &[1., 2., 3.]);
        m.record_cuda_graph_step(GraphActivity::default());
        m.record_model_step(m.start_stage(), 1, 1, 2, false, &[1., 2., 3.]);
        m.record_cuda_graph_step(GraphActivity::default());
        let record = m.finish(None).unwrap();
        assert_eq!(record.execution, "eager");
        assert_eq!(
            record.cuda_graph_fallback_reason,
            reason.or(Some("unsupported_component_contract"))
        );
    }
}
