use super::*;

fn metrics(enabled: bool) -> InferenceMetrics {
    InferenceMetrics::start(
        crate::test_support::request("fixture", "bf16", 3, &[9]),
        MetricsOptions::new().with_tracing(enabled),
        &candle::Device::Cpu,
    )
}

/// Disabled tracing must not construct a timer or retain request/logit data.
#[test]
fn disabled_tracing_does_not_collect() {
    let mut metrics = metrics(false);
    assert!(metrics.start_stage().is_none());
    assert!(
        metrics
            .decode_timer(&candle::Device::Cpu)
            .unwrap()
            .is_none()
    );
    metrics.record_tokenization(None, &[1, 2]);
    metrics.record_model_step(None, 0, 2, 1, false, &[f32::NAN]);
    metrics.record_eos();
    assert!(metrics.finish(None).is_none());
    assert!(metrics.serving.is_none());
}

#[test]
fn serving_metrics_do_not_collect_numerical_traces() {
    let mut metrics = InferenceMetrics::start(
        crate::test_support::request("fixture", "bf16", 3, &[9]),
        MetricsOptions::new().with_metrics(true),
        &candle::Device::Cpu,
    );
    metrics.record_tokenization(metrics.start_stage(), &[1, 2]);
    // A metrics-only request needs no logits at all.
    for (offset, token, eos) in [(0, 2, false), (2, 3, false), (3, 9, true)] {
        metrics.record_model_step(metrics.start_stage(), offset, 1, token, eos, &[]);
    }
    metrics.record_eos();
    assert!(metrics.active.is_none());
    assert!(
        metrics
            .decode_timer(&candle::Device::Cpu)
            .unwrap()
            .is_none()
    );
    let serving = metrics.serving.as_ref().unwrap().snapshot();
    assert_eq!(serving["prompt_tokens"], 2);
    assert_eq!(serving["generated_tokens"], 2);
    assert_eq!(serving["decode_tokens"], 1);
    assert_eq!(serving["stop_reason"], "eos");
    metrics.log_success();
    assert!(metrics.serving.is_none());
    assert!(metrics.start_stage().is_none());
}

#[test]
fn serving_metrics_and_traces_share_step_timing_and_request_identity() {
    let mut metrics = InferenceMetrics::start(
        crate::test_support::request("fixture", "bf16", 3, &[9]),
        MetricsOptions::new().with_metrics(true).with_tracing(true),
        &candle::Device::Cpu,
    );
    metrics.record_tokenization(metrics.start_stage(), &[1, 2]);
    metrics.record_model_step(metrics.start_stage(), 0, 2, 2, false, &[1., 2., 3.]);
    metrics.record_model_step(metrics.start_stage(), 2, 1, 2, false, &[1., 2., 3.]);
    metrics.record_length_limit();
    let trace = serde_json::to_value(metrics.finish(None).unwrap()).unwrap();
    let serving = metrics.serving.as_ref().unwrap().snapshot();
    assert_eq!(serving["request_id"], trace["request_id"]);
    assert_eq!(
        serving["prefill_ms"].as_f64().unwrap(),
        trace["summary"]["prefill_us"].as_f64().unwrap() / 1000.0
    );
    assert_eq!(
        serving["decode_ms"].as_f64().unwrap(),
        trace["summary"]["decode_us"].as_f64().unwrap() / 1000.0
    );
    assert_eq!(
        serving["decode_tokens_per_second"],
        trace["summary"]["decode_tokens_per_second"]
    );
    assert_eq!(
        serving["generated_tokens"],
        trace["generated_token_ids"].as_array().unwrap().len()
    );
    assert_eq!(serving["stop_reason"], "length");
    metrics.log_failure("generation_failed");
    assert!(metrics.serving.is_none());
}

#[test]
fn finalization_drains_each_mode_on_success_and_failure() {
    for (serving, tracing) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        for failed in [false, true] {
            let mut metrics = InferenceMetrics::start(
                crate::test_support::request("fixture", "f32", 3, &[9]),
                MetricsOptions::new()
                    .with_metrics(serving)
                    .with_tracing(tracing),
                &candle::Device::Cpu,
            );
            assert_eq!(metrics.serving.is_some(), serving);
            assert_eq!(metrics.active.is_some(), tracing);
            assert_eq!(metrics.needs_logits(), tracing);
            if failed {
                metrics.log_failure("generation_failed");
            } else {
                metrics.log_success();
            }
            assert!(metrics.serving.is_none());
            assert!(metrics.active.is_none());
            assert!(metrics.start_stage().is_none());
            // Repeated finalization must not recreate or retain either collector.
            metrics.log_success();
            metrics.log_failure("already_finished");
            assert!(metrics.finish(None).is_none());
        }
    }
}

#[test]
fn only_explicit_vram_collection_creates_the_memory_collector() {
    for serving in [false, true] {
        for tracing in [false, true] {
            for vram in [false, true] {
                let mut metrics = InferenceMetrics::start(
                    crate::test_support::request("fixture", "f32", 3, &[9]),
                    MetricsOptions::new()
                        .with_metrics(serving)
                        .with_tracing(tracing)
                        .with_vram_metrics(vram),
                    &candle::Device::Cpu,
                );
                assert_eq!(metrics.memory.is_some(), vram);
                assert_eq!(metrics.start_stage().is_some(), serving || tracing);
                // Failed requests also consume the independent memory collector.
                metrics.log_failure("generation_failed");
                assert!(metrics.memory.is_none());
                assert!(metrics.serving.is_none());
                assert!(metrics.active.is_none());
                metrics.log_success();
            }
        }
    }
}

#[test]
fn final_reports_share_vram_fields_and_omit_them_by_default() {
    let snapshot =
        memory::VramMetrics::start("fixture", 1, &candle::Device::Cpu).finish(None);
    let memory = serde_json::to_value(&snapshot).unwrap();
    let mut serving = ServingMetrics::new("fixture", 1);
    let mut trace = metrics(true).finish(None).unwrap();
    assert!(serving.snapshot().get("vram_start_bytes").is_none());
    assert!(
        serde_json::to_value(&trace).unwrap()["summary"]
            .get("vram_start_bytes")
            .is_none()
    );
    serving.record_vram(snapshot.clone());
    trace.record_vram(snapshot);
    let serving = serving.snapshot();
    let trace = serde_json::to_value(trace).unwrap();
    for (key, value) in memory.as_object().unwrap() {
        assert_eq!(&serving[key], value);
        assert_eq!(&trace["summary"][key], value);
    }
}
