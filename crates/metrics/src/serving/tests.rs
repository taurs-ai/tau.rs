use super::*;

#[test]
fn serving_rates_separate_prefill_interval_and_eos() {
    let mut metrics = ServingMetrics::new("fixture", 1);
    metrics.record_step(1_000_000.0, false); // first token belongs to prefill
    metrics.record_step(100_000.0, false);
    metrics.log(metrics.started + Duration::from_secs(5), "running", None);
    metrics.record_step(200_000.0, false);
    metrics.record_step(100_000.0, true); // EOS takes time, but is not output
    metrics.stop_reason = "eos";
    let report = metrics.report(metrics.started + Duration::from_secs(6), "ok", None);
    assert_eq!(report.generated_tokens, 3);
    assert_eq!(report.decode_tokens, 2);
    assert_eq!(report.prefill_ms, 1000.0);
    assert_eq!(report.decode_ms, 400.0);
    assert_eq!(report.decode_tokens_per_second, Some(5.0));
    assert!(
        (report.interval_decode_tokens_per_second.unwrap() - 10.0 / 3.0).abs() < 1e-10
    );
    assert_eq!(report.elapsed_ms, 6000.0);
}

#[test]
fn serving_progress_is_rate_limited_and_errors_remain_visible() {
    let mut metrics = ServingMetrics::new("fixture", 1);
    let start = metrics.started;
    assert!(!metrics.due(start + Duration::from_millis(4999)));
    assert!(metrics.due(start + Duration::from_secs(5)));
    metrics.log(start + Duration::from_secs(5), "running", None);
    assert!(!metrics.due(start + Duration::from_millis(9999)));
    assert!(metrics.due(start + Duration::from_secs(10)));
    let report = metrics.report(
        start + Duration::from_secs(6),
        "error",
        Some("generation_failed"),
    );
    assert_eq!(report.status, "error");
    assert_eq!(report.error, Some("generation_failed"));
    assert_eq!(report.decode_tokens_per_second, None);
    assert!(
        serde_json::to_string(&report)
            .unwrap()
            .contains("\"decode_tokens_per_second\":null")
    );
}

#[test]
fn serving_single_token_and_immediate_eos_have_no_decode_rate() {
    for eos in [false, true] {
        let mut metrics = ServingMetrics::new("fixture", 1);
        metrics.record_step(1000.0, eos);
        let report = metrics.report(Instant::now(), "ok", None);
        assert_eq!(report.generated_tokens, usize::from(!eos));
        assert_eq!(report.decode_tokens, 0);
        assert_eq!(report.decode_tokens_per_second, None);
    }
}
