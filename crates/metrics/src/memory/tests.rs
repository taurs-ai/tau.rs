use super::*;

#[test]
fn sampled_peak_includes_transient_observations_and_boundary_samples() {
    let mut snapshot =
        VramSnapshot::new(Some("GPU-test".to_owned()), VRAM_SAMPLE_INTERVAL);
    snapshot.vram_start_bytes = Some(100);
    for bytes in [100, 400, 150] {
        snapshot.observe(&Ok(bytes));
    }
    snapshot.complete(&Ok(150));
    assert_eq!(snapshot.vram_peak_sampled_bytes, Some(400));
    assert_eq!(snapshot.vram_peak_delta_bytes, Some(300));
    assert_eq!(snapshot.vram_end_bytes, Some(150));
    assert_eq!(snapshot.vram_samples, 3);
    assert_eq!(snapshot.vram_sample_interval_ms, 100);
}

#[test]
fn missing_samples_preserve_partial_peaks_and_unknown_baselines() {
    let mut snapshot = VramSnapshot::new(None, VRAM_SAMPLE_INTERVAL);
    snapshot.observe(&Err("start failed".to_owned()));
    snapshot.observe(&Ok(400));
    snapshot.observe(&Err("end failed".to_owned()));
    snapshot.complete(&Err("end failed".to_owned()));
    assert_eq!(snapshot.vram_start_bytes, None);
    assert_eq!(snapshot.vram_end_bytes, None);
    assert_eq!(snapshot.vram_peak_sampled_bytes, Some(400));
    assert_eq!(snapshot.vram_peak_delta_bytes, None);
    assert_eq!(snapshot.vram_error.as_deref(), Some("end failed"));
    assert_eq!(snapshot.vram_samples, 1);
}

#[test]
fn cpu_reports_null_without_creating_a_sampler() {
    let metrics = VramMetrics::start("cpu", 1, &candle::Device::Cpu);
    assert!(metrics.sampler.is_none());
    let snapshot = metrics.unavailable.clone();
    let json = serde_json::to_value(snapshot).unwrap();
    for key in [
        "vram_start_bytes",
        "vram_end_bytes",
        "vram_peak_sampled_bytes",
        "vram_peak_delta_bytes",
    ] {
        assert!(json[key].is_null());
    }
    assert!(json["vram_error"].as_str().unwrap().contains("CUDA device"));
    // Drop and finalization both work when NVML was never initialized.
    assert_eq!(metrics.finish(Some("failed")).vram_samples, 0);
}

#[test]
fn uuid_format_matches_nvml_without_using_cuda_ordinals() {
    assert_eq!(
        format_uuid([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 252, 253, 254, 255]),
        "GPU-00010203-0405-0607-0809-0a0bfcfdfeff"
    );
}

#[test]
#[ignore = "requires a usable CUDA GPU and NVML process-memory reporting"]
fn samples_live_cuda_process_memory() {
    let device = candle::Device::new_cuda(0).unwrap();
    let retained =
        candle::Tensor::zeros(1024 * 1024, candle::DType::F32, &device).unwrap();
    let metrics = VramMetrics::start("live", 1, &device);
    let snapshot = metrics.finish(None);
    assert!(snapshot.vram_error.is_none(), "{:?}", snapshot.vram_error);
    assert!(snapshot.vram_start_bytes.unwrap() >= 4 * 1024 * 1024);
    assert!(
        snapshot.vram_peak_sampled_bytes.unwrap() >= snapshot.vram_start_bytes.unwrap()
    );
    assert!(snapshot.vram_end_bytes.is_some());
    assert!(snapshot.vram_samples >= 2);
    drop(retained);
}
