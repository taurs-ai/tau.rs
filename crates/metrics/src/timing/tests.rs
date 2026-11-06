use super::*;

/// One request reuses its four events across steps. Intervals are collected only
/// after the existing readback has completed, with no model/cache dependency.
#[test]
#[ignore = "requires CUDA GPU 0; run alone with --test-threads=1"]
fn cuda_phases_reuse_events_and_preserve_readback_values() {
    let device = Device::new_cuda(0).unwrap();
    let input = candle::Tensor::new(&[1f32, 2., 3., 4.], &device)
        .unwrap()
        .to_dtype(candle::DType::BF16)
        .unwrap();
    let mut timer = DecodeTimer::new(&device).unwrap();
    for offset in [1., 2.] {
        timer.start_step();
        timer.begin_forward("model_forward").unwrap();
        let logits = input.affine(2., offset).unwrap();
        timer.end_forward().unwrap();
        timer.begin_conversion().unwrap();
        let converted = logits.to_dtype(candle::DType::F32).unwrap();
        timer.begin_readback().unwrap();
        let values = converted.to_vec1::<f32>().unwrap();
        drop(converted);
        timer.end_readback(values.len());
        timer.end_selection();
        let sample = timer.finish(false).unwrap();
        assert_eq!(
            values,
            [
                2. + offset as f32,
                4. + offset as f32,
                6. + offset as f32,
                8. + offset as f32
            ]
        );
        assert_eq!(sample.totals.logits_readback_bytes, 16);
        assert!(sample.totals.gpu_forward_us.unwrap() >= 0.);
        assert!(sample.totals.gpu_logits_to_f32_us.unwrap() >= 0.);
    }
}

/// Cold graph creation remains visible but cannot inflate the warmed aggregate.
#[test]
fn summaries_separate_capture_steps_and_keep_cpu_gpu_unavailable() {
    let mut summary = DecodePhaseSummary::default();
    for cold in [true, false, false] {
        summary.record(&DecodePhases {
            cold_graph_capture: cold,
            forward_scope: "model_forward",
            totals: DecodePhaseTotals {
                input_setup_host_us: if cold { 100. } else { 2. },
                logits_readback_bytes: 16,
                ..Default::default()
            },
        });
    }
    assert_eq!(summary.steps, 3);
    assert_eq!(summary.cold_graph_steps, 1);
    assert_eq!(summary.total.input_setup_host_us, 104.);
    assert_eq!(summary.steady.input_setup_host_us, 4.);
    assert_eq!(summary.total.logits_readback_bytes, 48);
    assert_eq!(summary.total.gpu_forward_us, None);
}
