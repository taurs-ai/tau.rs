use super::*;
use candle::Device;
use tau_metrics::GraphActivity;
use tau_models::CudaGraphStats;

/// Adapters preserve requested policy and every graph counter without letting
/// observation types validate or alter the model's execution settings.
#[test]
fn preserves_graph_observation_labels_and_counters() {
    for (mode, label) in [
        (crate::CudaGraphMode::None, "none"),
        (crate::CudaGraphMode::Layer, "layer"),
        (crate::CudaGraphMode::Full, "full"),
    ] {
        assert_eq!(
            serde_json::to_value(GraphPolicy::from(mode)).unwrap(),
            label
        );
    }
    let counts = CudaGraphStats {
        forward_graph_captures: 1,
        forward_graph_replays: 2,
        gdn_layer_graph_captures: 3,
        gdn_layer_graph_replays: 4,
        mlp_graph_captures: 5,
        mlp_graph_replays: 6,
    };
    assert_eq!(
        serde_json::to_value(GraphActivity::from(counts)).unwrap(),
        serde_json::to_value(counts).unwrap()
    );
    assert!(counts.validate_mode(crate::CudaGraphMode::Layer).is_err());
}
use tau_metrics::timing::*;

/// CPU diagnostics retain meaningful host spans and explicitly unavailable GPU
/// intervals, with a fresh sample rather than accumulated values on each step.
#[test]
fn cpu_phases_reset_and_preserve_readback_and_selection() {
    let logits = candle::Tensor::new(&[[[1f32, 4., 4., -2.]]], &Device::Cpu).unwrap();
    let mut timer = DecodeTimer::new(&Device::Cpu).unwrap();
    for cold in [true, false] {
        timer.start_step();
        timer.begin_forward("model_forward").unwrap();
        timer.end_forward().unwrap();
        let values =
            crate::generation::logits_values_profiled(&logits, &mut timer).unwrap();
        assert_eq!(
            crate::generation::next_token_from_values(&values).unwrap(),
            2
        );
        timer.end_selection();
        let sample = timer.finish(cold).unwrap();
        assert_eq!(sample.totals.logits_readback_bytes, 16);
        assert_eq!(sample.cold_graph_capture, cold);
        assert_eq!(sample.totals.gpu_forward_us, None);
        assert_eq!(sample.totals.gpu_logits_to_f32_us, None);
        assert!(sample.totals.forward_submit_host_us >= 0.0);
    }
}
