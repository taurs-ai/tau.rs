//! Ignored production-shape timing, shared by control and candidate builds.

use super::*;

#[test]
#[ignore = "production-shape CUDA timing; run alone with --nocapture"]
fn measures_prefill_recurrence() {
    let device = Device::new_cuda(0).unwrap();
    let mut cfg = cuda_test_config();
    cfg.key_head_dim = 128;
    cfg.value_head_dim = 128;
    cfg.num_key_heads = 16;
    cfg.num_value_heads = 32;
    for tokens in [27, 138, 256] {
        let mut layer = GatedDeltaNet::new(
            &cfg,
            true,
            VarBuilder::from_tensors(test_tensors(&cfg, &device), DType::BF16, &device),
        )
        .unwrap();
        let q = patterned_bf16((1, tokens, 32, 128), 1, 0.02, &device);
        let k = patterned_bf16((1, tokens, 32, 128), 3, 0.02, &device);
        let v = patterned_bf16((1, tokens, 32, 128), 7, 0.02, &device);
        let g = Tensor::full(-0.1f32, (1, tokens, 32), &device).unwrap();
        let b = Tensor::full(0.5f32, (1, tokens, 32), &device).unwrap();
        let mut samples = Vec::new();
        for round in 0..6 {
            layer.recurrent_state = None;
            device.synchronize().unwrap();
            let start = std::time::Instant::now();
            let output = layer.recurrent_delta_rule(&q, &k, &v, &g, &b).unwrap();
            device.synchronize().unwrap();
            if round > 0 {
                samples.push(start.elapsed().as_secs_f64() * 1000.);
            }
            assert!(
                output
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    .is_finite()
            );
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "prefill_recurrence tokens={tokens} heads=32 median_ms={:.6}",
            samples[2]
        );
    }
}
