use super::*;

/// Fingerprints cover unsampled values and preserve nonfinite evidence in JSON.
#[test]
fn logit_snapshot_checks_the_complete_vector() {
    let values = vec![1.0; 40];
    let a = LogitSnapshot::from_values(&values, 2);
    let mut changed = values;
    changed[1] = f32::NAN;
    changed[2] = f32::INFINITY;
    let b = LogitSnapshot::from_values(&changed, 2);
    assert_eq!(a.finite_count, 40);
    assert_eq!(b.finite_count, 38);
    assert_ne!(a.sha256_f32_le, b.sha256_f32_le);
    assert_eq!(b.selected_logit, None);
    assert!(serde_json::to_string(&b).is_ok());
    let single = LogitSnapshot::from_values(&[2.0], 0);
    assert_eq!(single.samples.len(), 1);
    assert_eq!(single.samples[0].token_id, 0);
}
