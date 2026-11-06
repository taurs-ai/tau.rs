//! Full-vector fingerprints and bounded diagnostic samples of host logits.

use serde::Serialize;
use sha2::{Digest, Sha256};

/// SHA-256 covers every FP32 logit in vocabulary order, encoded little-endian.
/// Samples aid diagnosis; they are not a substitute for a full-vector tolerance test.
/// The full vector is not retained, so a hash mismatch cannot quantify its error.
#[derive(Debug, Serialize)]
pub(super) struct LogitSnapshot {
    count: usize,
    finite_count: usize,
    sha256_f32_le: String,
    samples: Vec<LogitSample>,
    selected_logit: Option<f32>,
}

/// JSON cannot represent NaN or infinity, so nonfinite samples use `None`/null.
/// The full-vector hash and finite-count fields still preserve evidence of them.
#[derive(Debug, Serialize)]
struct LogitSample {
    token_id: usize,
    value: Option<f32>,
}

impl LogitSnapshot {
    /// Summarizes an existing host logits vector without another GPU transfer.
    pub(super) fn from_values(values: &[f32], token: u32) -> Self {
        let mut digest = Sha256::new();
        let mut finite_count = 0;
        for &value in values {
            digest.update(value.to_le_bytes());
            finite_count += usize::from(value.is_finite());
        }
        let count = values.len();
        let sample_count = count.min(16);
        let samples = (0..sample_count)
            .map(|i| {
                // Spread diagnostic samples across the vocabulary, not just the
                // highest-scoring tokens. Empty and single-value inputs are valid.
                let token_id =
                    i * count.saturating_sub(1) / sample_count.saturating_sub(1).max(1);
                let value = values[token_id];
                LogitSample {
                    token_id,
                    value: value.is_finite().then_some(value),
                }
            })
            .collect();
        Self {
            count,
            finite_count,
            sha256_f32_le: format!("{:x}", digest.finalize()),
            samples,
            selected_logit: values
                .get(token as usize)
                .copied()
                .filter(|value| value.is_finite()),
        }
    }
}

#[cfg(test)]
mod tests;
