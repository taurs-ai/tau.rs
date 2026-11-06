//! Fixed-size serving counters and rate-limited stderr reports.

use super::memory::VramSnapshot;
use serde::Serialize;
use std::time::{Duration, Instant};

/// Fixed-size counters; no token IDs, logits, hashes, or per-token allocations.
#[derive(Debug)]
pub(super) struct ServingMetrics {
    request_id: u64,
    model: String,
    started: Instant,
    // The previous reporting checkpoint and its cumulative decode totals. Taking
    // differences gives the rate for the newest interval without retaining samples.
    last_reported: Instant,
    last_decode_tokens: usize,
    last_decode_us: f64,
    prompt_tokens: usize,
    /// All returned tokens, including the first token produced by prefill; no EOS.
    generated_tokens: usize,
    // The first recorded model step is prefill, even for a one-token prompt.
    prefilled: bool,
    prefill_us: f64,
    /// Returned tokens produced after prefill. The first output and EOS are excluded.
    decode_tokens: usize,
    /// Sum of decode-step durations, including the work that predicts a final EOS.
    decode_us: f64,
    stop_reason: &'static str,
    vram: Option<VramSnapshot>,
}

/// A borrowed view of the counters for one log line, not a stored token trace.
/// Durations are displayed in milliseconds; rates are measured in tokens/second.
#[derive(Debug, Serialize)]
struct ServingReport<'a> {
    request_id: u64,
    model: &'a str,
    status: &'static str,
    prompt_tokens: usize,
    generated_tokens: usize,
    decode_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    /// Average over the request's completed decode steps, excluding prefill.
    decode_tokens_per_second: Option<f64>,
    /// Same calculation, using only decode work since the previous metrics log.
    interval_decode_tokens_per_second: Option<f64>,
    /// Wall time up to this checkpoint, including work between measured model steps.
    elapsed_ms: f64,
    stop_reason: &'static str,
    error: Option<&'a str>,
    #[serde(flatten)]
    vram: Option<&'a VramSnapshot>,
}

impl ServingMetrics {
    // Checked at token boundaries; there is no background logging thread.
    const LOG_INTERVAL: Duration = Duration::from_secs(5);

    /// Starts fresh counters for a request, independent of the model's retained caches.
    pub(super) fn new(model: &str, request_id: u64) -> Self {
        let started = Instant::now();
        Self {
            request_id,
            model: model.to_owned(),
            started,
            last_reported: started,
            last_decode_tokens: 0,
            last_decode_us: 0.0,
            prompt_tokens: 0,
            generated_tokens: 0,
            prefilled: false,
            prefill_us: 0.0,
            decode_tokens: 0,
            decode_us: 0.0,
            stop_reason: "unknown",
            vram: None,
        }
    }

    /// Accounts for a completed forward pass and its selected token.
    /// Prefill produces the first output token, so it contributes to total output
    /// but not decode throughput. EOS costs inference time but is never returned.
    pub(super) fn record_step(&mut self, duration_us: f64, is_eos: bool) {
        if self.prefilled {
            self.decode_us += duration_us;
            self.decode_tokens += usize::from(!is_eos);
        } else {
            self.prefill_us = duration_us;
            self.prefilled = true;
        }
        self.generated_tokens += usize::from(!is_eos);
    }

    /// A long prefill or slow token can delay logging beyond the nominal interval.
    fn due(&self, now: Instant) -> bool {
        now.duration_since(self.last_reported) >= Self::LOG_INTERVAL
    }

    /// Builds a snapshot without changing the counters or advancing the interval.
    /// Rate denominators use summed model-step time, not the wall-clock log interval.
    fn report<'a>(
        &'a self,
        now: Instant,
        status: &'static str,
        error: Option<&'a str>,
    ) -> ServingReport<'a> {
        fn rate(tokens: usize, duration_us: f64) -> Option<f64> {
            // No decode output means no meaningful rate. Serialize it as null,
            // rather than reporting zero throughput or dividing by zero.
            (tokens > 0 && duration_us > 0.0).then(|| tokens as f64 * 1e6 / duration_us)
        }
        ServingReport {
            request_id: self.request_id,
            model: &self.model,
            status,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            decode_tokens: self.decode_tokens,
            prefill_ms: self.prefill_us / 1000.0,
            decode_ms: self.decode_us / 1000.0,
            decode_tokens_per_second: rate(self.decode_tokens, self.decode_us),
            interval_decode_tokens_per_second: rate(
                self.decode_tokens - self.last_decode_tokens,
                self.decode_us - self.last_decode_us,
            ),
            elapsed_ms: now.duration_since(self.started).as_secs_f64() * 1000.0,
            stop_reason: self.stop_reason,
            error,
            vram: self.vram.as_ref(),
        }
    }

    /// Writes progress or a final summary to stderr, keeping stdout for full traces.
    fn log(&mut self, now: Instant, status: &'static str, error: Option<&str>) {
        eprintln!(
            "inference_metrics {}",
            serde_json::to_string(&self.report(now, status, error))
                .expect("serving metrics contain serializable values")
        );
        // Advance the interval baseline only after emitting its snapshot. The
        // request-wide totals stay intact for subsequent logs and the final summary.
        self.last_reported = now;
        self.last_decode_tokens = self.decode_tokens;
        self.last_decode_us = self.decode_us;
    }

    /// Retains the prompt count without storing any token IDs.
    pub(super) fn record_tokenization(&mut self, prompt_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
    }

    pub(super) fn record_stop_reason(&mut self, reason: &'static str) {
        self.stop_reason = reason;
    }

    pub(super) fn record_vram(&mut self, snapshot: VramSnapshot) {
        self.vram = Some(snapshot);
    }

    /// EOS is followed by a final summary, so it needs no progress log.
    pub(super) fn log_progress(&mut self, completed: Instant, is_eos: bool) {
        if !is_eos && self.due(completed) {
            self.log(completed, "running", None);
        }
    }

    /// Consuming the collector prevents a second final summary.
    pub(super) fn finish(mut self, error: Option<&str>) {
        self.log(
            Instant::now(),
            if error.is_some() { "error" } else { "ok" },
            error,
        );
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self.report(Instant::now(), "ok", None)).unwrap()
    }
}

#[cfg(test)]
mod tests;
