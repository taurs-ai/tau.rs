//! Independent opt-in serving metrics, VRAM samples and numerical traces.
//!
//! With all modes disabled there are no instrumentation clock reads or logs.
//! Serving metrics retain only counters/timers and log to stderr at token boundaries.
//! Numerical tracing buffers per-token records and emits JSON to stdout after
//! generation.
//!
//! Each generation request owns one collector. The engine records tokenization,
//! one prefill step for the whole prompt, and subsequent single-token decode steps.
//! It then finishes the collector with either a successful result or an error.
//! Metrics and tracing can run separately or together; when both are enabled,
//! they use the same request ID and the same measured duration for each model step.
//! VRAM collection independently samples process memory through a shared NVML
//! worker and attaches its final snapshot to any enabled serving/trace reports.
//!
//! Records describe completed token-forward outputs and host timings. They
//! do not contain intermediate activations, cache/state tensors or memory
//! addresses. Trace-only decode diagnostics also report CUDA-event
//! intervals; capture tests still validate graph/state boundaries separately.

mod config;
mod logits;
mod memory;
mod serving;
pub mod timing;
pub use config::{GraphActivity, GraphPolicy, MetricsOptions, RequestMetadata};
mod trace;

use memory::VramMetrics;
use serving::ServingMetrics;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use trace::{ActiveTrace, RequestTrace};

// IDs connect progress logs, the final metrics summary, and an optional trace.
// Relaxed ordering is enough: this counter provides uniqueness, not synchronization.
static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Owns independent opt-in collectors for one request. All are absent by default.
#[derive(Debug, Default)]
pub struct InferenceMetrics {
    active: Option<ActiveTrace>,
    serving: Option<ServingMetrics>,
    memory: Option<VramMetrics>,
}

impl InferenceMetrics {
    /// Numerical tracing consumes the full vocabulary for fingerprints and top-k
    /// diagnostics. Serving counters and VRAM collection do not need logits.
    pub fn needs_logits(&self) -> bool {
        self.active.is_some()
    }

    /// Creates phase instrumentation only for numerical tracing, never metrics-only
    /// serving. Event allocation is lazy until generation actually needs decode.
    pub fn decode_timer(
        &self,
        device: &candle::Device,
    ) -> candle::Result<Option<timing::DecodeTimer>> {
        self.active
            .is_some()
            .then(|| timing::DecodeTimer::new(device))
            .transpose()
    }

    /// Attaches a completed diagnostic sample to the last recorded decode step.
    pub fn record_decode_phases(&mut self, phases: timing::DecodePhases) {
        if let Some(active) = &mut self.active {
            active.record_decode_phases(phases);
        }
    }

    /// Creates only the collection modes explicitly enabled by the caller.
    pub fn start(
        request: RequestMetadata<'_>,
        options: MetricsOptions,
        device: &candle::Device,
    ) -> Self {
        let model = request.model;
        // Keep constructors lazy: disabled instrumentation reads no clocks,
        // allocates no collector state and does not advance the request sequence.
        let request_id =
            (options.tracing() || options.metrics() || options.vram_metrics())
                .then(|| REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed));
        Self {
            serving: options.metrics().then(|| {
                ServingMetrics::new(model, request_id.expect("metrics enabled"))
            }),
            active: options.tracing().then(|| {
                ActiveTrace::new(request_id.expect("tracing enabled"), &request, device)
            }),
            memory: options.vram_metrics().then(|| {
                VramMetrics::start(
                    model,
                    request_id.expect("VRAM metrics enabled"),
                    device,
                )
            }),
        }
    }

    /// Preserve the requested limit while explaining an earlier capacity-based stop.
    pub fn configure_output_limit(&mut self, effective: usize) {
        if let Some(active) = &mut self.active {
            active.configure_output_limit(effective);
        }
    }

    /// Records the chunk size needed to interpret graph replay and eager tails.
    pub fn configure_prefill(&mut self, chunk_size: usize) {
        if let Some(active) = &mut self.active {
            active.configure_prefill(chunk_size);
        }
    }

    /// Records requested policy separately from observed execution.
    pub fn configure_cuda_graph(
        &mut self,
        requested: GraphPolicy,
        fallback: Option<&'static str>,
    ) {
        if let Some(active) = &mut self.active {
            active.configure_cuda_graph(requested, fallback);
        }
    }

    /// Labels the most recently completed step using actual capture/replay counts.
    pub fn record_cuda_graph_step(&mut self, counts: GraphActivity) {
        if let Some(active) = &mut self.active {
            active.record_cuda_graph_step(counts);
        }
    }

    /// Shares existing timing boundaries without adding GPU synchronization.
    pub fn start_stage(&self) -> Option<Instant> {
        // `then_some(Instant::now())` would read the clock in disabled mode too.
        (self.active.is_some() || self.serving.is_some()).then(Instant::now)
    }

    /// Records an explicit runtime boundary such as tokenization or cache reset.
    pub fn record_stage(&mut self, timer: Option<Instant>, name: &'static str) {
        // Only traces retain stage history; serving counters include its wall time.
        if let (Some(active), Some(started)) = (&mut self.active, timer) {
            active.record_stage(started, name);
        }
    }

    /// Counts prompt tokens for metrics and retains their IDs only for tracing.
    pub fn record_tokenization(&mut self, timer: Option<Instant>, tokens: &[u32]) {
        self.record_stage(timer, "tokenization");
        if let Some(serving) = &mut self.serving {
            serving.record_tokenization(tokens.len());
        }
        if let Some(active) = &mut self.active {
            active.record_tokenization(tokens);
        }
    }

    /// Ends step timing before numerical trace work and records EOS candidates too.
    #[allow(clippy::too_many_arguments)]
    pub fn record_model_step(
        &mut self,
        timer: Option<Instant>,
        cache_offset: usize,
        input_tokens: usize,
        token_id: u32,
        is_eos: bool,
        logits: &[f32],
    ) {
        let Some(started) = timer else {
            return;
        };
        // Stop one shared timer before counters, hashing or logs. Both collectors
        // measure exactly the same inference work without another GPU wait.
        let completed = Instant::now();
        let duration_us = completed.duration_since(started).as_secs_f64() * 1e6;
        if let Some(serving) = &mut self.serving {
            serving.record_step(duration_us, is_eos);
        }
        if let Some(active) = &mut self.active {
            active.record_model_step(
                started,
                completed,
                duration_us,
                cache_offset,
                input_tokens,
                token_id,
                is_eos,
                logits,
            );
        }
        if let Some(serving) = &mut self.serving {
            serving.log_progress(completed, is_eos);
        }
    }

    /// Marks generation as stopped by the checkpoint's configured EOS.
    pub fn record_eos(&mut self) {
        self.record_stop_reason("eos");
    }

    /// Marks generation as stopped by the requested output limit.
    pub fn record_length_limit(&mut self) {
        self.record_stop_reason("length");
    }

    fn record_stop_reason(&mut self, reason: &'static str) {
        if let Some(serving) = &mut self.serving {
            serving.record_stop_reason(reason);
        }
        if let Some(active) = &mut self.active {
            active.record_stop_reason(reason);
        }
    }

    fn finish(&mut self, error: Option<&str>) -> Option<RequestTrace> {
        // Taking ownership makes trace finalization idempotent.
        self.active.take().map(|active| active.finish(error))
    }

    /// Finalizes each enabled collection mode after successful generation.
    pub fn log_success(&mut self) {
        self.log_finished(None);
    }

    /// Labels partial metrics/traces as failures so they cannot pass a baseline.
    pub fn log_failure(&mut self, reason: &str) {
        self.log_finished(Some(reason));
    }

    fn log_finished(&mut self, error: Option<&str>) {
        // Finish trace timing before either output is serialized or written;
        // consume collectors, then emit serving stderr before trace stdout.
        let mut trace = self.finish(error);
        if let Some(memory) = self.memory.take() {
            let snapshot = memory.finish(error);
            if let Some(serving) = &mut self.serving {
                serving.record_vram(snapshot.clone());
            }
            if let Some(record) = &mut trace {
                record.record_vram(snapshot);
            }
        }
        if let Some(serving) = self.serving.take() {
            serving.finish(error);
        }
        if let Some(record) = trace {
            record.log();
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_support;
