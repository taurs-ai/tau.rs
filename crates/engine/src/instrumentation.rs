//! Converts engine settings and graph observations at the metrics boundary.

use crate::{EngineOptions, ModelDType};
use tau_metrics::{GraphPolicy, InferenceMetrics, MetricsOptions, RequestMetadata};

/// Supplies borrowed facts and opt-in flags without exposing engine options to
/// the metrics crate. Building this metadata reads no clocks and allocates nothing.
pub(crate) fn start_request(
    model: &str,
    dtype: ModelDType,
    max_new_tokens: usize,
    options: EngineOptions,
    device: &candle::Device,
    eos_token_ids: &[u32],
) -> InferenceMetrics {
    InferenceMetrics::start(
        RequestMetadata {
            model,
            dtype: dtype.as_str(),
            max_new_tokens,
            eos_token_ids,
            optimize_kernel: options.optimize_kernel(),
            prefill_chunk_size: options.prefill_chunk_size(),
            graph_policy: GraphPolicy::from(options.cuda_graph()),
        },
        MetricsOptions::new()
            .with_tracing(options.tracing())
            .with_metrics(options.metrics())
            .with_vram_metrics(options.vram_metrics()),
        device,
    )
}

#[cfg(test)]
mod tests;
