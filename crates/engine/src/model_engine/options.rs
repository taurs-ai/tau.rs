//! Public engine settings and their construction-only loader conversion.

use super::kv_budget::KvCacheBudget;
use crate::model::{AttentionBlockSize, CudaGraphMode, ModelLoadOptions};

/// Authoritative engine configuration used for construction and runtime options.
/// Kernel selection is fixed when the model is built; it is not a live toggle.
/// The factory derives `ModelLoadOptions` from this value, then retains the original
/// options in the engine. A trace's kernel flag describes policy, not per-op hits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineOptions {
    optimize_kernel: bool,
    cuda_graph: CudaGraphMode,
    prefill_chunk_size: usize,
    attention_block_size: AttentionBlockSize,
    moe_prefill_config: Option<crate::MoePrefillConfig>,
    moe_decode_options: Option<crate::MoeDecodeOptions>,
    tracing: bool,
    metrics: bool,
    vram_metrics: bool,
    auto_kv_cache_budget: bool,
    pub(super) kv_cache_budget: Option<KvCacheBudget>,
}

impl Default for EngineOptions {
    /// Keeps default kernel selection aligned with standalone model loading.
    fn default() -> Self {
        Self {
            optimize_kernel: ModelLoadOptions::default().optimize_kernel(),
            cuda_graph: CudaGraphMode::None,
            prefill_chunk_size: Self::DEFAULT_PREFILL_CHUNK_SIZE,
            attention_block_size: AttentionBlockSize::default(),
            moe_prefill_config: None,
            moe_decode_options: None,
            tracing: false,
            metrics: false,
            vram_metrics: false,
            auto_kv_cache_budget: false,
            kv_cache_budget: None,
        }
    }
}

impl EngineOptions {
    /// Retains exact-profile construction callers. Auto budgets use the options
    /// builder below; both feed one construction-time source of configuration.
    pub fn with_moe_decode_config(
        mut self,
        config: Option<crate::MoeDecodeConfig>,
    ) -> Self {
        self.moe_decode_options = config.map(crate::MoeDecodeOptions::from);
        self
    }
    /// Legacy exact accessor. Automatic selections need device resolution;
    /// callers inspecting startup options use `moe_decode_options`.
    pub fn moe_decode_config(self) -> Option<crate::MoeDecodeConfig> {
        self.moe_decode_options
            .and_then(crate::MoeDecodeOptions::exact_config)
    }
    /// Carries startup selections to model construction, where device resolution
    /// happens once before token execution and graph capture.
    pub fn with_moe_decode_options(
        mut self,
        options: Option<crate::MoeDecodeOptions>,
    ) -> Self {
        self.moe_decode_options = options;
        self
    }
    pub fn moe_decode_options(self) -> Option<crate::MoeDecodeOptions> {
        self.moe_decode_options
    }

    /// Select compiled MoE prefill tiles before model allocation; runtime requests
    /// cannot change this setting or the retained dispatch/scratch layout.
    pub fn with_moe_prefill_config(
        mut self,
        config: Option<crate::MoePrefillConfig>,
    ) -> Self {
        self.moe_prefill_config = config;
        self
    }
    pub fn moe_prefill_config(self) -> Option<crate::MoePrefillConfig> {
        self.moe_prefill_config
    }

    /// Let the factory size Qwen3.5 CUDA KV caches from post-load free VRAM.
    /// tau-server enables this automatically; standalone model callers opt in.
    pub fn with_auto_kv_cache_budget(mut self, enabled: bool) -> Self {
        self.auto_kv_cache_budget = enabled;
        self
    }

    pub fn auto_kv_cache_budget(self) -> bool {
        self.auto_kv_cache_budget
    }

    /// Fixed prefill chunk size; the final shorter chunk always executes eagerly.
    pub const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;

    /// Selects a positive prefill chunk size. The loader rejects zero.
    pub fn with_prefill_chunk_size(mut self, prefill_chunk_size: usize) -> Self {
        self.prefill_chunk_size = prefill_chunk_size;
        self
    }

    pub fn prefill_chunk_size(self) -> usize {
        self.prefill_chunk_size
    }

    /// Context tokens per prepared decode attention tile, fixed at construction.
    pub fn with_attention_block_size(mut self, size: AttentionBlockSize) -> Self {
        self.attention_block_size = size;
        self
    }

    pub fn attention_block_size(self) -> AttentionBlockSize {
        self.attention_block_size
    }

    /// Creates an engine configuration with CUDA graphs and instrumentation disabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Chooses the kernel policy used to construct the model and label its traces.
    pub fn with_optimize_kernel(mut self, optimize_kernel: bool) -> Self {
        self.optimize_kernel = optimize_kernel;
        self
    }

    /// Returns the model's construction-time kernel policy.
    pub fn optimize_kernel(self) -> bool {
        self.optimize_kernel
    }

    /// Selects graph scope; supported regions may remain eager when appropriate.
    pub fn with_cuda_graph(mut self, cuda_graph: CudaGraphMode) -> Self {
        self.cuda_graph = cuda_graph;
        self
    }

    /// Returns the configured CUDA graph preference, not the active execution mode.
    pub fn cuda_graph(self) -> CudaGraphMode {
        self.cuda_graph
    }

    /// Enables host timing and output-logit fingerprints at token boundaries.
    /// Decode also collects host phases and CUDA-event timings when available.
    /// This does not enable CUDA capture or record intermediate layer/state tensors.
    pub fn with_tracing(mut self, tracing: bool) -> Self {
        self.tracing = tracing;
        self
    }

    /// Returns whether generation requests should collect execution traces.
    pub fn tracing(self) -> bool {
        self.tracing
    }

    /// Enables lightweight serving-rate logs without numerical tracing.
    pub fn with_metrics(mut self, metrics: bool) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn metrics(self) -> bool {
        self.metrics
    }

    /// Samples process VRAM independently of throughput logs and numerical traces.
    pub fn with_vram_metrics(mut self, vram_metrics: bool) -> Self {
        self.vram_metrics = vram_metrics;
        self
    }

    pub fn vram_metrics(self) -> bool {
        self.vram_metrics
    }
}

impl From<ModelLoadOptions> for EngineOptions {
    /// Preserves construction-only callers with graphs and instrumentation disabled.
    fn from(options: ModelLoadOptions) -> Self {
        Self {
            optimize_kernel: options.optimize_kernel(),
            cuda_graph: CudaGraphMode::None,
            prefill_chunk_size: Self::DEFAULT_PREFILL_CHUNK_SIZE,
            attention_block_size: options.attention_block_size(),
            moe_prefill_config: options.moe_prefill_config(),
            moe_decode_options: options.moe_decode_options(),
            tracing: false,
            metrics: false,
            vram_metrics: false,
            auto_kv_cache_budget: false,
            kv_cache_budget: None,
        }
    }
}

impl From<EngineOptions> for ModelLoadOptions {
    /// Projects only construction settings into the model loader.
    fn from(options: EngineOptions) -> Self {
        Self::new()
            .with_optimize_kernel(options.optimize_kernel())
            .with_attention_block_size(options.attention_block_size())
            .with_moe_prefill_config(options.moe_prefill_config())
            .with_moe_decode_options(options.moe_decode_options())
    }
}
