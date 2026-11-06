//! Instrumentation settings and observations, independent of engine/model types.

use serde::Serialize;

/// Selects collection modes; disabled defaults create no request collector state.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricsOptions {
    tracing: bool,
    metrics: bool,
    vram_metrics: bool,
}

impl MetricsOptions {
    /// Starts with every collection mode disabled.
    pub fn new() -> Self {
        Self::default()
    }
    /// Enables detailed numerical traces and decode-phase measurements.
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing = enabled;
        self
    }
    /// Enables fixed-size serving counters independently of numerical traces.
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.metrics = enabled;
        self
    }
    /// Enables the independent process-memory collector.
    pub fn with_vram_metrics(mut self, enabled: bool) -> Self {
        self.vram_metrics = enabled;
        self
    }
    /// Reports whether detailed tracing was requested.
    pub fn tracing(self) -> bool {
        self.tracing
    }
    /// Reports whether serving counters were requested.
    pub fn metrics(self) -> bool {
        self.metrics
    }
    /// Reports whether process-memory sampling was requested.
    pub fn vram_metrics(self) -> bool {
        self.vram_metrics
    }
}

/// Requested graph scope as a report label. This type never enables execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphPolicy {
    #[default]
    None,
    Layer,
    Full,
}

/// Borrowed request facts supplied by the caller, without execution ownership.
/// The collector copies them only when the corresponding collection mode is on.
#[derive(Debug, Clone, Copy)]
pub struct RequestMetadata<'a> {
    pub model: &'a str,
    pub dtype: &'static str,
    pub max_new_tokens: usize,
    pub eos_token_ids: &'a [u32],
    pub optimize_kernel: bool,
    pub prefill_chunk_size: usize,
    pub graph_policy: GraphPolicy,
}

/// Per-step observations supplied by graph owners. Scope enforcement and
/// cumulative-counter subtraction remain the execution layer's responsibility.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GraphActivity {
    pub forward_graph_captures: usize,
    pub forward_graph_replays: usize,
    pub gdn_layer_graph_captures: usize,
    pub gdn_layer_graph_replays: usize,
    pub mlp_graph_captures: usize,
    pub mlp_graph_replays: usize,
}

impl GraphActivity {
    /// Adds observations for request summaries without changing execution state.
    pub fn merge(self, other: Self) -> Self {
        Self {
            forward_graph_captures: self.forward_graph_captures
                + other.forward_graph_captures,
            forward_graph_replays: self.forward_graph_replays
                + other.forward_graph_replays,
            gdn_layer_graph_captures: self.gdn_layer_graph_captures
                + other.gdn_layer_graph_captures,
            gdn_layer_graph_replays: self.gdn_layer_graph_replays
                + other.gdn_layer_graph_replays,
            mlp_graph_captures: self.mlp_graph_captures + other.mlp_graph_captures,
            mlp_graph_replays: self.mlp_graph_replays + other.mlp_graph_replays,
        }
    }
    /// Labels actual replay rather than inferring activity from a requested policy.
    pub fn execution(self) -> &'static str {
        if self.scope() == "none" {
            "eager"
        } else {
            "cuda_graph"
        }
    }
    /// Describes only observed replay boundaries for the existing wire schema.
    pub fn scope(self) -> &'static str {
        match (
            self.forward_graph_replays > 0,
            self.gdn_layer_graph_replays > 0 || self.mlp_graph_replays > 0,
        ) {
            (false, false) => "none",
            (false, true) => "layer",
            (true, false) => "full",
            (true, true) => "mixed",
        }
    }
}
