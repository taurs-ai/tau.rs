//! Converts model-owned counters/policy into reporting-only observations.

use crate::CudaGraphStats;
use tau_metrics::{GraphActivity, GraphPolicy};

impl From<crate::CudaGraphMode> for GraphPolicy {
    /// Converts the requested execution setting into an observation label only.
    fn from(mode: crate::CudaGraphMode) -> Self {
        match mode {
            crate::CudaGraphMode::None => Self::None,
            crate::CudaGraphMode::Layer => Self::Layer,
            crate::CudaGraphMode::Full => Self::Full,
        }
    }
}

impl From<CudaGraphStats> for GraphActivity {
    /// Copies the six counters; execution still owns validation and cache state.
    fn from(stats: CudaGraphStats) -> Self {
        Self {
            forward_graph_captures: stats.forward_graph_captures,
            forward_graph_replays: stats.forward_graph_replays,
            gdn_layer_graph_captures: stats.gdn_layer_graph_captures,
            gdn_layer_graph_replays: stats.gdn_layer_graph_replays,
            mlp_graph_captures: stats.mlp_graph_captures,
            mlp_graph_replays: stats.mlp_graph_replays,
        }
    }
}
