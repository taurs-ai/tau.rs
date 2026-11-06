//! Reusable neural-network components shared by model-family implementations.
//!
//! Component modules own tensor algorithms and model-independent construction
//! configs. Model-family modules translate serialized checkpoint configuration into
//! these interfaces and provide the expected checkpoint subtree.

mod gated_attention;
mod gated_delta_net;
mod mlp;
mod rms_norm;
mod rope;
mod sparse_moe;

// Native operators accelerate measured model-specific hot paths. Components
// select a matching operator when optimized-kernel mode is enabled, preserving
// explicit tensor storage, allocation, and stream contracts.
mod cuda;

pub(crate) use cuda::decode::DecodeProgram;
pub(crate) use cuda::decode::write_metadata;
pub(crate) use cuda::graph::CapturedGraph;
pub use cuda::graph::new_capture_device;
#[cfg(test)]
pub(crate) use cuda::graph::tests::{
    address as graph_tensor_address, node_types as graph_node_types,
};

pub(crate) use gated_attention::{GatedAttention, GatedAttentionConfig};
pub(crate) use gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig, GdnPrefillState};
pub(crate) use mlp::Mlp;
pub(crate) use rms_norm::CenteredRmsNorm;
pub(crate) use rope::{PartialRotaryConfig, PartialRotaryEmbedding};
pub(crate) use sparse_moe::SparseMoeBlock;
pub(crate) use sparse_moe::{
    RouterScoreFunction, SharedExpertConfig, SharedExpertGateConfig, SparseMoeConfig,
    SparseMoeWeightLayout,
};

/// Shared decoder feed-forward interface. Module::forward is always ordinary
/// computation; forward_graph is strict graph execution, never eager fallback.
/// The decoder owns eligibility/fallback decisions for each execution phase.
pub(crate) trait GraphFeedForward: candle::Module + Clone + 'static {
    /// Executes an explicit eager decode step. Dense blocks use their ordinary
    /// MLP; sparse blocks can retain a single-token runner independently of
    /// same-shaped prefill tails, whose arithmetic follows the prefill contract.
    fn forward_decode(&self, input: &candle::Tensor) -> candle::Result<candle::Tensor> {
        self.forward(input)
    }
    /// Appends stateless feed-forward operations to a complete decode program.
    /// Blocks without a prepared implementation reject this path; capability
    /// checks must select narrower execution before capture or state mutation.
    fn prepare_decode_program(
        &self,
        _program: &mut DecodeProgram,
        _input: &candle::Tensor,
    ) -> candle::Result<candle::Tensor> {
        candle::bail!("feed-forward block does not support prepared decode")
    }
    /// Complete decode uses prepared operations, independently of layer capture.
    /// Keeping these capabilities separate lets MoE support full decode while
    /// multi-token prefill and legacy layer execution retain ordinary dispatch.
    fn supports_prepared_decode(&self, xs: &candle::Tensor) -> bool {
        self.supports_layer_graph(xs)
            && xs
                .dims3()
                .is_ok_and(|(batch, length, _)| batch == 1 && length == 1)
    }
    /// Full-layer capture requires a validated stateless dense feed-forward path.
    fn supports_layer_graph(&self, _xs: &candle::Tensor) -> bool {
        false
    }
    fn forward_graph(
        &mut self,
        _xs: &candle::Tensor,
        _phase: crate::GraphPhase,
    ) -> candle::Result<candle::Tensor> {
        candle::bail!("feed-forward block does not support CUDA graphs")
    }

    /// Reports captures/replays owned by this feed-forward block. Blocks without
    /// their own graph implementation report zero, including when run eagerly
    /// inside a larger captured forward owned by the model.
    fn cuda_graph_counts(&self) -> crate::CudaGraphStats {
        crate::CudaGraphStats::default()
    }
}
