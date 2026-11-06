//! Whole-layer GDN graph eligibility, capture, replay and observations.
//!
//! The decoder retains separate prefill/decode graph slots. GDN owns native
//! recurrent state; captured prefill keeps explicit state input/output buffers.

use super::{HybridDecoderLayer, TokenMixer, forward_block};
use crate::components::{CapturedGraph, GdnPrefillState, GraphFeedForward};
use crate::{CudaGraphStats, GraphPhase};
use candle::{Result as CandleResult, Tensor, Var};

/// Retains a complete layer graph and its buffers. Multi-token prefill has
/// separate state inputs/outputs; native decode state remains owned by GDN.
pub(super) struct GdnLayerGraph {
    graph: CapturedGraph<Box<dyn FnMut() -> CandleResult<()>>>,
    input: Var,
    output: Var,
    replays: usize,
    prefill_state: Option<GdnPrefillState>,
}

impl std::fmt::Debug for GdnLayerGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GdnLayerGraph")
            .field("replays", &self.replays)
            .finish_non_exhaustive()
    }
}

type GdnComputation = dyn FnMut(&Tensor) -> CandleResult<Tensor>;

impl<F: GraphFeedForward> HybridDecoderLayer<F> {
    /// Eligibility only: no cache mutation, warm-up, capture or replay.
    pub(super) fn supports_gdn_graph(
        &self,
        xs: &Tensor,
        phase: GraphPhase,
    ) -> CandleResult<bool> {
        let TokenMixer::Linear(gdn) = &self.token_mixer else {
            return Ok(false);
        };
        // Choose computation by token count, but graph ownership by phase below.
        // A full prefill chunk of size one can use the native single-token GDN
        // kernel too; it still belongs to the prefill graph, not the decode graph.
        // Multi-token chunks capture the same composed prefill path as eager.
        let native = xs.dim(1)? == 1;
        let supported = if native {
            gdn.supports_prepared_decode(xs)
        } else {
            phase == GraphPhase::Prefill && gdn.supports_prefill_graph(xs)
        };
        Ok(supported && self.feed_forward.supports_layer_graph(xs))
    }

    /// Strict whole-layer graph execution. Eligibility/fallback belongs to
    /// forward_graph_with_eager_fallback; errors here never retry the layer eagerly.
    pub(super) fn forward_gdn_graph(
        &mut self,
        xs: &Tensor,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        // Validate this input and capture a missing graph for the selected phase.
        self.ensure_gdn_graph(xs, phase)?;
        // Replay the prepared CUDA graph and update the layer's state.
        self.replay_gdn_graph(xs, phase)
    }

    /// Validates input compatibility and captures a graph only when its slot is empty.
    fn ensure_gdn_graph(&mut self, xs: &Tensor, phase: GraphPhase) -> CandleResult<()> {
        if !self.supports_gdn_graph(xs, phase)? {
            candle::bail!("GDN layer does not support this graph input/phase")
        }
        // Capture needs the special device policy established before weights and
        // buffers were allocated. Disabling per-allocation event tracking here
        // would be too late to make existing tensors safe for this graph lifecycle.
        let device = xs.device();
        if device.as_cuda_device()?.is_event_tracking() {
            candle::bail!(
                "GDN layer capture requires a capture device before loading tensors"
            )
        }
        // Retain one graph per phase across calls and request cache resets. A
        // prefill graph has the configured full-chunk shape; decode has one token.
        let slot = match phase {
            GraphPhase::Prefill => &self.prefill_graph,
            GraphPhase::Decode => &self.decode_graph,
        };
        // Replays keep the captured shapes and buffer addresses. A new shape in
        // an occupied slot is a contract error, not a request to resize its buffers
        // or capture an additional graph. Short prefill tails bypass this method.
        if slot
            .as_ref()
            .is_some_and(|graph| graph.input.shape() != xs.shape())
        {
            candle::bail!(
                "GDN layer graph input shape changed within a fixed graph phase"
            )
        }
        if slot.is_none() {
            let graph = self.capture_gdn_graph(xs)?;
            let slot = match phase {
                GraphPhase::Prefill => &mut self.prefill_graph,
                GraphPhase::Decode => &mut self.decode_graph,
            };
            *slot = Some(graph);
        }
        Ok(())
    }

    /// Allocates buffers, warms up, and captures a GDN layer after validation.
    /// The caller publishes the returned graph only after capture succeeds.
    fn capture_gdn_graph(&mut self, xs: &Tensor) -> CandleResult<GdnLayerGraph> {
        let TokenMixer::Linear(gdn) = &mut self.token_mixer else {
            unreachable!("validated GDN graph")
        };
        let native = xs.dim(1)? == 1;
        let device = xs.device();
        // First eligible call for this phase: allocate fixed activation slots
        // and seed the input for warm-up. Later calls copy new values into the
        // same storage; neither graph input nor output addresses may change.
        let input = Var::zeros(xs.shape(), xs.dtype(), device)?;
        input.set(xs)?;
        let output = Var::zeros(xs.shape(), xs.dtype(), device)?;
        // Multi-token prefill reads separate convolution/recurrent input
        // slots and writes output slots. Native single-token computation
        // instead updates the GDN runner's retained state buffers in place.
        let prefill_state = if native {
            None
        } else {
            Some(gdn.new_prefill_state()?)
        };
        let mut original_state = None;
        let mut compute_gdn: Box<GdnComputation> = if let Some(state) = &prefill_state {
            // Start from the layer's current state (or zeros for a fresh
            // request). The closure shares weights and fixed slots, but
            // uses private prefill cache handles during capture.
            gdn.seed_prefill_state(state)?;
            Box::new(gdn.prepared_prefill_operation(state))
        } else {
            // Prepare native state outside capture, then take independent
            // snapshots so we can undo the state advance caused by warm-up.
            gdn.prepare_optimized_decode()?;
            original_state = Some(gdn.snapshot_optimized_state()?);
            Box::new(gdn.prepared_decode_operation()?)
        };
        // Own weight/module handles in the closure so captured resources stay
        // alive with the graph, without borrowing the decoder between calls.
        let input_norm = self.input_layernorm.clone();
        let post_norm = self.post_attention_layernorm.clone();
        let mlp = self.feed_forward.clone();
        let mut operation: Box<dyn FnMut() -> CandleResult<()>> = {
            let input = input.as_detached_tensor();
            let output = output.clone();
            Box::new(move || {
                // Use the same math as eager, with prepared GDN state. Ordinary
                // MLP forward is recorded inside this graph, never nested replay.
                output.set(&forward_block(
                    &input,
                    &input_norm,
                    &post_norm,
                    &mut compute_gdn,
                    |hidden| mlp.forward(hidden),
                )?)
            })
        };
        {
            // Execute the exact operation eagerly once, caching Candle's
            // host-to-device kernel metadata needed during capture. Finish
            // warm-up before restoring state and recording the graph; keep
            // this cache policy scoped to the operation being captured.
            let _cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
            operation()?;
            device.synchronize()?;
        }
        // Native warm-up already consumed this token, so restore its starting
        // state before the real replay. Multi-token prefill needs no restore:
        // warm-up writes separate outputs and leaves its state inputs intact.
        if let Some((conv, recurrent)) = original_state {
            gdn.seed_optimized_state(Some(&conv), Some(&recurrent))?;
        }
        // Capture records GPU work rather than executing another model step.
        // The closure is retained to keep its resources alive, but replay will
        // not call it again. forward_gdn_graph calls replay_gdn_graph after
        // this method returns. Publish the graph only after capture succeeds.
        // SAFETY: The closure owns weights and fixed input/output/state buffers.
        // All temporary tensors are released inside capture. The owner serializes
        // state handoff, replay, output consumption and teardown on this stream.
        let graph = unsafe { CapturedGraph::capture(device, operation) }?;
        Ok(GdnLayerGraph {
            graph,
            input,
            output,
            replays: 0,
            prefill_state,
        })
    }

    /// Hot path: seed current state, copy input, replay, and publish final state.
    /// Called only after ensure_gdn_graph has established the shape and resources.
    fn replay_gdn_graph(
        &mut self,
        xs: &Tensor,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        let TokenMixer::Linear(gdn) = &mut self.token_mixer else {
            candle::bail!("GDN replay requires a GDN layer")
        };
        let slot = match phase {
            GraphPhase::Prefill => &mut self.prefill_graph,
            GraphPhase::Decode => &mut self.decode_graph,
        };
        // From here, newly captured and previously captured graphs use the same
        // path. Preparation stays outside capture because replay cannot execute
        // Rust cache bookkeeping or decide whether a request's state was reset.
        let graph = slot.as_mut().expect("ensure_gdn_graph prepared this phase");
        if let Some(state) = &graph.prefill_state {
            // Copy the latest authoritative state into the graph's input slots:
            // previous chunk outputs, eager/native state, or zeros after a reset.
            gdn.seed_prefill_state(state)?;
        } else {
            // Import eager/prefill state or zero after reset only when needed;
            // otherwise preserve the native buffers advanced by the last token.
            gdn.prepare_optimized_decode()?;
        }
        // Enqueue the new activations and replay on the same stream. Stream order
        // makes state/input copies finish before the recorded computation reads
        // them. Count successful launches; execution is asynchronous at this point.
        graph.input.set(xs)?;
        graph.graph.replay()?;
        graph.replays += 1;
        if let Some(state) = &graph.prefill_state {
            // Point the original GDN's cache handles at the output state slots.
            // No host readback is needed: later chunks, eager tails and native
            // decode consume that state in stream order. Native replay already
            // updated its authoritative state in place and needs no publication.
            gdn.publish_prefill_state(state);
        }
        // This is a view of the reusable output slot, not an independent copy.
        // The caller must consume it in stream order before another replay of
        // this layer overwrites it, and skip the ordinary layer path for this call.
        Ok(graph.output.as_detached_tensor())
    }

    /// Counts captures and replays across both phases; traces record step deltas.
    /// Begin with the feed-forward block's counters, then add one capture and
    /// the replay count for each retained GDN prefill/decode graph. Complete
    /// forward graphs are counted separately by the model owner.
    pub(crate) fn cuda_graph_counts(&self) -> CudaGraphStats {
        [&self.decode_graph, &self.prefill_graph]
            .into_iter()
            .flatten()
            .fold(self.feed_forward.cuda_graph_counts(), |counts, graph| {
                counts.merge(CudaGraphStats {
                    gdn_layer_graph_captures: 1,
                    gdn_layer_graph_replays: graph.replays,
                    ..Default::default()
                })
            })
    }
}

#[cfg(test)]
mod tests;
