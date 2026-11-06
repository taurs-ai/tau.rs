//! MLP-owned prefill/decode graph slots, eligibility, capture and replay.
//! Keep the graph, retained buffers and lifecycle operations together. Ordinary
//! MLP forward remains computation-only, including inside a whole GDN capture.

use super::Mlp;
use crate::components::GraphFeedForward;
use crate::components::cuda::graph::CapturedGraph;
use crate::{CudaGraphStats, GraphPhase};
use candle::{DType, Module, Result as CandleResult, Tensor, Var};

pub(super) struct MlpGraph {
    graph: CapturedGraph<Box<dyn FnMut() -> CandleResult<()>>>,
    input: Var,
    output: Var,
    replays: usize,
}

impl std::fmt::Debug for MlpGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlpGraph")
            .field("replays", &self.replays)
            .finish_non_exhaustive()
    }
}

impl GraphFeedForward for Mlp {
    /// Contributes ordinary MLP operations to the model's complete decode program;
    /// the model owner performs capture rather than replaying this MLP's own graph.
    fn prepare_decode_program(
        &self,
        program: &mut crate::components::DecodeProgram,
        input: &Tensor,
    ) -> CandleResult<Tensor> {
        Mlp::prepare_decode_program(self, program, input)
    }
    fn supports_layer_graph(&self, xs: &Tensor) -> bool {
        let hidden = self.hidden_size;

        self.gate_up_projection.is_packed()
            && self.dtype == DType::BF16
            && xs.dtype() == self.dtype
            && xs.is_contiguous()
            && xs.dims3().is_ok_and(|(batch, length, width)| {
                batch == 1 && length > 0 && width == hidden
            })
            && xs.device().is_cuda()
            && xs.device().same_device(&self.device)
    }

    /// Strict graph execution for attention-layer MLPs. The decoder has already
    /// selected this path; unsupported inputs are errors, never hidden eager work.
    /// GDN-layer MLPs use ordinary forward inside their enclosing layer capture.
    fn forward_graph(
        &mut self,
        xs: &Tensor,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        self.ensure_graph(xs, phase)?;
        self.replay_graph(xs, phase)
    }

    /// Counts one capture for each retained prefill/decode graph and adds their
    /// recorded replays. Empty graph slots contribute nothing to the snapshot.
    fn cuda_graph_counts(&self) -> CudaGraphStats {
        [&self.decode_graph, &self.prefill_graph]
            .into_iter()
            .flatten()
            .fold(CudaGraphStats::default(), |counts, graph| {
                counts.merge(CudaGraphStats {
                    mlp_graph_captures: 1,
                    mlp_graph_replays: graph.replays,
                    ..Default::default()
                })
            })
    }
}

impl Mlp {
    /// Cold path: validate an existing graph, or warm up and capture one.
    fn ensure_graph(&mut self, xs: &Tensor, phase: GraphPhase) -> CandleResult<()> {
        // The dispatcher owns fallback; a direct graph call must honor its name.
        if !self.supports_layer_graph(xs) {
            candle::bail!("MLP does not support this graph input")
        }
        // Prefill processes a fixed-size chunk, while decode processes one token.
        // Give each phase its own graph and buffers so alternating between them
        // does not replace a graph that will be reused on later calls or requests.
        let captured = match phase {
            GraphPhase::Prefill => &self.prefill_graph,
            GraphPhase::Decode => &self.decode_graph,
        };
        // Capture fixes the operation shapes and buffer addresses. Once this
        // phase has a graph, a different input shape is a contract error: this
        // owner is not a cache of graphs for arbitrary sequence lengths.
        if captured
            .as_ref()
            .is_some_and(|graph| graph.input.shape() != xs.shape())
        {
            candle::bail!("MLP graph input shape changed within a fixed graph phase")
        }
        if captured.is_none() {
            // Pay setup costs lazily on the first eligible call for this phase.
            // Event tracking must have been disabled before tensor allocation;
            // ordinary per-allocation CUDA events are not safe for this replay
            // lifecycle, so changing the device policy here would be too late.
            let device = xs.device();
            if device.as_cuda_device()?.is_event_tracking() {
                candle::bail!(
                    "MLP capture requires a capture device before loading tensors"
                )
            }
            // The graph must always read/write the same storage, even when each
            // caller supplies a newly allocated xs. Var::set copies values into
            // these persistent slots without changing their addresses. Seed the
            // input now so warm-up runs the exact computation we intend to capture.
            let input = Var::zeros(xs.shape(), xs.dtype(), device)?;
            input.set(xs)?;
            let output = Var::zeros(xs.shape(), xs.dtype(), device)?;
            // Retain the weights and slot handles in the closure for the graph's
            // entire lifetime. Cloning the MLP shares immutable weights, not its
            // graphs. Calling ordinary forward here captures only MLP computation
            // plus the output-slot copy; it cannot recursively capture a graph.
            let mut operation: Box<dyn FnMut() -> CandleResult<()>> = {
                let mlp = self.clone();
                let input = input.as_detached_tensor();
                let output = output.clone();
                Box::new(move || output.set(&mlp.forward(&input)?))
            };
            {
                // Run once eagerly to initialize CUDA work and cache Candle's
                // host-to-device kernel metadata before capture. Scope caching to
                // this operation and finish warm-up before starting capture. MLP
                // has no recurrent/KV state, so no state needs restoring afterward.
                let _cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
                operation()?;
                device.synchronize()?;
            }
            // Capture records the closure's GPU work and creates an executable
            // graph. Later replays launch that recorded work without calling the
            // Rust closure again. forward_graph calls replay_graph after this
            // method returns, including on the first invocation.
            // SAFETY: The closure owns its weights and fixed input/output storage.
            // All tensors predate capture on the private untracked device; use and
            // destruction remain serialized on this stream/thread by the model owner.
            let graph = unsafe { CapturedGraph::capture(device, operation) }?;
            // Publish the graph only after successful capture. Warm-up/capture
            // are setup, not replays, so the replay counter starts at zero.
            let slot = match phase {
                GraphPhase::Prefill => &mut self.prefill_graph,
                GraphPhase::Decode => &mut self.decode_graph,
            };
            *slot = Some(MlpGraph {
                graph,
                input,
                output,
                replays: 0,
            });
        }
        Ok(())
    }

    /// Hot path: copy activations into fixed storage and enqueue one replay.
    /// Called only after ensure_graph has validated/prepared the selected phase.
    fn replay_graph(&mut self, xs: &Tensor, phase: GraphPhase) -> CandleResult<Tensor> {
        // The selected slot now exists, either from this call or an earlier one.
        let graph = match phase {
            GraphPhase::Prefill => self.prefill_graph.as_mut().unwrap(),
            GraphPhase::Decode => self.decode_graph.as_mut().unwrap(),
        };
        // Copy this call's activations into the fixed input, then enqueue replay
        // on the same stream. Stream ordering puts the copy before computation;
        // no host synchronization is needed here. Count successful replay launches.
        graph.input.set(xs)?;
        graph.graph.replay()?;
        graph.replays += 1;
        // Return a tensor sharing the output slot, not an independent snapshot.
        // Downstream work must consume it on the serialized stream before a later
        // replay overwrites the slot; "detached" does not copy the underlying data.
        let output = graph.output.as_detached_tensor();
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
