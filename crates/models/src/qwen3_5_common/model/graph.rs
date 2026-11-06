//! Whole-forward graph ownership, prepared operation lifetime and state handoff.
use super::*;
use crate::components::{CapturedGraph, DecodeProgram, write_metadata};
use std::{cell::RefCell, rc::Rc};
use tau_metrics::timing::DecodeTimer;

type Operation = Box<dyn FnMut() -> CandleResult<()>>;
type CacheKey = Option<(candle::TensorId, candle::TensorId, usize, bool)>;

pub(super) struct FullDecodeGraph {
    // Drop the graph first so outstanding replay finishes before retained slots
    // and metadata owned by the remaining fields can be released.
    graph: Option<CapturedGraph<Operation>>,
    program: Rc<RefCell<DecodeProgram>>,
    metadata: Tensor,
    output: Tensor,
    pub(super) keys: Vec<CacheKey>,
}

impl std::fmt::Debug for FullDecodeGraph {
    /// Reports capture status without dumping retained tensors or driver handles.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullDecodeGraph")
            .field("captured", &self.graph.is_some())
            .finish_non_exhaustive()
    }
}

impl FullDecodeGraph {
    /// Records the already-warmed program once and rejects allocation/free nodes.
    /// Returns whether a new graph was created so capture counters stay accurate;
    /// later steps reuse the existing graph without recording the program again.
    fn ensure_capture(&mut self, device: &Device) -> CandleResult<bool> {
        if self.graph.is_some() {
            return Ok(false);
        }
        let compute = Rc::clone(&self.program);
        let operation: Operation = Box::new(move || compute.borrow().run());
        // SAFETY: The warmed program retains all weights, input/output slots and
        // mutable state, on this owner's private serialized stream. No allocation
        // occurs while recording. Replay/metadata/reset/drop use the same stream.
        let graph = unsafe { CapturedGraph::capture(device, operation) }?;
        if graph.allocation_nodes()? != 0 {
            candle::bail!("prepared forward unexpectedly captured allocation nodes")
        }
        self.graph = Some(graph);
        eprintln!(
            "tau-models CUDA graph: captured complete decode forward; allocation/free nodes=0"
        );
        Ok(true)
    }
}

impl<F: GraphFeedForward> HybridCausalLm<F> {
    /// Traces the same decode math with setup outside the full replay interval.
    /// Narrower modes bracket the complete ordinary forward, including any cold
    /// layer capture; capture counters identify those steps in the trace.
    pub(crate) fn forward_decode_token_timed(
        &mut self,
        token: u32,
        offset: usize,
        timing: &mut tau_metrics::timing::DecodeTimer,
    ) -> CandleResult<Tensor> {
        if let Some(logits) = self.try_full_decode(token, offset, true, Some(timing))? {
            return Ok(logits);
        }
        let input = Tensor::new(&[[token]], &self.model.device)?;
        timing.begin_forward("model_forward")?;
        let logits = self.forward_decode(&input, offset)?;
        timing.end_forward()?;
        Ok(logits)
    }

    /// Select before execution. A missing capability is ordinary policy behavior;
    /// once preparation starts, failures propagate without eager retries.
    /// Cache identities guard captured addresses. A supported step validates its
    /// offset, seeds native state, updates metadata, then runs the prepared eager
    /// program or graph and publishes the new logical KV length. `None` means the
    /// caller may select narrower coverage before any step has executed.
    pub(super) fn try_full_decode(
        &mut self,
        token: u32,
        offset: usize,
        capture: bool,
        mut timing: Option<&mut DecodeTimer>,
    ) -> CandleResult<Option<Tensor>> {
        if self.cuda_graph != CudaGraphMode::Full {
            return Ok(None);
        }
        let (vocabulary, hidden) = self.model.embed_tokens.embeddings().dims2()?;
        if token as usize >= vocabulary || offset > u32::MAX as usize {
            candle::bail!("decode token or position out of bounds")
        }
        if self.full_graph.as_ref().is_some_and(|g| {
            g.keys
                .iter()
                .zip(&self.model.layers)
                .any(|(key, layer)| *key != layer.decode_cache_key())
        }) {
            self.full_graph = None;
        }
        if self.full_graph.is_none() {
            let probe = self
                .model
                .embed_tokens
                .embeddings()
                .narrow(0, 0, 1)?
                .reshape((1, 1, hidden))?;
            if !self.model.device.is_cuda()
                || self.model.dtype != DType::BF16
                || !self
                    .model
                    .layers
                    .iter()
                    .all(|l| l.supports_full_graph(&probe))
            {
                return Ok(None);
            }
        }
        for layer in &self.model.layers {
            layer.validate_full_decode(offset)?;
        }
        if self.full_graph.is_none() {
            self.full_graph = Some(self.prepare_full_decode(token, offset)?);
        }
        for layer in &mut self.model.layers {
            layer.ensure_full_decode_state()?;
        }
        let owner = self.full_graph.as_mut().unwrap();
        write_metadata(&owner.metadata, [token, offset as u32])?;
        if capture && owner.ensure_capture(&self.model.device)? {
            self.full_graph_stats.forward_graph_captures += 1;
        }
        // Diagnostic events surround only execution, never graph recording,
        // library warm-up, metadata copies or state restoration.
        if let Some(timer) = timing.as_deref_mut() {
            timer.begin_forward(if capture {
                "complete_forward_replay"
            } else {
                "prepared_eager"
            })?;
        }
        if capture {
            owner.graph.as_mut().unwrap().replay()?;
            self.full_graph_stats.forward_graph_replays += 1;
        } else {
            owner.program.borrow().run()?;
        }
        if let Some(timer) = timing {
            timer.end_forward()?;
        }
        for layer in &mut self.model.layers {
            layer.publish_full_decode(offset);
        }
        Ok(Some(owner.output.clone()))
    }

    /// Builds the complete embedding-to-logits operation list and its fixed slots.
    /// Warm-up initializes library/kernel resources before capture. GDN snapshots
    /// restore the live state afterward; warming touches only the unused next KV
    /// slot, which the real decode step overwrites before consuming it.
    fn prepare_full_decode(
        &mut self,
        token: u32,
        offset: usize,
    ) -> CandleResult<FullDecodeGraph> {
        let device = self.model.device.clone();
        let mut program = DecodeProgram::new(&device)?;
        let metadata = Tensor::zeros(2, DType::U32, &device)?;
        write_metadata(&metadata, [token, offset as u32])?;
        let weights = self.model.embed_tokens.embeddings();
        let hidden_size = weights.dim(1)?;
        let hidden = program.buffer((1, 1, hidden_size))?;
        let mut hidden = program.elementwise(
            0,
            &[weights, &metadata],
            hidden,
            hidden_size,
            hidden_size,
            0,
            false,
        )?;
        let snapshots = self
            .model
            .layers
            .iter_mut()
            .map(|l| l.prepare_full_decode())
            .collect::<CandleResult<Vec<_>>>()?;
        for (layer_index, layer) in self.model.layers.iter_mut().enumerate() {
            #[cfg(test)]
            let _capture = crate::test_support::tensor_capture::scope(format!(
                "layer-{layer_index}-"
            ));
            #[cfg(not(test))]
            let _ = layer_index;
            hidden = layer.prepare_decode_program(&mut program, &hidden, &metadata)?;
        }
        let normalized = self
            .model
            .norm
            .prepare_decode_program(&mut program, &hidden)?;
        let output = program.linear(&normalized, &self.lm_head)?;
        // Warm-up writes only the unused KV slot at `offset`; live prefix and
        // logical lengths are unchanged. Restore every GDN state before capture.
        let warmed = program.run().and_then(|()| device.synchronize());
        let restored =
            self.model.layers.iter_mut().zip(&snapshots).try_for_each(
                |(layer, state)| layer.restore_full_decode(state.as_ref()),
            );
        warmed?;
        restored?;
        let program = Rc::new(RefCell::new(program));
        Ok(FullDecodeGraph {
            graph: None,
            program,
            metadata,
            output,
            keys: self
                .model
                .layers
                .iter()
                .map(|l| l.decode_cache_key())
                .collect(),
        })
    }

    /// Reference for exact same-arithmetic graph/eager tests, without a new CLI.
    /// Uses the same buffers, metadata and state handoff but queues the operation
    /// list directly, isolating replay from changes to attention arithmetic.
    #[cfg(test)]
    pub(crate) fn forward_prepared_eager(
        &mut self,
        token: u32,
        offset: usize,
    ) -> CandleResult<Tensor> {
        self.try_full_decode(token, offset, false, None)?
            .ok_or_else(|| candle::Error::Msg("prepared decode unsupported".to_owned()))
    }
}
