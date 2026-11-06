//! GDN state handoff and prepared computations for eager decode and CUDA graphs.
//!
//! The layer still owns all mutable state. These methods prepare/reseed its native
//! runner or build closures over fixed prefill state inputs/outputs. Actual graph
//! capture and replay remain in the decoder owner.

use super::{GatedDeltaNet, GdnPrefillState};
use candle::{DType, Result as CandleResult, Tensor, Var};

impl GatedDeltaNet {
    /// Appends projections, native GDN and the output projection to full decode.
    /// The native runner shares the layer's convolution/recurrent state and writes
    /// into a retained output slot, avoiding a new output allocation on each run.
    pub(crate) fn prepare_decode_program(
        &mut self,
        program: &mut crate::components::DecodeProgram,
        input: &Tensor,
    ) -> CandleResult<Tensor> {
        self.prepare_optimized_decode()?;
        let op = self
            .optimized_decode
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("native GDN unavailable".to_owned()))?;
        let qkvz = program.linear(input, &op.qkvz_proj)?;
        let ba = program.linear(input, &op.ba_proj)?;
        let output = program.buffer((1, 1, self.value_width))?;
        let target = output.clone();
        let runner = op.runner.clone();
        program.push(Box::new(move || runner.forward_into(&qkvz, &ba, &target)));
        program.linear(&output, &self.out_proj)
    }

    /// Restores native state into Candle tensors before a composed prefill/fallback
    /// call.
    pub(super) fn materialize_optimized_state(&mut self) -> CandleResult<()> {
        if !self.optimized_state_active {
            return Ok(());
        }
        let optimized = self.optimized_decode.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "Gated DeltaNet native state is active without a runner".to_owned(),
            )
        })?;
        let (conv_state, recurrent_state) = optimized.runner.export_state()?;
        self.conv_state = Some(conv_state);
        self.recurrent_state = Some(recurrent_state);
        self.optimized_state_active = false;
        Ok(())
    }

    /// A prefill graph captures the same multi-token computation as eager prefill.
    pub(crate) fn supports_prefill_graph(&self, xs: &Tensor) -> bool {
        self.optimized_decode.is_some()
            && xs.dtype() == DType::BF16
            && xs.device().same_device(self.conv_weight.device())
            && xs.is_contiguous()
            && xs.dims3().is_ok_and(|(batch, length, hidden)| {
                batch == 1 && length > 1 && hidden == self.hidden_size
            })
    }

    pub(crate) fn new_prefill_state(&self) -> CandleResult<GdnPrefillState> {
        let device = self.conv_weight.device();
        let conv = (1, self.conv_width, self.conv_kernel_size);
        let recurrent = (
            1,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        );
        Ok(GdnPrefillState {
            conv_input: Var::zeros(conv, self.conv_weight.dtype(), device)?,
            recurrent_input: Var::zeros(recurrent, DType::F32, device)?,
            conv_output: Var::zeros(conv, self.conv_weight.dtype(), device)?,
            recurrent_output: Var::zeros(recurrent, DType::F32, device)?,
        })
    }

    /// Seeds each replay from the authoritative eager/native/previous-chunk state.
    /// This runs outside capture and preserves all graph buffer addresses.
    pub(crate) fn seed_prefill_state(
        &mut self,
        state: &GdnPrefillState,
    ) -> CandleResult<()> {
        self.materialize_optimized_state()?;
        match (&self.conv_state, &self.recurrent_state) {
            (Some(conv), Some(recurrent)) => {
                state.conv_input.set(conv)?;
                state.recurrent_input.set(recurrent)?;
            }
            (None, None) => {
                state.conv_input.set(&state.conv_input.zeros_like()?)?;
                state
                    .recurrent_input
                    .set(&state.recurrent_input.zeros_like()?)?;
            }
            _ => candle::bail!("incomplete Gated DeltaNet prefill state"),
        }
        Ok(())
    }

    /// Makes the graph's final state available to an eager tail or native decode.
    pub(crate) fn publish_prefill_state(&mut self, state: &GdnPrefillState) {
        self.conv_state = Some(state.conv_output.as_detached_tensor());
        self.recurrent_state = Some(state.recurrent_output.as_detached_tensor());
        self.optimized_state_active = false;
    }

    /// Builds a GDN computation closure for multi-token prefill; this method
    /// itself neither executes the computation nor captures a CUDA graph.
    /// The decoder owns warm-up/capture/replay and seeds/publishes state outside
    /// the graph. The returned closure owns its handles (`use<>` borrows nothing
    /// from this layer), leaving the original layer available for state handoff.
    pub(crate) fn prepared_prefill_operation(
        &self,
        state: &GdnPrefillState,
    ) -> impl FnMut(&Tensor) -> CandleResult<Tensor> + use<> {
        // Clone handles to the four persistent buffers, not their contents:
        // convolution history and recurrent memory each have an input/output pair.
        // The decoder can update these same input buffers before every replay.
        let state = state.clone();
        // Build a private prefill GDN whose cache assignments cannot change
        // the original layer's cache fields. Weight clones share immutable tensor
        // storage; this does not allocate a second copy of the model's weights.
        let mut compute = Self {
            in_proj_qkv: self.in_proj_qkv.clone(),
            in_proj_z: self.in_proj_z.clone(),
            in_proj_b: self.in_proj_b.clone(),
            in_proj_a: self.in_proj_a.clone(),
            conv_weight: self.conv_weight.clone(),
            dt_bias: self.dt_bias.clone(),
            a_log: self.a_log.clone(),
            norm: self.norm.clone(),
            out_proj: self.out_proj.clone(),
            // Preserve the dimensions and activations used by ordinary prefill.
            hidden_size: self.hidden_size,
            conv_activation: self.conv_activation,
            gate_activation: self.gate_activation,
            num_key_heads: self.num_key_heads,
            num_value_heads: self.num_value_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            key_width: self.key_width,
            value_width: self.value_width,
            conv_width: self.conv_width,
            conv_kernel_size: self.conv_kernel_size,
            // Each invocation attaches the supplied state inputs below. Do not
            // inherit live sequence caches or the single-token native runner:
            // this closure must not share that runner's mutable decode workspace.
            conv_state: None,
            recurrent_state: None,
            optimize_prefill: self.optimize_prefill,
            reference_precision: self.reference_precision,
            #[cfg(test)]
            prefill_qk_rounding_override: self.prefill_qk_rounding_override,
            optimized_decode: None,
            optimized_state_active: false,
        };
        // Rust executes this closure during eager warm-up and again during capture.
        // Replay runs only the recorded GPU work, not these Rust assignments. The
        // recorded operations therefore must read/write the fixed state buffers.
        move |xs| {
            // Attach views of the incoming state without copying or modifying it.
            // Always start from these inputs, not a previous warm-up's outputs.
            // Separate output slots ensure warm-up cannot advance the input state
            // and accidentally make the first real replay process the chunk twice.
            compute.conv_state = Some(state.conv_input.as_detached_tensor());
            compute.recurrent_state = Some(state.recurrent_input.as_detached_tensor());
            // Run projections, causal convolution, gated recurrence, normalization,
            // and output projection for this chunk. The composed path uses the
            // same prefill convolution policy as eager execution. It produces
            // hidden activations and replaces compute's caches with the final state.
            let output = compute.forward_composed(xs)?;
            // Copy final state into persistent output slots; these device copies
            // become graph operations too. take() empties the private cache fields
            // and releases their temporary tensor handles after each copy is queued,
            // so a successful capture does not retain graph-created state temporaries
            // in the closure. Successful computation populates both caches.
            state.conv_output.set(&compute.conv_state.take().unwrap())?;
            state
                .recurrent_output
                .set(&compute.recurrent_state.take().unwrap())?;
            // Return GDN hidden activations, not logits or a complete decoder layer.
            // The outer graph adds layer norms, residuals and MLP computation. After
            // replay, the decoder publishes the state output slots on the original
            // GDN so the next chunk, eager tail or native decode can continue.
            Ok(output)
        }
    }

    /// Copies prefill state into retained native buffers, or zeroes them when absent.
    ///
    /// This is an explicit reseed even when native state is already active. Call it
    /// outside capture, ordered before decode/replay on Candle's stream. Successful
    /// seeding makes the native buffers authoritative without replacing their storage.
    pub(crate) fn seed_optimized_state(
        &mut self,
        conv_state: Option<&Tensor>,
        recurrent_state: Option<&Tensor>,
    ) -> CandleResult<()> {
        let optimized = self.optimized_decode.as_ref().ok_or_else(|| {
            candle::Error::Msg("Gated DeltaNet has no native decode runner".to_owned())
        })?;
        optimized.runner.seed_state(conv_state, recurrent_state)?;
        self.conv_state = None;
        self.recurrent_state = None;
        self.optimized_state_active = true;
        Ok(())
    }

    /// Prepares native decode once, preserving state on subsequent calls.
    ///
    /// Keep this outside capture: the first call can enqueue state copies or zeros.
    pub(crate) fn prepare_optimized_decode(&mut self) -> CandleResult<()> {
        if !self.optimized_state_active {
            let conv_state = self.conv_state.clone();
            let recurrent_state = self.recurrent_state.clone();
            self.seed_optimized_state(conv_state.as_ref(), recurrent_state.as_ref())?;
        }
        Ok(())
    }

    /// Runs only the prepared native computation, without initialization or fallback.
    ///
    /// This is the prepared decode boundary. Preparation, reseeding and cache reset
    /// belong outside it; replay will not execute these host-side checks.
    pub(super) fn forward_prepared_decode(&self, xs: &Tensor) -> CandleResult<Tensor> {
        if !self.optimized_state_active {
            candle::bail!("Gated DeltaNet native decode state is not prepared")
        }
        let optimized = self.optimized_decode.as_ref().ok_or_else(|| {
            candle::Error::Msg("Gated DeltaNet has no native decode runner".to_owned())
        })?;
        if !optimized.supports_input(xs) {
            candle::bail!(
                "Gated DeltaNet input does not support prepared native decode"
            )
        }
        optimized.forward(xs, &self.out_proj)
    }

    /// Checks the native contract without changing state or creating a graph.
    pub(crate) fn supports_prepared_decode(&self, xs: &Tensor) -> bool {
        self.optimized_decode
            .as_ref()
            .is_some_and(|op| op.supports_input(xs))
    }

    /// Owns the prepared compute handles, sharing this layer's native workspace.
    /// `use<>` ensures the returned closure borrows nothing from this layer, so
    /// its normal mutable reset/reseed methods remain available between replays.
    pub(crate) fn prepared_decode_operation(
        &self,
    ) -> CandleResult<impl Fn(&Tensor) -> CandleResult<Tensor> + use<>> {
        if !self.optimized_state_active {
            candle::bail!("Gated DeltaNet native decode state is not prepared")
        }
        let optimized = self
            .optimized_decode
            .as_ref()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet has no native decode runner".to_owned(),
                )
            })?
            .clone();
        let out_proj = self.out_proj.clone();
        Ok(move |xs: &Tensor| optimized.forward(xs, &out_proj))
    }

    /// Independent snapshots for restoring state advanced by capture warm-up.
    pub(crate) fn snapshot_optimized_state(&self) -> CandleResult<(Tensor, Tensor)> {
        self.optimized_decode
            .as_ref()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet has no native decode runner".to_owned(),
                )
            })?
            .runner
            .export_state()
    }

    #[cfg(test)]
    pub(crate) fn optimized_state_addresses(&self) -> CandleResult<(u64, u64)> {
        self.optimized_decode
            .as_ref()
            .unwrap()
            .runner
            .state_addresses()
    }

    #[cfg(test)]
    pub(crate) fn snapshot_current_state(&self) -> CandleResult<(Tensor, Tensor)> {
        if self.optimized_state_active {
            self.snapshot_optimized_state()
        } else {
            Ok((
                self.conv_state
                    .as_ref()
                    .expect("convolution state")
                    .copy()?,
                self.recurrent_state
                    .as_ref()
                    .expect("recurrent state")
                    .copy()?,
            ))
        }
    }
}
