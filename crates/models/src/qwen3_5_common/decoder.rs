//! Shared Qwen3.5 hybrid decoder-layer orchestration.
//!
//! Dense and MoE variants use the same token mixers, normalization order,
//! residual connections, and cache lifecycle. The feed-forward module is the
//! only family-specific part of a decoder layer.

mod graph;

use super::config::{HybridLayerType, HybridTextConfig};
use crate::GraphPhase;
use crate::components::{
    CenteredRmsNorm, GatedAttention, GatedAttentionConfig, GatedDeltaNet,
    GatedDeltaNetConfig, GraphFeedForward, PartialRotaryEmbedding,
};
use candle::{Module, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use graph::GdnLayerGraph;
use std::sync::Arc;

/// Token-mixing implementation selected by the serialized layer schedule.
#[derive(Debug)]
enum TokenMixer {
    Linear(Box<GatedDeltaNet>),
    Full(Box<GatedAttention>),
}

impl TokenMixer {
    /// Loads either Gated DeltaNet or gated full attention for one decoder layer.
    fn new(
        gated_attention_config: &GatedAttentionConfig,
        gated_delta_net_config: &GatedDeltaNetConfig,
        layer_type: HybridLayerType,
        rotary: Arc<PartialRotaryEmbedding>,
        options: crate::ModelLoadOptions,
        vb: &VarBuilder,
    ) -> CandleResult<Self> {
        let optimize_kernel = options.optimize_kernel();
        match layer_type {
            HybridLayerType::LinearAttention => Ok(Self::Linear(Box::new(
                GatedDeltaNet::new(
                    gated_delta_net_config,
                    optimize_kernel,
                    vb.pp("linear_attn"),
                )?
                .with_reference_precision(options.moe_reference_precision()),
            ))),
            HybridLayerType::FullAttention => {
                Ok(Self::Full(Box::new(GatedAttention::new(
                    gated_attention_config,
                    rotary,
                    options,
                    vb.pp("self_attn"),
                )?)))
            }
        }
    }

    /// Executes the token mixer eagerly; whole GDN graphs are owned by the decoder.
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Linear(layer) => layer.forward(xs),
            Self::Full(layer) => layer.forward(xs, attention_mask, offset),
        }
    }

    /// Clears sequence state owned by the selected token mixer.
    fn clear_cache(&mut self) {
        match self {
            Self::Linear(layer) => layer.clear_cache(),
            Self::Full(layer) => layer.clear_cache(),
        }
    }
}

/// Pre-normalized hybrid token mixer followed by a family-specific feed-forward.
/// <F> is MLP for dense model, and a SparseMoeBlock for MoE model.
#[derive(Debug)]
pub(crate) struct HybridDecoderLayer<F> {
    // Support per-layer cuda graph
    decode_graph: Option<GdnLayerGraph>,
    prefill_graph: Option<GdnLayerGraph>,

    token_mixer: TokenMixer,
    feed_forward: F,
    input_layernorm: CenteredRmsNorm,
    post_attention_layernorm: CenteredRmsNorm,
}

/// The layer's math, shared by eager execution and graph capture. Only the two
/// supplied operations vary; no execution policy or graph lifecycle lives here.
fn forward_block(
    xs: &Tensor,
    input_norm: &CenteredRmsNorm,
    post_norm: &CenteredRmsNorm,
    mut mix_tokens: impl FnMut(&Tensor) -> CandleResult<Tensor>,
    mut feed_forward: impl FnMut(&Tensor) -> CandleResult<Tensor>,
) -> CandleResult<Tensor> {
    let normalized = input_norm.forward(xs)?;
    #[cfg(test)]
    crate::test_support::tensor_capture::tensor("input-normalized", &normalized);
    let branch = mix_tokens(&normalized)?;
    #[cfg(test)]
    crate::test_support::tensor_capture::tensor("mixer", &branch);
    let (residual_sum, normalized) =
        post_norm.add_residual_and_normalize(xs, &branch)?;
    #[cfg(test)]
    crate::test_support::tensor_capture::tensor(
        "post-attention-residual",
        &residual_sum,
    );
    &residual_sum + feed_forward(&normalized)?
}

impl<F: GraphFeedForward> HybridDecoderLayer<F> {
    /// Checks the layer's arithmetic support before KV storage has been reserved.
    /// Attention cache size/layout is checked later, after choosing a reservation.
    pub(super) fn supports_full_operators(&self, input: &Tensor) -> bool {
        self.feed_forward.supports_prepared_decode(input)
            && match &self.token_mixer {
                TokenMixer::Linear(gdn) => gdn.supports_prepared_decode(input),
                TokenMixer::Full(attention) => attention.supports_prepared_operators(),
            }
    }
    /// Reserves prepared KV storage for attention layers. GDN owns fixed-size
    /// state and needs no context-length-dependent KV allocation here.
    pub(super) fn reserve_decode_cache(&mut self, capacity: usize) -> CandleResult<()> {
        if let TokenMixer::Full(a) = &mut self.token_mixer {
            a.reserve_decode_cache(capacity)?;
        }
        Ok(())
    }
    /// Reports attention storage/layout changes that invalidate a captured graph;
    /// a GDN layer has no KV reservation to reinterpret.
    pub(super) fn decode_reservation_changes(
        &self,
        capacity: usize,
        graph: bool,
    ) -> bool {
        match &self.token_mixer {
            TokenMixer::Full(a) => a.decode_reservation_changes(capacity, graph),
            _ => false,
        }
    }
    /// Returns the attention allocation/layout identity used to guard graph reuse.
    /// GDN's retained runner state is handled through its separate state handoff.
    pub(super) fn decode_cache_key(
        &self,
    ) -> Option<(candle::TensorId, candle::TensorId, usize, bool)> {
        match &self.token_mixer {
            TokenMixer::Full(a) => a.decode_cache_key(),
            _ => None,
        }
    }

    /// Checks the complete layer contract, including attention's current request
    /// reservation, before any operations are prepared or state is advanced.
    pub(super) fn supports_full_graph(&self, input: &Tensor) -> bool {
        self.feed_forward.supports_prepared_decode(input)
            && match &self.token_mixer {
                TokenMixer::Linear(gdn) => gdn.supports_prepared_decode(input),
                TokenMixer::Full(attention) => attention.supports_prepared_decode(),
            }
    }

    /// Checks attention's next KV position and capacity before a full-forward
    /// launch; GDN state does not use an explicit context-length offset.
    pub(super) fn validate_full_decode(&self, offset: usize) -> CandleResult<()> {
        if let TokenMixer::Full(a) = &self.token_mixer {
            a.validate_prepared_decode(offset)?;
        }
        Ok(())
    }

    /// Seeds the native GDN runner from the authoritative state and snapshots it
    /// for restoration after warm-up. Attention only writes the unused next KV
    /// slot while warming and therefore needs no full-prefix snapshot here.
    pub(super) fn prepare_full_decode(
        &mut self,
    ) -> CandleResult<Option<(Tensor, Tensor)>> {
        if let TokenMixer::Linear(gdn) = &mut self.token_mixer {
            gdn.prepare_optimized_decode()?;
            return gdn.snapshot_optimized_state().map(Some);
        }
        Ok(None)
    }

    /// Reseeds retained GDN buffers after prefill/reset when necessary, without
    /// replacing the addresses already captured by the graph.
    pub(super) fn ensure_full_decode_state(&mut self) -> CandleResult<()> {
        if let TokenMixer::Linear(gdn) = &mut self.token_mixer {
            gdn.prepare_optimized_decode()?;
        }
        Ok(())
    }

    /// Restores the GDN snapshot after warm-up so the first real replay starts
    /// from the same convolution and recurrent state as an eager decode step.
    pub(super) fn restore_full_decode(
        &mut self,
        state: Option<&(Tensor, Tensor)>,
    ) -> CandleResult<()> {
        if let (TokenMixer::Linear(gdn), Some((conv, recurrent))) =
            (&mut self.token_mixer, state)
        {
            gdn.seed_optimized_state(Some(conv), Some(recurrent))?;
        }
        Ok(())
    }

    /// Advances attention's host-visible KV length after the prepared launch.
    /// GDN state was already advanced in place by its queued native operation.
    pub(super) fn publish_full_decode(&mut self, offset: usize) {
        if let TokenMixer::Full(a) = &mut self.token_mixer {
            a.publish_prepared_decode(offset);
        }
    }

    /// Adds this layer's normalization, token mixer, residuals and feed-forward
    /// operations to the model's one decode program. Fixed output slots connect
    /// the operations directly, without a nested layer graph or replay-time walk.
    pub(super) fn prepare_decode_program(
        &mut self,
        program: &mut crate::components::DecodeProgram,
        input: &Tensor,
        metadata: &Tensor,
    ) -> CandleResult<Tensor> {
        let normalized = self
            .input_layernorm
            .prepare_decode_program(program, input)?;
        let branch = match &mut self.token_mixer {
            TokenMixer::Linear(gdn) => {
                gdn.prepare_decode_program(program, &normalized)?
            }
            TokenMixer::Full(attention) => {
                attention.prepare_decode_program(program, &normalized, metadata)?
            }
        };
        let residual = program.add(input, &branch)?;
        let normalized = self
            .post_attention_layernorm
            .prepare_decode_program(program, &residual)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("ff-input", &normalized);
        let branch = self
            .feed_forward
            .prepare_decode_program(program, &normalized)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("ff-output", &branch);
        program.add(&residual, &branch)
    }

    #[cfg(test)]
    pub(super) fn snapshot_current_state(&self) -> CandleResult<Vec<Tensor>> {
        match &self.token_mixer {
            TokenMixer::Linear(gdn) => {
                let (conv, recurrent) = gdn.snapshot_current_state()?;
                Ok(vec![conv, recurrent])
            }
            TokenMixer::Full(attention) => attention.snapshot_current_state(),
        }
    }

    /// Loads the shared layer structure around an already-constructed feed-forward.
    pub(crate) fn new(
        config: &HybridTextConfig,
        layer_type: HybridLayerType,
        rotary: Arc<PartialRotaryEmbedding>,
        feed_forward: F,
        options: crate::ModelLoadOptions,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let optimize_kernel = options.optimize_kernel();
        let gated_attention_config = config.gated_attention_config();
        let gated_delta_net_config = config.gated_delta_net_config();
        Ok(Self {
            decode_graph: None,
            prefill_graph: None,
            token_mixer: TokenMixer::new(
                &gated_attention_config,
                &gated_delta_net_config,
                layer_type,
                rotary,
                options,
                &vb,
            )?,
            feed_forward,
            input_layernorm: CenteredRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                optimize_kernel,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: CenteredRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                optimize_kernel,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    /// EAGER MODE: ordinary token mixing and MLP computation, even when this
    /// layer already retains graphs from a previous chunk or request.
    pub(crate) fn forward_eager(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
    ) -> CandleResult<Tensor> {
        self.forward_eager_phase(xs, attention_mask, offset, GraphPhase::Prefill)
    }

    /// Shares token mixing/norms/residuals while preserving the feed-forward
    /// phase contract, especially for one-token MoE prefill versus decode.
    pub(super) fn forward_eager_phase(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        forward_block(
            xs,
            &self.input_layernorm,
            &self.post_attention_layernorm,
            |hidden| self.token_mixer.forward(hidden, attention_mask, offset),
            |hidden| match phase {
                GraphPhase::Prefill => self.feed_forward.forward(hidden),
                GraphPhase::Decode => self.feed_forward.forward_decode(hidden),
            },
        )
    }

    /// Uses supported graph regions, with eager fallback for unsupported inputs.
    /// Capture/replay errors propagate without retrying eagerly.
    /// GDN graphs cover the whole layer; full-attention graphs cover only its MLP.
    /// Short prompt chunks bypass this method and go directly to forward_eager.
    pub(crate) fn forward_graph_with_eager_fallback(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        match &self.token_mixer {
            TokenMixer::Linear(_) => {
                if self.supports_gdn_graph(xs, phase)? {
                    // GRAPH MODE, GDN: the graph contains the complete layer.
                    self.forward_gdn_graph(xs, phase)
                } else {
                    // EAGER FALLBACK: do not substitute an MLP-only GDN graph.
                    self.forward_eager_phase(xs, attention_mask, offset, phase)
                }
            }
            TokenMixer::Full(_) => {
                // GRAPH MODE, ATTENTION: attention/norms/residuals stay eager.
                // Check MLP eligibility on its actual normalized input, just as
                // in the original path (normalization may change contiguity).
                forward_block(
                    xs,
                    &self.input_layernorm,
                    &self.post_attention_layernorm,
                    |hidden| self.token_mixer.forward(hidden, attention_mask, offset),
                    |hidden| {
                        if self.feed_forward.supports_layer_graph(hidden) {
                            self.feed_forward.forward_graph(hidden, phase)
                        } else {
                            // EAGER FALLBACK: unsupported dense input or MoE.
                            match phase {
                                GraphPhase::Prefill => {
                                    self.feed_forward.forward(hidden)
                                }
                                GraphPhase::Decode => {
                                    self.feed_forward.forward_decode(hidden)
                                }
                            }
                        }
                    },
                )
            }
        }
    }

    /// Clears stateful token-mixing caches; feed-forward modules are stateless.
    pub(crate) fn clear_cache(&mut self) {
        self.token_mixer.clear_cache();
    }

    /// Only full attention grows with context; DeltaNet already owns fixed state.
    pub(crate) fn reserve_kv_cache(&mut self, capacity: usize) -> CandleResult<()> {
        if let TokenMixer::Full(layer) = &mut self.token_mixer {
            layer.reserve_kv_cache(capacity)?;
        }
        Ok(())
    }

    pub(crate) fn release_kv_cache(&mut self) {
        if let TokenMixer::Full(layer) = &mut self.token_mixer {
            layer.release_kv_cache();
        }
    }

    pub(crate) fn set_kv_cache_capacity(
        &mut self,
        capacity: usize,
    ) -> CandleResult<()> {
        if let TokenMixer::Full(layer) = &mut self.token_mixer {
            layer.set_kv_cache_capacity(capacity)?;
        }
        Ok(())
    }

    /// Returns this layer's `(KV bytes, prefill-workspace bytes)` per context
    /// token, using `chunk_size` query tokens for the workspace estimate. Full
    /// attention delegates to its head/dtype calculation. GDN contributes zero
    /// here because its fixed-size state is charged through separate headroom.
    pub(crate) fn kv_memory_per_token(
        &self,
        chunk_size: usize,
    ) -> CandleResult<(usize, usize)> {
        match &self.token_mixer {
            TokenMixer::Full(layer) => layer.kv_memory_per_token(chunk_size),
            TokenMixer::Linear(_) => Ok((0, 0)),
        }
    }
}

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
