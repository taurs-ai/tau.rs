//! Shared Qwen3.5 hybrid text backbone and causal-language-model projection.
//!
//! Checkpoint façades provide the feed-forward constructor and exact VarBuilder
//! prefixes. Embedding, token-mixer construction, normalization, causal masking,
//! decoder traversal, final-token projection, and cache lifecycle are common.

use super::config::HybridTextConfig;
use super::decoder::HybridDecoderLayer;
use crate::components::GraphFeedForward;
use crate::components::{CenteredRmsNorm, PartialRotaryEmbedding};
use crate::{CudaGraphMode, CudaGraphStats, GraphPhase};
use candle::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_nn::{Linear, linear_no_bias};
use std::sync::Arc;
mod graph;
use graph::FullDecodeGraph;

/// Qwen3.5 hybrid text decoder before the language-model head.
#[derive(Debug)]
pub(crate) struct HybridTextModel<F> {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<HybridDecoderLayer<F>>,
    norm: CenteredRmsNorm,
    device: Device,
    dtype: DType,
}

#[cfg(test)]
mod cache_budget_tests {
    use super::super::config::{HybridLayerType, RopeParameters};
    use super::*;
    use crate::components::Mlp;
    use candle_nn::Activation;

    #[test]
    fn failed_layer_reservation_releases_all_partial_allocations() {
        let config = HybridTextConfig {
            vocab_size: 16,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            attention_bias: false,
            attention_dropout: 0.0,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            use_cache: true,
            tie_word_embeddings: true,
            rope_parameters: RopeParameters {
                rope_type: "default".to_owned(),
                rope_theta: 10000.0,
                partial_rotary_factor: 0.5,
            },
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            layer_types: vec![HybridLayerType::FullAttention; 2],
        };
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model = HybridCausalLm::new(
            &config,
            "test",
            crate::ModelLoadOptions::new().with_optimize_kernel(false),
            vb.clone(),
            vb,
            |_, vb| Mlp::new(16, 32, Activation::Silu, false, vb),
        )
        .unwrap();
        model.model.layers[1].set_kv_cache_capacity(2).unwrap();
        // Layer zero allocates successfully; layer one then rejects the budget.
        assert!(model.reserve_kv_cache(4).is_err());
        // Reconfiguration is only legal with no allocated buffers, proving rollback.
        model.set_kv_cache_capacity(2).unwrap();
        model.reserve_kv_cache(2).unwrap();
        assert!(model.set_kv_cache_capacity(2).is_err());
        assert_eq!(model.kv_memory_per_token(4).unwrap().0, 2 * 2 * 8 * 4);
    }
}

impl<F: GraphFeedForward> HybridTextModel<F> {
    /// Loads the shared backbone with a family-provided feed-forward constructor.
    fn new(
        config: &HybridTextConfig,
        family: &str,
        options: crate::ModelLoadOptions,
        vb: VarBuilder,
        mut build_feed_forward: impl FnMut(usize, VarBuilder) -> CandleResult<F>,
    ) -> CandleResult<Self> {
        let optimize_kernel = options.optimize_kernel();
        if vb.dtype() == DType::F64 {
            candle::bail!("{family} does not support f64")
        }
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        let rotary = Arc::new(PartialRotaryEmbedding::new(
            &config.partial_rotary_config(),
            if options.moe_reference_precision() {
                DType::F32
            } else {
                vb.dtype()
            },
            vb.device(),
        )?);
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (layer_index, &layer_type) in config.layer_types.iter().enumerate() {
            let layer_vb = layers_vb.pp(layer_index);
            let feed_forward = build_feed_forward(layer_index, layer_vb.pp("mlp"))?;
            layers.push(HybridDecoderLayer::new(
                config,
                layer_type,
                rotary.clone(),
                feed_forward,
                options,
                layer_vb,
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: CenteredRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                optimize_kernel,
                vb.pp("norm"),
            )?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Builds an additive causal mask for a chunk following a cached prefix.
    fn causal_mask(
        &self,
        batch_size: usize,
        sequence_length: usize,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let mut values = Vec::with_capacity(
            batch_size * sequence_length * (sequence_length + offset),
        );
        for _ in 0..batch_size {
            for query_index in 0..sequence_length {
                for key_index in 0..sequence_length + offset {
                    values.push(if key_index <= query_index + offset {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        }
        Tensor::new(values, &self.device)?
            .reshape((batch_size, 1, sequence_length, sequence_length + offset))?
            .to_dtype(self.dtype)
    }

    /// EAGER MODE: every decoder layer executes without capture or replay.
    fn forward_eager(
        &mut self,
        input: &Tensor,
        offset: usize,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        let (mut hidden, mask) = self.embed_and_mask(input, offset)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            #[cfg(test)]
            let _scope = crate::test_support::tensor_capture::root_scope(format!(
                "layer-{index}-"
            ));
            hidden = match phase {
                GraphPhase::Prefill => {
                    layer.forward_eager(&hidden, mask.as_ref(), offset)
                }
                GraphPhase::Decode => {
                    layer.forward_eager_phase(&hidden, mask.as_ref(), offset, phase)
                }
            }
            .map_err(|error| {
                candle::Error::Msg(format!("decoder layer {index}: {error}"))
            })?;
            #[cfg(test)]
            crate::test_support::tensor_capture::tensor("hidden", &hidden);
        }
        self.norm.forward(&hidden)
    }

    /// GRAPH MODE: layer dispatch selects whole-GDN or attention-MLP graphs,
    /// with explicit eager fallback for unsupported contracts.
    fn forward_graph(
        &mut self,
        input: &Tensor,
        offset: usize,
        phase: GraphPhase,
    ) -> CandleResult<Tensor> {
        let (mut hidden, mask) = self.embed_and_mask(input, offset)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            hidden = layer
                .forward_graph_with_eager_fallback(
                    &hidden,
                    mask.as_ref(),
                    offset,
                    phase,
                )
                .map_err(|error| {
                    candle::Error::Msg(format!("decoder layer {index}: {error}"))
                })?;
        }
        self.norm.forward(&hidden)
    }

    /// Prepares embeddings and the causal mask shared by both execution modes.
    fn embed_and_mask(
        &self,
        input: &Tensor,
        offset: usize,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        let (batch_size, sequence_length) = input.dims2()?;
        let hidden = self.embed_tokens.forward(input)?;
        let mask = if sequence_length > 1 {
            Some(self.causal_mask(batch_size, sequence_length, offset)?)
        } else {
            None
        };
        Ok((hidden, mask))
    }

    /// Clears sequence-dependent state from every hybrid decoder layer.
    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Exposes the primary runtime dtype to construction regressions.
    #[cfg(test)]
    fn dtype(&self) -> DType {
        self.dtype
    }

    /// Exposes the materialized device to construction regressions.
    #[cfg(test)]
    fn device(&self) -> &Device {
        &self.device
    }
}

/// Shared Qwen3.5 autoregressive text model with a vocabulary projection.
/// "Hybrid" refers to the mixture of Gated DeltaNet (GDN) and full attention layers.
#[derive(Debug)]
pub(crate) struct HybridCausalLm<F> {
    full_graph: Option<FullDecodeGraph>,
    full_graph_stats: CudaGraphStats,
    model: HybridTextModel<F>,
    lm_head: Linear,
    cuda_graph: CudaGraphMode,
}

impl<F: GraphFeedForward> HybridCausalLm<F> {
    /// Complete live state, in layer order; snapshots own independent storage.
    #[cfg(test)]
    pub(crate) fn snapshot_current_state(&self) -> CandleResult<Vec<Tensor>> {
        let mut state = Vec::new();
        for layer in &self.model.layers {
            state.extend(layer.snapshot_current_state()?);
        }
        Ok(state)
    }

    /// Loads the hybrid backbone and tied or independent language-model head.
    pub(crate) fn new(
        config: &HybridTextConfig,
        family: &str,
        options: crate::ModelLoadOptions,
        model_vb: VarBuilder,
        lm_head_vb: VarBuilder,
        build_feed_forward: impl FnMut(usize, VarBuilder) -> CandleResult<F>,
    ) -> CandleResult<Self> {
        let model = HybridTextModel::new(
            config,
            family,
            options,
            model_vb,
            build_feed_forward,
        )?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, lm_head_vb)?
        };
        Ok(Self {
            full_graph: None,
            full_graph_stats: CudaGraphStats::default(),
            model,
            lm_head,
            cuda_graph: CudaGraphMode::None,
        })
    }

    /// Executes eagerly and returns vocabulary logits for the chunk's final token.
    pub(crate) fn forward_eager(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let (_, sequence_length) = input.dims2()?;
        // Eager execution can still use CUDA and optimized kernels.
        self.model
            .forward_eager(input, offset, GraphPhase::Prefill)?
            .narrow(1, sequence_length - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Executes one prompt chunk. Full chunks may replay prefill graphs; tails
    /// are always eager. Only the final chunk needs the vocabulary projection.
    pub(crate) fn forward_prefill_chunk(
        &mut self,
        input: &Tensor,
        offset: usize,
        full_chunk: bool,
        last_chunk: bool,
    ) -> CandleResult<Option<Tensor>> {
        let (_, length) = input.dims2()?;
        let hidden = if self.cuda_graph.is_enabled() && full_chunk {
            // GRAPH MODE: only eligible full chunks may capture/replay.
            self.model
                .forward_graph(input, offset, GraphPhase::Prefill)?
        } else {
            // EAGER MODE: disabled graphs, short prompts and partial tails.
            self.model
                .forward_eager(input, offset, GraphPhase::Prefill)?
        };
        if last_chunk {
            Ok(Some(hidden.narrow(1, length - 1, 1)?.apply(&self.lm_head)?))
        } else {
            Ok(None)
        }
    }

    /// Enable only after the engine factory creates a private capture device.
    pub(crate) fn set_cuda_graph(&mut self, cuda_graph: CudaGraphMode) {
        self.cuda_graph = cuda_graph;
    }

    pub(crate) fn cuda_graph_mode(&self) -> CudaGraphMode {
        self.cuda_graph
    }

    pub(crate) fn cuda_graph_enabled(&self) -> bool {
        self.cuda_graph_mode().is_enabled()
    }

    /// Explicit decode entry point, distinct from a same-shaped one-token prefill.
    /// Full policy first tries the complete prepared forward; unsupported contracts
    /// select existing layer graphs or eager execution before a step is run.
    pub(crate) fn forward_decode(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> CandleResult<Tensor> {
        if self.cuda_graph == CudaGraphMode::Full && input.dims2()? == (1, 1) {
            let token = input.to_vec2::<u32>()?[0][0];
            if let Some(logits) = self.try_full_decode(token, offset, true, None)? {
                return Ok(logits);
            }
        }
        if !self.cuda_graph_enabled() || input.dims2()? != (1, 1) {
            // EAGER MODE: graphs are disabled or this is not batch-one decode.
            let (_, length) = input.dims2()?;
            return self
                .model
                .forward_eager(input, offset, GraphPhase::Decode)?
                .narrow(1, length - 1, 1)?
                .apply(&self.lm_head);
        }
        // GRAPH MODE: eligible GDN layers replay whole-layer graphs; attention
        // layers keep attention eager and replay only their MLP. The LM head
        // below remains eager; this does not capture the entire model.
        self.model
            .forward_graph(input, offset, GraphPhase::Decode)?
            .apply(&self.lm_head)
    }

    /// Snapshots observed graph activity across the model. Start with complete
    /// forward captures/replays, then add each layer's GDN and MLP counters for
    /// both prefill and decode. Generation compares snapshots before and after
    /// a step to report actual graph use rather than infer it from the policy.
    pub(crate) fn cuda_graph_counts(&self) -> CudaGraphStats {
        self.model
            .layers
            .iter()
            .map(|layer| layer.cuda_graph_counts())
            .fold(self.full_graph_stats, CudaGraphStats::merge)
    }

    /// Host token boundary avoids creating a new device tensor in full replay.
    /// The complete graph reads the token through retained metadata. If its
    /// contract is unsupported, construct the ordinary input for narrower decode.
    pub(crate) fn forward_decode_token(
        &mut self,
        token: u32,
        offset: usize,
    ) -> CandleResult<Tensor> {
        if let Some(logits) = self.try_full_decode(token, offset, true, None)? {
            return Ok(logits);
        }
        let input = Tensor::new(&[[token]], &self.model.device)?;
        self.forward_decode(&input, offset)
    }

    /// Clears full-attention and DeltaNet state before an unrelated request.
    pub(crate) fn clear_cache(&mut self) {
        self.model.clear_cache();
    }

    /// Reserve per-layer KV storage before prefill, outside all captured regions.
    /// Eligible full decode uses capacity buckets and its preferred K layout.
    /// Drop the old graph before any address/layout change, then reserve every
    /// layer; on failure release partial reservations rather than retaining them.
    pub(crate) fn reserve_kv_cache(&mut self, capacity: usize) -> CandleResult<()> {
        let prepared = if self.cuda_graph == CudaGraphMode::Full && capacity <= 8192 {
            let hidden = self.model.embed_tokens.embeddings().dim(1)?;
            let probe = self
                .model
                .embed_tokens
                .embeddings()
                .narrow(0, 0, 1)?
                .reshape((1, 1, hidden))?;
            self.model
                .layers
                .iter()
                .all(|layer| layer.supports_full_operators(&probe))
        } else {
            false
        };
        // Drop graph-owned KV views before a request can grow/reinterpret storage.
        if self.full_graph.as_ref().is_some_and(|g| {
            self.model.layers.iter().zip(&g.keys).any(|(layer, key)| {
                layer.decode_cache_key() != *key
                    || layer.decode_reservation_changes(capacity, prepared)
            })
        }) {
            self.full_graph = None;
        }
        let result = self.model.layers.iter_mut().try_for_each(|layer| {
            if prepared {
                layer.reserve_decode_cache(capacity)
            } else {
                layer.reserve_kv_cache(capacity)
            }
        });
        if result.is_err() {
            self.full_graph = None;
            // A failure in a later layer must not leave earlier, large reservations
            // resident. Any graph retaining the old buffers has already been dropped.
            for layer in &mut self.model.layers {
                layer.release_kv_cache();
            }
        }
        result
    }

    pub(crate) fn set_kv_cache_capacity(
        &mut self,
        capacity: usize,
    ) -> CandleResult<()> {
        for layer in &mut self.model.layers {
            layer.set_kv_cache_capacity(capacity)?;
        }
        Ok(())
    }

    /// Estimates `(KV bytes, peak prefill-workspace bytes)` per context token
    /// for batch-one cache sizing at the supplied prefill chunk size. Sum KV
    /// across layers because every layer retains its history; take the largest
    /// workspace estimate because attention layers execute sequentially.
    /// Fixed GDN/graph/projection costs are covered by separate budget headroom.
    /// This only computes a sizing heuristic; it allocates no memory.
    pub(crate) fn kv_memory_per_token(
        &self,
        chunk_size: usize,
    ) -> CandleResult<(usize, usize)> {
        let mut total_kv = 0usize;
        let mut peak_workspace = 0usize;
        for layer in &self.model.layers {
            let (kv, workspace) = layer.kv_memory_per_token(chunk_size)?;
            total_kv = total_kv.checked_add(kv).ok_or_else(|| {
                candle::Error::Msg("model KV memory estimate overflow".to_owned())
            })?;
            // Attention layers execute sequentially; their temporary storage is
            // reusable, whereas all layers' persistent K/V must fit simultaneously.
            peak_workspace = peak_workspace.max(workspace);
        }
        Ok((total_kv, peak_workspace))
    }

    /// Exposes the backbone's primary runtime dtype to construction regressions.
    #[cfg(test)]
    pub(crate) fn dtype(&self) -> DType {
        self.model.dtype()
    }

    /// Exposes the backbone device to family construction regressions.
    #[cfg(test)]
    pub(crate) fn device(&self) -> &Device {
        self.model.device()
    }
}
