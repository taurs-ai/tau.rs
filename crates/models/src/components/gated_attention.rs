//! General-purpose gated grouped-query full attention.
//!
//! The implementation uses a doubled query projection for query values and an
//! output gate, centered per-head Q/K normalization, partial RoPE, grouped-query
//! attention, and an incremental KV cache. Single-token CUDA/BF16 decode consumes
//! grouped KV directly; the reference/prefill path expands heads. The caller
//! supplies the causal mask and decides how frequently this mixer appears.

mod kv_cache;

use self::kv_cache::KvCache;
use super::rms_norm::CenteredRmsNorm;
use super::rope::PartialRotaryEmbedding;
use candle::{D, DType, Module, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_nn::{Linear, linear_b};
use std::sync::Arc;

/// Multiple one-row GEMV groups are reserved for contexts where history copies
/// dominate their launch overhead. Short decode retains the reference calculation.
const GROUPED_DECODE_MIN_CONTEXT: usize = 256;

/// Dimensions and numerical choices needed to construct gated full attention.
#[derive(Debug, Clone)]
pub(crate) struct GatedAttentionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) num_query_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) projection_bias: bool,
    pub(crate) norm_eps: f64,
}

impl GatedAttentionConfig {
    /// Checks dimensions required by projections and grouped-query expansion.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.num_query_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(
                "gated attention dimensions and head counts must be non-zero"
                    .to_owned(),
            );
        }
        if !self
            .num_query_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(
                "gated attention query heads must be divisible by key/value heads"
                    .to_owned(),
            );
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(
                "gated attention normalization epsilon must be positive".to_owned()
            );
        }
        Ok(())
    }
}

/// Gated grouped-query attention with exclusively owned mutable cache state.
/// Deliberately not Clone: tensor clones would share the writable KV allocation.
#[derive(Debug)]
pub(crate) struct GatedAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: CenteredRmsNorm,
    k_norm: CenteredRmsNorm,
    rotary: Arc<PartialRotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    query_width: usize,
    optimize_kernel: bool,
    attention_block_size: crate::AttentionBlockSize,
    fp32_gate: bool,
    bf16_context_weights: bool,
    kv_cache: KvCache,
}

impl GatedAttention {
    /// Restores explicit normalized/rotated KV for an isolated layer comparison.
    /// Validation inputs use the same positions and history in both engines.
    #[cfg(test)]
    pub(crate) fn seed_validation_cache(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        capacity: usize,
    ) {
        self.clear_cache();
        self.reserve_kv_cache(capacity).unwrap();
        if key.dim(2).unwrap() > 0 {
            self.kv_cache
                .append(
                    &key.to_dtype(self.kv_cache.dtype).unwrap(),
                    &value.to_dtype(self.kv_cache.dtype).unwrap(),
                    0,
                )
                .unwrap();
        }
    }
    /// Reserves bucketed KV storage for prepared decode before any graph captures
    /// its addresses. The request's visible token limit remains exact.
    pub(crate) fn reserve_decode_cache(&mut self, capacity: usize) -> CandleResult<()> {
        self.kv_cache.reserve_decode(capacity, true)
    }

    /// Predicts whether a reservation changes captured storage or its K layout.
    /// The model must drop the graph before growing or reinterpreting those buffers.
    pub(crate) fn decode_reservation_changes(
        &self,
        capacity: usize,
        graph: bool,
    ) -> bool {
        let allocation = self.kv_cache.reservation_capacity(capacity, graph);
        allocation > self.kv_cache.capacity()
            || self.kv_cache.transposed_keys
                != (self.kv_cache.prefer_transposed_keys && !graph && allocation >= 256)
    }
    /// Describes the KV allocation identities, capacity and K layout captured by
    /// a graph, so reuse can be rejected even when only the layout changes.
    pub(crate) fn decode_cache_key(
        &self,
    ) -> Option<(candle::TensorId, candle::TensorId, usize, bool)> {
        self.kv_cache.buffers.as_ref().map(|(k, v)| {
            (
                k.id(),
                v.id(),
                self.kv_cache.capacity(),
                self.kv_cache.transposed_keys,
            )
        })
    }

    /// Requires supported operators plus an allocated, request-bounded cache
    /// small enough for the prepared attention workspace and combine kernel.
    pub(crate) fn supports_prepared_decode(&self) -> bool {
        self.supports_prepared_operators()
            && self.kv_cache.request_limit.is_some()
            && self.kv_cache.capacity() > 0
            && self.kv_cache.capacity() <= 8192
    }

    /// Checks head shape, BF16 storage and bias-free projections without needing
    /// a cache reservation. This lets the model choose a layout before allocating.
    pub(crate) fn supports_prepared_operators(&self) -> bool {
        self.optimize_kernel
            && matches!(self.head_dim, 64 | 128 | 256)
            && self.kv_cache.dtype == DType::BF16
            && self.q_proj.bias().is_none()
            && self.k_proj.bias().is_none()
            && self.v_proj.bias().is_none()
            && self.o_proj.bias().is_none()
    }

    /// Rejects an invalid next-token offset, exhausted request budget or unsupported
    /// cache before any captured kernel can mutate KV or recurrent state.
    pub(crate) fn validate_prepared_decode(&self, offset: usize) -> CandleResult<()> {
        self.kv_cache.validate_append(1, 1, offset)?;
        if !self.supports_prepared_decode() {
            candle::bail!("unsupported prepared attention")
        }
        Ok(())
    }

    /// Publishes logical length only; the prepared kernels already wrote live KV.
    pub(crate) fn publish_prepared_decode(&mut self, offset: usize) {
        self.kv_cache.len = offset + 1;
    }

    /// Builds attention's fixed operation sequence: project Q/K/V, unpack and
    /// normalize Q/K, apply position-dependent RoPE, append KV, attend and gate.
    /// The returned output is produced in retained slots; live position and cache
    /// length come from device metadata rather than changing captured shapes.
    pub(crate) fn prepare_decode_program(
        &self,
        program: &mut super::DecodeProgram,
        input: &Tensor,
        metadata: &Tensor,
    ) -> CandleResult<Tensor> {
        let query_and_gate = program.linear(input, &self.q_proj)?;
        let key = program.linear(input, &self.k_proj)?;
        let value = program.linear(input, &self.v_proj)?;
        let (cached_key, cached_value) =
            self.kv_cache.buffers.as_ref().ok_or_else(|| {
                candle::Error::Msg("KV must be reserved before preparation".to_owned())
            })?;
        let capacity = self.kv_cache.capacity();
        let query = program.attention_prepare(
            &query_and_gate,
            &key,
            &value,
            self.q_norm.prepared_parameters(),
            self.k_norm.prepared_parameters(),
            self.rotary.prepared_parameters(),
            (cached_key, cached_value),
            metadata,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            capacity,
            self.kv_cache.transposed_keys,
        )?;
        let context = program.attention(
            &query,
            cached_key,
            cached_value,
            metadata,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            capacity,
            self.kv_cache.transposed_keys,
            self.attention_block_size,
            self.bf16_context_weights,
        )?;
        let gated = program.buffer((1, 1, self.query_width))?;
        let gated = program.elementwise(
            if self.fp32_gate { 6 } else { 3 },
            &[&context, &query_and_gate],
            gated,
            self.query_width,
            self.head_dim,
            0,
            false,
        )?;
        program.linear(&gated, &self.o_proj)
    }

    /// Test-only independent snapshots of the complete live KV prefix.
    #[cfg(test)]
    pub(crate) fn snapshot_current_state(&self) -> CandleResult<Vec<Tensor>> {
        let (key, value) = self.kv_cache.prefix()?;
        Ok(vec![key.copy()?, value.copy()?])
    }

    /// Loads a gated grouped-query attention layer and starts with an empty cache.
    pub(crate) fn new(
        cfg: &GatedAttentionConfig,
        rotary: Arc<PartialRotaryEmbedding>,
        options: crate::ModelLoadOptions,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let optimize_kernel = options.optimize_kernel();
        if let Err(message) = cfg.validate() {
            candle::bail!("{message}")
        }
        let query_width = cfg.num_query_heads * cfg.head_dim;
        let kv_width = cfg.num_key_value_heads * cfg.head_dim;
        let kv_cache = KvCache {
            buffers: None,
            transposed_keys: false,
            prefer_transposed_keys: optimize_kernel
                && vb.device().is_cuda()
                && vb.dtype() == DType::BF16,
            len: 0,
            request_limit: None,
            num_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            max_len: rotary.max_sequence_length(),
            dtype: vb.dtype(),
            device: vb.device().clone(),
        };
        Ok(Self {
            q_proj: linear_b(
                cfg.hidden_size,
                query_width * 2,
                cfg.projection_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_b(
                cfg.hidden_size,
                kv_width,
                cfg.projection_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                kv_width,
                cfg.projection_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(
                query_width,
                cfg.hidden_size,
                cfg.projection_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: CenteredRmsNorm::new(
                cfg.head_dim,
                cfg.norm_eps,
                optimize_kernel,
                vb.pp("q_norm"),
            )?,
            k_norm: CenteredRmsNorm::new(
                cfg.head_dim,
                cfg.norm_eps,
                optimize_kernel,
                vb.pp("k_norm"),
            )?,
            rotary,
            num_heads: cfg.num_query_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_query_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            query_width,
            optimize_kernel,
            attention_block_size: options.attention_block_size(),
            fp32_gate: options.moe_reference_precision(),
            bf16_context_weights: options.moe_reference_precision()
                && cfg.num_query_heads / cfg.num_key_value_heads >= 4,
            kv_cache,
        })
    }

    /// Runs full causal attention and appends the current keys and values to cache.
    ///
    /// `offset` identifies the first input position and must match the number of
    /// tokens already represented by the cache during incremental decoding.
    pub(crate) fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let (batch_size, sequence_length, _) = xs.dims3()?;
        self.kv_cache
            .validate_append(batch_size, sequence_length, offset)?;
        let query_and_gate = self.q_proj.forward(xs)?.reshape((
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let query_and_gate = query_and_gate.chunk(2, D::Minus1)?;
        let query = self.q_norm.forward(&query_and_gate[0])?.transpose(1, 2)?;
        let output_gate = query_and_gate[1].reshape((
            batch_size,
            sequence_length,
            self.query_width,
        ))?;
        let key = self
            .k_norm
            .forward(&self.k_proj.forward(xs)?.reshape((
                batch_size,
                sequence_length,
                self.num_kv_heads,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?;
        let value = self
            .v_proj
            .forward(xs)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_kv_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?;
        let (query, key) = self.rotary.apply(&query, &key, offset)?;

        self.kv_cache.append(&key, &value, offset)?;
        let (key, value) = self.kv_cache.prefix()?;

        #[cfg(test)]
        {
            use crate::test_support::tensor_capture;
            tensor_capture::tensor("attention-query", &query);
            tensor_capture::tensor("attention-key", &key);
            tensor_capture::tensor("attention-value", &value);
        }

        let context = self.attention_context(&query, &key, &value, attention_mask)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("attention-context", &context);
        let context = if self.fp32_gate && context.dtype() == DType::BF16 {
            if self.optimize_kernel && context.device().is_cuda() {
                super::cuda::sigmoid_mul_fp32(
                    &context.contiguous()?,
                    &output_gate.contiguous()?,
                )?
            } else {
                context
                    .to_dtype(DType::F32)?
                    .broadcast_mul(&candle_nn::ops::sigmoid(
                        &output_gate.to_dtype(DType::F32)?,
                    )?)?
                    .to_dtype(context.dtype())?
            }
        } else {
            context.broadcast_mul(&candle_nn::ops::sigmoid(&output_gate)?)?
        };
        self.o_proj.forward(&context)
    }

    /// Computes attention from already normalized/rotated Q and live cached K/V.
    /// Kept separate so tests can exercise head mapping and capacity strides directly.
    fn attention_context(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        #[cfg(test)]
        crate::test_support::tensor_capture::host(
            "attention-grouped",
            vec![1],
            vec![u8::from(self.supports_grouped_decode(
                query,
                key,
                value,
                attention_mask,
            )) as f32],
        );
        if self.supports_grouped_decode(query, key, value, attention_mask) {
            let query = query.contiguous()?.reshape((
                1,
                self.num_kv_heads,
                self.num_kv_groups,
                self.head_dim,
            ))?;
            let key_transposed = key.transpose(2, 3)?;
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut contexts = Vec::with_capacity(self.num_kv_groups);
            for group in 0..self.num_kv_groups {
                // Preserve one matrix row per query so cuBLAS retains the GEMV
                // reduction contract. KV head batches read the retained strides.
                let query = query.narrow(2, group, 1)?;
                let scores =
                    (super::cuda::cached_qk(&query, &key_transposed)? * scale)?;
                let probabilities =
                    candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                        .to_dtype(value.dtype())?;
                contexts.push(probabilities.matmul(value)?);
            }
            // Pack only this token's output, never the complete K/V history.
            return Tensor::cat(&contexts, 2)?.reshape((1, 1, self.query_width));
        }
        let (batch_size, _, sequence_length, _) = query.dims4()?;
        let key =
            candle_transformers::utils::repeat_kv(key.clone(), self.num_kv_groups)?
                .contiguous()?;
        let value =
            candle_transformers::utils::repeat_kv(value.clone(), self.num_kv_groups)?
                .contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // RoPE can return a non-contiguous view for single-token cached decode
        // even when its input was contiguous. Enforce Candle's CUDA matmul
        // contract at the operator boundary, after every layout-changing step.
        let query = query.contiguous()?;
        let key_transposed = key.transpose(2, 3)?.contiguous()?;
        let mut scores = (query.matmul(&key_transposed)? * scale)?;
        if let Some(mask) = attention_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probabilities =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                .to_dtype(value.dtype())?;
        probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, sequence_length, self.query_width))
    }

    /// Long-context decode reads transposed K and ordinary V directly. Keep short
    /// contexts on the existing path to avoid multiple small GEMV launch groups.
    /// The CUDA bridge supplies K's capacity leading dimension to cuBLAS.
    fn supports_grouped_decode(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> bool {
        if !self.optimize_kernel
            || attention_mask.is_some()
            || !query.device().is_cuda()
            || query.dtype() != DType::BF16
            || query.dims() != [1, self.num_heads, 1, self.head_dim]
        {
            return false;
        }
        let Ok((batch, heads, length, dim)) = key.dims4() else {
            return false;
        };
        batch == 1
            && heads == self.num_kv_heads
            && length >= GROUPED_DECODE_MIN_CONTEXT
            && dim == self.head_dim
            && value.shape() == key.shape()
            && [key, value].iter().all(|t| {
                t.dtype() == query.dtype() && t.device().same_device(query.device())
            })
            && key.stride()[2] == 1
            && key.stride()[3] >= length
            && self
                .head_dim
                .checked_mul(key.stride()[3])
                .is_some_and(|n| key.stride()[1] >= n)
            && value.stride()[3] == 1
            && value.stride()[2] == self.head_dim
            && length
                .checked_mul(self.head_dim)
                .is_some_and(|n| value.stride()[1] >= n)
    }

    /// Reserves the validated serving budget before any prompt tokens are processed.
    pub(crate) fn reserve_kv_cache(&mut self, capacity: usize) -> CandleResult<()> {
        self.kv_cache.reserve(capacity)
    }

    /// Invalidates cached values while retaining storage for the next request.
    pub(crate) fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }

    pub(crate) fn release_kv_cache(&mut self) {
        self.kv_cache.release();
    }

    pub(crate) fn set_kv_cache_capacity(
        &mut self,
        capacity: usize,
    ) -> CandleResult<()> {
        self.kv_cache.set_capacity_limit(capacity)
    }

    /// Returns `(persistent KV bytes, estimated prefill-workspace bytes)` charged
    /// for each context token in a batch-one request. Multiplying each value by
    /// the context length estimates its storage cost; `chunk_size` is the number
    /// of query tokens processed together during prefill.
    ///
    /// KV stores one key and value per KV head: 2 * KV heads * head dim * dtype
    /// bytes. Workspace also uses the larger query-head count and chunk size to
    /// allow expanded KV copies, score/softmax buffers and the attention mask.
    /// Prefill/reference attention still expands KV and materializes scores/softmax.
    /// Budget those temporaries as well: a KV-only percentage can admit contexts
    /// whose attention workspace exceeds all remaining VRAM. This is a heuristic,
    /// not an allocator guarantee; fixed graph/GDN/projection headroom is separate.
    /// Decode's smaller workspace must not reduce the prefill capacity budget.
    pub(crate) fn kv_memory_per_token(
        &self,
        chunk_size: usize,
    ) -> CandleResult<(usize, usize)> {
        let bytes = self.kv_cache.dtype.size_in_bytes();
        // Retained K and V use KV heads, without expanding to all query heads.
        let kv = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|n| n.checked_mul(bytes))
            .and_then(|n| n.checked_mul(2));
        // Twice the three expanded K/V/transposed-K outputs, allowing copy overlap.
        let copies = self
            .num_heads
            .checked_mul(self.head_dim)
            .and_then(|n| n.checked_mul(bytes))
            .and_then(|n| n.checked_mul(6));
        // Allow four model-dtype and four FP32 score-sized buffers, plus the mask.
        let score_bytes = bytes.checked_mul(4).and_then(|n| n.checked_add(16));
        let scores = self
            .num_heads
            .checked_mul(chunk_size)
            .and_then(|n| n.checked_mul(score_bytes?));
        let mask = chunk_size.checked_mul(bytes + 4);
        let workspace = copies
            .and_then(|n| n.checked_add(scores?))
            .and_then(|n| n.checked_add(mask?));
        match (kv, workspace) {
            (Some(kv), Some(workspace)) => Ok((kv, workspace)),
            _ => candle::bail!("KV memory estimate overflow"),
        }
    }
}

#[cfg(test)]
#[path = "gated_attention/tests.rs"]
mod tests;
