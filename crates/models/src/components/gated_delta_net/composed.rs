//! Composed GDN computation shared by ordinary execution and prefill capture.
//!
//! Projections, gates, head expansion, recurrence and output projection stay
//! together here. Convolution dispatch lives in its sibling module; state remains
//! owned by GatedDeltaNet, and the decoder still owns graph capture/replay.

use super::GatedDeltaNet;
use candle::{D, DType, Module, Result as CandleResult, Tensor};

/// L2-normalizes the last tensor dimension with a small zero-norm guard.
fn l2_normalize(xs: &Tensor) -> CandleResult<Tensor> {
    let denominator = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    xs.broadcast_div(&denominator)
}

/// Computes softplus using a form that remains stable for large positive values.
fn softplus(xs: &Tensor) -> CandleResult<Tensor> {
    let positive = xs.maximum(0.0)?;
    let correction = (xs.abs()?.neg()?.exp()? + 1.0)?.log()?;
    positive + correction
}

impl GatedDeltaNet {
    /// Repeats each key head so keys align with the larger set of value heads.
    fn repeat_key_heads(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, sequence_length, _, head_dim) = xs.dims4()?;
        let groups = self.num_value_heads / self.num_key_heads;
        xs.unsqueeze(3)?
            .broadcast_as((
                batch_size,
                sequence_length,
                self.num_key_heads,
                groups,
                head_dim,
            ))?
            .reshape((batch_size, sequence_length, self.num_value_heads, head_dim))
    }

    /// Applies the delta-rule memory update token by token in stable FP32.
    ///
    /// The final memory matrix is cached, so later decode calls continue the same
    /// recurrence instead of recomputing the prompt.
    pub(super) fn recurrent_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        decay: &Tensor,
        beta: &Tensor,
    ) -> CandleResult<Tensor> {
        let (batch_size, sequence_length, _, _) = query.dims4()?;
        let native_prefill = self.optimize_prefill && query.dtype() == DType::BF16;
        let mut state = match &self.recurrent_state {
            Some(state) => state.clone(),
            None => Tensor::zeros(
                (
                    batch_size,
                    self.num_value_heads,
                    self.key_head_dim,
                    self.value_head_dim,
                ),
                DType::F32,
                query.device(),
            )?,
        };
        // Prefill stores normalized Q/K in the activation dtype before its
        // matrix products. Decode's fused normalization keeps FP32 instead.
        let normalize = |input: &Tensor| -> CandleResult<Tensor> {
            let normalized = l2_normalize(&input.to_dtype(DType::F32)?)?;
            if self.round_prefill_normalized_qk() && sequence_length > 1 {
                normalized.to_dtype(input.dtype())?.to_dtype(DType::F32)
            } else {
                Ok(normalized)
            }
        };
        let query = (normalize(query)? / (self.key_head_dim as f64).sqrt())?;
        let key = normalize(key)?;
        let value = value.to_dtype(DType::F32)?;
        let decay = decay.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;
        if native_prefill
            && self.value_head_dim == 128
            && super::super::cuda::supports_prefill_recurrence(&query)
        {
            let (output, final_state) = super::super::cuda::prefill_recurrence_cuda(
                &query,
                &key,
                &value,
                &decay.exp()?,
                &beta,
                &state,
            )?;
            self.recurrent_state = Some(final_state);
            return Ok(output);
        }
        let mut outputs = Vec::with_capacity(sequence_length);

        for token_index in 0..sequence_length {
            let query_token = query.narrow(1, token_index, 1)?.squeeze(1)?;
            let key_token = key.narrow(1, token_index, 1)?.squeeze(1)?;
            let value_token = value.narrow(1, token_index, 1)?.squeeze(1)?;
            let beta_token =
                beta.narrow(1, token_index, 1)?.squeeze(1)?.unsqueeze(2)?;
            let decay_token = decay
                .narrow(1, token_index, 1)?
                .squeeze(1)?
                .exp()?
                .unsqueeze(2)?
                .unsqueeze(3)?;
            state = state.broadcast_mul(&decay_token)?;
            let memory = state.broadcast_mul(&key_token.unsqueeze(3)?)?.sum(2)?;
            let delta = (value_token - memory)?.broadcast_mul(&beta_token)?;
            let update = key_token
                .unsqueeze(3)?
                .broadcast_mul(&delta.unsqueeze(2)?)?;
            state = (state + update)?;
            let output = state
                .broadcast_mul(&query_token.unsqueeze(3)?)?
                .sum(2)?
                .unsqueeze(1)?;
            outputs.push(output);
        }

        self.recurrent_state = Some(state);
        let outputs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }

    /// Projects Q/K/V and gates through Candle, with eligible native convolution.
    ///
    /// Prefill flows through projection -> local causal filtering -> recurrent
    /// memory update -> gated normalization -> output projection. Chunk history
    /// and recurrent memory persist independently so later chunks can continue.
    pub(super) fn forward_composed(&mut self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, sequence_length, _) = xs.dims3()?;
        // Project all tokens into packed Q/K/V features. Move channels before
        // tokens so convolution can slide each channel's filter along the sequence.
        let mixed = self.in_proj_qkv.forward(xs)?.transpose(1, 2)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor(
            "gdn-projected",
            &mixed.transpose(1, 2)?,
        );
        let mixed = self.causal_convolution(&mixed)?.transpose(1, 2)?;
        #[cfg(test)]
        let mixed =
            crate::test_support::tensor_capture::substitute("gdn-convolved", mixed);
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("gdn-convolved", &mixed);
        // Return to token-major shape, then split the locally filtered features
        // into the query, key and value heads consumed by the recurrent update.
        let query = mixed.narrow(D::Minus1, 0, self.key_width)?.reshape((
            batch_size,
            sequence_length,
            self.num_key_heads,
            self.key_head_dim,
        ))?;
        let key = mixed
            .narrow(D::Minus1, self.key_width, self.key_width)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_heads,
                self.key_head_dim,
            ))?;
        let value = mixed
            .narrow(D::Minus1, self.key_width * 2, self.value_width)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_value_heads,
                self.value_head_dim,
            ))?;
        let gate = self.in_proj_z.forward(xs)?.reshape((
            batch_size,
            sequence_length,
            self.num_value_heads,
            self.value_head_dim,
        ))?;
        let beta = self.in_proj_b.forward(xs)?;
        let beta = if self.reference_precision {
            candle_nn::ops::sigmoid(&beta.to_dtype(DType::F32)?)?
                .to_dtype(beta.dtype())?
        } else {
            candle_nn::ops::sigmoid(&beta)?
        };
        let a = self.in_proj_a.forward(xs)?.to_dtype(DType::F32)?;
        let step = softplus(&a.broadcast_add(&self.dt_bias.to_dtype(DType::F32)?)?)?;
        let negative_a = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?.reshape((
            1,
            1,
            self.num_value_heads,
        ))?;
        let decay = step.broadcast_mul(&negative_a)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("gdn-beta", &beta);
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("gdn-decay", &decay);
        // Convolution supplies the short-range context; recurrence combines those
        // features with the longer-term per-head memory and advances it per token.
        let output = if self.optimize_prefill
            && self.key_head_dim == 128
            && self.value_head_dim == 128
            && super::super::cuda::supports_mixed_prefill(
                &mixed,
                self.num_key_heads,
                self.num_value_heads,
            ) {
            let initial = match &self.recurrent_state {
                Some(state) => state.clone(),
                None => Tensor::zeros(
                    (batch_size, self.num_value_heads, 128, 128),
                    DType::F32,
                    xs.device(),
                )?,
            };
            let (output, state) = super::super::cuda::mixed_prefill_cuda(
                &mixed,
                &decay.exp()?,
                &beta,
                &initial,
                self.num_key_heads,
                self.num_value_heads,
                self.round_prefill_normalized_qk(),
            )?;
            self.recurrent_state = Some(state);
            output
        } else {
            self.recurrent_delta_rule(
                &self.repeat_key_heads(&query)?,
                &self.repeat_key_heads(&key)?,
                &value,
                &decay,
                &beta,
            )?
        };
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("gdn-readout", &output);
        let output = self
            .norm
            .forward_prefill(
                &output,
                &gate,
                &self.gate_activation,
                self.optimize_prefill,
            )?
            .reshape((batch_size, sequence_length, self.value_width))?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("gdn-normalized", &output);
        self.out_proj.forward(&output)
    }
}
