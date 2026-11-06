//! Bias-free gated multilayer perceptron.
//!
//! The reference path uses independent gate and up projections. The optimized
//! CUDA path concatenates their weights once during construction, evaluates one
//! larger projection, and fuses SiLU with the gating multiply. A down projection
//! maps either result back to the input width.

mod graph;
mod projection;

use candle::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::Linear;
use candle_nn::{Activation, VarBuilder};
use graph::MlpGraph;
use projection::GateUpProjection;

/// Gated feed-forward network with bias-free gate, up, and down projections.
#[derive(Debug)]
pub(crate) struct Mlp {
    decode_graph: Option<MlpGraph>,
    prefill_graph: Option<MlpGraph>,
    hidden_size: usize,
    device: Device,
    dtype: DType,
    gate_up_projection: GateUpProjection,
    down_proj: Linear,
    activation: Activation,
}

/// Clones share immutable weights, but never share graph buffers or observations.
impl Clone for Mlp {
    fn clone(&self) -> Self {
        Self {
            decode_graph: None,
            prefill_graph: None,
            hidden_size: self.hidden_size,
            device: self.device.clone(),
            dtype: self.dtype,
            gate_up_projection: self.gate_up_projection.clone(),
            down_proj: self.down_proj.clone(),
            activation: self.activation,
        }
    }
}

impl Mlp {
    /// Applies per-row FP32 weights before the down projection's storage round.
    /// Routed experts use this explicitly; ordinary dense/shared MLP forward is
    /// unchanged. The result has the original activation dtype.
    pub(crate) fn forward_weighted(
        &self,
        xs: &Tensor,
        weights: &Tensor,
    ) -> CandleResult<Tensor> {
        let (rows, _) = xs.dims2()?;
        if weights.dims() != [rows, 1]
            || weights.dtype() != DType::F32
            || !weights.device().same_device(xs.device())
        {
            candle::bail!("weighted MLP requires one FP32 weight per input row")
        }
        let gated = self
            .gate_up_projection
            .forward_fp32_accum(xs, &self.activation)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("activation", &gated);
        let projected = if gated.dtype() == DType::BF16 {
            if self.gate_up_projection.is_packed() && gated.device().is_cuda() {
                super::cuda::linear_bf16_fp32(&gated, self.down_proj.weight())?
            } else {
                // Independent portable reference: convert only this expert's
                // weights, rather than constructing an FP32 copy of the bank.
                gated
                    .to_dtype(DType::F32)?
                    .matmul(&self.down_proj.weight().to_dtype(DType::F32)?.t()?)?
            }
        } else {
            self.down_proj.forward(&gated)?.to_dtype(DType::F32)?
        };
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("down", &projected);
        projected.broadcast_mul(weights)?.to_dtype(xs.dtype())
    }

    /// Prepares the packed gate/up projection, SiLU product and down projection
    /// in persistent slots, without creating a nested MLP graph.
    pub(crate) fn prepare_decode_program(
        &self,
        program: &mut super::DecodeProgram,
        input: &Tensor,
    ) -> CandleResult<Tensor> {
        let hidden = self
            .gate_up_projection
            .prepare_decode_program(program, input)?;
        program.linear(&hidden, &self.down_proj)
    }

    /// Loads a gated MLP from its bias-free gate, up, and down projections.
    pub(crate) fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let gate_proj = vb
            .pp("gate_proj")
            .get((intermediate_size, hidden_size), "weight")?;
        let up_proj = vb
            .pp("up_proj")
            .get((intermediate_size, hidden_size), "weight")?;
        let down_proj = vb
            .pp("down_proj")
            .get((hidden_size, intermediate_size), "weight")?;
        Self::from_separate_weights(
            gate_proj,
            up_proj,
            down_proj,
            activation,
            optimize_kernel,
        )
    }

    /// Selects separate or packed execution for individually loaded weights.
    fn from_separate_weights(
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        activation: Activation,
        optimize_kernel: bool,
    ) -> CandleResult<Self> {
        let gate_up_projection = GateUpProjection::from_separate(
            gate_proj,
            up_proj,
            activation,
            optimize_kernel,
        )?;
        Ok(Self {
            decode_graph: None,
            prefill_graph: None,
            hidden_size: down_proj.dim(0)?,
            device: down_proj.device().clone(),
            dtype: down_proj.dtype(),
            gate_up_projection,
            down_proj: Linear::new(down_proj, None),
            activation,
        })
    }

    /// Builds an MLP from a checkpoint tensor that already packs gate and up.
    pub(crate) fn from_packed_weights(
        gate_up_proj: Tensor,
        down_proj: Tensor,
        intermediate_size: usize,
        activation: Activation,
        optimize_kernel: bool,
    ) -> CandleResult<Self> {
        let gate_up_projection = GateUpProjection::from_packed(
            gate_up_proj,
            intermediate_size,
            activation,
            optimize_kernel,
        )?;
        Ok(Self {
            decode_graph: None,
            prefill_graph: None,
            hidden_size: down_proj.dim(0)?,
            device: down_proj.device().clone(),
            dtype: down_proj.dtype(),
            gate_up_projection,
            down_proj: Linear::new(down_proj, None),
            activation,
        })
    }
}

impl Module for Mlp {
    /// Applies the gated activation and projects the result back to hidden size.
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let gated = self.gate_up_projection.forward(xs, &self.activation)?;
        #[cfg(test)]
        {
            crate::test_support::tensor_capture::tensor("activation", &gated);
            let output = gated.apply(&self.down_proj)?;
            crate::test_support::tensor_capture::tensor("down", &output);
            Ok(output)
        }
        #[cfg(not(test))]
        gated.apply(&self.down_proj)
    }
}

#[cfg(test)]
mod tests;
