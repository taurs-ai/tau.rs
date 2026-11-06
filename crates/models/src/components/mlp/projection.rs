//! Immutable gate/up projection layouts, packing policy and gated computation.
//!
//! Constructors accept tensors and execution policy explicitly. Forward uses only
//! the selected projection weights, input and activation; it has no Mlp or graph
//! dependency. The layer owns the final down projection and graph lifecycle.

use crate::components::cuda;
use candle::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::Activation;
use candle_nn::Linear;

/// Gate/up projection layout selected when the MLP is constructed.
#[derive(Debug, Clone)]
pub(super) enum GateUpProjection {
    /// Independent checkpoint projections retained for portable reference execution.
    Separate { gate_proj: Linear, up_proj: Linear },
    /// Gate and up weights concatenated for one larger vendor-backed GEMM.
    Packed { gate_up_proj: Linear },
}

impl GateUpProjection {
    /// Gives weighted expert execution an FP32 accumulator before BF16 storage.
    /// Dense forward keeps its original projection policy. Unpacked/reference
    /// layouts retain their independent activation implementation.
    pub(super) fn forward_fp32_accum(
        &self,
        xs: &Tensor,
        activation: &Activation,
    ) -> CandleResult<Tensor> {
        if let Self::Packed { gate_up_proj } = self
            && xs.dtype() == DType::BF16
            && xs.device().is_cuda()
        {
            let projected = cuda::linear_bf16_fp32(xs, gate_up_proj.weight())?
                .to_dtype(DType::BF16)?;
            #[cfg(test)]
            crate::test_support::tensor_capture::tensor("gate_up", &projected);
            return silu_and_mul(&projected, activation);
        }
        self.forward(xs, activation)
    }

    /// Packs separate gate/up weights once when the native SiLU contract permits;
    /// otherwise retains the two checkpoint projections for reference execution.
    pub(super) fn from_separate(
        gate_proj: Tensor,
        up_proj: Tensor,
        activation: Activation,
        optimize_kernel: bool,
    ) -> CandleResult<Self> {
        let gate_up_projection = if supports_packed_silu_weights(
            &gate_proj,
            &up_proj,
            activation,
            optimize_kernel,
        ) {
            let gate_up = Tensor::cat(&[&gate_proj, &up_proj], 0)?;
            GateUpProjection::Packed {
                gate_up_proj: Linear::new(gate_up, None),
            }
        } else {
            GateUpProjection::Separate {
                gate_proj: Linear::new(gate_proj, None),
                up_proj: Linear::new(up_proj, None),
            }
        };
        Ok(gate_up_projection)
    }

    /// Uses an existing packed checkpoint view, or splits it into separate views
    /// when activation, device, dtype or kernel policy requires the reference path.
    pub(super) fn from_packed(
        gate_up_proj: Tensor,
        intermediate_size: usize,
        activation: Activation,
        optimize_kernel: bool,
    ) -> CandleResult<Self> {
        let use_packed = supports_packed_silu_tensor(
            &gate_up_proj,
            intermediate_size,
            activation,
            optimize_kernel,
        );
        let gate_up_projection = if use_packed {
            GateUpProjection::Packed {
                gate_up_proj: Linear::new(gate_up_proj, None),
            }
        } else {
            let gate_proj = gate_up_proj.narrow(0, 0, intermediate_size)?;
            let up_proj =
                gate_up_proj.narrow(0, intermediate_size, intermediate_size)?;
            GateUpProjection::Separate {
                gate_proj: Linear::new(gate_proj, None),
                up_proj: Linear::new(up_proj, None),
            }
        };
        Ok(gate_up_projection)
    }

    /// Records one packed gate/up projection followed by the SiLU product.
    /// Separate projections are rejected because this prepared path relies on
    /// the packed SiLU layout chosen during MLP construction.
    pub(super) fn prepare_decode_program(
        &self,
        program: &mut crate::components::DecodeProgram,
        input: &Tensor,
    ) -> CandleResult<Tensor> {
        let Self::Packed { gate_up_proj } = self else {
            candle::bail!("prepared MLP requires packed SiLU")
        };
        let projected = program.linear(input, gate_up_proj)?;
        program.silu(&projected)
    }

    /// Reports the fixed layout selected during construction. Graph eligibility
    /// adds its own stricter dtype/device/input contract in the graph module.
    pub(super) fn is_packed(&self) -> bool {
        matches!(self, Self::Packed { .. })
    }

    /// Produces activated gate * up, ready for the layer's down projection.
    /// This performs ordinary tensor computation, including during graph capture.
    pub(super) fn forward(
        &self,
        xs: &Tensor,
        activation: &Activation,
    ) -> CandleResult<Tensor> {
        let gated = match self {
            GateUpProjection::Separate { gate_proj, up_proj } => {
                let gate = gate_proj.forward(xs)?.apply(activation)?;
                let up = up_proj.forward(xs)?;
                (gate * up)?
            }
            GateUpProjection::Packed { gate_up_proj } => {
                let gate_up = gate_up_proj.forward(xs)?;
                #[cfg(test)]
                crate::test_support::tensor_capture::tensor("gate_up", &gate_up);
                silu_and_mul(&gate_up, activation)?
            }
        };
        Ok(gated)
    }
}

/// Checks whether separate weights can enter the CUDA packed-SwiGLU path.
fn supports_packed_silu_weights(
    gate_proj: &Tensor,
    up_proj: &Tensor,
    activation: Activation,
    optimize_kernel: bool,
) -> bool {
    optimize_kernel
        && activation == Activation::Silu
        && gate_proj.device().is_cuda()
        && gate_proj.device().same_device(up_proj.device())
        && gate_proj.dtype() == up_proj.dtype()
        && matches!(gate_proj.dtype(), DType::F32 | DType::BF16)
        && gate_proj.dims() == up_proj.dims()
}

/// Checks whether an existing packed weight has the expected CUDA layout.
fn supports_packed_silu_tensor(
    gate_up_proj: &Tensor,
    intermediate_size: usize,
    activation: Activation,
    optimize_kernel: bool,
) -> bool {
    intermediate_size
        .checked_mul(2)
        .is_some_and(|packed_width| {
            optimize_kernel
                && activation == Activation::Silu
                && gate_up_proj.device().is_cuda()
                && matches!(gate_up_proj.dtype(), DType::F32 | DType::BF16)
                && gate_up_proj
                    .dims()
                    .first()
                    .is_some_and(|&width| width == packed_width)
        })
}

/// Applies packed gating through the native kernel or its Candle fallback.
fn silu_and_mul(gate_up: &Tensor, activation: &Activation) -> CandleResult<Tensor> {
    if *activation == Activation::Silu && cuda::supports_silu_and_mul(gate_up) {
        return cuda::silu_and_mul_cuda(gate_up);
    }

    silu_and_mul_reference(gate_up, activation)
}

/// Computes activation and gating from the two halves using Candle operations.
fn silu_and_mul_reference(
    gate_up: &Tensor,
    activation: &Activation,
) -> CandleResult<Tensor> {
    let last_dimension = gate_up.rank().checked_sub(1).ok_or_else(|| {
        candle::Error::Msg(
            "packed gate/up projection must have rank at least one".to_owned(),
        )
    })?;
    let packed_width = gate_up.dim(last_dimension)?;
    if packed_width == 0 || packed_width % 2 != 0 {
        candle::bail!(
            "packed gate/up projection width must be positive and even, got \
             {packed_width}"
        )
    }
    let intermediate_size = packed_width / 2;
    let gate = gate_up
        .narrow(last_dimension, 0, intermediate_size)?
        .apply(activation)?;
    let up = gate_up.narrow(last_dimension, intermediate_size, intermediate_size)?;
    gate * up
}
