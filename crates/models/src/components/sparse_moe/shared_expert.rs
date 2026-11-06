//! Always-active expert branch, independent of routed top-k normalization.

use super::config::SharedExpertConfig;
use crate::components::Mlp;
use candle::{Module, Result as CandleResult, Tensor};
use candle_nn::{Activation, Linear, VarBuilder, linear_b};

/// Shared MLP belonging to one MoE layer, evaluated for every token alongside
/// the selected routed experts. Prefill processes every token in the prompt
/// chunk; decode processes the new token. Each layer has separate shared weights.
/// This branch operates independently of top-k selection and routed normalization.
///
/// When present, its learned gate scales the MLP output before addition to the
/// routed sum: `routed(x) + gate(x) * mlp(x)`. A small gate reduces the contribution
/// without skipping MLP computation. Without a gate, the full MLP output is added.
#[derive(Debug, Clone)]
pub(super) struct SharedExpert {
    mlp: Mlp,
    gate: Option<SharedExpertGate>,
    native_gate: bool,
}

/// Learned projection and activation producing one scalar gate per token in
/// both prefill and decode. The scalar multiplies every feature of that token's
/// shared-expert output before it is added to the weighted routed-expert sum.
///
/// With sigmoid activation, `gate(x) = sigmoid(w · x + b)` lies between zero and
/// one; the bias is optional. This gate is independent of the router's softmax:
/// it controls the shared contribution rather than selecting an expert.
#[derive(Debug, Clone)]
struct SharedExpertGate {
    projection: Linear,
    activation: Activation,
}

impl SharedExpert {
    /// Applies the CUDA FP32 gate/add contract while retaining the ordinary
    /// dense shared MLP. Other devices/dtypes keep the portable implementation.
    pub(super) fn combine(&self, xs: &Tensor, routed: &Tensor) -> CandleResult<Tensor> {
        if self.native_gate
            && let Some(gate) = &self.gate
            && gate.activation == Activation::Sigmoid
            && gate.projection.bias().is_none()
            && xs.device().is_cuda()
            && xs.dtype() == candle::DType::BF16
            && xs.dim(1)? <= 8192
        {
            #[cfg(test)]
            let _capture =
                crate::test_support::tensor_capture::scope("shared-".to_owned());
            let shared = self.mlp.forward(xs)?;
            #[cfg(test)]
            if crate::test_support::tensor_capture::active() {
                // Auxiliary dense gate projection, matching the reference's
                // diagnostic stage; the actual fused dot remains FP32.
                crate::test_support::tensor_capture::tensor(
                    "gate",
                    &gate.projection.forward(xs)?,
                );
            }
            return crate::components::cuda::moe_shared_combine(
                xs,
                gate.projection.weight(),
                &shared,
                routed,
            );
        }
        routed + self.forward(xs)?
    }

    /// Checks both dense MLP and scalar gate before the model mutates state.
    pub(super) fn supports_prepared_decode(&self, input: &Tensor) -> bool {
        use crate::components::GraphFeedForward;
        self.mlp.supports_prepared_decode(input)
            && self.gate.as_ref().is_none_or(|gate| {
                gate.activation == Activation::Sigmoid
                    && input.elem_count() <= 8192
                    && gate.projection.bias().is_none()
                    && gate.projection.weight().dtype() == input.dtype()
                    && gate.projection.weight().is_contiguous()
                    && gate
                        .projection
                        .weight()
                        .device()
                        .same_device(input.device())
            })
    }

    /// Reuses dense prepared MLP operations, then combines with the routed branch
    /// through the same FP32 dot/sigmoid/FMA used by ordinary MoE execution.
    pub(super) fn prepare_decode_combine(
        &self,
        program: &mut crate::components::DecodeProgram,
        input: &Tensor,
        routed: &Tensor,
    ) -> CandleResult<Tensor> {
        let output = self.mlp.prepare_decode_program(program, input)?;
        match &self.gate {
            Some(gate) => program.moe_shared_combine(
                input,
                gate.projection.weight(),
                &output,
                routed,
            ),
            None => program.add(routed, &output),
        }
    }

    /// Loads the shared MLP and its optional scalar gate.
    pub(super) fn new(
        cfg: &SharedExpertConfig,
        hidden_size: usize,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let gate = match &cfg.gate {
            Some(gate) => Some(SharedExpertGate {
                projection: linear_b(
                    hidden_size,
                    1,
                    gate.bias,
                    vb.pp("shared_expert_gate"),
                )?,
                activation: gate.activation,
            }),
            None => None,
        };
        Ok(Self {
            mlp: Mlp::new(
                hidden_size,
                cfg.intermediate_size,
                cfg.activation,
                optimize_kernel,
                vb.pp("shared_expert"),
            )?,
            gate,
            native_gate: optimize_kernel,
        })
    }

    /// Evaluates the shared expert and applies its learned gate when configured.
    pub(super) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        #[cfg(test)]
        let _capture = crate::test_support::tensor_capture::scope("shared-".to_owned());
        let output = self.mlp.forward(xs)?;
        match &self.gate {
            Some(gate) => {
                #[cfg(test)]
                let weight = {
                    let logits = gate.projection.forward(xs)?;
                    crate::test_support::tensor_capture::tensor("gate", &logits);
                    logits.apply(&gate.activation)?
                };
                #[cfg(not(test))]
                let weight = gate.projection.forward(xs)?.apply(&gate.activation)?;
                output.broadcast_mul(&weight)
            }
            None => Ok(output),
        }
    }
}
