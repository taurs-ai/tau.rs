//! Expert weight ownership, layout adaptation and portable expert execution.

use super::config::{SparseMoeConfig, SparseMoeWeightLayout};
use super::routing::RoutingPlan;
use crate::components::Mlp;
use candle::{DType, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
struct PackedExpertWeights {
    gate_up: Tensor,
    down: Tensor,
}

#[derive(Debug, Clone)]
pub(super) struct ExpertBank {
    experts: Vec<Mlp>,
    packed: Option<PackedExpertWeights>,
}

impl ExpertBank {
    pub(super) fn load(
        cfg: &SparseMoeConfig,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        match cfg.weight_layout {
            SparseMoeWeightLayout::Individual => {
                let mut experts = Vec::with_capacity(cfg.num_experts);
                for expert in 0..cfg.num_experts {
                    experts.push(Mlp::new(
                        cfg.hidden_size,
                        cfg.expert_intermediate_size,
                        cfg.expert_activation,
                        optimize_kernel,
                        vb.pp(expert),
                    )?);
                }
                Ok(Self {
                    experts,
                    packed: None,
                })
            }
            SparseMoeWeightLayout::Packed => {
                Self::load_packed(cfg, optimize_kernel, vb)
            }
        }
    }

    fn load_packed(
        cfg: &SparseMoeConfig,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let gate_up = vb.get(
            (
                cfg.num_experts,
                cfg.expert_intermediate_size * 2,
                cfg.hidden_size,
            ),
            "gate_up_proj",
        )?;
        let down = vb.get(
            (
                cfg.num_experts,
                cfg.hidden_size,
                cfg.expert_intermediate_size,
            ),
            "down_proj",
        )?;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for expert in 0..cfg.num_experts {
            // These are views into the original packed storage. Keeping both
            // forms lets reference MLPs and the GPU runner share weight bytes.
            let gate_up_view = gate_up.narrow(0, expert, 1)?.squeeze(0)?;
            let down_view = down.narrow(0, expert, 1)?.squeeze(0)?;
            experts.push(Mlp::from_packed_weights(
                gate_up_view,
                down_view,
                cfg.expert_intermediate_size,
                cfg.expert_activation,
                optimize_kernel,
            )?);
        }
        Ok(Self {
            experts,
            packed: Some(PackedExpertWeights { gate_up, down }),
        })
    }

    pub(super) fn packed_weights(&self) -> Option<(&Tensor, &Tensor)> {
        self.packed
            .as_ref()
            .map(|weights| (&weights.gate_up, &weights.down))
    }

    pub(super) fn forward_reference(
        &self,
        input: &Tensor,
        plan: &RoutingPlan,
    ) -> CandleResult<Tensor> {
        debug_assert_eq!(plan.expert_count(), self.experts.len());
        let (tokens, top_k) = plan.geometry();
        let hidden = input.dim(1)?;
        let assignments = tokens.checked_mul(top_k).ok_or_else(|| {
            candle::Error::Msg("MoE assignment count overflow".to_owned())
        })?;
        // Preserve each weighted BF16 result independently, then sum in routing
        // rank order in FP32. Expert scheduling must not dictate rounding/order.
        let mut staged =
            Tensor::zeros((assignments, hidden), input.dtype(), input.device())?;
        for (expert_index, expert) in self.experts.iter().enumerate() {
            let (tokens, weights) = plan.for_expert(expert_index);
            if tokens.is_empty() {
                continue;
            }
            #[cfg(test)]
            let _capture = crate::test_support::tensor_capture::scope(format!(
                "expert-{expert_index}-"
            ));
            #[cfg(test)]
            if crate::test_support::tensor_capture::active() {
                crate::test_support::tensor_capture::host(
                    "tokens",
                    vec![tokens.len()],
                    tokens.iter().map(|id| *id as f32).collect(),
                );
            }
            let indices = Tensor::new(tokens, input.device())?;
            let weights = Tensor::new(weights, input.device())?.reshape(((), 1))?;
            let expert_input = input.index_select(&indices, 0)?;
            let expert_output = expert.forward_weighted(&expert_input, &weights)?;
            #[cfg(test)]
            crate::test_support::tensor_capture::tensor(
                "weighted_down",
                &expert_output,
            );
            let slots =
                Tensor::new(plan.slots_for_expert(expert_index), input.device())?;
            staged = staged.index_add(&slots, &expert_output, 0)?;
        }
        let staged = staged.reshape((tokens, top_k, hidden))?;
        let mut output = Tensor::zeros((tokens, hidden), DType::F32, input.device())?;
        for rank in 0..top_k {
            let contribution = staged
                .narrow(1, rank, 1)?
                .squeeze(1)?
                .to_dtype(DType::F32)?;
            output = (&output + contribution)?;
        }
        output.to_dtype(input.dtype())
    }

    #[cfg(test)]
    pub(super) fn expert_for_test(&self, expert: usize) -> &Mlp {
        &self.experts[expert]
    }
}
