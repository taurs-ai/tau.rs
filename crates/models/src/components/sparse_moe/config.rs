//! Generic sparse-MoE dimensions, storage layout and shared-branch contracts.

use candle_nn::Activation;

/// Function that converts router logits into per-expert routing scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterScoreFunction {
    Softmax,
}

/// Serialized representation used for routed-expert projection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SparseMoeWeightLayout {
    /// Each expert owns separate gate, up, and down projection tensors.
    Individual,
    /// All experts share packed gate/up and down projection tensors.
    Packed,
}

/// Optional learned gate applied to an always-active shared expert.
#[derive(Debug, Clone)]
pub(crate) struct SharedExpertGateConfig {
    pub(crate) activation: Activation,
    pub(crate) bias: bool,
}

/// Construction choices for an always-active expert outside top-k routing.
#[derive(Debug, Clone)]
pub(crate) struct SharedExpertConfig {
    pub(crate) intermediate_size: usize,
    pub(crate) activation: Activation,
    pub(crate) gate: Option<SharedExpertGateConfig>,
}

/// Dimensions and routing behavior for a sparse mixture-of-experts block.
#[derive(Debug, Clone)]
pub(crate) struct SparseMoeConfig {
    pub(crate) hidden_size: usize,
    pub(crate) expert_intermediate_size: usize,
    pub(crate) expert_activation: Activation,
    pub(crate) num_experts: usize,
    pub(crate) experts_per_token: usize,
    pub(crate) router_bias: bool,
    pub(crate) router_score_function: RouterScoreFunction,
    pub(crate) normalize_selected_weights: bool,
    pub(crate) shared_expert: Option<SharedExpertConfig>,
    pub(crate) weight_layout: SparseMoeWeightLayout,
}

impl SparseMoeConfig {
    /// Checks dimensions needed by the router, routed experts, and shared path.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.expert_intermediate_size == 0
            || self.num_experts == 0
            || self.experts_per_token == 0
        {
            return Err(
                "sparse MoE dimensions and expert counts must be non-zero".to_owned()
            );
        }
        if self.experts_per_token > self.num_experts {
            return Err(
                "sparse MoE experts_per_token cannot exceed num_experts".to_owned()
            );
        }
        if self
            .shared_expert
            .as_ref()
            .is_some_and(|shared| shared.intermediate_size == 0)
        {
            return Err(
                "sparse MoE shared-expert intermediate size must be non-zero"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::valid_config;

    /// Rejects zero dimensions and top-k counts larger than the expert pool.
    #[test]
    fn validates_general_config() {
        valid_config().validate().unwrap();

        let mut invalid = valid_config();
        invalid.experts_per_token = invalid.num_experts + 1;
        assert!(invalid.validate().unwrap_err().contains("cannot exceed"));

        let mut invalid = valid_config();
        invalid.shared_expert.as_mut().unwrap().intermediate_size = 0;
        assert!(invalid.validate().unwrap_err().contains("shared-expert"));
    }
}
