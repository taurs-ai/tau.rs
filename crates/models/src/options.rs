//! Model construction choices and internal graph policy/observations.

/// Context tokens per tile in prepared BF16 decode attention. CUDA thread count
/// stays fixed; this geometry is chosen before workspace allocation and capture.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum AttentionBlockSize {
    Tokens32 = 32,
    #[default]
    Tokens64 = 64,
    Tokens128 = 128,
    Tokens256 = 256,
}

impl AttentionBlockSize {
    pub const fn tokens(self) -> usize {
        self as usize
    }
}

impl std::fmt::Display for AttentionBlockSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.tokens().fmt(f)
    }
}

impl std::str::FromStr for AttentionBlockSize {
    type Err = String;

    /// Check for supported attention block sizes.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "32" => Ok(Self::Tokens32),
            "64" => Ok(Self::Tokens64),
            "128" => Ok(Self::Tokens128),
            "256" => Ok(Self::Tokens256),
            _ => Err(format!(
                "invalid --attention-block-size value {value:?}; expected 32, 64, 128, or 256"
            )),
        }
    }
}

/// CUDA graph policy, independent of device, dtype and optimized-kernel selection.
/// Actual coverage is capability- and benefit-dependent and is reported separately.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaGraphMode {
    /// Disable capture and replay; retain ordinary eager kernel selection.
    #[default]
    None,
    /// Graphs stay within individual layers; any layer may remain eager.
    Layer,
    /// Maximum useful supported coverage, including cross-layer graphs when available.
    /// This does not require capturing every operation or the entire forward pass.
    Full,
}

impl CudaGraphMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Layer => "layer",
            Self::Full => "full",
        }
    }

    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl std::fmt::Display for CudaGraphMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for CudaGraphMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "layer" => Ok(Self::Layer),
            "full" => Ok(Self::Full),
            "true" | "false" => Err(format!(
                "invalid --cuda-graph value {value:?}; expected none, layer, or full; \
                 replace false with none and true with layer"
            )),
            _ => Err(format!(
                "invalid --cuda-graph value {value:?}; expected none, layer, or full"
            )),
        }
    }
}

/// Prefill and decode retain separate graphs, even when their shapes coincide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphPhase {
    Prefill,
    Decode,
}

/// Monotonic counts owned by component graphs; trace records use per-step deltas.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CudaGraphStats {
    pub forward_graph_captures: usize,
    pub forward_graph_replays: usize,
    pub gdn_layer_graph_captures: usize,
    pub gdn_layer_graph_replays: usize,
    pub mlp_graph_captures: usize,
    pub mlp_graph_replays: usize,
}

impl CudaGraphStats {
    pub fn merge(self, other: Self) -> Self {
        Self {
            forward_graph_captures: self.forward_graph_captures
                + other.forward_graph_captures,
            forward_graph_replays: self.forward_graph_replays
                + other.forward_graph_replays,
            gdn_layer_graph_captures: self.gdn_layer_graph_captures
                + other.gdn_layer_graph_captures,
            gdn_layer_graph_replays: self.gdn_layer_graph_replays
                + other.gdn_layer_graph_replays,
            mlp_graph_captures: self.mlp_graph_captures + other.mlp_graph_captures,
            mlp_graph_replays: self.mlp_graph_replays + other.mlp_graph_replays,
        }
    }

    pub fn since(self, before: Self) -> Self {
        Self {
            forward_graph_captures: self.forward_graph_captures
                - before.forward_graph_captures,
            forward_graph_replays: self.forward_graph_replays
                - before.forward_graph_replays,
            gdn_layer_graph_captures: self.gdn_layer_graph_captures
                - before.gdn_layer_graph_captures,
            gdn_layer_graph_replays: self.gdn_layer_graph_replays
                - before.gdn_layer_graph_replays,
            mlp_graph_captures: self.mlp_graph_captures - before.mlp_graph_captures,
            mlp_graph_replays: self.mlp_graph_replays - before.mlp_graph_replays,
        }
    }

    /// Enforce capture boundaries even when detailed numerical tracing is disabled.
    pub fn validate_mode(self, mode: CudaGraphMode) -> candle::Result<()> {
        if mode == CudaGraphMode::None && self != Self::default() {
            candle::bail!("CUDA graph activity observed with --cuda-graph none")
        }
        if mode == CudaGraphMode::Layer
            && (self.forward_graph_captures != 0 || self.forward_graph_replays != 0)
        {
            candle::bail!("cross-layer CUDA graph observed with --cuda-graph layer")
        }
        Ok(())
    }
}

/// Construction choices that affect the resulting model without changing its data.
/// The engine factory derives these from `EngineOptions`; they are not retained
/// in the loaded-model bundle or used to configure per-request tracing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLoadOptions {
    optimize_kernel: bool,
    attention_block_size: AttentionBlockSize,
    moe_reference_precision: bool,
    moe_prefill_config: Option<crate::MoePrefillConfig>,
    moe_decode_options: Option<crate::MoeDecodeOptions>,
}

impl Default for ModelLoadOptions {
    /// Enables matching tau.rs kernels unless the caller explicitly opts out.
    fn default() -> Self {
        Self {
            optimize_kernel: true,
            attention_block_size: AttentionBlockSize::default(),
            moe_reference_precision: false,
            moe_prefill_config: None,
            moe_decode_options: None,
        }
    }
}

impl ModelLoadOptions {
    /// Retains exact-profile construction callers. Auto budgets use the options
    /// builder below; both feed one construction-time source of configuration.
    pub fn with_moe_decode_config(
        mut self,
        config: Option<crate::MoeDecodeConfig>,
    ) -> Self {
        self.moe_decode_options = config.map(crate::MoeDecodeOptions::from);
        self
    }
    /// Legacy exact accessor. Automatic selections need device resolution;
    /// callers inspecting startup options use `moe_decode_options`.
    pub fn moe_decode_config(self) -> Option<crate::MoeDecodeConfig> {
        self.moe_decode_options
            .and_then(crate::MoeDecodeOptions::exact_config)
    }
    /// Carries startup selections to model construction, where device resolution
    /// happens once before token execution and graph capture.
    pub fn with_moe_decode_options(
        mut self,
        options: Option<crate::MoeDecodeOptions>,
    ) -> Self {
        self.moe_decode_options = options;
        self
    }
    pub fn moe_decode_options(self) -> Option<crate::MoeDecodeOptions> {
        self.moe_decode_options
    }

    /// Fix both MoE prefill projection geometries before device scratch is built.
    /// None selects the validated device default; an explicit choice must execute.
    pub fn with_moe_prefill_config(
        mut self,
        config: Option<crate::MoePrefillConfig>,
    ) -> Self {
        self.moe_prefill_config = config;
        self
    }

    pub fn moe_prefill_config(self) -> Option<crate::MoePrefillConfig> {
        self.moe_prefill_config
    }

    /// Selects MoE reference precision for gates, rotary tables and attention.
    /// GDN prefill retains its reference BF16 normalization boundary.
    /// Dense construction retains its existing default and arithmetic.
    pub(crate) fn with_moe_reference_precision(mut self) -> Self {
        self.moe_reference_precision = true;
        self
    }

    /// Reports the model-family precision preset selected before construction.
    pub(crate) fn moe_reference_precision(self) -> bool {
        self.moe_reference_precision
    }
    /// Creates options that prefer matching repository-owned kernels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables custom-first dispatch or forces Candle reference implementations.
    pub fn with_optimize_kernel(mut self, optimize_kernel: bool) -> Self {
        self.optimize_kernel = optimize_kernel;
        self
    }

    /// Returns whether model components should prefer matching custom kernels.
    pub fn optimize_kernel(self) -> bool {
        self.optimize_kernel
    }

    /// Selects prepared decode's fixed context tile before allocating graph scratch.
    pub fn with_attention_block_size(mut self, size: AttentionBlockSize) -> Self {
        self.attention_block_size = size;
        self
    }

    pub fn attention_block_size(self) -> AttentionBlockSize {
        self.attention_block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_policy_enforces_boundaries_without_requiring_coverage() {
        for mode in [
            CudaGraphMode::None,
            CudaGraphMode::Layer,
            CudaGraphMode::Full,
        ] {
            assert_eq!(mode.as_str().parse::<CudaGraphMode>().unwrap(), mode);
            assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_str());
            CudaGraphStats::default().validate_mode(mode).unwrap();
        }
        for activity in [
            CudaGraphStats {
                gdn_layer_graph_captures: 1,
                ..Default::default()
            },
            CudaGraphStats {
                mlp_graph_replays: 1,
                ..Default::default()
            },
            CudaGraphStats {
                forward_graph_captures: 1,
                ..Default::default()
            },
            CudaGraphStats {
                forward_graph_replays: 1,
                ..Default::default()
            },
        ] {
            assert!(activity.validate_mode(CudaGraphMode::None).is_err());
            assert_eq!(
                activity.validate_mode(CudaGraphMode::Layer).is_err(),
                activity.forward_graph_captures + activity.forward_graph_replays > 0
            );
            activity.validate_mode(CudaGraphMode::Full).unwrap();
            let total = activity.merge(activity);
            assert_eq!(total.since(activity), activity);
        }
        let layer = CudaGraphStats {
            mlp_graph_replays: 1,
            ..Default::default()
        };
        let full = CudaGraphStats {
            forward_graph_replays: 1,
            ..Default::default()
        };
        assert_eq!(
            tau_metrics::GraphActivity::from(CudaGraphStats::default()).scope(),
            "none"
        );
        assert_eq!(tau_metrics::GraphActivity::from(layer).scope(), "layer");
        assert_eq!(tau_metrics::GraphActivity::from(full).scope(), "full");
        assert_eq!(
            tau_metrics::GraphActivity::from(layer.merge(full)).scope(),
            "mixed"
        );
    }

    /// Verifies that both constructors enable optimized kernels.
    #[test]
    fn optimized_kernels_are_enabled_by_default() {
        assert!(ModelLoadOptions::default().optimize_kernel());
        assert!(ModelLoadOptions::new().optimize_kernel());
    }

    /// Verifies that callers can force the reference implementation.
    #[test]
    fn optimized_kernels_can_be_disabled_for_reference_execution() {
        let options = ModelLoadOptions::new().with_optimize_kernel(false);
        assert!(!options.optimize_kernel());
    }
}
