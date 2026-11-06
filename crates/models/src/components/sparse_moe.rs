//! Sparse-MoE composition and execution dispatch.
//!
//! Routing owns expert selection, the expert bank owns routed computation, and
//! the shared branch remains outside top-k normalization. This facade chooses
//! the supported backend and combines the two outputs.

mod config;
mod experts;
mod prepared;
mod routing;
mod shared_expert;

pub(crate) use config::{
    RouterScoreFunction, SharedExpertConfig, SharedExpertGateConfig, SparseMoeConfig,
    SparseMoeWeightLayout,
};

use super::cuda::{FusedMoeRunner, MoePrefillRunner};
use candle::{Module, Result as CandleResult, Tensor};
use candle_nn::{Activation, VarBuilder};
use experts::ExpertBank;
use routing::Router;
use shared_expert::SharedExpert;
use std::sync::Once;

static MOE_BACKEND_LOGGED: Once = Once::new();
static PREFILL_BACKEND_LOGGED: Once = Once::new();

/// Select an execution policy without confusing one-token prompt tails with
/// decode. Native prefill is tested explicitly before ordinary prefill selects it.
#[derive(Clone, Copy)]
enum Execution {
    #[cfg(test)]
    ReferencePrefill,
    GroupedPrefill,
    Decode,
}

/// MoE feed-forward block within one decoder layer, used during both
/// prefill and decode.
/// Routes each token to top-k experts and optionally adds a shared expert,
/// which may have its own learned gate.
/// Each decoder layer has its own router and separate expert weights.
#[derive(Debug, Clone)]
pub(crate) struct SparseMoeBlock {
    router: Router,
    experts: ExpertBank,
    shared_expert: Option<SharedExpert>,
    fused_moe_runner: Option<FusedMoeRunner>,
    prefill_runner: Option<MoePrefillRunner>,
    require_native_prefill: bool,
    // require using Tau's native MoE decode cuda kernels.
    require_native_decode: bool,
}

impl SparseMoeBlock {
    /// Apply an explicit startup tile selection before any prefill workspace is
    /// allocated. Unsupported models fail here instead of ignoring an override.
    pub(crate) fn with_prefill_config(
        mut self,
        config: Option<crate::MoePrefillConfig>,
    ) -> CandleResult<Self> {
        if let Some(config) = config {
            let Some(runner) = self.prefill_runner.as_mut() else {
                candle::bail!(
                    "explicit MoE prefill configuration requires the native CUDA/BF16 backend"
                )
            };
            runner.configure(config);
            self.require_native_prefill = true;
        }
        Ok(self)
    }

    /// Apply a startup decode choice to the existing routed runner. Unsupported
    /// blocks fail before graph construction instead of ignoring an override.
    /// With no override, eligible devices use their measured preset; other
    /// backends retain ordinary fallback behavior and baseline selection.
    pub(crate) fn with_decode_options(
        mut self,
        options: Option<crate::MoeDecodeOptions>,
    ) -> CandleResult<Self> {
        match (self.fused_moe_runner.as_mut(), options) {
            (Some(runner), Some(options)) => {
                runner.configure_decode(options)?;
                self.require_native_decode = true;
            }
            (Some(runner), None) => runner.configure_decode_defaults()?,
            (None, Some(_)) => {
                candle::bail!(
                    "explicit MoE decode configuration requires the native CUDA/BF16 backend"
                )
            }
            (None, None) => {}
        }
        Ok(self)
    }

    /// Compose one layer's current token rows using an explicit phase. Prefill
    /// uses device grouping when supported; decode keeps its one-token runner.
    /// Unsupported prefill inputs retain the portable reference expert loop.
    fn forward_phase(&self, xs: &Tensor, mode: Execution) -> CandleResult<Tensor> {
        let (batch_size, sequence_length, hidden_size) = xs.dims3()?;
        let flattened = xs.reshape(((), hidden_size))?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("input", &flattened);
        let logits = self.router.logits(&flattened)?;
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("router-projection", &logits);
        #[cfg(test)]
        let logits = crate::test_support::tensor_capture::router_logits(logits);
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("router", &logits);
        let routed = match mode {
            Execution::Decode => self.forward_routed(&flattened, &logits)?,
            Execution::GroupedPrefill
                if self.prefill_runner.as_ref().is_some_and(|runner| {
                    runner.supports_input(&flattened, &logits)
                }) =>
            {
                PREFILL_BACKEND_LOGGED.call_once(|| {
                    let config = self.prefill_runner.as_ref().unwrap().config();
                    eprintln!("tau-models sparse MoE prefill backend: grouped BF16/FP32 (device dispatch); gate_up={}; down={}", config.gate_up(), config.down());
                });
                self.prefill_runner
                    .as_ref()
                    .unwrap()
                    .forward(&flattened, &logits)?
            }
            _ => {
                if self.require_native_prefill
                    && matches!(mode, Execution::GroupedPrefill)
                {
                    candle::bail!(
                        "input cannot execute the requested native MoE prefill configuration"
                    )
                }
                let plan = self.router.reference_plan(&logits)?;
                self.experts.forward_reference(&flattened, &plan)?
            }
        };
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("routed", &routed);
        let output = match &self.shared_expert {
            Some(shared) => shared.combine(&flattened, &routed)?,
            None => routed,
        };
        #[cfg(test)]
        crate::test_support::tensor_capture::tensor("output", &output);
        output.reshape((batch_size, sequence_length, hidden_size))
    }

    pub(crate) fn new(
        cfg: &SparseMoeConfig,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        if let Err(message) = cfg.validate() {
            candle::bail!("{message}")
        }
        let router = Router::new(cfg, optimize_kernel, vb.pp("gate"))?;
        let experts = ExpertBank::load(cfg, optimize_kernel, vb.pp("experts"))?;
        // Block policy decides whether native execution is appropriate. The
        // CUDA runner separately validates hardware, dtype and tensor layout.
        let fused_moe_runner = if optimize_kernel
            && cfg.expert_activation == Activation::Silu
            && cfg.router_score_function == RouterScoreFunction::Softmax
            && cfg.normalize_selected_weights
        {
            match experts.packed_weights() {
                Some((gate_up, down)) => FusedMoeRunner::new(
                    gate_up.clone(),
                    down.clone(),
                    cfg.experts_per_token,
                )?,
                None => None,
            }
        } else {
            None
        };
        let uses_fused_decode = fused_moe_runner.is_some();
        MOE_BACKEND_LOGGED.call_once(|| {
            let backend = if uses_fused_decode {
                "fused-bf16-fp32-mma-decode (reference fallback for unsupported inputs)"
            } else {
                "Candle reference"
            };
            eprintln!("tau-models sparse MoE backend: {backend}");
        });
        let shared_expert = match &cfg.shared_expert {
            Some(shared) => Some(SharedExpert::new(
                shared,
                cfg.hidden_size,
                optimize_kernel,
                vb,
            )?),
            None => None,
        };
        let prefill_runner = fused_moe_runner
            .as_ref()
            .map(FusedMoeRunner::prefill_runner);
        Ok(Self {
            router,
            experts,
            shared_expert,
            fused_moe_runner,
            prefill_runner,
            require_native_prefill: false,
            require_native_decode: false,
        })
    }

    fn forward_routed(&self, input: &Tensor, logits: &Tensor) -> CandleResult<Tensor> {
        if let Some(runner) = &self.fused_moe_runner
            && runner.supports_input(input, logits)
        {
            #[cfg(test)]
            if crate::test_support::tensor_capture::active() {
                let output = runner.forward(input, logits)?;
                let (ids, weights) = runner.selected_routing()?;
                crate::test_support::tensor_capture::host(
                    "ids",
                    vec![1, ids.len()],
                    ids.iter().map(|id| *id as f32).collect(),
                );
                crate::test_support::tensor_capture::host(
                    "weights",
                    vec![1, weights.len()],
                    weights,
                );
                return Ok(output);
            }
            return runner.forward(input, logits);
        }
        if self.require_native_decode {
            candle::bail!(
                "input cannot execute the requested native MoE decode configuration"
            )
        }
        // Host routing and grouping are confined to the portable fallback.
        // Eligible GPU decode never constructs a host plan or reads back logits.
        let plan = self.router.reference_plan(logits)?;
        self.experts.forward_reference(input, &plan)
    }
}

impl Module for SparseMoeBlock {
    /// Ordinary module execution is prefill; decode is an explicit capability.
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward_phase(xs, Execution::GroupedPrefill)
    }
}

#[cfg(test)]
mod tests;
