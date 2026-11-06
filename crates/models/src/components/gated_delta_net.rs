//! General-purpose recurrent Gated DeltaNet token mixer.
//!
//! The implementation combines depthwise causal convolution, learned decay and
//! update gates, and a recurrent delta-rule memory matrix. Its construction depends
//! only on [`GatedDeltaNetConfig`], not on a surrounding model-family config. The
//! expected checkpoint subtree contains the canonical GDN projections and state
//! parameters (`in_proj_*`, `conv1d`, `dt_bias`, `A_log`, `norm`, and `out_proj`).
//! `A_log` and `norm.weight` load directly in FP32, and recurrent state is FP32.
//! Other weights and convolution state follow the primary runtime dtype. The
//! recurrence output rounds to the activation dtype before FP32 gated RMSNorm,
//! whose result returns to that dtype for the output projection.

// Keep layer construction and execution dispatch here. Private modules own
// composed math, convolution/history and preparation for decoder-owned graphs.
mod composed;
mod convolution;
#[cfg(test)]
pub(in crate::components) mod golden;
mod prepared;

use super::rms_norm::GatedRmsNorm;
use candle::{DType, Module, Result as CandleResult, Tensor, Var};
use candle_nn::Linear;
use candle_nn::{Activation, VarBuilder};
use std::sync::Once;

/// Reports the first Gated DeltaNet backend selected during model construction.
static GDN_BACKEND_LOGGED: Once = Once::new();

/// Dimensions and numerical choices needed to construct a Gated DeltaNet layer.
#[derive(Debug, Clone)]
pub(crate) struct GatedDeltaNetConfig {
    pub(crate) hidden_size: usize,
    pub(crate) conv_kernel_size: usize,
    pub(crate) key_head_dim: usize,
    pub(crate) value_head_dim: usize,
    pub(crate) num_key_heads: usize,
    pub(crate) num_value_heads: usize,
    pub(crate) norm_eps: f64,
    pub(crate) conv_activation: Activation,
    pub(crate) gate_activation: Activation,
}

impl GatedDeltaNetConfig {
    /// Checks dimensions required by projection, convolution, and head reshaping.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.conv_kernel_size == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.num_key_heads == 0
            || self.num_value_heads == 0
        {
            return Err(
                "Gated DeltaNet dimensions and head counts must be non-zero".to_owned()
            );
        }
        if !self.num_value_heads.is_multiple_of(self.num_key_heads) {
            return Err(
                "Gated DeltaNet value heads must be divisible by key heads".to_owned()
            );
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(
                "Gated DeltaNet normalization epsilon must be positive".to_owned()
            );
        }
        Ok(())
    }
}

/// Gated DeltaNet with depthwise causal convolution and recurrent state caching.
#[derive(Debug)]
pub(crate) struct GatedDeltaNet {
    hidden_size: usize,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: GatedRmsNorm,
    out_proj: Linear,
    conv_activation: Activation,
    gate_activation: Activation,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_width: usize,
    value_width: usize,
    conv_width: usize,
    conv_kernel_size: usize,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    // Independent of the stateful decode runner: prepared prefill clones retain
    // this policy while deliberately excluding mutable native decode workspace.
    optimize_prefill: bool,
    reference_precision: bool,
    #[cfg(test)]
    prefill_qk_rounding_override: Option<bool>,
    optimized_decode: Option<OptimizedGatedDeltaNetDecode>,
    optimized_state_active: bool,
}

/// Persistent prefill state inputs/outputs. Separate input and output slots let
/// a graph read the previous chunk's state and publish the next one without any
/// Rust cache assignment inside replay. The decoder graph retains these handles.
#[derive(Debug, Clone)]
pub(crate) struct GdnPrefillState {
    conv_input: Var,
    recurrent_input: Var,
    conv_output: Var,
    recurrent_output: Var,
}

/// Packed projections and the stateful CUDA core used for single-token decode.
#[derive(Debug, Clone)]
struct OptimizedGatedDeltaNetDecode {
    qkvz_proj: Linear,
    ba_proj: Linear,
    runner: super::cuda::GatedDeltaNetRunner,
}

impl OptimizedGatedDeltaNetDecode {
    /// Reports whether the dynamic hidden-state tensor can use native decode.
    fn supports_input(&self, xs: &Tensor) -> bool {
        self.runner.supports_input(xs)
    }

    /// Runs two packed projections, the stateful CUDA core, and the output projection.
    fn forward(&self, xs: &Tensor, out_proj: &Linear) -> CandleResult<Tensor> {
        let qkvz = self.qkvz_proj.forward(xs)?;
        let ba = self.ba_proj.forward(xs)?;
        out_proj.forward(&self.runner.forward(&qkvz, &ba)?)
    }
}

impl GatedDeltaNet {
    /// Seeds independent same-input layer tests before execution. These tensors
    /// are copied so replay cannot mutate the oracle's initial-state storage.
    #[cfg(test)]
    pub(crate) fn seed_validation_state(&mut self, conv: &Tensor, state: &Tensor) {
        assert_eq!(conv.dims(), [1, self.conv_width, self.conv_kernel_size]);
        assert_eq!(
            state.dims(),
            [
                1,
                self.num_value_heads,
                self.key_head_dim,
                self.value_head_dim
            ]
        );
        self.clear_cache();
        self.conv_state = Some(conv.force_contiguous().unwrap());
        self.recurrent_state = Some(state.force_contiguous().unwrap());
    }
    /// Keeps the shared runner while selecting the MoE reference reduction tree.
    pub(crate) fn with_reference_precision(mut self, enabled: bool) -> Self {
        self.reference_precision = enabled;
        if let Some(decode) = &mut self.optimized_decode {
            decode.runner.set_reference_reduction(enabled);
        }
        self
    }

    /// Selects the existing prefill storage boundary. Component tests can vary
    /// this one boundary without changing gate arithmetic or the reduction tree.
    /// Production retains the model's reference-precision selection unchanged.
    fn round_prefill_normalized_qk(&self) -> bool {
        #[cfg(test)]
        if let Some(round) = self.prefill_qk_rounding_override {
            return round;
        }
        self.reference_precision
    }

    /// Loads one GDN layer and initializes its sequence caches empty.
    pub(crate) fn new(
        cfg: &GatedDeltaNetConfig,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        if let Err(message) = cfg.validate() {
            candle::bail!("{message}")
        }
        let key_width = cfg.num_key_heads * cfg.key_head_dim;
        let value_width = cfg.num_value_heads * cfg.value_head_dim;
        let conv_width = key_width * 2 + value_width;
        let in_proj_qkv_weight =
            vb.get((conv_width, cfg.hidden_size), "in_proj_qkv.weight")?;
        let in_proj_z_weight =
            vb.get((value_width, cfg.hidden_size), "in_proj_z.weight")?;
        let in_proj_b_weight =
            vb.get((cfg.num_value_heads, cfg.hidden_size), "in_proj_b.weight")?;
        let in_proj_a_weight =
            vb.get((cfg.num_value_heads, cfg.hidden_size), "in_proj_a.weight")?;
        let conv_weight =
            vb.get((conv_width, 1, cfg.conv_kernel_size), "conv1d.weight")?;
        let dt_bias = vb.get(cfg.num_value_heads, "dt_bias")?;
        // Request FP32 directly from storage: BF16 then FP32 would irreversibly
        // round the master parameters in mixed-precision checkpoints.
        let fp32_vb = vb.to_dtype(DType::F32);
        let a_log = fp32_vb.get(cfg.num_value_heads, "A_log")?;
        let norm_weight = fp32_vb.pp("norm").get(cfg.value_head_dim, "weight")?;
        let out_proj_weight =
            vb.get((cfg.hidden_size, value_width), "out_proj.weight")?;
        let optimized_decode = if optimize_kernel
            && cfg.conv_activation == Activation::Silu
            && cfg.gate_activation == Activation::Silu
            && vb.device().is_cuda()
            && vb.dtype() == DType::BF16
        {
            let qkvz_weight =
                Tensor::cat(&[&in_proj_qkv_weight, &in_proj_z_weight], 0)?
                    .contiguous()?;
            let ba_weight = Tensor::cat(&[&in_proj_b_weight, &in_proj_a_weight], 0)?
                .contiguous()?;
            let negative_a = a_log.to_dtype(DType::F32)?.exp()?.neg()?.contiguous()?;
            super::cuda::GatedDeltaNetRunner::new(
                cfg.hidden_size,
                cfg.conv_kernel_size,
                cfg.key_head_dim,
                cfg.value_head_dim,
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.norm_eps,
                conv_weight.clone(),
                dt_bias.clone(),
                negative_a,
                norm_weight.clone(),
            )?
            .map(|runner| OptimizedGatedDeltaNetDecode {
                qkvz_proj: Linear::new(qkvz_weight, None),
                ba_proj: Linear::new(ba_weight, None),
                runner,
            })
        } else {
            None
        };
        GDN_BACKEND_LOGGED.call_once(|| {
            let backend = if optimized_decode.is_some() {
                "native BF16 decode + eligible prefill convolution/recurrence/norm"
            } else {
                "Candle reference"
            };
            eprintln!("tau-models Gated DeltaNet backend: {backend}");
        });
        Ok(Self {
            hidden_size: cfg.hidden_size,
            in_proj_qkv: Linear::new(in_proj_qkv_weight, None),
            in_proj_z: Linear::new(in_proj_z_weight, None),
            in_proj_b: Linear::new(in_proj_b_weight, None),
            in_proj_a: Linear::new(in_proj_a_weight, None),
            conv_weight,
            dt_bias,
            a_log,
            norm: GatedRmsNorm::from_weight(norm_weight, cfg.norm_eps),
            out_proj: Linear::new(out_proj_weight, None),
            conv_activation: cfg.conv_activation,
            gate_activation: cfg.gate_activation,
            num_key_heads: cfg.num_key_heads,
            num_value_heads: cfg.num_value_heads,
            key_head_dim: cfg.key_head_dim,
            value_head_dim: cfg.value_head_dim,
            key_width,
            value_width,
            conv_width,
            conv_kernel_size: cfg.conv_kernel_size,
            conv_state: None,
            recurrent_state: None,
            optimize_prefill: optimize_kernel,
            reference_precision: false,
            #[cfg(test)]
            prefill_qk_rounding_override: None,
            optimized_decode,
            optimized_state_active: false,
        })
    }

    /// Calculates filtered Q/K/V and publishes raw history only after success.
    /// The convolution module receives explicit inputs and never mutates this layer.
    fn causal_convolution(&mut self, xs: &Tensor) -> CandleResult<Tensor> {
        let (output, next_history) = convolution::forward(
            xs,
            &self.conv_weight,
            self.conv_state.as_ref(),
            self.conv_activation,
            self.optimize_prefill,
        )?;
        // Future chunks/decode consume raw projections, not activated outputs.
        self.conv_state = Some(next_history);
        Ok(output)
    }

    /// Selects native single-token decode or the portable stateful implementation.
    pub(crate) fn forward(&mut self, xs: &Tensor) -> CandleResult<Tensor> {
        if self
            .optimized_decode
            .as_ref()
            .is_some_and(|optimized| optimized.supports_input(xs))
        {
            self.prepare_optimized_decode()?;
            return self.forward_prepared_decode(xs);
        }

        self.materialize_optimized_state()?;
        self.forward_composed(xs)
    }

    /// Invalidates sequence state; the next eager call initializes a new sequence.
    ///
    /// Native buffers are retained and lazily reseeded. The graph owner must
    /// explicitly seed/zero them before replay, which bypasses this host bookkeeping.
    pub(crate) fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
        self.optimized_state_active = false;
    }
}

#[cfg(test)]
mod tests;
