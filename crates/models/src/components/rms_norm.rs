//! Root-mean-square normalization variants.
//!
//! [`CenteredRmsNorm`] stores its learned scale as a zero-centered adjustment and
//! applies `1 + weight`. [`GatedRmsNorm`] stores the effective scale directly and
//! modulates its normalized output with an activation supplied by the caller. Both
//! variants calculate normalization in FP32 before restoring the input dtype.

use candle::{D, DType, Module, Result as CandleResult, Tensor};
use candle_nn::{Activation, VarBuilder};
use tau_cuda_kernels::RmsNormVectorBytes;

/// RMSNorm whose learned scale is stored as an offset from one.
#[derive(Debug, Clone)]
pub(crate) struct CenteredRmsNorm {
    weight: Tensor,
    eps: f64,
    optimize_kernel: bool,
    preferred_vector_bytes: RmsNormVectorBytes,
}

impl CenteredRmsNorm {
    /// Borrows immutable normalization assets for a fused prepared operation.
    pub(super) fn prepared_parameters(&self) -> (&Tensor, f64) {
        (&self.weight, self.eps)
    }

    /// Retains this norm's scale and epsilon in the decode program and prepares
    /// a fixed output slot for the existing native centered RMSNorm kernel.
    pub(crate) fn prepare_decode_program(
        &self,
        program: &mut super::DecodeProgram,
        input: &Tensor,
    ) -> CandleResult<Tensor> {
        program.norm(input, &self.weight, self.eps)
    }

    /// Returns the rounded residual sum and its normalized value without changing
    /// either input. The decoder still needs the sum after the feed-forward block.
    pub(crate) fn add_residual_and_normalize(
        &self,
        residual: &Tensor,
        branch: &Tensor,
    ) -> CandleResult<(Tensor, Tensor)> {
        if self.optimize_kernel
            && super::cuda::supports_fused_residual_rms_norm(
                residual,
                branch,
                &self.weight,
                self.eps,
            )
        {
            return super::cuda::fused_residual_rms_norm_cuda(
                residual,
                branch,
                &self.weight,
                self.eps,
                self.preferred_vector_bytes,
            );
        }
        self.add_residual_and_normalize_unfused(residual, branch)
    }

    /// Retains the pre-fusion two-operation path for fallback and A/B checks.
    /// With optimized kernels enabled this still uses the current native norm;
    /// with them disabled both operations use Candle.
    pub(crate) fn add_residual_and_normalize_unfused(
        &self,
        residual: &Tensor,
        branch: &Tensor,
    ) -> CandleResult<(Tensor, Tensor)> {
        let sum = (residual + branch)?;
        let normalized = self.forward(&sum)?;
        Ok((sum, normalized))
    }

    /// Loads the learned zero-centered scale for one normalization site.
    pub(crate) fn new(
        size: usize,
        eps: f64,
        optimize_kernel: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self {
            preferred_vector_bytes: preferred_vector_bytes(&weight),
            weight,
            eps,
            optimize_kernel,
        })
    }

    /// Builds a normalization layer from an explicit weight for CUDA A/B tests.
    #[cfg(test)]
    pub(crate) fn from_weight_for_test(
        weight: Tensor,
        eps: f64,
        optimize_kernel: bool,
    ) -> Self {
        Self {
            preferred_vector_bytes: preferred_vector_bytes(&weight),
            weight,
            eps,
            optimize_kernel,
        }
    }
}

impl Module for CenteredRmsNorm {
    /// Normalizes the last dimension in FP32, then applies `1 + weight`.
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        if self.optimize_kernel
            && super::cuda::supports_centered_rms_norm(xs, &self.weight, self.eps)
        {
            return super::cuda::centered_rms_norm_cuda(
                xs,
                &self.weight,
                self.eps,
                self.preferred_vector_bytes,
            );
        }

        centered_rms_norm_reference(xs, &self.weight, self.eps)
    }
}

/// Resolve the portable CUDA policy once while constructing each norm layer.
/// B200/SM100 and RTX/SM120 both support 16-byte accesses; no unmeasured GPU-name
/// heuristic or startup timing probe is needed. A measured device-specific policy
/// can replace this preference later without changing the operator's interface.
fn preferred_vector_bytes(weight: &Tensor) -> RmsNormVectorBytes {
    if !weight.device().is_cuda() || !matches!(weight.dtype(), DType::F32 | DType::BF16)
    {
        return RmsNormVectorBytes::Scalar;
    }
    let row_bytes = weight
        .elem_count()
        .checked_mul(weight.dtype().size_in_bytes());
    RmsNormVectorBytes::for_row_stride(row_bytes.unwrap_or(0))
}

/// Candle implementation retained for unsupported native contracts and A/B tests.
fn centered_rms_norm_reference(
    xs: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> CandleResult<Tensor> {
    let dtype = xs.dtype();
    let normalized = normalize_in_f32(xs, eps)?;
    let weight = (weight.to_dtype(DType::F32)? + 1.0)?;
    normalized.broadcast_mul(&weight)?.to_dtype(dtype)
}

/// RMSNorm whose output is modulated by a separately supplied learned gate.
#[derive(Debug, Clone)]
pub(crate) struct GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GatedRmsNorm {
    /// Builds the layer from an already-loaded weight shared with a CUDA runner.
    pub(crate) fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Exposes the loaded scale to precision-preservation tests.
    #[cfg(test)]
    pub(crate) fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Prefill readout stays FP32 until the required BF16 rounding inside norm.
    pub(crate) fn forward_prefill(
        &self,
        xs: &Tensor,
        gate: &Tensor,
        activation: &Activation,
        optimize: bool,
    ) -> CandleResult<Tensor> {
        if optimize
            && *activation == Activation::Silu
            && super::cuda::supports_prefill_norm(xs, gate, &self.weight, self.eps)
        {
            return super::cuda::prefill_norm_cuda(xs, gate, &self.weight, self.eps);
        }
        self.forward(&xs.to_dtype(gate.dtype())?, gate, activation)
    }

    /// Normalizes the input and modulates it with an activated gate.
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        gate: &Tensor,
        activation: &Activation,
    ) -> CandleResult<Tensor> {
        let dtype = xs.dtype();
        let normalized = normalize_in_f32(xs, self.eps)?
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        let gate = gate.to_dtype(DType::F32)?.apply(activation)?;
        normalized.broadcast_mul(&gate)?.to_dtype(dtype)
    }
}

/// Divides each vector by its root-mean-square magnitude using FP32 arithmetic.
fn normalize_in_f32(xs: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let xs = xs.to_dtype(DType::F32)?;
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    let inverse_rms = (variance + eps)?.sqrt()?.recip()?;
    xs.broadcast_mul(&inverse_rms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use std::collections::HashMap;

    /// Loading resolves the policy before any forward/capture, without timing GPU
    /// work. CPU/reference execution does not need a CUDA vector preference.
    #[test]
    fn vector_policy_is_resolved_during_layer_construction() {
        for device in [Device::Cpu, Device::new_cuda(0).unwrap()] {
            for size in [1, 2, 3, 4, 256, 2560, 5120] {
                let weight = Tensor::zeros(size, DType::F32, &device).unwrap();
                let vb = VarBuilder::from_tensors(
                    HashMap::from([("weight".to_owned(), weight)]),
                    DType::F32,
                    &device,
                );
                let layer = CenteredRmsNorm::new(size, 1e-6, true, vb).unwrap();
                let expected = if device.is_cuda() {
                    RmsNormVectorBytes::for_row_stride(size * 4)
                } else {
                    RmsNormVectorBytes::Scalar
                };
                assert_eq!(layer.preferred_vector_bytes, expected);
                assert_eq!(layer.clone().preferred_vector_bytes, expected);
            }
        }
    }
}
