//! Runtime floating-point dtypes supported by model-backed inference.

use candle::DType;
use std::fmt;

/// Primary floating-point dtype for bulk model weights and activations.
///
/// Model components may retain precision-sensitive parameters and state in FP32
/// during BF16 execution. Checkpoint storage dtypes are validated separately.
///
/// tau.rs deliberately exposes only dtypes that have an end-to-end model
/// validation path. Add another variant only when model loading, inference tensor
/// creation, numerical tolerances, and the serving interface support it together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDType {
    /// 32-bit floating point, used for correctness references and compatibility.
    F32,
    /// Brain floating point, used by supported modern model artifacts.
    BF16,
}

impl ModelDType {
    /// Returns the stable command-line and diagnostic spelling of the dtype.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::BF16 => "bf16",
        }
    }

    /// Converts the validated model dtype into Candle's runtime dtype.
    pub(crate) const fn candle_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::BF16 => DType::BF16,
        }
    }
}

impl fmt::Display for ModelDType {
    /// Formats the dtype using its stable command-line spelling.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Derives tau.rs's supported runtime dtypes from Hugging Face metadata.
///
/// FP32 remains available as the correctness and compatibility path. BF16 is
/// enabled only when the resolved `torch_dtype` or `dtype` metadata declares BF16.
/// This does not require uniform BF16 storage: family-specific tensor contracts
/// determine which weights may be stored in FP32. Unknown or currently unsupported
/// declarations, such as FP16, retain only F32 in this generic metadata helper.
pub(crate) fn supported_dtypes_from_torch_dtype(
    torch_dtype: Option<&str>,
) -> Vec<ModelDType> {
    let mut supported = vec![ModelDType::F32];
    if matches!(torch_dtype, Some("bfloat16" | "bf16")) {
        supported.push(ModelDType::BF16);
    }
    supported
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that BF16 checkpoints retain both supported execution dtypes.
    #[test]
    fn bf16_model_artifacts_support_f32_and_bf16() {
        assert_eq!(
            supported_dtypes_from_torch_dtype(Some("bfloat16")),
            vec![ModelDType::F32, ModelDType::BF16]
        );
    }

    /// Verifies that absent or unsupported declarations fall back to F32 support.
    #[test]
    fn missing_or_unsupported_declarations_retain_f32() {
        assert_eq!(
            supported_dtypes_from_torch_dtype(None),
            vec![ModelDType::F32]
        );
        assert_eq!(
            supported_dtypes_from_torch_dtype(Some("float16")),
            vec![ModelDType::F32]
        );
    }
}
