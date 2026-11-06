//! Runtime text decoding, logit readback and greedy sampling.

mod greedy;
pub(crate) use greedy::GpuGreedy;

use crate::InferenceError;
use candle::{DType, IndexOp, Tensor as CandleTensor};
#[cfg(test)]
pub(crate) use tau_models::DEFAULT_MAX_NEW_TOKENS;
pub(crate) use tau_models::GenerationConfig;
use tokenizers::Tokenizer;

/// Decodes generated token ids into assistant text.
///
/// The input ids are completion tokens only, not the prompt. The function returns
/// decoded UTF-8 text with special tokens skipped, or `InferenceError` when the
/// tokenizer cannot decode the ids.
pub(crate) fn decode_generated_tokens(
    tokenizer: &Tokenizer,
    tokens: &[u32],
) -> Result<String, InferenceError> {
    if tokens.is_empty() {
        return Ok(String::new());
    }

    tokenizer
        .decode(tokens, true)
        .map(|text| text.trim_end().to_owned())
        .map_err(|source| {
            InferenceError::Backend(format!("tokenizer decode failed: {source}"))
        })
}

/// Copies final-token logits to the host for greedy selection and optional tracing.
///
/// The input tensor has shape `[1, 1, vocab]` from the LM head. Conversion and
/// readback errors remain visible to the shared generation runtime.
/// These are complete forward outputs, not intermediate activations. A future
/// captured path must keep this host readback outside the captured sequence.
pub(crate) fn logits_values(logits: &CandleTensor) -> Result<Vec<f32>, InferenceError> {
    logits
        .i((0, 0))
        .and_then(|logits| logits.to_dtype(DType::F32))
        .and_then(|logits| logits.to_vec1::<f32>())
        .map_err(candle_inference_error)
}

/// Runs the same view/cast/readback as ordinary generation with explicit trace
/// boundaries. The blocking host copy may wait for forward and conversion; its
/// wall time must not be added to the overlapping CUDA event durations.
pub(crate) fn logits_values_profiled(
    logits: &CandleTensor,
    timing: &mut tau_metrics::timing::DecodeTimer,
) -> Result<Vec<f32>, InferenceError> {
    timing.begin_conversion().map_err(candle_inference_error)?;
    let converted = logits
        .i((0, 0))
        .and_then(|values| values.to_dtype(DType::F32))
        .map_err(candle_inference_error)?;
    timing.begin_readback().map_err(candle_inference_error)?;
    let values = converted.to_vec1::<f32>().map_err(candle_inference_error)?;
    // Charge temporary-buffer release to readback, keeping the following host
    // interval focused on greedy selection and the EOS check.
    drop(converted);
    timing.end_readback(values.len());
    Ok(values)
}

/// Selects a token by deterministic greedy argmax over host logits.
///
/// Each slice index is a vocabulary token ID. The CPU scan ignores NaNs and
/// selects the greatest remaining logit using `f32::total_cmp`: infinities are
/// valid candidates, and positive zero ranks above negative zero. If multiple
/// logits compare equal, the last (highest-index) token wins. Selection uses the
/// raw logits directly, without softmax or random sampling.
///
/// Returns `InferenceError::Backend` if the slice is empty, all logits are NaN,
/// or the selected index cannot fit in `u32`. EOS tokens participate normally;
/// the caller checks whether the selected token ends generation. The slice is
/// borrowed so tracing can inspect the same values without another GPU readback.
pub(crate) fn next_token_from_values(values: &[f32]) -> Result<u32, InferenceError> {
    let index = values
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index)
        .ok_or_else(|| {
            InferenceError::Backend("model logits are all NaN".to_owned())
        })?;

    u32::try_from(index).map_err(|_| {
        InferenceError::Backend(format!("token index {index} does not fit in u32"))
    })
}

/// Converts a Candle runtime error into an inference error.
///
/// The input is a Candle error from tensor creation, forward execution, or tensor
/// reads. The function returns `InferenceError::Backend` with a human-readable
/// message and cannot fail.
pub(crate) fn candle_inference_error(source: candle::Error) -> InferenceError {
    InferenceError::Backend(format!("Candle model runtime failed: {source}"))
}
