//! Stateless per-channel causal filtering for GDN prefill/fallback.
//!
//! The calculation depends only on explicit tensor and policy arguments. It
//! returns activated output and next raw history; the layer owns state publication.
//! All inputs are read-only, including during graph warm-up.

use crate::components::cuda;
use candle::{Result as CandleResult, Tensor};
use candle_nn::Activation;

/// Gives each projected Q/K/V feature local token context before GDN recurrence.
///
/// Input is [batch, channels, tokens], where channels pack the Q/K/V features.
/// Each channel has its own learned filter: it combines values along the token
/// axis without mixing channels. The filter is causal, so an output uses only
/// the current token and earlier tokens, never future tokens.
///
/// For the production four-tap filter, each output is conceptually:
/// SiLU(w0 * x[t-3] + w1 * x[t-2] + w2 * x[t-1] + w3 * x[t]).
/// BF16 execution rounds the weighted sum before applying SiLU. These filtered
/// Q/K/V values then feed the recurrent update, which maintains longer-term
/// memory. The convolution's short history is separate from that memory.
///
/// `weight` has shape [channels, 1, K], with one filter per input channel.
/// `previous` supplies K raw values per channel, or `None` starts from zeros.
/// Returns (activated output, next raw history) without changing any arguments
/// or retaining state. The caller publishes history only after success.
pub(super) fn forward(
    xs: &Tensor,
    weight: &Tensor,
    previous: Option<&Tensor>,
    activation: Activation,
    optimize_kernel: bool,
) -> CandleResult<(Tensor, Tensor)> {
    let (batch_size, input_channels, sequence_length) = xs.dims3()?;
    let (channels, channels_per_filter, kernel_size) = weight.dims3()?;
    // The weights define the channel count and window. Validate this depthwise
    // contract before choosing either path; no layer configuration is needed.
    if channels == 0
        || kernel_size == 0
        || channels_per_filter != 1
        || input_channels != channels
        || xs.dtype() != weight.dtype()
        || !xs.device().same_device(weight.device())
    {
        candle::bail!(
            "causal convolution requires matching input/weight dtype and device, \
             with input [B,C,T] and nonempty weights [C,1,K]"
        )
    }
    if previous.is_some_and(|history| {
        history.dims() != [batch_size, channels, kernel_size]
            || history.dtype() != xs.dtype()
            || !history.device().same_device(xs.device())
    }) {
        candle::bail!(
            "causal convolution history must have shape [B,C,K] and match \
             the input dtype and device"
        )
    }
    // Continue the previous chunk's raw projected values, or use zero padding
    // at the start of a request. Keep K slots to match the decode state format;
    // the first new output uses the last K-1 history values plus input token 0.
    let previous = match previous {
        Some(state) => state.clone(),
        None => {
            Tensor::zeros((batch_size, channels, kernel_size), xs.dtype(), xs.device())?
        }
    };
    // The native kernel computes every channel/token window in one launch.
    // It reads previous history and this chunk directly, avoiding a full
    // concatenation and separate convolution launches for individual channels.
    let (output, next) = if optimize_kernel
        && cuda::supports_prefill_convolution(xs, &previous, weight)
    {
        let convolved = cuda::prefill_convolution_cuda(xs, &previous, weight)?;
        // Copy only the next raw window into independent compact storage.
        // History stays read-only during convolution, including graph warm-up.
        let next = if sequence_length >= kernel_size {
            // This chunk is long enough to replace every history slot.
            xs.narrow(2, sequence_length - kernel_size, kernel_size)?
                .force_contiguous()?
        } else {
            // A short chunk replaces only part of the window. For K=4, T=2,
            // the next history is [previous[2], previous[3], input[0], input[1]].
            let suffix =
                previous.narrow(2, sequence_length, kernel_size - sequence_length)?;
            Tensor::cat(&[&suffix, xs], 2)?.contiguous()?
        };
        (convolved.apply(&activation)?, next)
    } else {
        forward_reference(xs, weight, &previous, activation)?
    };
    // Return raw projections as history. The caller decides when to publish it;
    // neither ordinary calls nor graph warm-up can mutate the incoming history.
    Ok((output, next))
}

/// Original concatenation/grouped-convolution expression, retained as fallback.
fn forward_reference(
    xs: &Tensor,
    weight: &Tensor,
    previous: &Tensor,
    activation: Activation,
) -> CandleResult<(Tensor, Tensor)> {
    let sequence_length = xs.dim(2)?;
    let (channels, _, kernel_size) = weight.dims3()?;
    // Materialize the same history-plus-input sequence that the native kernel
    // reads virtually. Its trailing K raw values are the next chunk's history.
    let extended = Tensor::cat(&[previous, xs], 2)?;
    let extended_length = extended.dim(2)?;
    let next = extended.narrow(2, extended_length - kernel_size, kernel_size)?;
    // Setting groups=channels makes each learned filter operate independently.
    // K history values plus T inputs produce T+1 valid convolution outputs.
    let convolved = extended.conv1d(weight, 0, 1, 1, channels)?;
    // Output 0 uses only old history, so discard it. Outputs 1..=T correspond
    // exactly to this chunk's tokens; apply activation after convolution rounds.
    let output = convolved
        .narrow(2, 1, sequence_length)?
        .apply(&activation)?;
    Ok((output, next))
}
