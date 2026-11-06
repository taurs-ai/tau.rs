//! Candle bridges for Gated DeltaNet prefill and single-token decode.
//!
//! Prefill operators are stateless; single-token decode owns retained workspace.
//! The private modules keep each operator's validation and launch logic together,
//! while these re-exports preserve the component-facing API.

mod decode;
mod norm;
#[cfg(test)]
mod norm_tests;
pub(in crate::components) use norm::{prefill_norm_cuda, supports_prefill_norm};
mod prefill;
mod preparation;
#[cfg(test)]
mod preparation_tests;
mod recurrence;
pub(in crate::components) use preparation::{
    mixed_prefill_cuda, supports_mixed_prefill,
};
pub(in crate::components) use recurrence::{
    prefill_recurrence_cuda, supports_prefill_recurrence,
};

pub(in crate::components) use decode::GatedDeltaNetRunner;
pub(in crate::components) use prefill::{
    prefill_convolution_cuda, supports_prefill_convolution,
};

/// Converts a cudarc driver error into Candle's CUDA error representation.
/// Keeps the underlying diagnostic text so allocation/copy/launch failures can
/// propagate through ordinary Candle `Result` values to the model caller.
fn cuda_driver_error(error: impl std::fmt::Display) -> candle::Error {
    candle::Error::Cuda(Box::<dyn std::error::Error + Send + Sync>::from(
        error.to_string(),
    ))
}
