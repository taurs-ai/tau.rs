//! CUDA RMSNorm operator entry points with stable component-facing exports.
//!
//! Each operator keeps contract checks, pointer guards, allocation and native
//! launch together. Components own reference fallback; launch failures propagate.

mod centered;
mod residual;

pub(in crate::components) use centered::{
    centered_rms_norm_cuda, supports_centered_rms_norm,
};
pub(in crate::components) use residual::{
    fused_residual_rms_norm_cuda, supports_fused_residual_rms_norm,
};

#[cfg(test)]
mod fused_bench;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
