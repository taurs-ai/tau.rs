mod contracts;
mod correctness;
mod graph;
mod support;

use super::super::{fused_residual_rms_norm_cuda, supports_fused_residual_rms_norm};
use crate::components::rms_norm::CenteredRmsNorm;
use candle::{DType, Device, Module, Tensor};
use half::bf16;
use support::*;
use tau_cuda_kernels::RmsNormVectorBytes;
