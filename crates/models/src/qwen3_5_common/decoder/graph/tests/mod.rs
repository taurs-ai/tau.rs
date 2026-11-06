//! Whole-layer graph dispatch, state lifetime and prefill transitions.

mod dispatch;
mod lifecycle;
mod prefill;

use super::*;
use crate::components::{graph_node_types, graph_tensor_address};
use crate::qwen3_5_common::decoder::test_support::*;
use candle::{DType, Device};
