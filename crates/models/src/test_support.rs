//! Timing/readback scaffolding for isolated model-state and graph tests.

use candle::{DType, IndexOp, Tensor};
use tau_metrics::timing::DecodeTimer;

/// Reads model logits with explicit measurement boundaries, without importing
/// engine sampling policy. Engine tests exercise the actual generation helper.
pub(crate) fn timed_logits(output: &Tensor, timer: &mut DecodeTimer) -> Vec<f32> {
    timer.begin_conversion().unwrap();
    let converted = output.i((0, 0)).unwrap().to_dtype(DType::F32).unwrap();
    timer.begin_readback().unwrap();
    let values = converted.to_vec1::<f32>().unwrap();
    drop(converted);
    timer.end_readback(values.len());
    values
}

pub(crate) mod tensor_capture;
