//! Inference engine APIs, runtime composition and generation for tau.rs.

pub mod chat;
pub mod engine;
mod instrumentation;
pub use tau_models as model;
mod generation;
pub mod model_engine;
pub mod qwen2_engine;
pub mod qwen3_5_engine;
pub mod qwen3_5_moe_engine;
pub mod qwen3_engine;

pub mod safetensor_engine {
    pub use crate::qwen2_engine::*;
}

pub mod safetensors_engine {
    pub use crate::qwen2_engine::*;
}

pub use candle::Device;
pub use chat::{
    ChatCompletion, ChatEngine, ChatFinishReason, ChatInferenceEngine, ChatMessage,
    ChatOptions, ChatTemplateOptions, ChatTemplateSelection,
};
pub use engine::{InferenceEngine, LOOPBACK_MODEL_ID, LoopbackEngine};
pub use model::{
    AttentionBlockSize, CudaGraphMode, ModelDType, ModelLoadError, ModelLoadOptions,
    ModelShard, MoeDecodeConfig, MoeDecodeOptions, MoeDecodeProfile,
    MoeDecodeResolution, MoeDecodeSelection, MoePrefillConfig, MoePrefillTile,
    TensorInfo,
};
pub use model_engine::{
    EngineOptions, load_model_engine, load_model_engine_with_chat_template,
    load_model_engine_with_chat_template_selection, load_model_engine_with_dtype,
    load_model_engine_with_options, load_model_engine_with_runtime_options,
};
pub use qwen2_engine::Qwen2SafetensorsEngine;
pub use qwen3_5_engine::Qwen3_5SafetensorsEngine;
pub use qwen3_5_moe_engine::Qwen3_5MoeSafetensorsEngine;
pub use qwen3_engine::Qwen3SafetensorsEngine;

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Builds a tensor from an explicit shape and flat `f32` data buffer.
    ///
    /// The shape dimensions are multiplied to compute the expected element count. The
    /// function returns `ShapeMismatch` when the input data length does not match
    /// that shape.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, InferenceError> {
        let expected_len = shape.iter().product::<usize>();
        if expected_len != data.len() {
            return Err(InferenceError::ShapeMismatch {
                expected_len,
                actual_len: data.len(),
            });
        }

        Ok(Self { shape, data })
    }

    /// Builds a one-dimensional tensor from a flat vector.
    ///
    /// The input vector becomes the tensor data, and the output shape is
    /// `[data.len()]`. This constructor cannot fail because the shape is derived
    /// from the provided data.
    pub fn vector(data: Vec<f32>) -> Self {
        Self {
            shape: vec![data.len()],
            data,
        }
    }

    /// Returns the tensor shape as a borrowed slice of dimensions.
    ///
    /// The function takes an immutable tensor reference and returns a view into the
    /// tensor. It performs no allocation and has no error cases.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor data as an immutable flat `f32` slice.
    ///
    /// The output borrows from the tensor and is valid for the caller's borrow. The
    /// function has no error handling because it only exposes already-owned data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the tensor data as a mutable flat `f32` slice.
    ///
    /// The caller supplies a mutable tensor borrow and receives mutable access to the
    /// backing data. The function does not validate values or return errors.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    Backend(String),
    ShapeMismatch {
        expected_len: usize,
        actual_len: usize,
    },
    Unsupported(String),
}

impl fmt::Display for InferenceError {
    /// Formats an inference error for logs, CLI output, and propagated error messages.
    ///
    /// The formatter is the output sink, and the return value reports formatting
    /// success or failure from that sink. No inference logic is executed here.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(message) => write!(f, "backend error: {message}"),
            Self::ShapeMismatch {
                expected_len,
                actual_len,
            } => write!(
                f,
                "tensor shape expects {expected_len} values but got {actual_len}"
            ),
            Self::Unsupported(message) => {
                write!(f, "unsupported inference operation: {message}")
            }
        }
    }
}

impl Error for InferenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that tensor construction rejects data whose length does not match the
    /// shape.
    ///
    /// The test constructs an invalid 2x2 tensor with three values and expects a
    /// `ShapeMismatch` output. It fails by panicking if validation accepts the bad
    /// input or reports the wrong error.
    #[test]
    fn validates_tensor_shape() {
        let err = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(
            err,
            InferenceError::ShapeMismatch {
                expected_len: 4,
                actual_len: 3
            }
        );
    }
}
