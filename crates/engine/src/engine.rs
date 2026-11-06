//! Inference engine behavior shared by servers, CLIs, and future runtime callers.

use crate::InferenceError;

pub const LOOPBACK_MODEL_ID: &str = "tau-loopback";

pub trait InferenceEngine {
    /// Returns the public model id exposed through model-level APIs.
    ///
    /// The method borrows engine state and returns a stable model identifier. It has no
    /// error channel because model selection is part of engine construction.
    fn model_id(&self) -> &str;

    /// Returns the execution backend name shown by diagnostics.
    ///
    /// Model-backed engines return their storage/runtime family, and lightweight
    /// engines return their own execution name. The method borrows state and cannot
    /// fail.
    fn backend_name(&self) -> &str;

    /// Returns the execution backend version shown by diagnostics.
    ///
    /// Engines return the version of the code path serving requests. The method
    /// borrows state and cannot fail.
    fn backend_version(&self) -> &str;

    /// Runs one flat `f32` inference request through the engine.
    ///
    /// Inputs are immutable values and mutable output storage. The default
    /// implementation returns `InferenceError::Unsupported` so engines only implement
    /// this method when they support flat `f32` inference.
    fn infer_f32(
        &mut self,
        _input: &[f32],
        _output: &mut [f32],
    ) -> Result<(), InferenceError> {
        Err(InferenceError::Unsupported(
            "f32 inference is not supported by this engine".to_owned(),
        ))
    }

    /// Runs one plain text inference request through the engine.
    ///
    /// The input is model-level text that does not require the caller to manage
    /// tokenization. The default implementation returns `InferenceError::Unsupported`
    /// so engines only implement this method when they support text inference.
    fn infer_text(&mut self, _input: &str) -> Result<String, InferenceError> {
        Err(InferenceError::Unsupported(
            "text inference is not supported by this engine".to_owned(),
        ))
    }
}

/// A generic loopback inference engine which simply "talks back"
pub struct LoopbackEngine;

impl LoopbackEngine {
    /// Creates the built-in loopback inference engine for testing / development.
    ///
    /// The function takes no input and returns an engine that performs inference
    /// without loading a model. It cannot fail.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoopbackEngine {
    /// Creates the default loopback inference engine.
    ///
    /// The function takes no input and returns `LoopbackEngine::new()`. It cannot fail.
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngine for LoopbackEngine {
    /// Returns the public model id for the built-in loopback model.
    ///
    /// The method takes an immutable engine reference and returns `tau-loopback`. It
    /// cannot fail.
    fn model_id(&self) -> &str {
        LOOPBACK_MODEL_ID
    }

    /// Returns the execution name for loopback diagnostics.
    ///
    /// Loopback reports its model id as the execution backend name.
    /// The method cannot fail.
    fn backend_name(&self) -> &str {
        LOOPBACK_MODEL_ID
    }

    /// Returns the core crate version for loopback diagnostics.
    ///
    /// Loopback reports the compiled core package version. The method cannot fail.
    fn backend_version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    /// Copies flat `f32` input values into output storage.
    ///
    /// The input slice is echoed into the output slice. The function returns
    /// `InferenceError::Backend` when output storage is smaller than input.
    fn infer_f32(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError> {
        if output.len() < input.len() {
            Err(InferenceError::Backend(
                "output tensor is smaller than input tensor".to_owned(),
            ))
        } else {
            output[..input.len()].copy_from_slice(input);
            Ok(())
        }
    }

    /// Echoes plain text input as the text inference result.
    ///
    /// The input is copied into an owned output string. The function cannot fail for
    /// valid Rust string input.
    fn infer_text(&mut self, input: &str) -> Result<String, InferenceError> {
        Ok(input.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TextOnlyEngine;

    impl InferenceEngine for TextOnlyEngine {
        /// Returns the test model identifier.
        fn model_id(&self) -> &str {
            "text-only"
        }

        /// Returns the test backend identifier.
        fn backend_name(&self) -> &str {
            "text-only"
        }

        /// Returns the fixed backend version used by the test.
        fn backend_version(&self) -> &str {
            "test"
        }
    }

    /// Verifies that engines can rely on the default unsupported `f32` inference
    /// implementation.
    ///
    /// The test uses a minimal engine that does not override `infer_f32` and expects
    /// `InferenceError::Unsupported` instead of requiring every engine to define the
    /// method.
    #[test]
    fn default_infer_f32_rejects_unsupported() {
        let mut engine = TextOnlyEngine;
        let mut output = [0.0; 1];

        let error = engine.infer_f32(&[1.0], &mut output).unwrap_err();

        assert_eq!(
            error,
            InferenceError::Unsupported(
                "f32 inference is not supported by this engine".to_owned(),
            )
        );
    }

    /// Verifies that loopback mode copies input values into output storage.
    ///
    /// The test creates a loopback engine, supplies flat `f32` input, and expects
    /// identical output. It fails by panicking if loopback execution returns an error
    /// or changes the values.
    #[test]
    fn loopback_engine_copies_input() {
        let mut engine = LoopbackEngine::new();
        let mut output = [0.0; 3];

        engine.infer_f32(&[1.0, 2.5, -3.0], &mut output).unwrap();

        assert_eq!(output, [1.0, 2.5, -3.0]);
    }

    /// Verifies that loopback mode reports shape errors for undersized output.
    ///
    /// The test creates a loopback engine and supplies output storage shorter than the
    /// input. It expects `InferenceError::Backend` instead of a partial write or
    /// panic.
    #[test]
    fn loopback_engine_rejects_small_output() {
        let mut engine = LoopbackEngine::new();
        let mut output = [0.0; 1];

        let error = engine.infer_f32(&[1.0, 2.0], &mut output).unwrap_err();

        assert!(matches!(error, InferenceError::Backend(_)));
    }

    /// Verifies that loopback text inference echoes input text.
    ///
    /// The test supplies plain text input and expects the same text as output. It fails
    /// by panicking if loopback text inference errors or changes the content.
    #[test]
    fn loopback_engine_echoes_text() {
        let mut engine = LoopbackEngine::new();

        let output = engine.infer_text("hello tau").unwrap();

        assert_eq!(output, "hello tau");
    }
}
