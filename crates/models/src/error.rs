//! Errors produced while validating and constructing model runtimes.

use super::dtype::ModelDType;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum ModelLoadError {
    InvalidModelDir(PathBuf),
    MissingFile(PathBuf),
    Io(PathBuf, std::io::Error),
    Json(PathBuf, String),
    InvalidConfig(String),
    UnsupportedDType {
        architecture: String,
        requested: ModelDType,
        supported: Vec<ModelDType>,
    },
    UnsupportedDTypeOnDevice {
        requested: ModelDType,
        device: String,
    },
    Tokenizer(PathBuf, String),
    Safetensors(PathBuf, ::safetensors::SafeTensorError),
    Candle(String),
}

impl fmt::Display for ModelLoadError {
    /// Formats model loading errors for CLI output and logs.
    ///
    /// The formatter receives details about missing files, malformed JSON,
    /// tokenizer failures, or safetensors failures. The return value only reports
    /// formatter success or failure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelDir(path) => {
                write!(f, "model path is not a directory: {}", path.display())
            }
            Self::MissingFile(path) => {
                write!(f, "model file is missing: {}", path.display())
            }
            Self::Io(path, source) => {
                write!(f, "failed to read {}: {source}", path.display())
            }
            Self::Json(path, message) => {
                write!(f, "failed to parse {}: {message}", path.display())
            }
            Self::InvalidConfig(message) => {
                write!(f, "invalid model config: {message}")
            }
            Self::UnsupportedDType {
                architecture,
                requested,
                supported,
            } => {
                let supported = supported
                    .iter()
                    .map(|dtype| dtype.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "model architecture {architecture} does not support dtype \
                     {requested}; supported dtypes: {supported}"
                )
            }
            Self::UnsupportedDTypeOnDevice { requested, device } => write!(
                f,
                "dtype {requested} is not supported on the Candle {device} backend; \
                 select a compatible device or use f32"
            ),
            Self::Tokenizer(path, message) => {
                write!(f, "failed to load tokenizer {}: {message}", path.display())
            }
            Self::Safetensors(path, source) => {
                write!(f, "failed to load safetensors {}: {source}", path.display())
            }
            Self::Candle(message) => {
                write!(f, "failed to initialize Candle model runtime: {message}")
            }
        }
    }
}

impl Error for ModelLoadError {
    /// Returns the wrapped lower-level error when one exists.
    ///
    /// The method exposes I/O and safetensors sources and returns `None` for
    /// string-only validation errors. It cannot fail.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(_, source) => Some(source),
            Self::Safetensors(_, source) => Some(source),
            _ => None,
        }
    }
}
