//! Chat-local failures, independent of model loading and execution backends.

use std::{error::Error, fmt, path::PathBuf};

/// Errors retain source paths and separate startup failures from request errors.
#[derive(Debug)]
pub enum ChatTemplateError {
    Io(PathBuf, std::io::Error),
    Json(PathBuf, String),
    InvalidConfig(String),
    InvalidOptions(String),
    Render(String),
}

impl fmt::Display for ChatTemplateError {
    /// Gives standalone callers useful messages without model/backend terminology.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, source) => {
                write!(f, "failed to read {}: {source}", path.display())
            }
            Self::Json(path, message) => {
                write!(f, "failed to parse {}: {message}", path.display())
            }
            Self::InvalidConfig(message) | Self::Render(message) => {
                write!(f, "chat template: {message}")
            }
            Self::InvalidOptions(message) => {
                write!(f, "chat template controls: {message}")
            }
        }
    }
}

impl Error for ChatTemplateError {
    /// Preserves the underlying I/O failure for diagnostics and error chains.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(_, source) => Some(source),
            _ => None,
        }
    }
}
