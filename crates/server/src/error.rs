//! Server error type and conversions.

use std::error::Error;
use std::fmt;
use tau_engine::InferenceError;

#[derive(Debug)]
pub enum ServerError {
    Io(std::io::Error),
    Backend(InferenceError),
    BadRequest(String),
}

impl fmt::Display for ServerError {
    /// Formats server errors for CLI and log output.
    ///
    /// The formatter receives text describing I/O, backend, or bad-request
    /// failures. The return value reports formatting success or failure from the
    /// formatter.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => write!(f, "io error: {source}"),
            Self::Backend(source) => write!(f, "{source}"),
            Self::BadRequest(message) => write!(f, "bad request: {message}"),
        }
    }
}

impl Error for ServerError {
    /// Returns the wrapped lower-level error when one exists.
    ///
    /// The method exposes I/O and backend errors as sources and returns `None` for
    /// plain bad-request messages. It cannot fail.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Backend(source) => Some(source),
            Self::BadRequest(_) => None,
        }
    }
}

impl From<std::io::Error> for ServerError {
    /// Converts a standard I/O error into the server error type.
    ///
    /// The input error is preserved as the source. The function always returns
    /// `ServerError::Io` and cannot fail.
    fn from(source: std::io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<InferenceError> for ServerError {
    /// Converts a backend inference error into the server error type.
    ///
    /// The input error is preserved as the source. The function always returns
    /// `ServerError::Backend` and cannot fail.
    fn from(source: InferenceError) -> Self {
        Self::Backend(source)
    }
}
