//! Chat values, template selection and prompt rendering for tau.rs.
//!
//! This crate formats conversations without loading models, tokenizing text or
//! executing inference. Engine traits and adapters remain in tau-engine.

mod error;
mod template_options;
mod templates;
mod types;

pub use error::ChatTemplateError;
pub use template_options::ChatTemplateOptions;
pub use templates::{ChatTemplateSelection, CheckpointChatTemplate};
pub use types::{ChatCompletion, ChatFinishReason, ChatMessage, ChatOptions};
