//! Hugging Face tokenizer loading and validation.

use crate::ModelLoadError;
use crate::metadata::require_file;
use std::path::Path;
use tokenizers::Tokenizer;

const TOKENIZER_PROBE_TEXT: &str = "hello tau";

/// Loads `tokenizer.json` from a model directory.
///
/// The input is the model root path. The function returns a Hugging Face
/// tokenizer instance or `ModelLoadError` if the tokenizer file is missing or
/// invalid.
pub fn load_tokenizer(root: &Path) -> Result<Tokenizer, ModelLoadError> {
    let path = root.join("tokenizer.json");
    require_file(&path)?;
    Tokenizer::from_file(&path)
        .map_err(|source| ModelLoadError::Tokenizer(path, source.to_string()))
}

/// Validates basic tokenizer behavior and returns its vocabulary size.
///
/// The input tokenizer must support encode and decode. The function returns the
/// vocabulary size including added tokens or `ModelLoadError` if the tokenizer
/// does not round-trip a small probe string.
pub fn validate_tokenizer(tokenizer: &Tokenizer) -> Result<usize, ModelLoadError> {
    let encoding = tokenizer
        .encode(TOKENIZER_PROBE_TEXT, false)
        .map_err(|source| {
            ModelLoadError::InvalidConfig(format!("tokenizer encode failed: {source}"))
        })?;
    if encoding.get_ids().is_empty() {
        return Err(ModelLoadError::InvalidConfig(
            "tokenizer produced no tokens for probe text".to_owned(),
        ));
    }

    let decoded = tokenizer
        .decode(encoding.get_ids(), false)
        .map_err(|source| {
            ModelLoadError::InvalidConfig(format!("tokenizer decode failed: {source}"))
        })?;
    if decoded != TOKENIZER_PROBE_TEXT {
        return Err(ModelLoadError::InvalidConfig(format!(
            "tokenizer decoded probe as {decoded:?}"
        )));
    }

    Ok(tokenizer.get_vocab_size(true))
}
