//! HF-compatible tokenizer metadata and template-file discovery.

use super::{
    SelectionPolicy, TemplateCatalog, TemplateEntry, exists, read, template_error,
};
use crate::ChatTemplateError;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::path::Path;

/// Reads optional tokenizer metadata independently of tensor loading.
pub(super) fn config(root: &Path) -> Result<Value, ChatTemplateError> {
    let path = root.join("tokenizer_config.json");
    if !exists(&path)? {
        return Ok(Value::Object(Map::new()));
    }
    serde_json::from_str(&read(&path)?)
        .map_err(|e| ChatTemplateError::Json(path, e.to_string()))
}

/// Preserves the token strings supplied by HF tokenizer metadata.
pub(super) fn special_tokens(config: &Value) -> Map<String, Value> {
    let mut tokens = Map::new();
    for key in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(value) = config.get(key)
            && let Some(text) = value
                .as_str()
                .or_else(|| value.get("content").and_then(Value::as_str))
        {
            tokens.insert(key.to_owned(), Value::from(text));
        }
    }
    tokens
}

/// Independent files replace embedded metadata as one catalog. The standalone
/// default is inserted before extra files.
pub(super) fn catalog(
    root: &Path,
    config: &Value,
) -> Result<Option<TemplateCatalog>, ChatTemplateError> {
    let mut entries = Vec::new();
    let standalone = root.join("chat_template.jinja");
    if exists(&standalone)? {
        entries.push(entry("default", read(&standalone)?, &standalone));
    }
    let directory = root.join("additional_chat_templates");
    if exists(&directory)?
        && std::fs::metadata(&directory)
            .map_err(|e| ChatTemplateError::Io(directory.clone(), e))?
            .is_dir()
    {
        let files = std::fs::read_dir(&directory)
            .map_err(|e| ChatTemplateError::Io(directory.clone(), e))?;
        // Preserve discovery order: the first entry is the implicit selection.
        // Sorting could change which prompt template is used.
        for file in files {
            let path = file
                .map_err(|e| ChatTemplateError::Io(directory.clone(), e))?
                .path();
            if path.extension().and_then(|v| v.to_str()) != Some("jinja") {
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|v| v.to_str())
                .ok_or_else(|| template_error("non-UTF-8 template filename"))?;
            let next = entry(name, read(&path)?, &path);
            if let Some(old) = entries.iter_mut().find(|e| e.name == name) {
                // An extra default replaces the earlier value without moving
                // its position, just like Python dicts.
                *old = next;
            } else {
                entries.push(next);
            }
        }
    }
    // Collection fallback behavior matches SGLang: use the first entry unless
    // a name is requested, even when a later entry is named default.
    let policy = if entries.is_empty() {
        return embedded(root, config);
    } else if entries.len() == 1 && entries[0].name == "default" {
        SelectionPolicy::Single
    } else {
        SelectionPolicy::FirstAvailable
    };
    Ok(Some(TemplateCatalog {
        entries,
        convention: "hf_tokenizer",
        policy,
    }))
}

/// Normalizes both embedded shapes without reordering keys. Duplicate list
/// names fail explicitly instead of silently replacing an earlier template.
fn embedded(
    root: &Path,
    config: &Value,
) -> Result<Option<TemplateCatalog>, ChatTemplateError> {
    let path = root.join("tokenizer_config.json");
    let (entries, policy) = match config.get("chat_template") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(source)) => (
            vec![entry("default", source.clone(), &path)],
            SelectionPolicy::Single,
        ),
        Some(Value::Object(templates)) => {
            let mut entries = Vec::new();
            for (name, source) in templates {
                entries.push(entry(name, template_text(source)?, &path));
            }
            (entries, SelectionPolicy::FirstAvailable)
        }
        Some(Value::Array(templates)) => {
            let mut entries = Vec::new();
            let mut names = HashSet::new();
            for template in templates {
                let name =
                    template
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            template_error("named entry requires a string name")
                        })?;
                if !names.insert(name) {
                    return Err(template_error(format!(
                        "duplicate template name {name:?}"
                    )));
                }
                let source = template.get("template").ok_or_else(|| {
                    template_error("named entry requires template text")
                })?;
                entries.push(entry(name, template_text(source)?, &path));
            }
            (entries, SelectionPolicy::FirstAvailable)
        }
        _ => return Err(template_error("unsupported chat_template metadata shape")),
    };
    Ok(Some(TemplateCatalog {
        entries,
        convention: "hf_tokenizer",
        policy,
    }))
}

/// Rejects malformed source values before a renderer can stringify them silently.
fn template_text(value: &Value) -> Result<String, ChatTemplateError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| template_error("template source must be a string"))
}

/// Carries source provenance into a format-neutral catalog entry.
fn entry(name: &str, source: String, path: &Path) -> TemplateEntry {
    TemplateEntry {
        name: name.to_owned(),
        source,
        origin: path.display().to_string(),
    }
}
