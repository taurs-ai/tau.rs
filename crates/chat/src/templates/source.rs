//! Ordered catalogs and startup selection, independent of checkpoint packaging.

mod hugging_face;

use crate::ChatTemplateError;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// Construction-only source preferences. A file override wins over a name.
/// Request variables belong to ChatTemplateOptions, not this selection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatTemplateSelection {
    file: Option<PathBuf>,
    name: Option<String>,
}

impl ChatTemplateSelection {
    /// Uses the checkpoint's recognized source convention and default policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects an external Jinja file without changing the stored preferred name.
    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.file = Some(Into::<PathBuf>::into(path));
        self
    }

    /// Selects a checkpoint entry; source readers decide how unnamed sources behave.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(Into::<String>::into(name));
        self
    }

    /// Borrows the explicit file override, if requested.
    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    /// Borrows the preferred checkpoint name, independent of the tensor format.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Lets model-family dispatch reject preferences it cannot honor.
    pub fn is_explicit(&self) -> bool {
        self.file.is_some() || self.name.is_some()
    }
}

/// Source-specific policies stay explicit rather than becoming tensor-loader rules.
#[derive(Debug, Clone, Copy)]
pub(super) enum SelectionPolicy {
    Single,
    FirstAvailable,
}

impl SelectionPolicy {
    /// Stable labels make implicit selection auditable in serving provenance.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::FirstAvailable => "first_available",
        }
    }
}

/// One source entry; origin identifies metadata/file precedence without rendering.
pub(super) struct TemplateEntry {
    pub(super) name: String,
    pub(super) source: String,
    pub(super) origin: String,
}

/// Readers normalize packaging here; the selector never reads checkpoint files.
pub(super) struct TemplateCatalog {
    pub(super) entries: Vec<TemplateEntry>,
    pub(super) convention: &'static str,
    pub(super) policy: SelectionPolicy,
}

impl TemplateCatalog {
    /// Applies the reader's policy and reports errors before any prompt inference.
    /// A single source ignores names; collections apply their declared policy.
    pub(super) fn select(
        &self,
        preferred: Option<&str>,
    ) -> Result<&TemplateEntry, ChatTemplateError> {
        let first = self
            .entries
            .first()
            .ok_or_else(|| template_error("empty named template collection"))?;
        if matches!(self.policy, SelectionPolicy::FirstAvailable)
            && let Some(name) = preferred
        {
            return self.entries.iter().find(|e| e.name == name).ok_or_else(|| {
                template_error(format!(
                    "unknown template name {name:?}; available: {:?}",
                    self.entries
                        .iter()
                        .map(|e| e.name.as_str())
                        .collect::<Vec<_>>()
                ))
            });
        }
        Ok(first)
    }
}

/// Loads recognized metadata into the neutral catalog. Additional conventions
/// can add readers here without changing rendering, HTTP requests or inference.
pub(super) fn load(
    root: &Path,
    selection: &ChatTemplateSelection,
) -> Result<(Option<TemplateCatalog>, Map<String, Value>), ChatTemplateError> {
    let config = hugging_face::config(root)?;
    let tokens = hugging_face::special_tokens(&config);
    let catalog = if let Some(path) = selection.file() {
        if path.extension().and_then(|v| v.to_str()) != Some("jinja") {
            return Err(template_error("--chat-template accepts .jinja files"));
        }
        Some(TemplateCatalog {
            entries: vec![TemplateEntry {
                name: "default".to_owned(),
                source: read(path)?.trim_matches('\n').replace("\\n", "\n"),
                origin: path.display().to_string(),
            }],
            convention: "explicit_jinja",
            policy: SelectionPolicy::Single,
        })
    } else {
        hugging_face::catalog(root, &config)?
    };
    if catalog.is_none() && selection.name().is_some() {
        return Err(template_error(
            "requested a template name but no catalog exists",
        ));
    }
    Ok((catalog, tokens))
}

/// Retains asset paths in UTF-8 and filesystem errors.
fn read(path: &Path) -> Result<String, ChatTemplateError> {
    std::fs::read_to_string(path)
        .map_err(|e| ChatTemplateError::Io(path.to_path_buf(), e))
}

/// Distinguishes missing optional assets from permission or I/O failures.
fn exists(path: &Path) -> Result<bool, ChatTemplateError> {
    path.try_exists()
        .map_err(|e| ChatTemplateError::Io(path.to_path_buf(), e))
}

/// Gives source and renderer setup errors a common, readable prefix.
pub(super) fn template_error(message: impl std::fmt::Display) -> ChatTemplateError {
    ChatTemplateError::InvalidConfig(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same ordered catalog can use a reader's policy without guessing by name.
    #[test]
    fn selection_obeys_policy_order_and_explicit_names() {
        let mut catalog = TemplateCatalog {
            entries: ["alternate", "default"]
                .map(|name| TemplateEntry {
                    name: name.to_owned(),
                    source: name.to_owned(),
                    origin: "test".to_owned(),
                })
                .into_iter()
                .collect(),
            convention: "test",
            policy: SelectionPolicy::FirstAvailable,
        };
        assert_eq!(catalog.select(None).unwrap().name, "alternate");
        assert_eq!(catalog.select(Some("default")).unwrap().name, "default");
        assert!(catalog.select(Some("missing")).is_err());
        catalog.policy = SelectionPolicy::Single;
        assert_eq!(catalog.select(Some("missing")).unwrap().name, "alternate");
        catalog.entries.clear();
        assert!(catalog.select(None).is_err());
    }
}
