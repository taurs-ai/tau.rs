//! Text generation configuration and decoding helpers.

use crate::ModelLoadError;
use crate::metadata::{optional_json_usize, parse_json_file};
use serde_json::Value;
use std::path::Path;

pub const DEFAULT_MAX_NEW_TOKENS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationConfig {
    pub eos_token_ids: Vec<u32>,
    pub max_new_tokens: usize,
}

impl GenerationConfig {
    /// Returns whether a token id should stop generation.
    ///
    /// The input token id is compared with the resolved generation settings.
    /// The method returns `true` for stop tokens and has no error cases.
    pub fn is_eos_token(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }
}

/// Loads generation settings, falling back to model config only if absent.
///
/// `generation_config.json` takes precedence. Without it, settings come from
/// `config.json`, with non-null nested `text_config` fields taking precedence
/// over top-level fields. Invalid files and unusable settings remain errors.
pub(crate) fn load_generation_config(
    root: &Path,
) -> Result<GenerationConfig, ModelLoadError> {
    let path = root.join("generation_config.json");
    let (value, source) = if path
        .try_exists()
        .map_err(|error| ModelLoadError::Io(path.clone(), error))?
    {
        (parse_json_file(&path)?, "generation_config.json")
    } else {
        let config = parse_json_file(&root.join("config.json"))?;
        (model_generation_settings(&config), "config.json")
    };
    let eos_token_ids = required_json_u32_list(&value, "eos_token_id", source)?;
    if eos_token_ids.is_empty() {
        return Err(ModelLoadError::InvalidConfig(format!(
            "{source} must include at least one EOS token"
        )));
    }

    let max_new_tokens = optional_json_usize(&value, "max_new_tokens")?
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    if max_new_tokens == 0 {
        return Err(ModelLoadError::InvalidConfig(format!(
            "{source} max_new_tokens must be greater than zero"
        )));
    }

    Ok(GenerationConfig {
        eos_token_ids,
        max_new_tokens,
    })
}

/// Selects the runtime's generation fields from flat or nested model metadata.
fn model_generation_settings(config: &Value) -> Value {
    let mut settings = serde_json::Map::new();
    for field in ["eos_token_id", "max_new_tokens"] {
        let value = config
            .get("text_config")
            .and_then(|text| text.get(field))
            .filter(|value| !value.is_null())
            .or_else(|| config.get(field))
            .filter(|value| !value.is_null());
        if let Some(value) = value {
            settings.insert(field.to_owned(), value.clone());
        }
    }
    Value::Object(settings)
}

/// Extracts required `u32` token ids from a JSON field.
///
/// The input value must be an object with either a non-negative integer or an
/// array of non-negative integers under `field`. The function returns token ids
/// or `ModelLoadError::InvalidConfig` for absent or unsupported values.
fn required_json_u32_list(
    value: &Value,
    field: &str,
    source: &str,
) -> Result<Vec<u32>, ModelLoadError> {
    let value = value.get(field).ok_or_else(|| {
        ModelLoadError::InvalidConfig(format!("{source} must include {field:?}"))
    })?;

    if let Some(values) = value.as_array() {
        return values
            .iter()
            .map(|value| json_value_to_u32(value, field))
            .collect();
    }

    json_value_to_u32(value, field).map(|value| vec![value])
}

/// Converts one JSON value into a token id.
///
/// The inputs are a JSON value and the field name being parsed for diagnostics.
/// The function returns a `u32` token id or `ModelLoadError::InvalidConfig` when
/// the value is not a non-negative integer representable as `u32`.
fn json_value_to_u32(value: &Value, field: &str) -> Result<u32, ModelLoadError> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| {
            ModelLoadError::InvalidConfig(format!(
                "config field {field:?} must be a non-negative u32 integer"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Keeps each test's model metadata isolated and removes it on drop.
    struct ModelDir(PathBuf);

    impl ModelDir {
        fn new(config: Option<&str>, generation: Option<&str>) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "tau-generation-config-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            let directory = Self(root);
            for (filename, contents) in [
                ("config.json", config),
                ("generation_config.json", generation),
            ] {
                if let Some(contents) = contents {
                    fs::write(directory.0.join(filename), contents).unwrap();
                }
            }
            directory
        }

        fn load(&self) -> Result<GenerationConfig, ModelLoadError> {
            load_generation_config(&self.0)
        }
    }

    impl Drop for ModelDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn generation_file_takes_precedence_over_model_settings() {
        let root = ModelDir::new(
            Some(r#"{"text_config":{"eos_token_id":7,"max_new_tokens":99}}"#),
            Some(r#"{"eos_token_id":[248046,248044],"max_new_tokens":32}"#),
        );
        let config = root.load().unwrap();
        assert_eq!(config.eos_token_ids, [248_046, 248_044]);
        assert_eq!(config.max_new_tokens, 32);
        assert!(config.is_eos_token(248_046));
        assert!(!config.is_eos_token(7));
    }

    #[test]
    fn generation_file_does_not_inherit_model_token_limit() {
        let root = ModelDir::new(
            Some(r#"{"eos_token_id":7,"max_new_tokens":99}"#),
            Some(r#"{"eos_token_id":248044}"#),
        );
        let config = root.load().unwrap();
        assert_eq!(config.eos_token_ids, [248_044]);
        assert_eq!(config.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
    }

    #[test]
    fn generation_file_does_not_require_reading_model_config() {
        for model in [None, Some("invalid json")] {
            let root = ModelDir::new(model, Some(r#"{"eos_token_id":248044}"#));
            assert!(root.load().unwrap().is_eos_token(248_044));
        }
    }

    #[test]
    fn falls_back_to_flat_model_settings() {
        let root = ModelDir::new(
            Some(r#"{"eos_token_id":[151643,151645],"max_new_tokens":64}"#),
            None,
        );
        let config = root.load().unwrap();
        assert_eq!(config.eos_token_ids, [151_643, 151_645]);
        assert_eq!(config.max_new_tokens, 64);
    }

    #[test]
    fn falls_back_to_qwen3_5_text_settings_and_default_limit() {
        let root = ModelDir::new(
            Some(r#"{"model_type":"qwen3_5","text_config":{"eos_token_id":248044}}"#),
            None,
        );
        let config = root.load().unwrap();
        assert_eq!(config.eos_token_ids, [248_044]);
        assert_eq!(config.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
    }

    #[test]
    fn nested_model_settings_override_top_level_per_field() {
        let root = ModelDir::new(
            Some(
                r#"{
                "eos_token_id":7,"max_new_tokens":99,
                "text_config":{"eos_token_id":[248046,248044],"max_new_tokens":32}
            }"#,
            ),
            None,
        );
        let config = root.load().unwrap();
        assert_eq!(config.eos_token_ids, [248_046, 248_044]);
        assert_eq!(config.max_new_tokens, 32);

        for text in [json!({}), json!({"eos_token_id":null})] {
            let model = json!({
                "eos_token_id":248044,"max_new_tokens":64,"text_config":text
            });
            let root = ModelDir::new(Some(&model.to_string()), None);
            let config = root.load().unwrap();
            assert_eq!(config.eos_token_ids, [248_044]);
            assert_eq!(config.max_new_tokens, 64);
        }

        let root = ModelDir::new(
            Some(r#"{"max_new_tokens":64,"text_config":{"eos_token_id":248044}}"#),
            None,
        );
        let config = root.load().unwrap();
        assert_eq!(config.eos_token_ids, [248_044]);
        assert_eq!(config.max_new_tokens, 64);
    }

    #[test]
    fn null_model_token_limit_uses_default() {
        let root = ModelDir::new(
            Some(r#"{"text_config":{"eos_token_id":248044,"max_new_tokens":null}}"#),
            None,
        );
        assert_eq!(root.load().unwrap().max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
    }

    #[test]
    fn malformed_generation_file_does_not_fall_back() {
        let root = ModelDir::new(Some(r#"{"eos_token_id":248044}"#), Some("{"));
        assert!(matches!(root.load(), Err(ModelLoadError::Json(path, _))
            if path == root.0.join("generation_config.json")));
    }

    #[test]
    fn invalid_generation_settings_do_not_fall_back() {
        for generation in [
            json!({}),
            json!({"eos_token_id":[]}),
            json!({"eos_token_id":null}),
            json!({"eos_token_id":-1}),
            json!({"eos_token_id":248044,"max_new_tokens":0}),
        ] {
            let root = ModelDir::new(
                Some(r#"{"eos_token_id":248044,"max_new_tokens":32}"#),
                Some(&generation.to_string()),
            );
            assert!(
                matches!(root.load(), Err(ModelLoadError::InvalidConfig(_))),
                "unexpected fallback for {generation}"
            );
        }
    }

    #[test]
    fn generation_config_directory_is_not_treated_as_absent() {
        let root = ModelDir::new(Some(r#"{"eos_token_id":248044}"#), None);
        fs::create_dir(root.0.join("generation_config.json")).unwrap();
        assert!(root.load().is_err());
    }

    #[test]
    fn fallback_reports_missing_or_malformed_model_file() {
        let root = ModelDir::new(None, None);
        assert!(matches!(root.load(), Err(ModelLoadError::MissingFile(path))
            if path == root.0.join("config.json")));
        let root = ModelDir::new(Some("{"), None);
        assert!(matches!(root.load(), Err(ModelLoadError::Json(path, _))
            if path == root.0.join("config.json")));
    }

    #[test]
    fn fallback_rejects_invalid_eos_and_token_limits() {
        for settings in [
            json!({}),
            json!({"eos_token_id":[]}),
            json!({"eos_token_id":null}),
            json!({"eos_token_id":-1}),
            json!({"eos_token_id":4294967296_u64}),
            json!({"eos_token_id":[248044,"invalid"]}),
            json!({"eos_token_id":248044,"max_new_tokens":0}),
            json!({"eos_token_id":248044,"max_new_tokens":-1}),
            json!({"eos_token_id":248044,"max_new_tokens":"32"}),
        ] {
            for model in [settings.clone(), json!({"text_config":settings})] {
                let root = ModelDir::new(Some(&model.to_string()), None);
                assert!(
                    matches!(root.load(), Err(ModelLoadError::InvalidConfig(_))),
                    "accepted invalid model settings: {model}"
                );
            }
        }
    }

    #[test]
    fn fallback_validation_errors_identify_model_config() {
        for model in [
            json!({"text_config":{}}),
            json!({"text_config":{"eos_token_id":[]}}),
            json!({"text_config":{"eos_token_id":248044,"max_new_tokens":0}}),
        ] {
            let root = ModelDir::new(Some(&model.to_string()), None);
            let error = root.load().unwrap_err().to_string();
            assert!(error.contains("config.json"), "{error}");
            assert!(!error.contains("generation_config.json"), "{error}");
        }
    }
}
