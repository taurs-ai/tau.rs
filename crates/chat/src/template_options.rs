//! Request-local Jinja variables and normalization of reasoning options.

use crate::ChatTemplateError;
use serde_json::{Map, Value};

#[cfg(test)]
mod tests;

/// Validated template context, independent of generation and model execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    variables: Map<String, Value>,
}

impl ChatTemplateOptions {
    /// Normalizes public request fields once for both HTTP and Rust callers.
    /// Explicit kwargs win over inferred thinking flags and top-level effort.
    /// Missing values remain undefined so checkpoint defaults still apply.
    /// Normalization of reasoning aliases and precedence matches SGLang.
    pub fn from_request_fields(
        reasoning_effort: Option<&Value>,
        reasoning: Option<&Value>,
        chat_template_kwargs: Option<&Value>,
    ) -> Result<Self, ChatTemplateError> {
        let mut variables = match chat_template_kwargs {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(values)) => values.clone(),
            _ => return Err(invalid("chat_template_kwargs must be an object")),
        };
        for key in variables.keys() {
            if matches!(
                key.as_str(),
                "messages"
                    | "conversation"
                    | "tools"
                    | "documents"
                    | "add_generation_prompt"
                    | "continue_final_message"
                    | "chat_template"
                    | "tokenize"
                    | "return_dict"
                    | "return_tensors"
                    | "return_assistant_tokens_mask"
                    | "padding"
                    | "truncation"
                    | "max_length"
                    | "tokenizer_kwargs"
            ) {
                return Err(invalid(format!(
                    "chat_template_kwargs.{key} is a reserved or unsupported control"
                )));
            }
        }
        let mut raw_effort = reasoning_effort.filter(|v| !v.is_null());
        let mut thinking = None;
        match reasoning {
            None | Some(Value::Null) => {}
            Some(Value::Object(values)) => {
                // Effort in the reasoning object overrides the top-level field.
                // Within that object, `effort` takes precedence over its alias.
                if let Some(effort) = values
                    .get("effort")
                    .filter(|v| !v.is_null())
                    .or_else(|| values.get("reasoning_effort").filter(|v| !v.is_null()))
                {
                    raw_effort = Some(effort);
                }
                let enabled = values
                    .get("enabled")
                    .filter(|v| !v.is_null())
                    .or_else(|| values.get("enable").filter(|v| !v.is_null()));
                match enabled {
                    Some(Value::Bool(true)) => thinking = Some(true),
                    None | Some(Value::Bool(false)) => {}
                    _ => return Err(invalid("reasoning.enabled must be a boolean")),
                }
            }
            _ => return Err(invalid("reasoning must be an object")),
        }
        let effort = raw_effort.map(normalize_effort).transpose()?;
        if let Some(effort) = &effort {
            thinking = Some(effort.as_str() != Some("none"));
        }
        if let Some(thinking) = thinking {
            for key in ["thinking", "enable_thinking"] {
                variables.entry(key).or_insert(Value::Bool(thinking));
            }
        }
        // Build renderer kwargs in order: normalized effort, then explicit kwargs
        // (including any explicit null or conflicting thinking flag).
        let mut result = Map::new();
        if let Some(effort) = effort {
            result.insert("reasoning_effort".to_owned(), effort);
        }
        result.extend(variables);
        Ok(Self { variables: result })
    }

    /// Exposes immutable context; renderers must never add request globals.
    pub fn variables(&self) -> &Map<String, Value> {
        &self.variables
    }

    /// Allows legacy engines to reject controls rather than silently ignore them.
    pub fn require_supported(&self, supported: bool) -> Result<(), ChatTemplateError> {
        if !supported && !self.variables.is_empty() {
            return Err(invalid("this engine does not support template controls"));
        }
        Ok(())
    }
}

/// Accepts documented effort tiers and finite fractional values. The selected
/// template enforces checkpoint-specific restrictions through its own errors.
fn normalize_effort(value: &Value) -> Result<Value, ChatTemplateError> {
    if let Some(tier) = value.as_str()
        && matches!(
            tier,
            "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
        )
    {
        return Ok(value.clone());
    }
    let number = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(number) => number.trim().parse::<f64>().ok(),
        _ => None,
    };
    match number {
        Some(number) if number.is_finite() && (0.0..=0.99).contains(&number) => {
            Ok(Value::from(number))
        }
        _ => Err(invalid(
            "invalid reasoning_effort: expected a supported tier \
             or a number in [0, 0.99]",
        )),
    }
}

/// Keeps errors recognizable across direct engine calls and HTTP bad requests.
fn invalid(message: impl std::fmt::Display) -> ChatTemplateError {
    ChatTemplateError::InvalidOptions(message.to_string())
}
