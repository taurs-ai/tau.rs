//! OpenAI-compatible model list request handling.

use crate::http_response::json_escape;
use tau_engine::ChatInferenceEngine;

/// Builds the OpenAI-compatible `/v1/models` response.
///
/// The input engine supplies the current public model id, and the output is a JSON
/// list response. The function cannot fail because all dynamic strings are
/// JSON-escaped before formatting.
pub(super) fn openai_models(engine: &dyn ChatInferenceEngine) -> String {
    format!(
        concat!(
            "{{",
            "\"object\":\"list\",",
            "\"data\":[{{",
            "\"id\":\"{}\",",
            "\"object\":\"model\",",
            "\"created\":0,",
            "\"owned_by\":\"tau.rs\"",
            "}}]",
            "}}\n"
        ),
        json_escape(engine.model_id())
    )
}
