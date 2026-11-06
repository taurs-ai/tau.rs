//! HTTP response construction and JSON response helpers.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HttpResponse {
    pub(crate) status_code: u16,
    pub(crate) reason: &'static str,
    pub(crate) content_type: &'static str,
    pub(crate) body: String,
}

impl HttpResponse {
    /// Builds a successful HTTP response with a content type and body.
    ///
    /// The caller supplies the response MIME type and body payload. The function
    /// returns a 200 response and cannot fail.
    pub(crate) fn ok(content_type: &'static str, body: impl Into<String>) -> Self {
        Self {
            status_code: 200,
            reason: "OK",
            content_type,
            body: Into::<String>::into(body),
        }
    }

    /// Builds an empty HTTP 204 response for CORS preflight requests.
    ///
    /// The function takes no input and returns an empty response body. It cannot
    /// fail.
    pub(crate) fn no_content() -> Self {
        Self {
            status_code: 204,
            reason: "No Content",
            content_type: "text/plain; charset=utf-8",
            body: String::new(),
        }
    }

    /// Builds an OpenAI-style JSON HTTP 400 response.
    ///
    /// The input message describes the invalid request. The function returns a
    /// JSON error payload and cannot fail.
    pub(crate) fn json_bad_request(message: impl Into<String>) -> Self {
        Self {
            status_code: 400,
            reason: "Bad Request",
            content_type: "application/json",
            body: openai_error_json(
                Into::<String>::into(message),
                "invalid_request_error",
                "bad_request",
            ),
        }
    }

    /// Builds an OpenAI-style JSON HTTP 404 response for missing models.
    ///
    /// The input message describes the missing resource. The function returns a
    /// JSON error payload and cannot fail.
    pub(crate) fn json_not_found(message: impl Into<String>) -> Self {
        Self {
            status_code: 404,
            reason: "Not Found",
            content_type: "application/json",
            body: openai_error_json(
                Into::<String>::into(message),
                "invalid_request_error",
                "model_not_found",
            ),
        }
    }

    /// Builds a plain-text HTTP 404 response.
    ///
    /// The function takes no input and returns a fixed not-found body. It cannot
    /// fail.
    pub(crate) fn not_found() -> Self {
        Self {
            status_code: 404,
            reason: "Not Found",
            content_type: "text/plain; charset=utf-8",
            body: "not found\n".to_owned(),
        }
    }
}

/// Builds an OpenAI-compatible JSON error response body.
///
/// The inputs provide the message, error type, and code fields. The output is JSON
/// text, and the function cannot fail because all fields are escaped before
/// formatting.
fn openai_error_json(message: String, error_type: &str, code: &str) -> String {
    format!(
        concat!(
            "{{\"error\":{{",
            "\"message\":\"{}\",",
            "\"type\":\"{}\",",
            "\"param\":null,",
            "\"code\":\"{}\"",
            "}}}}\n"
        ),
        json_escape(&message),
        json_escape(error_type),
        json_escape(code)
    )
}

/// Escapes a Rust string so it can be safely embedded in JSON text.
///
/// The input is valid UTF-8, and the output escapes quotes, backslashes, and
/// control characters. The function writes to an owned string and cannot fail.
pub(crate) fn json_escape(value: &str) -> String {
    use std::fmt::Write as _;

    let mut output = String::new();
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0000}'..='\u{001f}' => {
                write!(&mut output, "\\u{:04x}", ch as u32)
                    .expect("write to string failed");
            }
            _ => output.push(ch),
        }
    }

    output
}
