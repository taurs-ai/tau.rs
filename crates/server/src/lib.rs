//! Server-facing inference orchestration and HTTP APIs.

mod chat_completions_handler;
mod config;
mod error;
mod http_response;
mod models_handler;

use crate::chat_completions_handler::{ChatCompletionError, chat_completions};
use crate::http_response::HttpResponse;
use crate::models_handler::openai_models;
pub use config::HttpServerConfig;
pub use error::ServerError;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use tau_engine::ChatInferenceEngine;
pub use tau_engine::{
    ModelDType, ModelLoadError, ModelShard, Qwen2SafetensorsEngine,
    Qwen3_5MoeSafetensorsEngine, Qwen3_5SafetensorsEngine, Qwen3SafetensorsEngine,
    TensorInfo, load_model_engine, load_model_engine_with_dtype,
};

pub const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:18081";

/// Binds the HTTP listener and serves requests with engine state.
///
/// The input engine is already fully constructed, which lets callers choose
/// loopback or a model-backed engine. The config supplies bind address plus
/// optional request limit. The function returns `ServerError` for
/// bind/listen/connection I/O failures and logs per-request handler errors before
/// continuing.
pub fn serve_http(
    mut engine: Box<dyn ChatInferenceEngine>,
    config: HttpServerConfig,
) -> Result<(), ServerError> {
    let listener = TcpListener::bind(config.listen_addr())?;
    let actual_addr = listener.local_addr()?;
    eprintln!(
        "tau-server listening on http://{} with model {}",
        actual_addr,
        engine.model_id()
    );

    for (handled_request_index, stream) in listener.incoming().enumerate() {
        let stream = stream?;
        if let Err(error) = handle_connection(stream, engine.as_mut())
            && !is_client_disconnect(&error)
        {
            eprintln!("request error: {error}");
        }

        if config
            .max_requests()
            .is_some_and(|max_requests| handled_request_index + 1 >= max_requests)
        {
            break;
        }
    }

    Ok(())
}

/// Returns whether an error only means the client closed its socket early.
///
/// Browsers, health checks, and aborted curl requests can disconnect before the
/// server finishes writing. Those cases do not indicate a bad model or a failed
/// server, so they are intentionally left out of request-error logs.
fn is_client_disconnect(error: &ServerError) -> bool {
    matches!(
        error,
        ServerError::Io(source)
            if matches!(
                source.kind(),
                ErrorKind::BrokenPipe
                    | ErrorKind::ConnectionReset
                    | ErrorKind::ConnectionAborted
            )
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

/// Handles one TCP connection by reading a request, routing it, and writing a response.
///
/// The input stream is an accepted client connection and `engine` is mutable shared
/// engine state. The function returns `ServerError` for request parsing or response
/// writing failures.
fn handle_connection(
    mut stream: TcpStream,
    engine: &mut dyn ChatInferenceEngine,
) -> Result<(), ServerError> {
    let request = read_request(&stream)?;
    let response = route_request(request, engine);
    write_response(&mut stream, response)?;
    Ok(())
}

/// Reads a minimal HTTP/1.1 request from a TCP stream.
///
/// The input stream is cloned for buffered reading, and the output contains method,
/// path, and UTF-8 body. The function returns `ServerError` for empty/malformed
/// requests, invalid content length, read failures, or non-UTF-8 bodies.
fn read_request(stream: &TcpStream) -> Result<HttpRequest, ServerError> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();

    if reader.read_line(&mut request_line)? == 0 {
        return Err(ServerError::BadRequest("empty request".to_owned()));
    }

    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| ServerError::BadRequest("missing HTTP method".to_owned()))?
        .to_owned();
    let path = request_parts
        .next()
        .ok_or_else(|| ServerError::BadRequest("missing HTTP path".to_owned()))?
        .to_owned();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            return Err(ServerError::BadRequest(
                "headers ended before blank line".to_owned(),
            ));
        }

        let header = header.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }

        if let Some((name, value)) = header.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value
                .trim()
                .parse::<usize>()
                .map_err(|source| ServerError::BadRequest(source.to_string()))?;
        }
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    let body = String::from_utf8(body).map_err(|source| {
        ServerError::BadRequest(format!("request body is not UTF-8: {source}"))
    })?;

    Ok(HttpRequest { method, path, body })
}

/// Dispatches an HTTP request to the matching API endpoint.
///
/// The input request contains the parsed method, path, and body, while `engine`
/// provides mutable inference state for inference calls. The function always returns an
/// HTTP response and converts endpoint-specific errors into status-coded response
/// bodies.
fn route_request(
    request: HttpRequest,
    engine: &mut dyn ChatInferenceEngine,
) -> HttpResponse {
    let path = request
        .path
        .split_once('?')
        .map_or(request.path.as_str(), |(path, _)| path);

    match (request.method.as_str(), path) {
        ("OPTIONS", _) => HttpResponse::no_content(),
        ("GET", "/healthz") => HttpResponse::ok("text/plain; charset=utf-8", "ok\n"),

        // OpenAI-compatible endpoints used by Open WebUI and the human CLI.
        ("GET", "/v1/models") => {
            HttpResponse::ok("application/json", openai_models(engine))
        }
        ("POST", "/v1/chat/completions") => {
            match chat_completions(engine, &request.body) {
                Ok(response) => response,
                Err(ChatCompletionError::BadRequest(message)) => {
                    HttpResponse::json_bad_request(message)
                }
                Err(ChatCompletionError::ModelNotFound(model)) => {
                    HttpResponse::json_not_found(format!(
                        "model {model:?} was not found"
                    ))
                }
            }
        }

        _ => HttpResponse::not_found(),
    }
}

/// Writes an HTTP response to a TCP stream.
///
/// The input response supplies status, content type, body, and CORS headers are added
/// here. The function returns `ServerError::Io` if writing or flushing the stream
/// fails.
fn write_response(
    stream: &mut TcpStream,
    response: HttpResponse,
) -> Result<(), ServerError> {
    write!(
        stream,
        concat!(
            "HTTP/1.1 {} {}\r\n",
            "Content-Type: {}\r\n",
            "Content-Length: {}\r\n",
            "Access-Control-Allow-Origin: *\r\n",
            "Access-Control-Allow-Headers: authorization, content-type\r\n",
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n",
            "Connection: close\r\n",
            "\r\n",
            "{}"
        ),
        response.status_code,
        response.reason,
        response.content_type,
        response.body.len(),
        response.body
    )?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tau_engine::{ChatEngine, Device, LoopbackEngine, ModelDType};

    /// Creates an isolated loopback engine for HTTP routing tests.
    fn loopback_chat_engine() -> ChatEngine<LoopbackEngine> {
        ChatEngine::new(LoopbackEngine::new())
    }

    /// Resolves the generated Qwen3.5 MoE fixture used by ignored route tests.
    fn qwen3_5_moe_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_MOE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.5-MoE-Tiny-Dev")
            })
    }

    /// Resolves the generated dense Qwen3.5 fixture used by ignored route tests.
    fn qwen3_5_dense_test_model_path() -> PathBuf {
        std::env::var("TAU_QWEN3_5_DENSE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.8-Dense-Tiny-Dev")
            })
    }

    /// Exercises every public route without sharing the MoE test path.
    fn assert_qwen3_5_dense_openai_routes(
        engine: &mut dyn ChatInferenceEngine,
        max_tokens: usize,
    ) -> (HttpResponse, HttpResponse) {
        let model_id = engine.model_id().to_owned();

        let health = route_request(
            HttpRequest {
                method: "GET".to_owned(),
                path: "/healthz".to_owned(),
                body: String::new(),
            },
            engine,
        );
        assert_eq!(health.status_code, 200);
        assert_eq!(health.body, "ok\n");

        let preflight = route_request(
            HttpRequest {
                method: "OPTIONS".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: String::new(),
            },
            engine,
        );
        assert_eq!(preflight.status_code, 204);
        assert!(preflight.body.is_empty());

        let models = route_request(
            HttpRequest {
                method: "GET".to_owned(),
                path: "/v1/models".to_owned(),
                body: String::new(),
            },
            engine,
        );
        assert_eq!(models.status_code, 200);
        assert!(models.body.contains(&format!(r#""id":"{model_id}""#)));

        let completion = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: format!(
                    concat!(
                        r#"{{"model":"{}","max_completion_tokens":{},"#,
                        r#""messages":[{{"role":"user","content":"Say hello."}}]}}"#,
                    ),
                    model_id, max_tokens
                ),
            },
            engine,
        );
        assert_eq!(completion.status_code, 200, "{}", completion.body);
        assert!(completion.body.contains(r#""object":"chat.completion""#));
        assert!(completion.body.contains(r#""finish_reason":"length""#));
        assert!(completion.body.contains(r#""usage":{"prompt_tokens":"#));
        assert!(
            completion
                .body
                .contains(&format!(r#""model":"{model_id}""#))
        );

        let stream = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: format!(
                    concat!(
                        r#"{{"model":"{}","max_tokens":{},"stream":true,"#,
                        r#""messages":[{{"role":"user","content":["#,
                        r#"{{"type":"text","text":"Say hello again."}}]}}]}}"#,
                    ),
                    model_id, max_tokens
                ),
            },
            engine,
        );
        assert_eq!(stream.status_code, 200, "{}", stream.body);
        assert_eq!(stream.content_type, "text/event-stream; charset=utf-8");
        assert!(stream.body.contains(r#""object":"chat.completion.chunk""#));
        assert!(stream.body.contains(&format!(r#""model":"{model_id}""#)));
        assert!(stream.body.ends_with("data: [DONE]\n\n"));
        (completion, stream)
    }

    /// Exercises every public route with a loaded Qwen3.5 MoE chat engine.
    fn assert_qwen3_5_moe_openai_routes(
        engine: &mut dyn ChatInferenceEngine,
        max_tokens: usize,
    ) {
        let model_id = engine.model_id().to_owned();

        let health = route_request(
            HttpRequest {
                method: "GET".to_owned(),
                path: "/healthz".to_owned(),
                body: String::new(),
            },
            engine,
        );
        assert_eq!(health.status_code, 200);
        assert_eq!(health.body, "ok\n");

        let preflight = route_request(
            HttpRequest {
                method: "OPTIONS".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: String::new(),
            },
            engine,
        );
        assert_eq!(preflight.status_code, 204);
        assert!(preflight.body.is_empty());

        let models = route_request(
            HttpRequest {
                method: "GET".to_owned(),
                path: "/v1/models".to_owned(),
                body: String::new(),
            },
            engine,
        );
        assert_eq!(models.status_code, 200);
        assert!(models.body.contains(&format!(r#""id":"{model_id}""#)));

        let completion = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: format!(
                    concat!(
                        r#"{{"model":"{}","max_completion_tokens":{},"#,
                        r#""messages":[{{"role":"user","content":"Say hello."}}]}}"#,
                    ),
                    model_id, max_tokens
                ),
            },
            engine,
        );
        assert_eq!(completion.status_code, 200, "{}", completion.body);
        assert!(completion.body.contains(r#""object":"chat.completion""#));
        assert!(
            completion
                .body
                .contains(&format!(r#""model":"{model_id}""#))
        );

        let stream = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: format!(
                    concat!(
                        r#"{{"model":"{}","max_tokens":{},"stream":true,"#,
                        r#""messages":[{{"role":"user","content":["#,
                        r#"{{"type":"text","text":"Say hello again."}}]}}]}}"#,
                    ),
                    model_id, max_tokens
                ),
            },
            engine,
        );
        assert_eq!(stream.status_code, 200, "{}", stream.body);
        assert_eq!(stream.content_type, "text/event-stream; charset=utf-8");
        assert!(stream.body.contains(r#""object":"chat.completion.chunk""#));
        assert!(stream.body.contains(&format!(r#""model":"{model_id}""#)));
        assert!(stream.body.ends_with("data: [DONE]\n\n"));
    }

    /// Verifies that ordinary client disconnects are recognized as non-actionable.
    ///
    /// The test constructs representative socket write/read failures and expects
    /// the server to classify them as client disconnects rather than model or
    /// request failures.
    #[test]
    fn recognizes_client_disconnect_errors() {
        assert!(is_client_disconnect(&ServerError::Io(
            std::io::Error::from(ErrorKind::BrokenPipe)
        )));
        assert!(is_client_disconnect(&ServerError::Io(
            std::io::Error::from(ErrorKind::ConnectionReset)
        )));
        assert!(!is_client_disconnect(&ServerError::BadRequest(
            "bad json".to_owned()
        )));
    }

    /// Verifies that `/v1/models` lists the built-in loopback model.
    ///
    /// The test routes an in-memory GET request and expects a 200 JSON response
    /// containing `tau-loopback`. It fails by panicking if routing or model
    /// serialization changes.
    #[test]
    fn lists_loopback_model() {
        let mut engine = loopback_chat_engine();
        let response = route_request(
            HttpRequest {
                method: "GET".to_owned(),
                path: "/v1/models".to_owned(),
                body: String::new(),
            },
            &mut engine,
        );

        assert_eq!(response.status_code, 200);
        assert!(response.body.contains(r#""id":"tau-loopback""#));
    }

    /// Verifies that loopback chat echoes the latest user message.
    ///
    /// The test sends an in-memory OpenAI chat request and expects a 200 completion
    /// with the user text in assistant content. It fails by panicking on wrong
    /// status or body content.
    #[test]
    fn loopback_chat_completion_echoes_last_user_message() {
        let mut engine = loopback_chat_engine();
        let response = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: concat!(
                    r#"{"model":"tau-loopback","#,
                    r#""messages":["#,
                    r#"{"role":"system","content":"ignore"},"#,
                    r#"{"role":"user","content":"hello tau"}]}"#,
                )
                .to_owned(),
            },
            &mut engine,
        );

        assert_eq!(response.status_code, 200);
        assert!(response.body.contains(r#""object":"chat.completion""#));
        assert!(response.body.contains(r#""content":"hello tau""#));
    }

    /// Verifies that loopback chat escapes response content as valid JSON.
    ///
    /// The test sends content containing a newline and quotes, then expects escaped
    /// JSON output. It fails by panicking if the response is not successful or
    /// escaping regresses.
    #[test]
    fn loopback_chat_completion_escapes_response_content() {
        let mut engine = loopback_chat_engine();
        let response = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: concat!(
                    r#"{"model":"tau-loopback","#,
                    r#""messages":["#,
                    r#"{"role":"user","content":"hello\n\"hb\""}]}"#,
                )
                .to_owned(),
            },
            &mut engine,
        );

        assert_eq!(response.status_code, 200);
        assert!(response.body.contains(r#""content":"hello\n\"hb\"""#));
    }

    /// Verifies that loopback chat can return streaming Server-Sent Events.
    ///
    /// The test sends `stream: true` and expects an SSE content type plus a `[DONE]`
    /// marker. It fails by panicking if streaming routing or payload formatting
    /// changes.
    #[test]
    fn loopback_chat_completion_supports_streaming() {
        let mut engine = loopback_chat_engine();
        let response = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: concat!(
                    r#"{"model":"tau-loopback","#,
                    r#""stream":true,"#,
                    r#""messages":["#,
                    r#"{"role":"user","content":"stream me"}]}"#,
                )
                .to_owned(),
            },
            &mut engine,
        );

        assert_eq!(response.status_code, 200);
        assert_eq!(response.content_type, "text/event-stream; charset=utf-8");
        assert!(response.body.contains(r#""content":"stream me""#));
        assert!(response.body.contains("data: [DONE]"));
    }

    /// Verifies that unknown OpenAI chat model ids are rejected.
    ///
    /// The test sends a missing model id and expects a 404 OpenAI-style error body. It
    /// fails by panicking if the server accepts the model or changes the error
    /// code.
    #[test]
    fn rejects_unknown_chat_model() {
        let mut engine = loopback_chat_engine();
        let response = route_request(
            HttpRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                body: concat!(
                    r#"{"model":"missing","#,
                    r#""messages":["#,
                    r#"{"role":"user","content":"hello"}]}"#,
                )
                .to_owned(),
            },
            &mut engine,
        );

        assert_eq!(response.status_code, 404);
        assert!(response.body.contains(r#""code":"model_not_found""#));
    }

    /// Exercises every public HTTP route with the generated model on CPU/F32.
    #[test]
    #[ignore = "loads and runs the local generated Qwen3.5 MoE development model"]
    fn serves_qwen3_5_moe_through_openai_routes() {
        let path = qwen3_5_moe_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());
        let mut engine =
            load_model_engine_with_dtype(&path, Device::Cpu, ModelDType::F32).unwrap();
        assert_qwen3_5_moe_openai_routes(engine.as_mut(), 2);
    }

    /// Exercises multi-token buffered and streaming routes on CUDA/BF16.
    #[test]
    #[ignore = "requires a CUDA GPU for Qwen3.5 MoE BF16 HTTP execution"]
    fn serves_qwen3_5_moe_in_bf16_on_cuda_through_openai_routes() {
        let path = qwen3_5_moe_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());
        let device =
            Device::new_cuda(0).expect("BF16 Qwen3.5 server test requires CUDA GPU 0");
        let mut engine =
            load_model_engine_with_dtype(&path, device, ModelDType::BF16).unwrap();

        assert_qwen3_5_moe_openai_routes(engine.as_mut(), 4);
    }

    /// Exercises every public HTTP route with dense Qwen3.5 on CPU/F32.
    #[test]
    #[ignore = "loads and runs the local generated Qwen3.8 dense model"]
    fn serves_qwen3_5_dense_through_openai_routes() {
        let path = qwen3_5_dense_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());
        let mut engine =
            load_model_engine_with_dtype(&path, Device::Cpu, ModelDType::F32).unwrap();

        assert_qwen3_5_dense_openai_routes(engine.as_mut(), 2);
    }

    /// Exercises multi-token dense buffered and streaming routes on CUDA/BF16.
    #[test]
    #[ignore = "requires a CUDA GPU for dense Qwen3.5 BF16 HTTP execution"]
    fn serves_qwen3_5_dense_in_bf16_on_cuda_through_openai_routes() {
        let path = qwen3_5_dense_test_model_path();
        assert!(path.is_dir(), "test model is absent: {}", path.display());
        let device = Device::new_cuda(0)
            .expect("dense Qwen3.5 BF16 server test requires CUDA GPU 0");
        let mut engine =
            load_model_engine_with_dtype(&path, device, ModelDType::BF16).unwrap();

        assert_qwen3_5_dense_openai_routes(engine.as_mut(), 4);
    }
    /// Public factory creates a capture device; repeat routes reuse graphs and reset
    /// state.
    #[test]
    #[ignore = "requires the dense tiny fixture and CUDA GPU 0"]
    fn serves_cuda_graph_decode_through_openai_routes() {
        for mode in [
            tau_engine::CudaGraphMode::Layer,
            tau_engine::CudaGraphMode::Full,
        ] {
            let path = qwen3_5_dense_test_model_path();
            let device = Device::new_cuda(0).unwrap();
            let original = device.clone();
            let mut engine = tau_engine::load_model_engine_with_runtime_options(
                &path,
                device,
                ModelDType::BF16,
                tau_engine::EngineOptions::new()
                    .with_cuda_graph(mode)
                    .with_prefill_chunk_size(4),
            )
            .unwrap();
            assert!(
                original.as_cuda_device().unwrap().is_event_tracking(),
                "factory must not mutate caller's device"
            );
            let mut eager = tau_engine::load_model_engine_with_runtime_options(
                &path,
                Device::new_cuda(0).unwrap(),
                ModelDType::BF16,
                tau_engine::EngineOptions::new()
                    .with_cuda_graph(tau_engine::CudaGraphMode::None)
                    .with_prefill_chunk_size(4),
            )
            .unwrap();
            let payload = |body: &str| {
                let value = serde_json::from_str::<serde_json::Value>(body).unwrap();
                (value.get("choices").cloned(), value.get("usage").cloned())
            };
            for _ in 0..2 {
                let (actual, stream) =
                    assert_qwen3_5_dense_openai_routes(engine.as_mut(), 4);
                let (expected, eager_stream) =
                    assert_qwen3_5_dense_openai_routes(eager.as_mut(), 4);
                assert_eq!(payload(&actual.body), payload(&expected.body));
                let chunks = |body: &str| {
                    body.lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|line| *line != "[DONE]")
                        .map(&payload)
                        .collect::<Vec<_>>()
                };
                assert_eq!(chunks(&stream.body), chunks(&eager_stream.body));
            }
        }
    }
}
