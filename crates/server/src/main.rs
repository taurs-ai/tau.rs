mod args;
mod device;
mod dtype;

use args::{ServerArgs, parse_args, usage};
use device::DevicePreference;
use dtype::DTypePreference;
use std::env;
use std::process::ExitCode;
use tau_engine::{
    ChatEngine, ChatInferenceEngine, LoopbackEngine,
    load_model_engine_with_chat_template_selection,
};
use tau_server::{HttpServerConfig, serve_http};

/// Starts the `tau-server` binary and maps success or failure to a process exit code.
///
/// The function reads no direct input beyond process arguments consumed by `run`. It
/// returns success for a clean server shutdown and failure after printing any startup
/// or runtime error.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Parses command-line options, loads an optional model, and runs the HTTP server.
///
/// Inputs come from `env::args`, including model path, device preference, listen
/// address, and `--once`. The function returns `Ok(())` after the server stops or
/// a string error for invalid arguments, model loading failure, listen-address
/// parsing failure, or server runtime errors.
fn run() -> Result<(), String> {
    let mut args = env::args();
    let binary = args.next().unwrap_or_else(|| "tau-server".to_owned());

    let parsed = parse_args(args, &binary)?;
    let listen_addr = parsed.listen_addr.parse().map_err(|source| {
        format!(
            "invalid listen address {:?}: {source}\n\n{}",
            parsed.listen_addr,
            usage(&binary)
        )
    })?;
    let config =
        HttpServerConfig::new(listen_addr).with_max_requests(parsed.once.then_some(1));
    let engine = load_engine(&parsed)?;

    serve_http(engine, config).map_err(|error| error.to_string())
}

/// Loads the engine selected by parsed CLI arguments.
///
/// The input args may specify a model directory or neither. The function returns a
/// boxed inference engine or a string error when model loading fails.
fn load_engine(parsed: &ServerArgs) -> Result<Box<dyn ChatInferenceEngine>, String> {
    if parsed.moe_decode_options.is_some() && parsed.model_path.is_none() {
        return Err("--moe-decode-config requires --model".to_owned());
    }
    if parsed.moe_prefill_config.is_some() && parsed.model_path.is_none() {
        return Err("--moe-prefill-config requires --model".to_owned());
    }
    if parsed.attention_block_size != tau_engine::AttentionBlockSize::default()
        && parsed.model_path.is_none()
    {
        return Err("--attention-block-size requires --model".to_owned());
    }
    if (parsed.chat_template.is_some() || parsed.chat_template_name.is_some())
        && parsed.model_path.is_none()
    {
        return Err("chat template selection requires --model".to_owned());
    }
    if parsed.cuda_graph.is_enabled() && parsed.model_path.is_none() {
        return Err(format!(
            "--cuda-graph {} requires --model",
            parsed.cuda_graph
        ));
    }
    if parsed.device == DevicePreference::Cpu && parsed.dtype == DTypePreference::BF16 {
        return Err(DTypePreference::cpu_bf16_error());
    }

    let resolved_device = match (&parsed.model_path, parsed.device) {
        (Some(_), _) | (None, DevicePreference::Gpu) => Some(parsed.device.resolve()?),
        _ => None,
    };

    if let Some(model_path) = &parsed.model_path {
        let resolved =
            resolved_device.expect("model path always resolves an inference device");
        resolved.log_selection();
        let dtype = parsed.dtype.resolve(resolved.is_cpu())?;
        eprintln!("tau-server inference dtype: {dtype}");

        return load_model_engine_with_chat_template_selection(
            model_path,
            resolved.into_device(),
            dtype,
            parsed.engine_options(),
            &parsed.chat_template_selection(),
        )
        .map_err(|error| error.to_string());
    }

    Ok(Box::new(ChatEngine::new(LoopbackEngine::new())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tau_engine::{ChatMessage, ChatOptions};

    /// Explicit tiles cannot disappear into loopback or a CPU reference engine.
    #[test]
    fn explicit_moe_prefill_configuration_requires_a_supported_model() {
        let args = parse_args(
            ["--moe-prefill-config=16,64,64,4,1".to_owned()].into_iter(),
            "tau-server",
        )
        .unwrap();
        assert!(
            load_engine(&args)
                .err()
                .unwrap()
                .contains("requires --model")
        );
        let args = parse_args(
            [
                "--model",
                "/absent/model",
                "--device",
                "cpu",
                "--moe-prefill-config=16,64,64,4,1",
            ]
            .into_iter()
            .map(str::to_owned),
            "tau-server",
        )
        .unwrap();
        assert!(
            load_engine(&args)
                .err()
                .unwrap()
                .contains("requires CUDA/BF16")
        );
    }

    /// A template override must not be silently ignored by loopback mode.
    #[test]
    fn chat_template_requires_a_model() {
        let args = parse_args(
            ["--chat-template=chat.jinja".to_owned()].into_iter(),
            "tau-server",
        )
        .unwrap();
        assert!(
            load_engine(&args)
                .err()
                .unwrap()
                .contains("requires --model")
        );
    }

    /// Rejects graph modes without a model before resolving devices or loading weights.
    #[test]
    fn cuda_graph_requires_a_model() {
        for mode in ["layer", "full"] {
            let args = parse_args(
                ["--cuda-graph".to_owned(), mode.to_owned()].into_iter(),
                "tau-server",
            )
            .unwrap();
            assert!(
                load_engine(&args)
                    .err()
                    .unwrap()
                    .contains("requires --model")
            );
        }
    }

    /// Compares optimized and reference execution on the same CUDA chat request.
    #[test]
    #[ignore = "requires CUDA GPU 0 and the generated dense Qwen3.5 model"]
    fn optimize_kernel_false_matches_custom_kernel_completion() {
        let model = std::env::var("TAU_QWEN3_5_DENSE_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").expect("HOME is not set"))
                    .join("work/models/Qwen3.8-Dense-Tiny-Dev")
            });
        assert!(model.is_dir(), "test model is absent: {}", model.display());
        let base_args = vec![
            "--model".to_owned(),
            model.to_string_lossy().into_owned(),
            "--device".to_owned(),
            "gpu".to_owned(),
            "--dtype".to_owned(),
            "bf16".to_owned(),
        ];
        let messages = [ChatMessage::new("user", "Say hello.")];
        let options = ChatOptions::new().with_max_tokens(Some(4));

        let custom = {
            let args = parse_args(base_args.clone().into_iter(), "tau-server").unwrap();
            let mut engine = load_engine(&args).unwrap();
            engine
                .chat_with_options(&messages, options.clone())
                .unwrap()
        };
        let reference = {
            let mut reference_args = base_args;
            reference_args.extend(["--optimize-kernel".to_owned(), "false".to_owned()]);
            let args = parse_args(reference_args.into_iter(), "tau-server").unwrap();
            let mut engine = load_engine(&args).unwrap();
            engine.chat_with_options(&messages, options).unwrap()
        };

        assert_eq!(custom, reference);
    }
}
