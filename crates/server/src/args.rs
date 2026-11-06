//! Command-line settings, value parsing and help for the server binary.

use crate::{device::DevicePreference, dtype::DTypePreference};
use std::path::PathBuf;
use tau_chat::ChatTemplateSelection;
use tau_engine::{
    AttentionBlockSize, CudaGraphMode, EngineOptions, MoeDecodeOptions,
    MoeDecodeSelection, MoePrefillConfig, MoePrefillTile,
};
use tau_server::DEFAULT_LISTEN_ADDR;

/// Parsed settings for the binary; runtime construction stays in the entry point.
#[derive(Debug)]
pub(super) struct ServerArgs {
    pub(super) model_path: Option<PathBuf>,
    pub(super) chat_template: Option<PathBuf>,
    pub(super) chat_template_name: Option<String>,
    pub(super) device: DevicePreference,
    pub(super) dtype: DTypePreference,
    pub(super) optimize_kernel: bool,
    pub(super) cuda_graph: CudaGraphMode,
    pub(super) prefill_chunk_size: usize,
    pub(super) attention_block_size: AttentionBlockSize,
    pub(super) moe_prefill_config: Option<MoePrefillConfig>,
    pub(super) moe_decode_options: Option<MoeDecodeOptions>,
    pub(super) tracing: bool,
    pub(super) metrics: bool,
    pub(super) vram_metrics: bool,
    pub(super) listen_addr: String,
    pub(super) once: bool,
}

impl ServerArgs {
    /// Resolves startup preferences separately from model execution settings.
    /// File precedence is applied by the source owner, so both requests remain
    /// visible in selection logs and neither argument spelling changes policy.
    pub(super) fn chat_template_selection(&self) -> ChatTemplateSelection {
        let mut selection = ChatTemplateSelection::new();
        if let Some(path) = &self.chat_template {
            selection = selection.with_file(path);
        }
        if let Some(name) = &self.chat_template_name {
            selection = selection.with_name(name);
        }
        selection
    }

    /// Carries model runtime preferences from the CLI into the engine configuration.
    pub(super) fn engine_options(&self) -> EngineOptions {
        EngineOptions::new()
            .with_auto_kv_cache_budget(true)
            .with_optimize_kernel(self.optimize_kernel)
            .with_cuda_graph(self.cuda_graph)
            .with_prefill_chunk_size(self.prefill_chunk_size)
            .with_attention_block_size(self.attention_block_size)
            .with_moe_prefill_config(self.moe_prefill_config)
            .with_moe_decode_options(self.moe_decode_options)
            .with_tracing(self.tracing)
            .with_metrics(self.metrics)
            .with_vram_metrics(self.vram_metrics)
    }
}

/// Converts raw CLI arguments into structured `tau-server` settings.
///
/// The iterator supplies arguments after the binary name, and `binary` is used in usage
/// text. The function returns parsed options or a usage string when flags are missing
/// values, duplicated in an unsupported way, or otherwise invalid.
pub(super) fn parse_args(
    args: impl Iterator<Item = String>,
    binary: &str,
) -> Result<ServerArgs, String> {
    let mut model_path = None;
    let mut chat_template = None;
    let mut chat_template_name = None;
    let mut device = DevicePreference::Auto;
    let mut dtype = DTypePreference::Auto;
    let mut optimize_kernel = true;
    let mut cuda_graph = CudaGraphMode::None;
    let mut prefill_chunk_size = EngineOptions::DEFAULT_PREFILL_CHUNK_SIZE;
    let mut attention_block_size = AttentionBlockSize::default();
    let mut moe_decode_up = None;
    let mut moe_decode_down = None;
    let mut moe_prefill_tile = None;
    let mut moe_prefill_down_tile = None;
    let mut tracing = false;
    let mut metrics = false;
    let mut vram_metrics = false;
    let mut listen_addr = DEFAULT_LISTEN_ADDR.to_owned();
    let mut once = false;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => {
                model_path = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or_else(|| usage(binary))?,
                );
            }
            "--chat-template-name" | "--hf-chat-template-name" => {
                chat_template_name = Some(
                    args.next()
                        .filter(|v| !v.is_empty())
                        .ok_or_else(|| usage(binary))?,
                );
            }
            "--chat-template" => {
                chat_template = Some(PathBuf::from(
                    args.next()
                        .filter(|v| !v.is_empty())
                        .ok_or_else(|| usage(binary))?,
                ));
            }
            "--device" => {
                let value = args.next().ok_or_else(|| usage(binary))?;
                device = DevicePreference::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --device value {value:?}; expected auto, gpu, or \
                         cpu\n\n{}",
                        usage(binary)
                    )
                })?;
            }
            "--dtype" => {
                // Handles --dtype auto|fp32|bf16
                let value = args.next().ok_or_else(|| usage(binary))?;
                dtype = parse_dtype(&value, binary)?;
            }
            "--listen" | "--addr" => {
                listen_addr = args.next().ok_or_else(|| usage(binary))?;
            }
            // Handle one HTTP request and exit; useful for testing.
            "--once" => {
                once = true;
            }
            "--tracing" => {
                tracing = true;
            }
            "--metrics" => {
                metrics = true;
            }
            "--vram-metrics" => {
                vram_metrics = true;
            }
            // Prefer matching optimized CUDA kernels when enabled.
            "--optimize-kernel" => {
                let value = args.next().ok_or_else(|| usage(binary))?;
                optimize_kernel = parse_bool("--optimize-kernel", &value, binary)?;
            }
            // Select CUDA graph capture/replay for supported prefill and decode work.
            "--cuda-graph" => {
                let value = args.next().ok_or_else(|| usage(binary))?;
                cuda_graph = value.parse::<CudaGraphMode>()?;
            }
            "--moe-decode-config" | "--moe-decode-down-config" => {
                let value = args
                    .next()
                    .ok_or_else(|| usage(binary))?
                    .parse::<MoeDecodeSelection>()?;
                if arg == "--moe-decode-config" {
                    moe_decode_up = Some(value);
                } else {
                    moe_decode_down = Some(value);
                }
            }
            "--moe-prefill-config" | "--moe-prefill-down-config" => {
                let value = args
                    .next()
                    .ok_or_else(|| usage(binary))?
                    .parse::<MoePrefillTile>()?;
                if arg == "--moe-prefill-config" {
                    moe_prefill_tile = Some(value);
                } else {
                    moe_prefill_down_tile = Some(value);
                }
            }
            "--attention-block-size" => {
                attention_block_size = args
                    .next()
                    .ok_or_else(|| usage(binary))?
                    .parse::<AttentionBlockSize>()?;
            }
            // Chunked prefill has two purposes:
            // - Reduce peak temporary memory; the KV cache still grows with context.
            // - Give eligible graph regions a fixed chunk shape for capture/replay.
            // Short prompts and partial final chunks always run eagerly.
            "--prefill-chunk-size" => {
                let value = args.next().ok_or_else(|| usage(binary))?;
                prefill_chunk_size = parse_prefill_chunk_size(&value, binary)?;
            }
            "-h" | "--help" => {
                println!("{}", usage(binary));
                std::process::exit(0);
            }
            _ if arg.starts_with("--chat-template-name=")
                || arg.starts_with("--hf-chat-template-name=") =>
            {
                let (_, value) = arg.split_once('=').expect("checked name prefix");
                if value.is_empty() {
                    return Err(usage(binary));
                }
                chat_template_name = Some(value.to_owned());
            }
            _ if arg.starts_with("--chat-template=") => {
                let value = arg.strip_prefix("--chat-template=").unwrap();
                if value.is_empty() {
                    return Err(usage(binary));
                }
                chat_template = Some(PathBuf::from(value));
            }
            _ if arg.starts_with("--dtype=") => {
                // Handles --dtype=auto|fp32|bf16
                let value = arg
                    .strip_prefix("--dtype=")
                    .expect("dtype prefix was checked");
                dtype = parse_dtype(value, binary)?;
            }
            _ if arg.starts_with("--optimize-kernel=") => {
                let value = arg
                    .strip_prefix("--optimize-kernel=")
                    .expect("hyphenated optimize-kernel prefix was checked");
                optimize_kernel = parse_bool("--optimize-kernel", value, binary)?;
            }
            _ if arg.starts_with("--cuda-graph=") => {
                let value = arg.strip_prefix("--cuda-graph=").unwrap();
                cuda_graph = value.parse::<CudaGraphMode>()?;
            }
            _ if arg.starts_with("--prefill-chunk-size=") => {
                prefill_chunk_size = parse_prefill_chunk_size(
                    arg.strip_prefix("--prefill-chunk-size=").unwrap(),
                    binary,
                )?;
            }
            _ if arg.starts_with("--moe-decode-config=")
                || arg.starts_with("--moe-decode-down-config=") =>
            {
                let (flag, value) = arg.split_once('=').unwrap();
                let value = value.parse::<MoeDecodeSelection>()?;
                if flag == "--moe-decode-config" {
                    moe_decode_up = Some(value);
                } else {
                    moe_decode_down = Some(value);
                }
            }
            _ if arg.starts_with("--moe-prefill-config=")
                || arg.starts_with("--moe-prefill-down-config=") =>
            {
                let (flag, value) = arg.split_once('=').unwrap();
                let value = value.parse::<MoePrefillTile>()?;
                if flag == "--moe-prefill-config" {
                    moe_prefill_tile = Some(value);
                } else {
                    moe_prefill_down_tile = Some(value);
                }
            }
            _ if arg.starts_with("--attention-block-size=") => {
                attention_block_size = arg
                    .strip_prefix("--attention-block-size=")
                    .unwrap()
                    .parse::<AttentionBlockSize>()?;
            }
            _ => return Err(usage(binary)),
        }
    }

    let moe_prefill_config = match (moe_prefill_tile, moe_prefill_down_tile) {
        (None, None) => None,
        (Some(up), down) => Some(MoePrefillConfig::new(up, down.unwrap_or(up))?),
        (None, Some(_)) => {
            return Err(
                "--moe-prefill-down-config requires --moe-prefill-config".to_owned()
            );
        }
    };
    let moe_decode_options = if moe_decode_up.is_some() || moe_decode_down.is_some() {
        Some(MoeDecodeOptions {
            gate_up: moe_decode_up.unwrap_or_default(),
            down: moe_decode_down.unwrap_or_default(),
        })
    } else {
        None
    };
    Ok(ServerArgs {
        model_path,
        chat_template,
        chat_template_name,
        device,
        dtype,
        optimize_kernel,
        cuda_graph,
        prefill_chunk_size,
        attention_block_size,
        moe_prefill_config,
        moe_decode_options,
        tracing,
        metrics,
        vram_metrics,
        listen_addr,
        once,
    })
}

/// Parses a nonzero token count before model loading.
fn parse_prefill_chunk_size(value: &str, binary: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| {
            format!(
                "invalid --prefill-chunk-size value {value:?}; \
                 expected a positive integer\n\n{}",
                usage(binary)
            )
        })
}

/// Parses one dtype option value and attaches usage text to invalid input.
fn parse_dtype(value: &str, binary: &str) -> Result<DTypePreference, String> {
    DTypePreference::parse(value).ok_or_else(|| {
        format!(
            "invalid --dtype value {value:?}; expected auto, f32, or bf16\n\n{}",
            usage(binary)
        )
    })
}

/// Parses an explicit boolean and identifies the flag in argument errors.
fn parse_bool(flag: &str, value: &str, binary: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!(
            "invalid {flag} value {value:?}; expected true or false\n\n{}",
            usage(binary)
        )),
    }
}

/// Builds the `tau-server` usage text for help and argument errors.
///
/// The binary name is interpolated into the output string. The function performs no
/// validation and cannot fail.
pub(super) fn usage(binary: &str) -> String {
    format!(
        r#"Usage: {binary} [OPTIONS]

  Serve a local safetensors model with an OpenAI-compatible HTTP API.
  Without --model, run tau-loopback to echo the latest user message.

Model options:
  -m, --model PATH                Local safetensors model directory.
  --chat-template PATH           Override the checkpoint's chat template with
                                  a .jinja file. Default: checkpoint template.
                                  Takes precedence over named selection.
                                  Requires --model; Qwen3.5 dense/MoE only.
  --chat-template-name NAME      Select a named checkpoint chat template.
                                  Collections default to their first entry.
                                  Single strings ignore this setting.
                                  Requires --model; Qwen3.5 dense/MoE only.
  --hf-chat-template-name NAME   Alias for --chat-template-name.
  --device auto|gpu|cpu           Execution device. [default: auto]
                                  auto: CUDA GPU 0, falling back to CPU.
                                  gpu: require CUDA GPU 0; cpu: use CPU.
  --dtype auto|f32|bf16           Model precision. [default: auto]
                                  auto: BF16 on GPU, F32 on CPU.
                                  BF16 requires compatible weights and CUDA.

Execution options:
  --optimize-kernel true|false    Prefer optimized kernels. [default: true]
                                  false: use Candle reference kernels.
  --cuda-graph none|layer|full    CUDA graph policy. [default: none]
                                  none: disable capture and replay.
                                  layer: graphs only within each layer, never
                                  across layers; any layer may remain eager
                                  when more appropriate or faster.
                                  full: maximum useful, supported graph use;
                                  may mix graph and eager execution. Does not
                                  require capturing the whole forward pass.
                                  Requires --model. Supported dense BF16 decode
                                  can use one complete-forward graph; prefill
                                  keeps layer graphs. Unsupported regions
                                  execute eagerly.
  --attention-block-size 32|64|128|256
                                  Context tokens per prepared decode attention
                                  tile, not CUDA threads per block. [default: {attention_block_size}]
                                  Non-default sizes require CUDA/BF16 dense
                                  Qwen3.5, optimized kernels and full graphs.
  --moe-decode-config PROFILE     Gate/up decode: baseline, tiled:N,K,W,P,
                                  mma8:N,K,W,P or auto:N,K,W_max,P.
                                  Logical M=1; auto caps W and reports effective
                                  settings; exact forms reject invalid W.
                                  Omitted decode flags use device defaults.
  --moe-decode-down-config PROFILE
                                  Independent down profile. With one override,
                                  the other projection uses baseline.
                                  Explicit profiles require CUDA/BF16 MoE.
  --moe-prefill-config M,N,K,W,S  Explicit grouped MoE prefill tile; W is warps,
                                  S is pipeline stages. Default: device policy.
                                  M=16|32, N=16|32|64, K=16|32|64|128,
                                  W=1|2|4|8, S=1|2; static shared memory <=48 KiB.
                                  W must divide (M/16)*(N/16).
                                  Requires CUDA/BF16 MoE and optimized kernels.
  --moe-prefill-down-config M,N,K,W,S
                                  Override down projection; must share M.
                                  Default: same as --moe-prefill-config.
  --prefill-chunk-size N          Prefill chunk size in tokens. [default: {chunk_size}]
                                  Must be positive. Full chunks may use graphs;
                                  shorter chunks always execute eagerly.

Logging options:
  --metrics                       Log live decode t/s to stderr. [default: off]
                                  Every ~5s during generation and at completion.
                                  Requires --model; independent of --tracing.
  --tracing                       Log numerical traces to stdout.
                                  Includes token IDs and logit checksums.
                                  Decode phases include CUDA-event timings
                                  and host setup/readback/selection spans.
                                  Adds diagnostic overhead; off for benchmarks.
                                  [default: off; requires --model]
  --vram-metrics                  Sample process VRAM; log peak to stderr.
                                  [default: off; requires --model]
                                  Independent of --metrics and --tracing.

Server options:
  --listen, --addr HOST:PORT      HTTP listen address.
                                  [default: {listen_addr}]
  --once                          Handle one HTTP request, then exit.
                                  [default: off]
  -h, --help                      Show this message and exit.

Examples:
  # GPU serving with live throughput
  {binary} --model ~/models/Qwen3.8-27B \
    --device gpu --dtype bf16 --cuda-graph full --metrics

  # Echo server for API checks
  {binary} --listen {listen_addr}"#,
        binary = binary,
        listen_addr = DEFAULT_LISTEN_ADDR,
        chunk_size = EngineOptions::DEFAULT_PREFILL_CHUNK_SIZE,
        attention_block_size = AttentionBlockSize::default(),
    )
}

#[cfg(test)]
mod tests;
