//! Runtime dtype selection for model-backed server inference.

use tau_engine::ModelDType;

/// Dtype behavior requested through the `--dtype` command-line option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DTypePreference {
    /// Use the backend default: F32 on CPU and BF16 on accelerators.
    Auto,
    /// Force 32-bit floating-point model execution.
    F32,
    /// Force brain floating-point model execution.
    BF16,
}

impl DTypePreference {
    /// Parses the stable command-line spelling of a dtype preference.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "f32" => Some(Self::F32),
            "bf16" => Some(Self::BF16),
            _ => None,
        }
    }

    /// Resolves the preference against the selected inference backend.
    ///
    /// Auto chooses F32 on CPU and BF16 on accelerator backends. Explicit BF16
    /// rejects CPU so the server fails before entering model loading.
    pub(crate) fn resolve(self, is_cpu: bool) -> Result<ModelDType, String> {
        match self {
            Self::Auto => Ok(if is_cpu {
                ModelDType::F32
            } else {
                ModelDType::BF16
            }),
            Self::F32 => Ok(ModelDType::F32),
            Self::BF16 if is_cpu => Err(Self::cpu_bf16_error()),
            Self::BF16 => Ok(ModelDType::BF16),
        }
    }

    /// Returns the startup error for an explicit CPU/BF16 combination.
    pub(crate) fn cpu_bf16_error() -> String {
        "--dtype bf16 is not supported with --device cpu; use --dtype auto or \
         --dtype f32"
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{ServerArgs, parse_args};
    use crate::device::DevicePreference;
    use crate::load_engine;
    use tau_server::DEFAULT_LISTEN_ADDR;

    /// Confirms an omitted dtype option selects automatic resolution.
    #[test]
    fn parses_default_dtype_as_auto() {
        let args = parse_args(std::iter::empty(), "tau-server").unwrap();
        assert_eq!(args.dtype, DTypePreference::Auto);
    }

    /// Confirms automatic dtype resolution chooses safe CPU and GPU defaults.
    #[test]
    fn resolves_auto_dtype_for_the_selected_backend() {
        assert_eq!(
            DTypePreference::Auto.resolve(true).unwrap(),
            ModelDType::F32
        );
        assert_eq!(
            DTypePreference::Auto.resolve(false).unwrap(),
            ModelDType::BF16
        );
        assert_eq!(DTypePreference::F32.resolve(true).unwrap(), ModelDType::F32);
    }

    /// Accepts every dtype spelling in separated and equals-style CLI forms.
    #[test]
    fn parses_explicit_dtype_modes_with_both_cli_forms() {
        for (value, expected) in [
            ("auto", DTypePreference::Auto),
            ("f32", DTypePreference::F32),
            ("bf16", DTypePreference::BF16),
        ] {
            let separated = parse_args(
                ["--dtype".to_owned(), value.to_owned()].into_iter(),
                "tau-server",
            )
            .unwrap();
            assert_eq!(separated.dtype, expected);

            let equals =
                parse_args([format!("--dtype={value}")].into_iter(), "tau-server")
                    .unwrap();
            assert_eq!(equals.dtype, expected);
        }
    }

    /// Rejects dtype names that the runtime cannot resolve.
    #[test]
    fn rejects_unknown_dtype_mode() {
        let error = parse_args(
            ["--dtype".to_owned(), "f16".to_owned()].into_iter(),
            "tau-server",
        )
        .unwrap_err();
        assert!(error.contains("invalid --dtype value"));
    }

    /// Fails startup before loading a model for the unsupported CPU/BF16 pairing.
    #[test]
    fn startup_rejects_explicit_bf16_with_cpu() {
        let args = ServerArgs {
            model_path: None,
            chat_template: None,
            chat_template_name: None,
            device: DevicePreference::Cpu,
            dtype: DTypePreference::BF16,
            optimize_kernel: true,
            cuda_graph: tau_engine::CudaGraphMode::None,
            prefill_chunk_size: tau_engine::EngineOptions::DEFAULT_PREFILL_CHUNK_SIZE,
            attention_block_size: tau_engine::AttentionBlockSize::default(),
            moe_prefill_config: None,
            moe_decode_options: None,
            tracing: false,
            metrics: false,
            vram_metrics: false,
            listen_addr: DEFAULT_LISTEN_ADDR.to_owned(),
            once: true,
        };

        let result = load_engine(&args);

        match result {
            Ok(_) => panic!("server unexpectedly accepted BF16 on CPU"),
            Err(error) => {
                assert!(error.contains("--dtype bf16 is not supported"));
                assert!(error.contains("--dtype auto or --dtype f32"));
            }
        }
    }
}
