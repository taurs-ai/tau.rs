//! Command-line device selection for model inference.
//!
//! The server keeps the public choices (`auto`, `gpu`, and `cpu`) separate from
//! Candle's concrete `Device`. This lets `auto` fall back gracefully while `gpu`
//! remains a strict request for accelerator execution.

use tau_engine::Device;

/// Device behavior requested through the `--device` command-line option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DevicePreference {
    /// Prefer GPU 0, but use the CPU when no supported GPU is available.
    Auto,
    /// Require GPU 0 and fail startup rather than falling back to the CPU.
    Gpu,
    /// Always load and run the model on the CPU.
    Cpu,
}

/// Concrete Candle device selected for model loading and inference.
///
/// The label is used in startup diagnostics. A fallback reason is retained only
/// when `auto` tried CUDA and selected the CPU instead.
pub(crate) struct ResolvedDevice {
    device: Device,
    label: &'static str,
    fallback_reason: Option<String>,
}

impl DevicePreference {
    /// Parses the stable command-line spelling of a device preference.
    ///
    /// Unknown values return `None` so the argument parser can attach usage text to
    /// its error message.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "gpu" => Some(Self::Gpu),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    /// Resolves a preference into the Candle device used by the model engine.
    ///
    /// CPU resolution cannot fail. Auto mode records a CUDA error and falls back to
    /// CPU, while GPU mode turns that same error into a startup failure.
    pub(crate) fn resolve(self) -> Result<ResolvedDevice, String> {
        match self {
            Self::Cpu => Ok(ResolvedDevice::cpu()),
            Self::Auto => match try_cuda_device() {
                Ok(device) => Ok(ResolvedDevice::cuda(device)),
                Err(error) => Ok(ResolvedDevice {
                    fallback_reason: Some(error),
                    ..ResolvedDevice::cpu()
                }),
            },
            Self::Gpu => try_cuda_device()
                .map(ResolvedDevice::cuda)
                .map_err(|error| {
                    format!(
                        "--device gpu requested, but no usable CUDA GPU was detected: \
                         {error}"
                    )
                }),
        }
    }
}

impl ResolvedDevice {
    /// Creates a resolved CPU selection without an accelerator fallback.
    fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            label: "cpu",
            fallback_reason: None,
        }
    }

    /// Creates a resolved selection for CUDA GPU 0.
    fn cuda(device: Device) -> Self {
        Self {
            device,
            label: "cuda:0",
            fallback_reason: None,
        }
    }

    /// Reports the selected inference device during model startup.
    ///
    /// Auto fallback includes the original CUDA error so operators can distinguish a
    /// deliberate CPU choice from a missing or unusable accelerator.
    pub(crate) fn log_selection(&self) {
        if let Some(reason) = &self.fallback_reason {
            eprintln!(
                "tau-server device auto selected cpu; CUDA unavailable: {reason}"
            );
        } else {
            eprintln!("tau-server inference device: {}", self.label);
        }
    }

    /// Returns whether inference resolved to Candle's CPU backend.
    pub(crate) fn is_cpu(&self) -> bool {
        self.device.is_cpu()
    }

    /// Consumes the diagnostic wrapper and returns the Candle device.
    pub(crate) fn into_device(self) -> Device {
        self.device
    }
}

/// Attempts to construct Candle's handle for CUDA GPU 0.
///
/// Candle reports runtime initialization failures through the returned error.
fn try_cuda_device() -> Result<Device, String> {
    // CUDA is the first supported accelerator backend. Add ROCm probing here later
    // while preserving the public `--device gpu` mode.
    Device::new_cuda(0).map_err(|source| source.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::parse_args;

    /// The CLI uses auto-selection when `--device` is omitted.
    #[test]
    fn parses_default_device_as_auto() {
        let args = parse_args(std::iter::empty(), "tau-server").unwrap();
        assert_eq!(args.device, DevicePreference::Auto);
    }

    /// Every documented device spelling maps to its matching preference.
    #[test]
    fn parses_explicit_device_modes() {
        for (value, expected) in [
            ("auto", DevicePreference::Auto),
            ("gpu", DevicePreference::Gpu),
            ("cpu", DevicePreference::Cpu),
        ] {
            let args = parse_args(
                ["--device".to_owned(), value.to_owned()].into_iter(),
                "tau-server",
            )
            .unwrap();
            assert_eq!(args.device, expected);
        }
    }

    /// Unsupported device names produce a useful argument error.
    #[test]
    fn rejects_unknown_device_mode() {
        let error = parse_args(
            ["--device".to_owned(), "tpu".to_owned()].into_iter(),
            "tau-server",
        )
        .unwrap_err();
        assert!(error.contains("invalid --device value"));
    }

    /// Explicit CPU selection resolves without a fallback diagnostic.
    #[test]
    fn resolves_cpu_device() {
        let resolved = DevicePreference::Cpu.resolve().unwrap();
        assert_eq!(resolved.label, "cpu");
        assert!(resolved.device.is_cpu());
        assert!(resolved.fallback_reason.is_none());
    }
}
