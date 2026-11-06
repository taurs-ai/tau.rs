//! Decode selection resolved once by model construction, shared by ordinary and
//! prepared execution. No environment lookups or tuning occur in the hot path.

use super::FusedMoeRunner;
use candle::Result;
use tau_cuda_kernels::{MoeDecodeConfig, MoeDecodeOptions, MoeDecodeProfile};

impl FusedMoeRunner {
    /// Resolve startup options on this device before graph/scratch use.
    pub(in crate::components) fn configure_decode(
        &mut self,
        options: impl Into<MoeDecodeOptions>,
    ) -> Result<()> {
        self.configure_decode_options(Some(Into::<MoeDecodeOptions>::into(options)))
    }

    /// Apply the measured device preset when model construction has no override.
    /// Generic runner construction stays baseline for reference/operator callers;
    /// SparseMoeBlock invokes this before preparing ordinary or graph execution.
    pub(in crate::components) fn configure_decode_defaults(&mut self) -> Result<()> {
        self.configure_decode_options(None)
    }

    fn configure_decode_options(
        &mut self,
        options: Option<MoeDecodeOptions>,
    ) -> Result<()> {
        use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
        };
        let stream = self
            .gate_up_weights
            .device()
            .as_cuda_device()?
            .cuda_stream();
        let context = stream.context();
        let capability = context
            .attribute(MAJOR)
            .ok()
            .zip(context.attribute(MINOR).ok());
        let d = self.dimensions;
        let name = context.name().ok();
        let preset = device_default_config(
            capability,
            name.as_deref(),
            d.hidden_size,
            d.intermediate_size,
        );
        let uses_device_default = options.is_none();
        let Some(options) = options.or(preset.map(MoeDecodeOptions::from)) else {
            return Ok(());
        };
        // Automatic layout eligibility is scoped to the locally accepted model
        // geometry and device family. Other devices keep the matching old layout;
        // exact candidates remain available for later target validation.
        let allow_mma8 = auto_layout_eligible(
            capability,
            name.as_deref(),
            d.hidden_size,
            d.intermediate_size,
        );
        // Measured preferences only apply when the request's N/K/stages and warp
        // ceiling permit them. The local down control benefits from fewer warps.
        let gate_up_preferred = preset.map(|config| config.gate_up);
        let down_preferred = preset.map(|config| config.down).or_else(|| {
            allow_mma8.then(|| {
                MoeDecodeProfile::tiled([32, 64, 2, 1])
                    .expect("validated local down preset")
            })
        });
        let mut gate_up = options
            .gate_up
            .resolve_with_preferred(
                capability,
                2 * d.intermediate_size,
                d.hidden_size,
                allow_mma8,
                gate_up_preferred,
            )
            .map_err(candle::Error::Msg)?;
        let mut down = options
            .down
            .resolve_with_preferred(
                capability,
                d.hidden_size,
                d.intermediate_size,
                allow_mma8,
                down_preferred,
            )
            .map_err(candle::Error::Msg)?;
        if uses_device_default {
            gate_up.reason = "device_default";
            down.reason = "device_default";
        }
        self.config = tau_cuda_kernels::MoeDecodeConfig {
            gate_up: gate_up.profile,
            down: down.profile,
        };
        self.resolution = Some([gate_up, down]);
        eprintln!(
            "tau-models MoE decode: gate_up requested={} effective={} mma_columns={} columns_per_warp={} shared_upper_bytes={} reason={}; down requested={} effective={} mma_columns={} columns_per_warp={} shared_upper_bytes={} reason={}",
            gate_up.requested,
            gate_up.profile,
            gate_up.profile.fragment_columns(),
            gate_up.profile.columns_per_warp(),
            gate_up.profile.shared_memory_bytes(),
            gate_up.reason,
            down.requested,
            down.profile,
            down.profile.fragment_columns(),
            down.profile.columns_per_warp(),
            down.profile.shared_memory_bytes(),
            down.reason
        );
        Ok(())
    }
}

/// Device/model geometry with accepted target measurements. The preset serves
/// omitted model options and acts as a preference inside explicit auto ceilings.
/// Other devices/shapes retain baseline defaults until their own validation.
fn device_default_config(
    capability: Option<(i32, i32)>,
    name: Option<&str>,
    hidden: usize,
    intermediate: usize,
) -> Option<MoeDecodeConfig> {
    let measured_device = (capability == Some((9, 0)) && name == Some("NVIDIA H200"))
        || (capability == Some((10, 3)) && name == Some("NVIDIA B300 SXM6 PC"));
    if !measured_device || hidden != 2048 || intermediate != 512 {
        return None;
    }
    let profile = MoeDecodeProfile::mma8([32, 128, 4, 2])
        .expect("validated H200/B300 decode preset");
    Some(MoeDecodeConfig {
        gate_up: profile,
        down: profile,
    })
}

/// Device policy is separate from mathematical support: exact candidates can
/// be tested elsewhere, while automatic mode keeps unmeasured devices/shapes on
/// the established matching layout until their own acceptance measurements.
fn auto_layout_eligible(
    capability: Option<(i32, i32)>,
    name: Option<&str>,
    hidden: usize,
    intermediate: usize,
) -> bool {
    device_default_config(capability, name, hidden, intermediate).is_some()
        || (capability == Some((12, 0))
            && name == Some("NVIDIA GeForce RTX 5060 Ti")
            && hidden == 2048
            && intermediate == 512)
}

#[cfg(test)]
mod tests {
    use super::{auto_layout_eligible, device_default_config};
    #[test]
    fn automatic_policy_is_limited_to_the_measured_device_and_shape() {
        assert!(auto_layout_eligible(
            Some((12, 0)),
            Some("NVIDIA GeForce RTX 5060 Ti"),
            2048,
            512
        ));
        assert!(auto_layout_eligible(
            Some((9, 0)),
            Some("NVIDIA H200"),
            2048,
            512
        ));
        assert!(!auto_layout_eligible(
            Some((12, 0)),
            Some("other"),
            2048,
            512
        ));
        assert!(!auto_layout_eligible(Some((12, 0)), None, 2048, 512));
        assert!(!auto_layout_eligible(
            Some((12, 0)),
            Some("NVIDIA GeForce RTX 5060 Ti"),
            2560,
            512
        ));
    }

    #[test]
    fn default_presets_are_scoped_to_measured_devices_and_geometry() {
        let config =
            device_default_config(Some((9, 0)), Some("NVIDIA H200"), 2048, 512)
                .unwrap();
        assert_eq!(config.gate_up.words(), [2, 32, 128, 4, 2]);
        assert_eq!(config.down, config.gate_up);
        assert_eq!(
            device_default_config(
                Some((10, 3)),
                Some("NVIDIA B300 SXM6 PC"),
                2048,
                512
            ),
            Some(config)
        );
        for (capability, name, hidden, intermediate) in [
            (None, Some("NVIDIA H200"), 2048, 512),
            (Some((12, 0)), Some("NVIDIA GeForce RTX 5060 Ti"), 2048, 512),
            (Some((10, 0)), Some("NVIDIA B200"), 2048, 512),
            (Some((9, 0)), Some("NVIDIA H100"), 2048, 512),
            (Some((9, 0)), None, 2048, 512),
            (Some((9, 0)), Some("NVIDIA H200"), 2560, 512),
            (Some((9, 0)), Some("NVIDIA H200"), 2048, 1024),
            (Some((10, 3)), Some("other"), 2048, 512),
            (Some((10, 3)), Some("NVIDIA B300 SXM6 PC"), 2560, 512),
        ] {
            assert!(
                device_default_config(capability, name, hidden, intermediate).is_none()
            );
        }
    }

    #[test]
    fn measured_auto_preference_respects_dimensions_and_warp_ceiling() {
        let preset =
            device_default_config(Some((9, 0)), Some("NVIDIA H200"), 2048, 512)
                .unwrap();
        let resolve = |value: &str| {
            value
                .parse::<tau_cuda_kernels::MoeDecodeSelection>()
                .unwrap()
                .resolve_with_preferred(
                    Some((9, 0)),
                    1024,
                    2048,
                    true,
                    Some(preset.gate_up),
                )
                .unwrap()
        };
        let selected = resolve("auto:32,128,8,2");
        assert_eq!(selected.profile, preset.gate_up);
        assert_eq!(selected.reason, "measured_preset");
        assert_eq!(
            resolve("auto:32,128,2,2").profile.words(),
            [1, 32, 128, 2, 2]
        );
        assert_eq!(resolve("auto:64,64,8,1").profile.words(), [2, 64, 64, 8, 1]);
        let baseline = resolve("baseline");
        assert_eq!(
            baseline.profile,
            tau_cuda_kernels::MoeDecodeProfile::default()
        );
        assert_eq!(baseline.reason, "exact");
    }
}
