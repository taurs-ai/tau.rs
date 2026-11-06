//! Construction-time controls for single-token routed projections.
//!
//! Server startup and fixture replay share this catalog. Logical M is always one;
//! tiled kernels pad to a fixed 16-row MMA representation and retain K16 FP32
//! accumulation. Gate/up and down choose independently without per-token tuning.

/// One compiled decode projection. Its five words also identify actual device
/// execution in test receipts: backend (0 baseline, 1 WMMA16, 2 MMA8), N, staged K, warps,
/// and pipeline stages. Fields stay private so uncompiled profiles cannot launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct MoeDecodeProfile {
    backend: i32,
    n: i32,
    k: i32,
    warps: i32,
    stages: i32,
}

impl Default for MoeDecodeProfile {
    fn default() -> Self {
        Self {
            backend: 0,
            n: 16,
            k: 16,
            warps: 1,
            stages: 1,
        }
    }
}

impl MoeDecodeProfile {
    /// Validate compiled native shapes before allocating model scratch. Warps
    /// own complete output fragments; no split-K or cross-warp sum is introduced.
    pub fn tiled(values: [i32; 4]) -> Result<Self, String> {
        Self::compiled(1, values)
    }

    /// Native eight-column fragments for exact single-token experiments. The
    /// same projection kernel/staging pipeline serves both expert projections.
    pub fn mma8(values: [i32; 4]) -> Result<Self, String> {
        Self::compiled(2, values)
    }

    fn compiled(backend: i32, values: [i32; 4]) -> Result<Self, String> {
        let [n, k, warps, stages] = values;
        if ![32, 64, 128].contains(&n)
            || ![32, 64, 128].contains(&k)
            || ![1, 2, 4, 8].contains(&warps)
            || ![1, 2].contains(&stages)
            || (n / if backend == 2 { 8 } else { 16 }) % warps != 0
        {
            return Err(format!(
                "unsupported decode profile {values:?}; N=32|64|128, K=32|64|128, warps=1|2|4|8 dividing N/{}, stages=1|2",
                if backend == 2 { 8 } else { 16 }
            ));
        }
        let profile = Self {
            backend,
            n,
            k,
            warps,
            stages,
        };
        if profile.shared_memory_bytes() > 48 * 1024 {
            return Err(
                "decode profile exceeds 48 KiB static shared-memory limit".to_owned()
            );
        }
        Ok(profile)
    }

    pub const fn words(self) -> [i32; 5] {
        [self.backend, self.n, self.k, self.warps, self.stages]
    }

    /// Output ownership used by construction diagnostics and benchmark identity.
    pub const fn columns_per_warp(self) -> i32 {
        self.n / self.warps
    }
    pub const fn fragment_columns(self) -> i32 {
        if self.backend == 2 { 8 } else { 16 }
    }

    /// Conservative bound for BF16 operand tiles and the layout's FP32 result.
    /// The finer layout stores directly, so its unused result can be optimized
    /// out. Startup rejects excessive storage; occupancy still needs measurement.
    pub const fn shared_memory_bytes(self) -> usize {
        let rows = if self.backend == 2 { 1 } else { 16 };
        (2 * self.stages * (rows + self.n) * (self.k + 8) + 4 * rows * self.n) as usize
    }

    /// Explicit candidates require async-copy capable hardware. Both kernels
    /// mask output/K tails; misaligned operand rows use scalar staging. Automatic
    /// selection is resolved separately; omitted overrides retain baseline.
    pub fn validate_device_shape(
        self,
        capability: Option<(i32, i32)>,
        columns: usize,
        inner: usize,
    ) -> Result<(), String> {
        if columns == 0
            || inner == 0
            || columns > i32::MAX as usize
            || inner > i32::MAX as usize
        {
            return Err("invalid decode projection dimensions".to_owned());
        }
        if self.backend != 0 && capability.is_none_or(|(major, _)| major < 8) {
            return Err(
                "explicit native decode profile requires SM80 or newer; no silent fallback"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// Bounded offline tuning catalog; all listed variants have compiled kernels.
    pub fn supported() -> Vec<Self> {
        let mut profiles = vec![Self::default()];
        for backend in [1, 2] {
            for n in [32, 64, 128] {
                for k in [32, 64, 128] {
                    for warps in [1, 2, 4, 8] {
                        for stages in [1, 2] {
                            if let Ok(p) =
                                Self::compiled(backend, [n, k, warps, stages])
                            {
                                profiles.push(p);
                            }
                        }
                    }
                }
            }
        }
        profiles
    }
}

impl std::fmt::Display for MoeDecodeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.backend == 0 {
            f.write_str("baseline")
        } else {
            write!(
                f,
                "{}:{},{},{},{}",
                if self.backend == 2 { "mma8" } else { "tiled" },
                self.n,
                self.k,
                self.warps,
                self.stages
            )
        }
    }
}
impl std::str::FromStr for MoeDecodeProfile {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        if value == "baseline" {
            return Ok(Self::default());
        }
        let (backend, fields) = if let Some(v) = value.strip_prefix("tiled:") {
            (1, v)
        } else if let Some(v) = value.strip_prefix("mma8:") {
            (2, v)
        } else {
            return Err("expected baseline, tiled:N,K,W,P or mma8:N,K,W,P".to_owned());
        };
        let values = fields
            .split(',')
            .map(str::parse::<i32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "expected integer decode N,K,warps,stages")?;
        Self::compiled(
            backend,
            values
                .try_into()
                .map_err(|_| "expected four decode values N,K,warps,stages")?,
        )
    }
}

/// Per-projection startup selection. None in load options uses the unchanged
/// automatic baseline; an explicit value must execute or fail before capture.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoeDecodeConfig {
    pub gate_up: MoeDecodeProfile,
    pub down: MoeDecodeProfile,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_catalog_rejects_resource_overflow_and_partial_warp_ownership() {
        for p in MoeDecodeProfile::supported() {
            assert_eq!(p.to_string().parse::<MoeDecodeProfile>().unwrap(), p);
            assert!(p.shared_memory_bytes() <= 48 * 1024);
        }
        for p in [
            "",
            "tiled:64,64,3,1",
            "tiled:32,64,4,1",
            "tiled:128,128,8,2",
            "tiled:64,64,4,0",
            "tiled:64,64,4",
            "tiled:64,0,4,1",
            "baseline:64",
        ] {
            assert!(p.parse::<MoeDecodeProfile>().is_err(), "{p}");
        }
    }
    #[test]
    fn explicit_candidates_fail_unsupported_devices_while_auto_keeps_baseline() {
        let p: MoeDecodeProfile = "tiled:64,64,4,2".parse().unwrap();
        for capability in [None, Some((7, 5))] {
            assert!(p.validate_device_shape(capability, 1024, 2048).is_err());
            MoeDecodeConfig::default()
                .gate_up
                .validate_device_shape(capability, 1024, 2048)
                .unwrap();
        }
        p.validate_device_shape(Some((12, 0)), 33, 29).unwrap();
        assert!(p.validate_device_shape(Some((12, 0)), 0, 29).is_err());
        assert_eq!(
            MoeDecodeConfig::default().gate_up.words(),
            [0, 16, 16, 1, 1]
        );
    }
}
