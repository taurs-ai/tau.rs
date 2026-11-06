//! Requested decode controls and deterministic construction-time resolution.
//!
//! CLI/model options carry this request; native launchers receive only the exact
//! compiled profile in the resolution. Warp budgets never become launch sizes.

use super::{MoeDecodeConfig, MoeDecodeProfile};

/// Startup selection for one routed-expert projection: gate/up or down.
/// `MoeDecodeOptions` carries both selections to model construction, which
/// resolves them before ordinary decode or graph capture. Logical M is fixed at
/// one token; these controls tune how each selected expert's matrix-vector work
/// is distributed and staged on the GPU. Selection does not tune during decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeDecodeSelection {
    /// Keep a concrete backend and N/K/warp/stage profile, including baseline.
    /// Used for reproducible benchmark controls and explicit tuning experiments;
    /// unsupported settings fail instead of changing the selected profile.
    Exact(MoeDecodeProfile),
    /// Resolve `auto:N,K,W_max,P` once for the device and projection dimensions.
    /// N/K/P stay fixed; a measured preset or the legal-profile heuristic chooses
    /// the layout and effective warp count within the ceiling. Benchmark results
    /// determine which settings are fastest on a particular GPU and model shape.
    Auto {
        /// N: output columns computed by one thread block for one expert.
        /// Supported tiles are 32/64/128 columns. Larger tiles reuse the staged
        /// input across more outputs and need fewer blocks, but require more
        /// shared storage/accumulators and can leave too few blocks to fill the
        /// GPU. Smaller tiles expose more independent blocks while repeating
        /// input staging and per-block setup.
        n: i32,
        /// K: input features staged per chunk of the projection's dot product.
        /// Supported chunks are 32/64/128 features. Larger chunks reduce the
        /// number of staging/synchronization rounds, but consume more shared
        /// memory and can reduce the number of resident blocks. This controls
        /// operand staging; FP32 accumulation still proceeds in K16 steps.
        k: i32,
        /// W_max: positive ceiling on warps per block; each warp has 32 threads.
        /// Effective W comes from legal 1/2/4/8-warp profiles and may be smaller
        /// because of fragment ownership, resource limits or a measured preset.
        /// Each warp owns N / effective_W output columns. More warps reduce work
        /// per warp, but consume more thread capacity per block and can reduce
        /// block residency; using the entire budget need not be fastest.
        max_warps: i32,
        /// P: number of shared-memory operand buffers/async-copy pipeline stages.
        /// Supported values are one or two. Two stages allow the next chunk's
        /// memory loads to overlap computation on the current chunk, potentially
        /// hiding load latency. Extra buffering consumes more shared memory and
        /// can lower block residency; short dot products may offer little overlap.
        stages: i32,
    },
}
impl Default for MoeDecodeSelection {
    fn default() -> Self {
        Self::Exact(MoeDecodeProfile::default())
    }
}
impl From<MoeDecodeProfile> for MoeDecodeSelection {
    fn from(profile: MoeDecodeProfile) -> Self {
        Self::Exact(profile)
    }
}
impl std::fmt::Display for MoeDecodeSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact(profile) => profile.fmt(f),
            Self::Auto {
                n,
                k,
                max_warps,
                stages,
            } => write!(f, "auto:{n},{k},{max_warps},{stages}"),
        }
    }
}
impl std::str::FromStr for MoeDecodeSelection {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        let Some(values) = value.strip_prefix("auto:") else {
            return value.parse().map(Self::Exact);
        };
        let values = values
            .split(',')
            .map(str::parse::<i32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "expected auto:N,K,max_warps,stages")?;
        let [n, k, max_warps, stages]: [i32; 4] = values
            .try_into()
            .map_err(|_| "expected four auto values N,K,max_warps,stages")?;
        if max_warps <= 0
            || !MoeDecodeProfile::supported().iter().any(|p| {
                let [_, pn, pk, _, ps] = p.words();
                p.words()[0] != 0 && pn == n && pk == k && ps == stages
            })
        {
            return Err("unsupported automatic decode dimensions/stages or nonpositive warp budget".to_owned());
        }
        Ok(Self::Auto {
            n,
            k,
            max_warps,
            stages,
        })
    }
}

/// Effective launch and selection explanation retained for startup diagnostics,
/// validation receipts and benchmark deduplication. The native ABI uses `profile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDecodeResolution {
    pub requested: MoeDecodeSelection,
    pub profile: MoeDecodeProfile,
    pub reason: &'static str,
}
impl MoeDecodeSelection {
    /// Resolve at construction without GPU allocation or tuning. The caller
    /// supplies device geometry
    /// and whether the new layout is locally accepted for automatic use there.
    /// Unknown hardware retains the old matching layout; N/K/stages never change.
    pub fn resolve(
        self,
        capability: Option<(i32, i32)>,
        columns: usize,
        inner: usize,
        mma8_accepted: bool,
    ) -> Result<MoeDecodeResolution, String> {
        self.resolve_with_preferred(capability, columns, inner, mma8_accepted, None)
    }

    /// A measured preset may beat the largest-warp heuristic. It is considered
    /// only if it is in the same legal N/K/stage/budget/device candidate set.
    pub fn resolve_with_preferred(
        self,
        capability: Option<(i32, i32)>,
        columns: usize,
        inner: usize,
        mma8_accepted: bool,
        preferred: Option<MoeDecodeProfile>,
    ) -> Result<MoeDecodeResolution, String> {
        let mut reason = "exact";
        let profile = match self {
            Self::Exact(profile) => profile,
            Self::Auto {
                n,
                k,
                max_warps,
                stages,
            } => {
                let candidates: Vec<_> = MoeDecodeProfile::supported()
                    .into_iter()
                    .filter(|p| {
                        let [backend, pn, pk, w, ps] = p.words();
                        backend != 0
                            && pn == n
                            && pk == k
                            && ps == stages
                            && w <= max_warps
                            && (backend != 2 || mma8_accepted)
                            && p.validate_device_shape(capability, columns, inner)
                                .is_ok()
                    })
                    .collect();
                if let Some(preferred) = preferred.filter(|p| candidates.contains(p)) {
                    reason = "measured_preset";
                    preferred
                } else {
                    reason = "largest_legal_warps";
                    candidates
                        .into_iter()
                        .max_by_key(|p| {
                            let [backend, _, _, w, _] = p.words();
                            (w, -backend)
                        })
                        .ok_or_else(|| {
                            "no legal profile satisfies the explicit decode request"
                                .to_owned()
                        })?
                }
            }
        };
        profile.validate_device_shape(capability, columns, inner)?;
        Ok(MoeDecodeResolution {
            requested: self,
            profile,
            reason,
        })
    }
}

/// Startup options for gate/up and down, carried by the engine and model loader.
/// Model construction resolves each selection to a device launch profile.
/// Existing exact config builders convert here, retaining one source of settings.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoeDecodeOptions {
    pub gate_up: MoeDecodeSelection,
    pub down: MoeDecodeSelection,
}
impl From<MoeDecodeConfig> for MoeDecodeOptions {
    fn from(config: MoeDecodeConfig) -> Self {
        Self {
            gate_up: MoeDecodeSelection::from(config.gate_up),
            down: MoeDecodeSelection::from(config.down),
        }
    }
}
impl MoeDecodeOptions {
    /// Legacy exact-option accessor. Automatic selections need device resolution;
    /// callers inspect them through the options accessors. Returning None here
    /// never drives native eligibility internally.
    pub fn exact_config(self) -> Option<MoeDecodeConfig> {
        match (self.gate_up, self.down) {
            (MoeDecodeSelection::Exact(gate_up), MoeDecodeSelection::Exact(down)) => {
                Some(MoeDecodeConfig { gate_up, down })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caps_budget_without_changing_dimensions_and_deduplicates_effective_profiles() {
        let resolve = |v: &str| {
            v.parse::<MoeDecodeSelection>()
                .unwrap()
                .resolve(Some((12, 0)), 1024, 2048, false)
                .unwrap()
        };
        assert_eq!(resolve("auto:32,64,8,2").profile.words(), [1, 32, 64, 2, 2]);
        assert_eq!(resolve("auto:32,64,3,2").profile.words(), [1, 32, 64, 2, 2]);
        assert_eq!(
            resolve("auto:32,64,8,2").profile,
            resolve("auto:32,64,64,2").profile
        );
        assert_eq!(resolve("auto:64,64,1,1").profile.words(), [1, 64, 64, 1, 1]);
    }
    #[test]
    fn finer_layout_can_increase_warps_only_when_accepted_and_within_budget() {
        let request = "auto:32,64,8,2".parse::<MoeDecodeSelection>().unwrap();
        let resolved = request.resolve(Some((12, 0)), 1024, 2048, true).unwrap();
        assert_eq!(resolved.profile.words(), [2, 32, 64, 4, 2]);
        assert_eq!(resolved.profile.columns_per_warp(), 8);
        let limited = "auto:32,64,3,2"
            .parse::<MoeDecodeSelection>()
            .unwrap()
            .resolve(Some((12, 0)), 1024, 2048, true)
            .unwrap();
        assert_eq!(limited.profile.words(), [1, 32, 64, 2, 2]);
        let resource = "auto:128,64,8,2".parse::<MoeDecodeSelection>().unwrap();
        assert!(resource.resolve(Some((12, 0)), 2048, 512, false).is_err());
        assert_eq!(
            resource
                .resolve(Some((12, 0)), 2048, 512, true)
                .unwrap()
                .profile
                .words(),
            [2, 128, 64, 8, 2]
        );
    }

    #[test]
    fn measured_preference_never_escapes_request_constraints() {
        let request = "auto:32,64,8,1".parse::<MoeDecodeSelection>().unwrap();
        let preferred = MoeDecodeProfile::tiled([32, 64, 2, 1]).unwrap();
        let result = request
            .resolve_with_preferred(Some((12, 0)), 2048, 512, true, Some(preferred))
            .unwrap();
        assert_eq!(result.profile, preferred);
        assert_eq!(result.reason, "measured_preset");
        let limited = "auto:32,64,1,1".parse::<MoeDecodeSelection>().unwrap();
        assert_eq!(
            limited
                .resolve_with_preferred(Some((12, 0)), 2048, 512, true, Some(preferred))
                .unwrap()
                .profile
                .words(),
            [1, 32, 64, 1, 1]
        );
        let wrong = MoeDecodeProfile::mma8([64, 64, 8, 1]).unwrap();
        assert_ne!(
            request
                .resolve_with_preferred(Some((12, 0)), 2048, 512, true, Some(wrong))
                .unwrap()
                .profile,
            wrong
        );
    }

    #[test]
    fn rejects_bad_budgets_resources_devices_and_keeps_exact_semantics() {
        for v in [
            "auto:32,64,0,1",
            "auto:32,64,-1,1",
            "auto:33,64,4,1",
            "auto:32,0,4,1",
            "auto:32,64,4,3",
            "auto:128,128,4,2",
            "auto:32,64,4",
            "tiled:32,64,4,1",
        ] {
            assert!(v.parse::<MoeDecodeSelection>().is_err(), "{v}");
        }
        let p = "auto:32,64,8,2".parse::<MoeDecodeSelection>().unwrap();
        assert!(p.resolve(Some((7, 5)), 1024, 2048, false).is_err());
        assert!(p.resolve(None, 1024, 2048, false).is_err());
        assert!(p.resolve(Some((12, 0)), 0, 2048, false).is_err());
        let options = MoeDecodeOptions {
            gate_up: p,
            down: MoeDecodeSelection::default(),
        };
        assert!(options.exact_config().is_none());
        let exact = MoeDecodeConfig::default();
        assert_eq!(MoeDecodeOptions::from(exact).exact_config(), Some(exact));
    }
}
