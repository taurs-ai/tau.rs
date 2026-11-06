//! Compiled grouped-prefill geometries shared by launchers, loaders and CLI.
//!
//! These are spatial/loading tiles, not model dimensions. Each specialization
//! retains consecutive K=16 MMA accumulation and the existing BF16 epilogues.

// Native specializations use static shared arrays. Larger allocations require
// a separate dynamic-memory/opt-in implementation, independent of GPU capacity.
const MAX_STATIC_SHARED_BYTES: usize = 48 * 1024;

/// One projection's cooperative CUDA block, selected before scratch allocation.
/// The native launcher uses the same five signed words at its FFI boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct MoePrefillTile {
    m: i32,
    n: i32,
    k: i32,
    warps: i32,
    stages: i32,
}

impl Default for MoePrefillTile {
    /// Retains the accepted single-warp implementation as an explicit control.
    fn default() -> Self {
        Self {
            m: 16,
            n: 16,
            k: 16,
            warps: 1,
            stages: 1,
        }
    }
}

impl MoePrefillTile {
    /// Validate the bounded compiled set without allocating or touching CUDA.
    /// Warps divide the number of 16x16 output fragments, so every warp owns
    /// complete accumulators and no cross-warp numerical reduction is needed.
    pub fn new(values: [i32; 5]) -> Result<Self, String> {
        let [m, n, k, warps, stages] = values;
        if ![16, 32].contains(&m)
            || ![16, 32, 64].contains(&n)
            || ![16, 32, 64, 128].contains(&k)
            || ![1, 2, 4, 8].contains(&warps)
            || ![1, 2].contains(&stages)
            || (m / 16 * (n / 16)) % warps != 0
        {
            return Err(format!(
                "unsupported MoE prefill tile {values:?}; expected \
                M=16|32, N=16|32|64, K=16|32|64|128, warps=1|2|4|8, stages=1|2; \
                warps must divide (M/16)*(N/16)"
            ));
        }
        let tile = Self {
            m,
            n,
            k,
            warps,
            stages,
        };
        if tile.shared_memory_upper_bound() > MAX_STATIC_SHARED_BYTES {
            return Err(format!(
                "MoE prefill tile {tile} exceeds the 48 KiB static shared-memory limit ({} bytes)",
                tile.shared_memory_upper_bound()
            ));
        }
        Ok(tile)
    }

    /// Exact startup configuration and GPU execution-receipt representation.
    pub const fn values(self) -> [i32; 5] {
        [self.m, self.n, self.k, self.warps, self.stages]
    }

    /// Assignment rows used by both dispatch padding and projection worklists.
    pub const fn rows(self) -> usize {
        self.m as usize
    }

    /// Declared cooperative operand/result storage, including the shared-row
    /// skew. Startup validation uses this bound before selecting a kernel;
    /// occupancy also depends on hardware and compiler register allocation.
    /// The legacy control uses less shared memory than this conservative bound.
    pub const fn shared_memory_upper_bound(self) -> usize {
        let [m, n, k, _, stages] = self.values();
        (2 * stages * (m + n) * (k + 8) + 4 * m * n) as usize
    }

    /// Enumerate only supported variants for bounded offline tuning and gates.
    pub fn supported() -> Vec<Self> {
        let mut result = Vec::new();
        for m in [16, 32] {
            for n in [16, 32, 64] {
                for k in [16, 32, 64, 128] {
                    for warps in [1, 2, 4, 8] {
                        for stages in [1, 2] {
                            if let Ok(tile) = Self::new([m, n, k, warps, stages]) {
                                result.push(tile);
                            }
                        }
                    }
                }
            }
        }
        result
    }
}

impl std::fmt::Display for MoePrefillTile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{},{},{},{}",
            self.m, self.n, self.k, self.warps, self.stages
        )
    }
}

impl std::str::FromStr for MoePrefillTile {
    type Err = String;
    /// CLI and numerical probes share this parser; malformed/unsupported values
    /// fail before model allocation rather than silently selecting another tile.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let values = value
            .split(',')
            .map(str::parse::<i32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "expected MoE prefill M,N,K,warps,stages".to_owned())?;
        Self::new(values.try_into().map_err(|_| {
            "expected five MoE prefill values: M,N,K,warps,stages".to_owned()
        })?)
    }
}

/// Separate gate/up and down loading geometries over one shared dispatch plan.
/// Model/engine options retain an optional explicit value; automatic selection
/// stays with the CUDA runner that knows the device architecture.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoePrefillConfig {
    gate_up: MoePrefillTile,
    down: MoePrefillTile,
}

impl MoePrefillConfig {
    /// Rejects inconsistent padding before either projection can consume slots.
    pub fn new(gate_up: MoePrefillTile, down: MoePrefillTile) -> Result<Self, String> {
        if gate_up.rows() != down.rows() {
            return Err("MoE prefill projections must share M".to_owned());
        }
        Ok(Self { gate_up, down })
    }
    pub const fn gate_up(self) -> MoePrefillTile {
        self.gate_up
    }
    pub const fn down(self) -> MoePrefillTile {
        self.down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K=128 must be selectable without allowing a two-stage tile whose declared
    /// buffers exceed the portable static shared-memory allocation contract.
    #[test]
    fn k128_profiles_require_bounded_shared_memory() {
        for profile in ["16,64,128,4,1", "16,64,128,4,2", "32,64,128,8,1"] {
            assert_eq!(profile.parse::<MoePrefillTile>().unwrap().values()[2], 128);
        }
        let error = "32,64,128,8,2".parse::<MoePrefillTile>().unwrap_err();
        assert!(error.contains("48 KiB"), "{error}");
        assert_eq!(
            "16,64,128,4,1"
                .parse::<MoePrefillTile>()
                .unwrap()
                .shared_memory_upper_bound(),
            25_856
        );
        assert_eq!(
            "16,64,128,4,2"
                .parse::<MoePrefillTile>()
                .unwrap()
                .shared_memory_upper_bound(),
            47_616
        );
        assert_eq!(MoePrefillTile::supported().len(), 116);
    }

    #[test]
    fn compiled_tiles_round_trip_and_reject_incomplete_warp_ownership() {
        for tile in MoePrefillTile::supported() {
            assert_eq!(tile.to_string().parse::<MoePrefillTile>().unwrap(), tile);
        }
        for value in [
            "",
            "16,32,64,4",
            "16,32,64,4,2,1",
            "16,16,16,4,2",
            "32,64,64,16,2",
            "16,32,0,1,1",
            "16,32,64,2,3",
            "-16,32,64,2,1",
        ] {
            assert!(value.parse::<MoePrefillTile>().is_err(), "{value}");
        }
        let a = "16,32,64,2,2".parse().unwrap();
        let b = "32,64,32,4,1".parse().unwrap();
        assert!(MoePrefillConfig::new(a, b).is_err());
        assert_eq!(MoePrefillTile::default().values(), [16, 16, 16, 1, 1]);
    }
}
