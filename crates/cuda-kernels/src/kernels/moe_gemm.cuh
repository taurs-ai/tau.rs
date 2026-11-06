#pragma once
#include <cuda_runtime.h>

namespace tau_moe {
// Matches Rust's repr(C) MoePrefillTile. The prefill composer passes validated
// startup choices; launchers still reject every uncompiled combination.
struct GemmTile { int m, n, k, warps, stages; };

// Keep the Rust catalog and native instantiation guard on the same portable
// static-allocation contract. Opt-in dynamic storage is a separate kernel policy.
constexpr int MaxStaticSharedBytes = 48 * 1024;
template<int M, int N, int K, int S>
constexpr int shared_memory_upper_bound() {
    return 2 * S * (M + N) * (K + 8) + 4 * M * N;
}
}

extern "C" int tau_launch_moe_prefill_gemm(const void*, const void*, const float*,
    const int*, const int*, void*, int, int, int, int, int,
    const tau_moe::GemmTile*, int*, void*);
