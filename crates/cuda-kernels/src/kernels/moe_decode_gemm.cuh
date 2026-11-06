#pragma once
#include <cuda_runtime.h>
namespace tau_moe {
// Matches repr(C) MoeDecodeProfile. M=1 logically, with a fixed MMA row tile 16.
struct DecodeProfile { int backend, n, k, warps, stages; };
}
extern "C" int tau_launch_moe_decode_gemm(const void*, const void*, void*,
    int, int, int, const tau_moe::DecodeProfile*, int*, void*);
