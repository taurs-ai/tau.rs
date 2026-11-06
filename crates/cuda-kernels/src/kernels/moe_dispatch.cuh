#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Router schema shared by device dispatch and expert projections. Tile retains
// the control kernel geometry; configurable prefill receives its M explicitly.
namespace tau_moe {
constexpr int Experts = 256, TopK = 8, Tile = 16;
constexpr int Status = 2 * Experts, PaddedRows = Status + 1;

// Combine stored BF16 contributions in route-rank order, as used by both native
// prefill and decode. No atomic reduction or intermediate BF16 addition is used.
__device__ __forceinline__ __nv_bfloat16 combine_ranked(
    const __nv_bfloat16* contributions, int width, int column) {
    float sum = 0.f;
    #pragma unroll
    for (int rank = 0; rank < TopK; ++rank)
        sum = __fadd_rn(sum, __bfloat162float(contributions[(size_t)rank * width + column]));
    return __float2bfloat16_rn(sum);
}
}
