#pragma once
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

namespace tau_moe {
using Bf16 = __nv_bfloat16;
// Shared operand loader for cooperative prefill and single-token decode.
// Callers commit/wait pipeline groups and keep source allocations alive.
// Aligned real-model operands use asynchronous 16-byte copies. Odd shapes and
// padded rows use scalar masks; every consumed shared value is initialized.
__device__ __forceinline__ void copy_eight(Bf16* dst, const Bf16* src,
    bool valid_row, int valid_k) {
    if (valid_row && valid_k >= 8 && (reinterpret_cast<uintptr_t>(src) & 15) == 0)
        __pipeline_memcpy_async(dst, src, 16);
    else {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            dst[j] = valid_row && j < valid_k ? src[j] : __float2bfloat16_rn(0.f);
    }
}

}
