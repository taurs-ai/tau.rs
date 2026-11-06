#pragma once
#include <cuda_runtime.h>
namespace tau_precision {
// Keep softmax and sigmoid on the same FP32 exponential/division contract.
__device__ __forceinline__ float exponential(float value) {
    float out, scaled=__fmul_rn(value,__int_as_float(0x3fb8aa3b));
    asm("ex2.approx.f32 %0,%1;":"=f"(out):"f"(scaled));return out;
}
// Full-range FP32 division preserves finite softmax tails.
__device__ __forceinline__ float divide(float value,float denominator) {
    float out;asm("div.full.f32 %0,%1,%2;":"=f"(out):"f"(value),"f"(denominator));return out;
}
// Preserve the interleaved reduction tree; reassociation changes routing bits.
__device__ __forceinline__ float sum_eight(const float* p) {
    return __fadd_rn(__fadd_rn(__fadd_rn(p[0],p[2]),__fadd_rn(p[4],p[6])),
        __fadd_rn(__fadd_rn(p[1],p[3]),__fadd_rn(p[5],p[7])));
}
// Reuse the exponential/division contract without a BF16 intermediate.
__device__ __forceinline__ float sigmoid(float value) {
    return divide(1.f,__fadd_rn(1.f,exponential(-value)));
}
}
