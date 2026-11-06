// GDN readout rounds through BF16 before FP32 gated RMSNorm. Keep each
// separately rounded expression and Candle's reduction tree intact.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
__global__ void gated_norm(const float* input, const __nv_bfloat16* gate,
    const float* weight, __nv_bfloat16* output, float epsilon) {
    const int col = threadIdx.x;
    const long long index = static_cast<long long>(blockIdx.x) * 128 + col;
    const float value = __bfloat162float(__float2bfloat16(input[index]));
    __shared__ float squares[128];
    squares[col] = __fmul_rn(value, value);
    for (int stride = 64; stride > 0; stride /= 2) {
        __syncthreads();
        if (col < stride) {
            squares[col] = __fadd_rn(squares[col], squares[col + stride]);
        }
    }
    __syncthreads();
    const float inverse = __fdiv_rn(1.0f,
        sqrtf(__fadd_rn(__fmul_rn(squares[0], 1.0f / 128), epsilon)));
    const float z = __bfloat162float(gate[index]);
    const float activated = __fdiv_rn(z, __fadd_rn(1.0f, expf(-z)));
    const float normalized = __fmul_rn(__fmul_rn(value, inverse), weight[col]);
    output[index] = __float2bfloat16(__fmul_rn(normalized, activated));
}
}  // namespace

extern "C" int tau_launch_gdn_prefill_norm(const float* input, const void* gate,
    const float* weight, void* output, int rows, float epsilon, void* stream) {
    if (!input || !gate || !weight || !output || !stream || rows <= 0 ||
        !std::isfinite(epsilon) || epsilon <= 0) { return -1; }
    gated_norm<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(input,
        static_cast<const __nv_bfloat16*>(gate), weight,
        static_cast<__nv_bfloat16*>(output), epsilon);
    return static_cast<int>(cudaGetLastError());
}
