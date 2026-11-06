// Read strided BF16 convolution output directly. Normalize Q/K and map grouped
// heads while packing the FP32 recurrence input; no expanded Q/K tensors.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
__global__ void prepare(
    const __nv_bfloat16* mixed, const float* alpha,
    const __nv_bfloat16* beta, float* output, int tokens,
    int key_heads, int value_heads, long long token_stride,
    long long channel_stride, int round_normalized) {
    const int row = threadIdx.x;
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int key_head = head / (value_heads / key_heads);
    const long long base = static_cast<long long>(token) * token_stride;
    const float q = __bfloat162float(mixed[base +
        (key_head * 128 + row) * channel_stride]);
    const float k = __bfloat162float(mixed[base +
        ((key_heads + key_head) * 128 + row) * channel_stride]);
    __shared__ float qs[128], ks[128];
    qs[row] = __fmul_rn(q, q);
    ks[row] = __fmul_rn(k, k);
    for (int stride = 64; stride > 0; stride /= 2) {
        __syncthreads();
        if (row < stride) {
            qs[row] = __fadd_rn(qs[row], qs[row + stride]);
            ks[row] = __fadd_rn(ks[row], ks[row + stride]);
        }
    }
    __syncthreads();
    const long long offset = (static_cast<long long>(token) * value_heads + head) * 386;
    float normalized_q = __fdiv_rn(q, sqrtf(__fadd_rn(qs[0], 1e-6f)));
    float normalized_k = __fdiv_rn(k, sqrtf(__fadd_rn(ks[0], 1e-6f)));
    // Reference prefill materializes normalized vectors in BF16 before query
    // scaling. Decode normalizes directly inside the FP32 recurrence instead.
    if (round_normalized) {
        normalized_q = __bfloat162float(__float2bfloat16_rn(normalized_q));
        normalized_k = __bfloat162float(__float2bfloat16_rn(normalized_k));
    }
    output[offset + row] = __fmul_rn(normalized_q,
        static_cast<float>(1.0 / std::sqrt(128.0)));
    output[offset + 128 + row] = normalized_k;
    output[offset + 256 + row] = __bfloat162float(mixed[base +
        ((2 * key_heads + head) * 128 + row) * channel_stride]);
    if (row == 0) {
        output[offset + 384] = alpha[token * value_heads + head];
        output[offset + 385] = __bfloat162float(beta[token * value_heads + head]);
    }
}
}  // namespace

extern "C" int tau_launch_gdn_prefill_prepare(
    const void* mixed, const float* alpha, const void* beta, float* output,
    int tokens, int key_heads, int value_heads,
    long long token_stride, long long channel_stride, int round_normalized, void* stream) {
    if (!mixed || !alpha || !beta || !output || !stream || tokens <= 1 ||
        key_heads <= 0 || value_heads <= 0 || value_heads % key_heads != 0 ||
        token_stride < 0 || channel_stride < 0) { return -1; }
    prepare<<<dim3(tokens, value_heads), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(mixed), alpha,
        static_cast<const __nv_bfloat16*>(beta), output, tokens,
        key_heads, value_heads, token_stride, channel_stride, round_normalized);
    return static_cast<int>(cudaGetLastError());
}
