// Multi-token, four-tap depthwise convolution. Inputs/history are read-only;
// Candle applies SiLU only after this kernel rounds the convolution to BF16.
// Each Q/K/V feature is filtered independently across the current token and its
// three predecessors, providing local context before GDN's recurrent update.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>

namespace {
constexpr int kThreads = 256;

__global__ void bf16_gdn_prefill_convolution(
    const __nv_bfloat16* input,
    const __nv_bfloat16* history,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int channels, int tokens,
    long long input_channel_stride, long long input_token_stride,
    long long history_channel_stride, long long history_token_stride) {
    // One thread writes one [channel, token] result. Every window reads raw
    // projected values, so outputs can run in parallel without depending on
    // other threads' filtered results or modifying shared history.
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<long long>(channels) * tokens) return;
    const int channel = static_cast<int>(index / tokens);
    const int token = static_cast<int>(index % tokens);
    float products[4];
    #pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        // Treat history[4] and input[T] as one sequence without concatenating them.
        // Skip the history-only window at position 0: output token 0 uses
        // history[1], history[2], history[3], input[0]. Later windows slide forward
        // until all four values come from this chunk. The newest value is always
        // input[token], so the filter never reads a future token.
        const long long position = static_cast<long long>(token) + 1 + tap;
        const __nv_bfloat16 value = position < 4
            ? history[channel * history_channel_stride + position * history_token_stride]
            : input[channel * input_channel_stride + (position - 4) * input_token_stride];
        products[tap] = __fmul_rn(__bfloat162float(value),
            __bfloat162float(weight[static_cast<long long>(channel) * 4 + tap]));
    }
    // Retain the accepted pairwise FP32 reduction. cuBLAS reduction order varies
    // with GPU/shape; near BF16 midpoints its rounded result can differ by one
    // BF16 step. Accuracy is checked against unoptimized FP32, not bitwise
    // cuBLAS agreement. Explicit adds keep this kernel's arithmetic unchanged.
    const float sum = __fadd_rn(__fadd_rn(products[0], products[1]),
                               __fadd_rn(products[2], products[3]));
    // Activation runs afterward in Candle. History is also updated separately
    // from raw input, preserving the window across prefill chunks and into decode.
    output[index] = __float2bfloat16_rn(sum);
}
} // namespace

extern "C" int tau_launch_bf16_gdn_prefill_convolution(
    const void* input, const void* history, const void* weight, void* output,
    int channels, int tokens,
    long long input_channel_stride, long long input_token_stride,
    long long history_channel_stride, long long history_token_stride,
    void* stream) {
    const long long count = static_cast<long long>(channels) * tokens;
    if (!input || !history || !weight || !output || channels <= 0 || tokens <= 1
        || count > INT_MAX || input_channel_stride < 0 || input_token_stride < 0
        || history_channel_stride < 0 || history_token_stride < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    bf16_gdn_prefill_convolution<<<static_cast<unsigned>((count + kThreads - 1) / kThreads),
        kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(history),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output), channels, tokens,
        input_channel_stride, input_token_stride, history_channel_stride, history_token_stride);
    return static_cast<int>(cudaGetLastError());
}
