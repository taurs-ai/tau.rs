// Multi-token FP32 GDN recurrence. Each lane retains one complete state column
// across the sequence. Inputs are normalized/scaled by the reference preparation.
// Separate state input/output makes eager execution and CUDA capture identical.
#include <cuda_runtime.h>

namespace {
// Reproduce Candle fast_sum's 128-element tree, including separately rounded
// products. Compile-time indices keep the retained state in registers. Splitting
// even/odd leaves recreates the tree's first 0+64, then 0+32, ..., finally 0+1.
template<int Count, int Start = 0, int Step = 1>
__device__ __forceinline__ float dot(
    const float (&state)[128], const volatile float* vector) {
    if constexpr (Count == 1) {
        return __fmul_rn(state[Start], vector[Start]);
    } else {
        return __fadd_rn(dot<Count / 2, Start, Step * 2>(state, vector),
                        dot<Count / 2, Start + Step, Step * 2>(state, vector));
    }
}

__global__ void recurrent_prefill(
    const float* packed, const float* initial, float* output,
    int tokens, int heads) {
    const int lane = threadIdx.x;
    const int head = blockIdx.y;
    const int column = blockIdx.x * 32 + lane;
    const long long state_offset = (static_cast<long long>(head) * 128) * 128 + column;
    __shared__ volatile float query[128];
    __shared__ volatile float key[128];
    float state[128];
    #pragma unroll
    for (int row = 0; row < 128; ++row) {
        state[row] = initial[state_offset + row * 128];
    }
    for (int token = 0; token < tokens; ++token) {
        const long long offset = (static_cast<long long>(token) * heads + head) * 386;
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            query[group * 32 + lane] = packed[offset + group * 32 + lane];
            key[group * 32 + lane] = packed[offset + 128 + group * 32 + lane];
        }
        __syncwarp();
        const float alpha = packed[offset + 384];
        const float beta = packed[offset + 385];
        #pragma unroll
        for (int row = 0; row < 128; ++row) {
            state[row] = __fmul_rn(state[row], alpha);
        }
        const float prediction = dot<128>(state, key);
        const float delta = __fmul_rn(
            __fsub_rn(packed[offset + 256 + column], prediction), beta);
        #pragma unroll
        for (int row = 0; row < 128; ++row) {
            state[row] = __fadd_rn(state[row], __fmul_rn(key[row], delta));
        }
        output[(static_cast<long long>(token) * heads + head) * 128 + column] =
            dot<128>(state, query);
        __syncwarp();
    }
    float* final_state = output + static_cast<long long>(tokens) * heads * 128;
    #pragma unroll
    for (int row = 0; row < 128; ++row) {
        final_state[state_offset + row * 128] = state[row];
    }
}
}  // namespace

extern "C" int tau_launch_gdn_prefill_recurrence(
    const float* input, const float* initial, float* output,
    int tokens, int heads, void* stream) {
    if (!input || !initial || !output || !stream || tokens <= 1 || heads <= 0) {
        return -1;
    }
    recurrent_prefill<<<dim3(4, heads), 32, 0,
        static_cast<cudaStream_t>(stream)>>>(input, initial, output, tokens, heads);
    return static_cast<int>(cudaGetLastError());
}
