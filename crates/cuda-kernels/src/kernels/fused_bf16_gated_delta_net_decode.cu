// Stateful single-token Gated DeltaNet decode kernels with BF16 input/output.
//
// Projected inputs, convolution weights/history, dt_bias, and convolved Q/K/V
// are BF16. Negative-A, normalization weights, and recurrent state are FP32.
//
// The first kernel updates the short depthwise-convolution history and produces
// activated Q/K/V vectors. For 128x128 heads, four independent value-column tiles
// update each FP32 state matrix; a final kernel normalizes the complete head.
// Other dimensions retain the original fused recurrence/normalization kernel.
// Both paths round recurrence output through BF16 before gated RMSNorm.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include "fp32_precision.cuh"

namespace {

constexpr int kBlockSize = 256;

// Evaluates SiLU in FP32 for both the convolution and normalization gates.
__device__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// Evaluates the stable softplus form used by the Candle reference path.
__device__ float stable_softplus(float value) {
    return fmaxf(value, 0.0f) + logf(expf(-fabsf(value)) + 1.0f);
}

// Reduces one FP32 value per thread to a block-wide sum.
__device__ float block_sum(float value, float* scratch) {
    const int thread = static_cast<int>(threadIdx.x);
    scratch[thread] = value;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float sum = scratch[0];
    // Callers reuse the same scratch array immediately. Keep every thread from
    // beginning that next reduction until the current sum has been observed.
    __syncthreads();
    return sum;
}

// Shifts every causal-convolution channel, appends the current projection, and
// computes only the newest depthwise output. Each channel is independent.
__global__ void update_causal_convolution(
    const __nv_bfloat16* qkvz,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state,
    __nv_bfloat16* convolved_qkv,
    int conv_width,
    int conv_kernel_size) {
    const int channel =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= conv_width) {
        return;
    }

    __nv_bfloat16* channel_state =
        conv_state + static_cast<long long>(channel) * conv_kernel_size;
    const __nv_bfloat16* channel_weight =
        conv_weight + static_cast<long long>(channel) * conv_kernel_size;
    float convolved = 0.0f;
    for (int tap = 0; tap < conv_kernel_size - 1; ++tap) {
        const __nv_bfloat16 shifted = channel_state[tap + 1];
        channel_state[tap] = shifted;
        convolved += __bfloat162float(shifted) *
                     __bfloat162float(channel_weight[tap]);
    }
    const __nv_bfloat16 current = qkvz[channel];
    channel_state[conv_kernel_size - 1] = current;
    convolved += __bfloat162float(current) *
                 __bfloat162float(channel_weight[conv_kernel_size - 1]);
    convolved_qkv[channel] = __float2bfloat16(silu(convolved));
}

// Updates one value head's recurrent matrix and fuses the output's gated RMSNorm.
// Shared memory holds normalized Q/K vectors, per-value deltas and rounded output,
// while the persistent matrix remains in global FP32 storage.
__global__ void recurrent_update_and_gated_rms_norm(
    const __nv_bfloat16* qkvz,
    const __nv_bfloat16* ba,
    const __nv_bfloat16* convolved_qkv,
    const __nv_bfloat16* dt_bias,
    const float* negative_a,
    const float* norm_weight,
    float* recurrent_state,
    __nv_bfloat16* output,
    int conv_width,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    float epsilon) {
    extern __shared__ float shared_values[];
    float* query = shared_values;
    float* key = query + key_head_dim;
    float* delta = key + key_head_dim;
    float* rounded_output = delta + value_head_dim;
    __shared__ float reduction[kBlockSize];

    const int thread = static_cast<int>(threadIdx.x);
    const int value_head = static_cast<int>(blockIdx.x);
    const int groups = num_value_heads / num_key_heads;
    const int key_head = value_head / groups;
    const int key_width = num_key_heads * key_head_dim;
    const int value_width = num_value_heads * value_head_dim;
    const int query_offset = key_head * key_head_dim;
    const int key_offset = key_width + query_offset;
    const int value_offset = 2 * key_width + value_head * value_head_dim;

    float query_square_sum = 0.0f;
    float key_square_sum = 0.0f;
    for (int column = thread; column < key_head_dim; column += kBlockSize) {
        const float query_value =
            __bfloat162float(convolved_qkv[query_offset + column]);
        const float key_value =
            __bfloat162float(convolved_qkv[key_offset + column]);
        query_square_sum += query_value * query_value;
        key_square_sum += key_value * key_value;
    }
    const float query_norm = block_sum(query_square_sum, reduction);
    const float key_norm = block_sum(key_square_sum, reduction);
    const float query_scale =
        rsqrtf(query_norm + 1e-6f) * rsqrtf(static_cast<float>(key_head_dim));
    const float key_scale = rsqrtf(key_norm + 1e-6f);
    for (int column = thread; column < key_head_dim; column += kBlockSize) {
        query[column] =
            __bfloat162float(convolved_qkv[query_offset + column]) * query_scale;
        key[column] =
            __bfloat162float(convolved_qkv[key_offset + column]) * key_scale;
    }

    const float beta_unrounded =
        1.0f / (1.0f + expf(-__bfloat162float(ba[value_head])));
    const float beta = __bfloat162float(__float2bfloat16(beta_unrounded));
    const float step = stable_softplus(
        __bfloat162float(ba[num_value_heads + value_head]) +
        __bfloat162float(dt_bias[value_head]));
    const float alpha = expf(step * negative_a[value_head]);
    float* head_state = recurrent_state +
        static_cast<long long>(value_head) * key_head_dim * value_head_dim;
    __syncthreads();

    if (thread < value_head_dim) {
        float memory = 0.0f;
        for (int row = 0; row < key_head_dim; ++row) {
            const long long index =
                static_cast<long long>(row) * value_head_dim + thread;
            const float decayed = head_state[index] * alpha;
            head_state[index] = decayed;
            memory += decayed * key[row];
        }
        const float value =
            __bfloat162float(convolved_qkv[value_offset + thread]);
        delta[thread] = (value - memory) * beta;
    }
    __syncthreads();

    if (thread < value_head_dim) {
        float recurrent_output = 0.0f;
        for (int row = 0; row < key_head_dim; ++row) {
            const long long index =
                static_cast<long long>(row) * value_head_dim + thread;
            const float updated = head_state[index] + key[row] * delta[thread];
            head_state[index] = updated;
            recurrent_output += updated * query[row];
        }
        rounded_output[thread] =
            __bfloat162float(__float2bfloat16(recurrent_output));
    }
    __syncthreads();

    float square_sum = 0.0f;
    for (int column = thread; column < value_head_dim; column += kBlockSize) {
        square_sum += rounded_output[column] * rounded_output[column];
    }
    const float inverse_rms = rsqrtf(
        block_sum(square_sum, reduction) /
            static_cast<float>(value_head_dim) +
        epsilon);
    const int gate_offset = conv_width + value_head * value_head_dim;
    for (int column = thread; column < value_head_dim; column += kBlockSize) {
        const float gate = __bfloat162float(qkvz[gate_offset + column]);
        const float normalized = rounded_output[column] * inverse_rms *
            norm_weight[column] * silu(gate);
        output[value_head * value_head_dim + column] =
            __float2bfloat16(normalized);
    }
}

// The 128-element reduction below follows the original block tree: first pair
// positions 0/64 and 32/96, then combine them, then reduce the 32 lanes. Explicit
// rounded products/adds prevent an FMA from changing the old normalization sums.
// Preserving this order also limits accumulated drift across long stateful runs.
__device__ float square_sum_128(const float (&values)[4]) {
    const float a = __fadd_rn(__fmul_rn(values[0], values[0]),
                             __fmul_rn(values[2], values[2]));
    const float b = __fadd_rn(__fmul_rn(values[1], values[1]),
                             __fmul_rn(values[3], values[3]));
    float sum = __fadd_rn(a, b);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    return __shfl_sync(0xffffffff, sum, 0);
}

// Reduce the reference's parallel dot-product leaves in a fixed FP32 tree.
// The caller reuses the same register array for prediction and final readout.
__device__ __forceinline__ float tree_sum_128(float (&values)[128]) {
    #pragma unroll
    for (int stride = 64; stride; stride /= 2) {
        #pragma unroll
        for (int i = 0; i < stride; ++i)
            values[i] = __fadd_rn(values[i], values[i + stride]);
    }
    return values[0];
}

// One warp owns a 128-key x 32-value tile. Lane j owns one value column, keeping
// its 128 FP32 state entries in registers until prediction and correction finish.
// Adjacent lanes load adjacent values from the canonical [head, key, value]
// layout. No state transpose or cross-block recurrence synchronization is needed.
template<bool Reference>
__global__ void tiled_recurrent_update_128(
    const __nv_bfloat16* ba,
    const __nv_bfloat16* convolved_qkv,
    const __nv_bfloat16* dt_bias,
    const float* negative_a,
    float* recurrent_state,
    __nv_bfloat16* rounded_output,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim) {
    const int lane = static_cast<int>(threadIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(blockIdx.x) * 32 + lane;
    const int key_head = head / (num_value_heads / num_key_heads);
    const int key_width = num_key_heads * 128;
    // A warp-wide shared-memory broadcast supplies each row's Q/K scalar.
    // Volatile keeps all 128 broadcast keys from remaining live in registers
    // between the two unrolled loops, leaving register space for the state tile.
    __shared__ volatile float normalized_query[128];
    __shared__ volatile float normalized_key[128];
    float query[4], key[4];
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        const int index = key_head * 128 + group * 32 + lane;
        query[group] = __bfloat162float(convolved_qkv[index]);
        key[group] = __bfloat162float(convolved_qkv[key_width + index]);
    }
    // Keep the head dimension as a runtime operand, even for this specialization.
    // Constant-folding rsqrtf(128) differs from the old GPU rsqrt instruction and
    // can change BF16 readout near cancellation (and real-model greedy tokens).
    const float query_square_sum = square_sum_128(query);
    const float key_square_sum = square_sum_128(key);
    const float query_scale = rsqrtf(query_square_sum + 1e-6f) *
        rsqrtf(static_cast<float>(key_head_dim));
    const float key_scale = rsqrtf(key_square_sum + 1e-6f);
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        if constexpr(Reference) {
            normalized_query[group * 32 + lane] = __fmul_rn(
                tau_precision::divide(query[group], sqrtf(query_square_sum + 1e-6f)),
                (float)(1.0 / std::sqrt(128.0)));
            normalized_key[group * 32 + lane] = tau_precision::divide(
                key[group], sqrtf(key_square_sum + 1e-6f));
        } else {
            normalized_query[group * 32 + lane] = query[group] * query_scale;
            normalized_key[group * 32 + lane] = key[group] * key_scale;
        }
    }
    __syncwarp();
    // Beta's BF16 rounding and FP32 decay match the generic/native contract.
    const float beta = __bfloat162float(__float2bfloat16(Reference
        ? tau_precision::sigmoid(__bfloat162float(ba[head]))
        : 1.0f / (1.0f + expf(-__bfloat162float(ba[head])))));
    const float step = stable_softplus(__bfloat162float(ba[num_value_heads + head]) +
                                      __bfloat162float(dt_bias[head]));
    const float alpha = expf(step * negative_a[head]);
    float* state = recurrent_state + static_cast<long long>(head) * 128 * 128 + column;
    float retained[128];
    float prediction = 0.0f;
    float products[128];
    // Compile-time indices keep retained[] in registers. A spill would defeat
    // the intended single state load/store and is checked by the benchmark.
    #pragma unroll
    for (int row = 0; row < 128; ++row) {
        retained[row] = state[row * 128] * alpha;
        const float k = normalized_key[row];
        if constexpr(Reference) products[row] = __fmul_rn(retained[row], k);
        else prediction = fmaf(retained[row], k, prediction);
    }
    // Parallel reference dot products reduce pairs separated by half the
    // remaining width. A serial FMA chain has different rounding near MoE ties.
    if constexpr(Reference) {
        prediction = tree_sum_128(products);
    }
    const float value = __bfloat162float(convolved_qkv[2 * key_width + head * 128 + column]);
    const float delta = (value - prediction) * beta;
    float result = 0.0f;
    #pragma unroll
    for (int row = 0; row < 128; ++row) {
        const float k = normalized_key[row];
        const float q = normalized_query[row];
        const float updated = fmaf(k, delta, retained[row]);
        state[row * 128] = updated;
        if constexpr(Reference) products[row]=__fmul_rn(updated,q);
        else result = fmaf(updated, q, result);
    }
    if constexpr(Reference) {
        result = tree_sum_128(products);
    }
    // Each tile writes disjoint columns. The following launch on the same stream
    // waits for all tiles before computing the full-head normalization statistic.
    rounded_output[head * 128 + column] = __float2bfloat16(result);
}

// Normalization spans all four recurrence tiles, so it is a separate short pass.
// One warp reads four values per lane and applies direct FP32 norm weights and
// a SiLU gate; these are GDN weights, not centered RMSNorm's (1 + weight).
__global__ void gated_rms_norm_128(
    const __nv_bfloat16* qkvz,
    const __nv_bfloat16* rounded_output,
    const float* norm_weight,
    __nv_bfloat16* output,
    int conv_width,
    float epsilon) {
    const int lane = static_cast<int>(threadIdx.x);
    const int head = static_cast<int>(blockIdx.x);
    float values[4];
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        values[group] = __bfloat162float(rounded_output[head * 128 + group * 32 + lane]);
    }
    const float inverse_rms = rsqrtf(square_sum_128(values) / 128.0f + epsilon);
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        const int column = group * 32 + lane;
        const float gate = __bfloat162float(qkvz[conv_width + head * 128 + column]);
        output[head * 128 + column] = __float2bfloat16(
            values[group] * inverse_rms * norm_weight[column] * silu(gate));
    }
}

}  // namespace

// Enqueues causal convolution and recurrent gated normalization on the caller's
// stream. State buffers are updated in place; the launcher allocates nothing and
// performs no host transfer or synchronization.
extern "C" int tau_launch_fused_bf16_gated_delta_net_decode(
    const void* qkvz,
    const void* ba,
    const void* conv_weight,
    const void* dt_bias,
    const void* negative_a,
    const float* norm_weight,
    void* conv_state,
    void* recurrent_state,
    void* convolved_qkv,
    void* recurrence_output,
    void* output,
    int conv_width,
    int conv_kernel_size,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    float epsilon,
    int reference_reduction,
    void* stream) {
    if (qkvz == nullptr || ba == nullptr || conv_weight == nullptr ||
        dt_bias == nullptr || negative_a == nullptr || norm_weight == nullptr ||
        conv_state == nullptr || recurrent_state == nullptr ||
        convolved_qkv == nullptr || output == nullptr || stream == nullptr ||
        conv_width <= 0 || conv_kernel_size <= 0 || num_key_heads <= 0 ||
        num_value_heads <= 0 || key_head_dim <= 0 || value_head_dim <= 0 ||
        key_head_dim > kBlockSize || value_head_dim > kBlockSize ||
        num_value_heads % num_key_heads != 0 || !std::isfinite(epsilon) ||
        epsilon <= 0.0f) {
        return -1;
    }
    const bool tiled = key_head_dim == 128 && value_head_dim == 128;
    // Reject before convolution mutates state if the tiled workspace is absent.
    if (tiled && recurrence_output == nullptr) {
        return -1;
    }

    auto* cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int conv_grid = (conv_width + kBlockSize - 1) / kBlockSize;
    update_causal_convolution<<<conv_grid, kBlockSize, 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(qkvz),
        static_cast<const __nv_bfloat16*>(conv_weight),
        static_cast<__nv_bfloat16*>(conv_state),
        static_cast<__nv_bfloat16*>(convolved_qkv),
        conv_width,
        conv_kernel_size);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    if (tiled) {
        if(reference_reduction) {
        tiled_recurrent_update_128<true><<<dim3(4, num_value_heads), 32, 0, cuda_stream>>>(
            static_cast<const __nv_bfloat16*>(ba),
            static_cast<const __nv_bfloat16*>(convolved_qkv),
            static_cast<const __nv_bfloat16*>(dt_bias),
            static_cast<const float*>(negative_a),
            static_cast<float*>(recurrent_state),
            static_cast<__nv_bfloat16*>(recurrence_output),
            num_key_heads, num_value_heads, key_head_dim);
        } else {
        tiled_recurrent_update_128<false><<<dim3(4, num_value_heads), 32, 0, cuda_stream>>>(
            static_cast<const __nv_bfloat16*>(ba),
            static_cast<const __nv_bfloat16*>(convolved_qkv),
            static_cast<const __nv_bfloat16*>(dt_bias),
            static_cast<const float*>(negative_a),
            static_cast<float*>(recurrent_state),
            static_cast<__nv_bfloat16*>(recurrence_output),
            num_key_heads, num_value_heads, key_head_dim);
        }
        status = cudaPeekAtLastError();
        if (status != cudaSuccess) {
            return static_cast<int>(status);
        }
        gated_rms_norm_128<<<num_value_heads, 32, 0, cuda_stream>>>(
            static_cast<const __nv_bfloat16*>(qkvz),
            static_cast<const __nv_bfloat16*>(recurrence_output), norm_weight,
            static_cast<__nv_bfloat16*>(output), conv_width, epsilon);
        return static_cast<int>(cudaPeekAtLastError());
    }

    const std::size_t shared_bytes = static_cast<std::size_t>(
        2 * key_head_dim + 2 * value_head_dim) * sizeof(float);
    recurrent_update_and_gated_rms_norm<<<
        num_value_heads, kBlockSize, shared_bytes, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(qkvz),
        static_cast<const __nv_bfloat16*>(ba),
        static_cast<const __nv_bfloat16*>(convolved_qkv),
        static_cast<const __nv_bfloat16*>(dt_bias),
        static_cast<const float*>(negative_a),
        norm_weight,
        static_cast<float*>(recurrent_state),
        static_cast<__nv_bfloat16*>(output),
        conv_width,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        epsilon);
    status = cudaPeekAtLastError();
    return static_cast<int>(status);
}
