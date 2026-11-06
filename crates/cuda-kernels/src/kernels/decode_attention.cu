// Address-stable one-token decode primitives. Metadata is [token_id, position].
// KV/activations/output stay BF16. Scores and normalization stay FP32; an
// explicit MoE preset rounds unnormalized probability products to BF16.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "fp32_precision.cuh"

using B = __nv_bfloat16;
__device__ float f(B v) { return __bfloat162float(v); }
__device__ B b(float v) { return __float2bfloat16_rn(v); }

// Preserve the ordinary BF16 activation's exp/add/reciprocal rounding, including
// when sigmoid is fused into a multiply. Both vector and scalar gates share it.
__device__ B sigmoid_bf16(B value) {
    B one = b(1.f);
    return one / (one + hexp(-value));
}

__global__ void elementwise(int kind, const B* x, const B* y, B* out,
    const unsigned* meta, int n, int width, int capacity, int transposed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (kind == 0) out[i] = x[(size_t)meta[0] * width + i]; // embedding
    if (kind == 1) out[i] = b(f(x[i]) + f(y[i])); // residual
    if (kind == 2) out[i] = x[(i / width) * (2 * width) + i % width]; // Q
    if (kind == 3) { // BF16 sigmoid, then BF16 product
        B gate = y[(i / width) * (2 * width) + width + i % width];
        out[i] = x[i] * sigmoid_bf16(gate);
    }
    if (kind == 6 || kind == 7) { // FP32 sigmoid/product; one final BF16 store
        int index = kind == 6 ? (i / width) * (2 * width) + width + i % width : i;
        out[i] = b(__fmul_rn(f(x[i]), tau_precision::sigmoid(f(y[index]))));
    }
    if (kind == 4) { // append one K or V token, never touch the stale tail
        int h = i / width, d = i % width;
        size_t index = transposed ? ((size_t)h * width + d) * capacity + meta[1]
            : ((size_t)h * capacity + meta[1]) * width + d;
        out[index] = x[i];
    }
    if (kind == 5) {
        // Ordinary scalar sigmoid rounds exp, addition and reciprocal in BF16.
        // Computing all of sigmoid in FP32 and rounding only its result changes
        // the shared expert's contribution near BF16 rounding boundaries.
        out[i] = x[i] * sigmoid_bf16(y[0]);
    }
}

__global__ void rotary(const B* x, const B* cos, const B* sin, B* out,
    const unsigned* meta, int n, int dim, int rotary_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % dim;
    if (d >= rotary_dim) { out[i] = x[i]; return; }
    int half = rotary_dim / 2;
    int j = d % half;
    size_t row = (size_t)meta[1] * half + j;
    B a = x[i - d + j], v = x[i - d + j + half];
    out[i] = d < half ? a * cos[row] - v * sin[row]
                      : a * sin[row] + v * cos[row];
}

__device__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}

template<bool Maximum>
__device__ float block_reduce(float x, float* partial) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int delta = 16; delta; delta >>= 1) {
        float other = __shfl_down_sync(0xffffffff, x, delta);
        x = Maximum ? fmaxf(x, other) : x + other;
    }
    if (!lane) partial[warp] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? partial[threadIdx.x] : (Maximum ? -INFINITY : 0.f);
    for (int delta = 16; delta; delta >>= 1) {
        float other = __shfl_down_sync(0xffffffff, x, delta);
        x = Maximum ? fmaxf(x, other) : x + other;
    }
    if (!threadIdx.x) partial[0] = x;
    __syncthreads();
    float result = partial[0];
    // Every warp must finish the broadcast before a later reduction reuses
    // partial[0]; otherwise a fast warp could overwrite another warp's result.
    __syncthreads();
    return result;
}

// Match the prepared centered norm's one-warp vector reduction. Each lane owns
// eight values; Elements selects the original 16/8/4-byte grouping and sum order.
template<int Elements>
__device__ void normalize_prepared_vector(const B* input, const B* weight,
    B* normalized, float epsilon) {
    if (threadIdx.x < 32) {
        float values[8];
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = (i / Elements) * 32 * Elements + threadIdx.x * Elements + i % Elements;
            values[i] = f(input[d]);
            sum += values[i] * values[i];
        }
        sum = warp_sum(sum);
        sum = __shfl_sync(0xffffffff, sum, 0);
        float inverse = rsqrtf(sum / 256 + epsilon);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = (i / Elements) * 32 * Elements + threadIdx.x * Elements + i % Elements;
            normalized[d] = b(values[i] * inverse * (1.f + f(weight[d])));
        }
    }
}

// One block owns one Q or K head. Normalized BF16 values are published in shared
// memory before partial RoPE; K blocks also append the unchanged V head. The
// stream orders the later attention kernel after every head's cache writes.
template<bool Fp32Rope>
__global__ void attention_prepare(const B* query_gate, const B* key, const B* value,
    const B* query_weight, const B* key_weight, const void* cos_arg, const void* sin_arg,
    B* query_output, B* cached_key, B* cached_value, const unsigned* meta,
    int heads, int dim, int capacity, int rotary_dim, int transposed,
    float query_epsilon, float key_epsilon) {
    __shared__ B normalized[256];
    __shared__ float partial[8];
    bool is_key = blockIdx.x >= heads;
    int head = is_key ? blockIdx.x - heads : blockIdx.x;
    const B* input = is_key ? key + head * dim : query_gate + head * 2 * dim;
    const B* weight = is_key ? key_weight : query_weight;
    float epsilon = is_key ? key_epsilon : query_epsilon;
    // The separate Q-extraction output is always aligned. K norm consumes its
    // projection directly, so a sliced K input can narrow its vector grouping.
    uintptr_t alignment = reinterpret_cast<uintptr_t>(weight);
    if (is_key) alignment |= reinterpret_cast<uintptr_t>(input);
    if (dim == 256 && alignment % 4 == 0) {
        if (alignment % 16 == 0) normalize_prepared_vector<8>(input, weight, normalized, epsilon);
        else if (alignment % 8 == 0) normalize_prepared_vector<4>(input, weight, normalized, epsilon);
        else normalize_prepared_vector<2>(input, weight, normalized, epsilon);
    } else {
        // Other supported head widths and unaligned scales use the original
        // 256-thread scalar reduction, including zero contributions past dim.
        float sum = 0.f;
        if (threadIdx.x < dim) {
            float x = f(input[threadIdx.x]);
            sum += x * x;
        }
        sum = block_reduce<false>(sum, partial);
        float inverse = rsqrtf(sum / dim + epsilon);
        if (threadIdx.x < dim)
            normalized[threadIdx.x] = b(f(input[threadIdx.x]) * inverse * (1.f + f(weight[threadIdx.x])));
    }
    __syncthreads();
    int d = threadIdx.x;
    if (d < dim) {
        B result = normalized[d];
        if (d < rotary_dim) {
            int half = rotary_dim / 2, j = d % half;
            size_t row = (size_t)meta[1] * half + j;
            B a = normalized[j], v = normalized[j + half];
            // FP32 tables select FP32 rotation with one final storage round.
            // Existing BF16 tables retain the original product/sum boundaries.
            if constexpr(Fp32Rope) {
                float c=((const float*)cos_arg)[row], s=((const float*)sin_arg)[row];
                result=b(d<half ? __fmaf_rn(f(a),c,-__fmul_rn(f(v),s))
                    : __fmaf_rn(f(v),c,__fmul_rn(f(a),s)));
            } else {
                const B* cos=(const B*)cos_arg;const B* sin=(const B*)sin_arg;
                result=d<half?a*cos[row]-v*sin[row]:a*sin[row]+v*cos[row];
            }
        }
        if (!is_key) query_output[head * dim + d] = result;
        else {
            size_t index = transposed ? ((size_t)head * dim + d) * capacity + meta[1]
                : ((size_t)head * capacity + meta[1]) * dim + d;
            cached_key[index] = result;
            cached_value[((size_t)head * capacity + meta[1]) * dim + d] = value[head * dim + d];
        }
    }
}

// One block per query head/context tile; grouped heads read retained KV without
// expanding it. Large capacities use a second FP32 combine kernel.
// Only [0, position] is read, so unused capacity can contain arbitrary NaNs.
template<int Tile, bool Bf16Products>
__global__ void attention(const B* q, const B* k, const B* v, B* out,
    const unsigned* meta, int heads, int kv_heads, int dim, int capacity,
    int transposed, float* workspace, int splits) {
    extern __shared__ float scores[];
    __shared__ float partial[8];
    int h = blockIdx.x, kh = h / (heads / kv_heads);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int start = splits > 1 ? blockIdx.y * Tile : 0;
    int length = (int)meta[1] + 1 - start;
    if (splits > 1) length = min(length, Tile);
    float* destination = workspace + ((size_t)h * splits + blockIdx.y) * (dim + 2);
    if (length <= 0) {
        if (!threadIdx.x) { destination[0] = -INFINITY; destination[1] = 0.f; }
        for (int d = threadIdx.x; d < dim; d += blockDim.x) destination[d+2] = 0.f;
        return;
    }
    for (int t = warp; t < length; t += 8) {
        float dot = 0.f;
        for (int d = lane; d < dim; d += 32) {
            size_t index = transposed ? ((size_t)kh * dim + d) * capacity + t + start
                : ((size_t)kh * capacity + t + start) * dim + d;
            // Separate multiply and addition avoid silently introducing FMA.
            dot = __fadd_rn(dot, __fmul_rn(f(q[h * dim + d]), f(k[index])));
        }
        dot = warp_sum(dot);
        if (!lane) scores[t] = dot * rsqrtf((float)dim);
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int t = threadIdx.x; t < length; t += blockDim.x)
        maximum = fmaxf(maximum, scores[t]);
    maximum = block_reduce<true>(maximum, partial);
    float denominator = 0.f;
    for (int t = threadIdx.x; t < length; t += blockDim.x) {
        scores[t] = Bf16Products ? tau_precision::exponential(scores[t] - maximum) : expf(scores[t] - maximum);
        denominator += scores[t];
    }
    denominator = block_reduce<false>(denominator, partial);
    if (splits > 1 && !threadIdx.x) {
        destination[0] = maximum;
        destination[1] = denominator;
    }
    for (int t = threadIdx.x; t < length; t += blockDim.x)
        scores[t] = Bf16Products ? f(b(scores[t])) : scores[t] / denominator;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        // Eight independent accumulators expose multiple context loads at once,
        // rather than putting every global load behind a single sum dependency.
        float accum[8] = {};
        int t = 0;
        for (; t + 7 < length; t += 8) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                accum[j] = __fadd_rn(accum[j], __fmul_rn(scores[t+j],
                    f(v[((size_t)kh * capacity + start+t+j) * dim + d])));
        }
        float sum = ((accum[0] + accum[1]) + (accum[2] + accum[3]))
                  + ((accum[4] + accum[5]) + (accum[6] + accum[7]));
        for (; t < length; ++t) sum = __fadd_rn(sum,
            __fmul_rn(scores[t], f(v[((size_t)kh * capacity + start+t) * dim + d])));
        if (Bf16Products) sum /= denominator;
        if (splits > 1) destination[d+2] = sum;
        else out[h * dim + d] = b(sum);
    }
}

// Merge normalized tile contexts with stable FP32 softmax weights. Empty tiles
// explicitly contain zero sums/outputs, so request resets cannot expose stale data.
template<int MaxSplits, bool Bf16Products>
__global__ void combine_attention(const float* workspace, B* out, int dim, int splits) {
    __shared__ float weights[MaxSplits];
    int h = blockIdx.x;
    const float* source = workspace + (size_t)h * splits * (dim + 2);
    if (!threadIdx.x) {
        float maximum = -INFINITY, denominator = 0.f;
        for (int s = 0; s < splits; ++s)
            if (source[s*(dim+2)+1] > 0.f) maximum = fmaxf(maximum, source[s*(dim+2)]);
        for (int s = 0; s < splits; ++s) {
            float sum = source[s*(dim+2)+1];
            weights[s] = sum > 0.f ? sum * expf(source[s*(dim+2)] - maximum) : 0.f;
            denominator += weights[s];
        }
        for (int s = 0; s < splits; ++s) weights[s] /= denominator;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.f;
        for (int s = 0; s < splits; ++s)
            sum = __fadd_rn(sum, __fmul_rn(weights[s], source[s*(dim+2)+d+2]));
        out[h * dim + d] = b(sum);
    }
}

extern "C" int tau_launch_decode_elementwise(int kind, const void* x,
    const void* y, void* out, const unsigned* meta, int n, int width,
    int capacity, int transposed, void* stream) {
    if (kind < 0 || kind > 7 || !x || !out || n <= 0 || width <= 0 ||
        ((kind == 0 || kind == 4) && !meta) ||
        ((kind == 1 || kind == 3 || kind == 5 || kind == 6 || kind == 7) && !y)) return 1;
    elementwise<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        kind, (const B*)x, (const B*)y, (B*)out, meta, n, width, capacity, transposed);
    return (int)cudaGetLastError();
}

extern "C" int tau_launch_decode_rope(const void* x, const void* cos,
    const void* sin, void* out, const unsigned* meta, int n, int dim,
    int rotary_dim, void* stream) {
    if (!x || !cos || !sin || !out || !meta || n <= 0 || dim <= 0 ||
        rotary_dim <= 0 || rotary_dim > dim || rotary_dim % 2) return 1;
    rotary<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((const B*)x,
        (const B*)cos, (const B*)sin, (B*)out, meta, n, dim, rotary_dim);
    return (int)cudaGetLastError();
}

extern "C" int tau_launch_decode_attention_prepare(const void* query_gate,
    const void* key, const void* value, const void* query_weight, const void* key_weight,
    const void* cos, const void* sin, void* query_output, void* cached_key,
    void* cached_value, const unsigned* metadata, int heads, int kv_heads, int dim,
    int capacity, int rotary_dim, int transposed, int fp32_rope, float query_epsilon,
    float key_epsilon, void* stream) {
    if (!query_gate || !key || !value || !query_weight || !key_weight || !cos || !sin ||
        !query_output || !cached_key || !cached_value || !metadata || heads <= 0 ||
        kv_heads <= 0 || heads % kv_heads || dim <= 0 || dim > 256 ||
        capacity <= 0 || capacity > 8192 || rotary_dim <= 0 || rotary_dim > dim ||
        rotary_dim % 2 || (transposed != 0 && transposed != 1) ||
        (fp32_rope != 0 && fp32_rope != 1) ||
        !(query_epsilon > 0.f) || !isfinite(query_epsilon) ||
        !(key_epsilon > 0.f) || !isfinite(key_epsilon)) return 1;
    if(fp32_rope) {
    attention_prepare<true><<<heads + kv_heads, 256, 0, (cudaStream_t)stream>>>(
        (const B*)query_gate, (const B*)key, (const B*)value,
        (const B*)query_weight, (const B*)key_weight, cos, sin,
        (B*)query_output, (B*)cached_key, (B*)cached_value, metadata,
        heads, dim, capacity, rotary_dim, transposed, query_epsilon, key_epsilon);
    } else {
    attention_prepare<false><<<heads + kv_heads, 256, 0, (cudaStream_t)stream>>>(
        (const B*)query_gate, (const B*)key, (const B*)value,
        (const B*)query_weight, (const B*)key_weight, cos, sin,
        (B*)query_output, (B*)cached_key, (B*)cached_value, metadata,
        heads, dim, capacity, rotary_dim, transposed, query_epsilon, key_epsilon);
    }
    return (int)cudaGetLastError();
}

// Geometry and scratch are fixed before capture. Keep the original combine
// storage for smaller split counts. At capacity 8192, tile 32 needs 256 splits;
// dispatch combine storage from the same split count used for scratch indexing.
template<int Tile, bool Bf16Products>
int launch_attention(const void* q, const void* k,
    const void* v, void* out, const unsigned* meta, float* workspace, int heads, int kv_heads,
    int dim, int capacity, int transposed, void* stream) {
    int splits = (capacity + Tile - 1) / Tile;
    attention<Tile,Bf16Products><<<dim3(heads, splits), 256, min(capacity, Tile) * sizeof(float), (cudaStream_t)stream>>>(
        (const B*)q, (const B*)k, (const B*)v, (B*)out, meta, heads, kv_heads,
        dim, capacity, transposed, workspace, splits);
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) return (int)code;
    if (splits > 128) combine_attention<256,Bf16Products><<<heads, 256, 0, (cudaStream_t)stream>>>(
        workspace, (B*)out, dim, splits);
    else if (splits > 32) combine_attention<128,Bf16Products><<<heads, 256, 0, (cudaStream_t)stream>>>(
        workspace, (B*)out, dim, splits);
    else if (splits > 1) combine_attention<32,Bf16Products><<<heads, 256, 0, (cudaStream_t)stream>>>(
        workspace, (B*)out, dim, splits);
    return (int)cudaGetLastError();
}

extern "C" int tau_launch_decode_attention(const void* q, const void* k,
    const void* v, void* out, const unsigned* meta, float* workspace, int heads, int kv_heads,
    int dim, int capacity, int transposed, int block_size, int bf16_products, void* stream) {
    if (!q || !k || !v || !out || !meta || !workspace || heads <= 0 || kv_heads <= 0 ||
        heads % kv_heads || dim <= 0 || dim > 256 || capacity <= 0 || capacity > 8192) return 1;
    switch (block_size) {
        case 32: return bf16_products ? launch_attention<32,true>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream) : launch_attention<32,false>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream);
        case 64: return bf16_products ? launch_attention<64,true>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream) : launch_attention<64,false>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream);
        case 128: return bf16_products ? launch_attention<128,true>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream) : launch_attention<128,false>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream);
        case 256: return bf16_products ? launch_attention<256,true>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream) : launch_attention<256,false>(q,k,v,out,meta,workspace,heads,kv_heads,dim,capacity,transposed,stream);
        default: return 1;
    }
}
