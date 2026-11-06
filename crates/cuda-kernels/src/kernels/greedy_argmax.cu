// Deterministic vocabulary argmax. All comparisons use integer keys so signed
// zeros and subnormals retain their FP32 total order, independent of FP math.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {
constexpr unsigned kThreads = 256;
constexpr unsigned kItemsPerBlock = 1024;

struct Candidate {
    unsigned key;
    unsigned token;
    unsigned valid;
};

// NaNs never enter the order. Validity is separate from the key because a
// vocabulary containing only negative infinities still has a valid maximum.
__device__ Candidate candidate(unsigned bits, unsigned token) {
    if ((bits & 0x7fffffffU) > 0x7f800000U) return {0, 0, 0};
    unsigned key = bits ^ ((bits & 0x80000000U) ? 0xffffffffU : 0x80000000U);
    return {key, token, 1};
}

// Valid candidates use the total order (FP32 key, token ID). Its maximum is
// associative, so warp/block partitioning cannot change the CPU reference winner.
__device__ Candidate better(Candidate a, Candidate b) {
    if (!b.valid) return a;
    if (!a.valid || b.key > a.key || (b.key == a.key && b.token > a.token)) return b;
    return a;
}

__device__ Candidate warp_max(Candidate value) {
    for (int offset = 16; offset; offset /= 2) {
        Candidate other = {
            __shfl_down_sync(0xffffffffU, value.key, offset),
            __shfl_down_sync(0xffffffffU, value.token, offset),
            __shfl_down_sync(0xffffffffU, value.valid, offset)};
        if ((threadIdx.x & 31) + offset < 32) value = better(value, other);
    }
    return value;
}

// Every thread participates, including invalid tail lanes. First reduce each
// warp, then let one full warp reduce the eight shared candidates.
__device__ Candidate block_max(Candidate value, Candidate* partial) {
    value = warp_max(value);
    if (!(threadIdx.x & 31)) partial[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < kThreads / 32 ? partial[threadIdx.x] : Candidate{0, 0, 0};
        value = warp_max(value);
    }
    return value;
}

// BF16 expands exactly by placing its bits in the high half of an FP32 word.
// Reading representations directly avoids flushing subnormals during conversion.
template<bool Bf16>
__device__ unsigned logit_bits(const void* logits, size_t index) {
    if constexpr (Bf16) return unsigned(static_cast<const uint16_t*>(logits)[index]) << 16;
    else return __float_as_uint(static_cast<const float*>(logits)[index]);
}

template<bool Bf16>
__global__ void vocabulary_max(const void* logits, unsigned count, unsigned* scratch) {
    __shared__ Candidate partial[kThreads / 32];
    Candidate best{0, 0, 0};
    size_t first = size_t(blockIdx.x) * kItemsPerBlock;
    for (size_t index = first + threadIdx.x;
         index < count && index < first + kItemsPerBlock; index += kThreads) {
        best = better(best, candidate(logit_bits<Bf16>(logits, index), unsigned(index)));
    }
    best = block_max(best, partial);
    if (!threadIdx.x) {
        scratch[blockIdx.x * 3] = best.key;
        scratch[blockIdx.x * 3 + 1] = best.token;
        scratch[blockIdx.x * 3 + 2] = best.valid;
    }
}

__global__ void finalize_max(const unsigned* scratch, unsigned blocks, unsigned* result) {
    __shared__ Candidate partial[kThreads / 32];
    Candidate best{0, 0, 0};
    for (unsigned index = threadIdx.x; index < blocks; index += kThreads) {
        best = better(best, {scratch[index * 3], scratch[index * 3 + 1], scratch[index * 3 + 2]});
    }
    best = block_max(best, partial);
    if (!threadIdx.x) {
        // Always overwrite both words, even when every input was NaN.
        result[0] = best.token;
        result[1] = best.valid;
    }
}

size_t block_count(unsigned count) {
    return count / kItemsPerBlock + (count % kItemsPerBlock != 0);
}
} // namespace

extern "C" size_t tau_greedy_argmax_workspace_words(unsigned count) {
    return block_count(count) * 3;
}

extern "C" int tau_launch_greedy_argmax(const void* logits, int dtype, unsigned count,
    unsigned* scratch, size_t scratch_words, unsigned* result, void* stream) {
    if (!logits || !scratch || !result || !count || (dtype != 0 && dtype != 1) ||
        scratch_words < tau_greedy_argmax_workspace_words(count)) return int(cudaErrorInvalidValue);
    unsigned blocks = unsigned(block_count(count));
    cudaStream_t ordered = static_cast<cudaStream_t>(stream);
    if (dtype == 0) vocabulary_max<false><<<blocks, kThreads, 0, ordered>>>(logits, count, scratch);
    else vocabulary_max<true><<<blocks, kThreads, 0, ordered>>>(logits, count, scratch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
    finalize_max<<<1, kThreads, 0, ordered>>>(scratch, blocks, result);
    return int(cudaGetLastError());
}
