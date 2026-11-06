// Token/rank assignment grouping stays entirely on the caller's CUDA stream.
// Integer atomics allocate unique row slots; numerical combination is ordered.
#include "moe_dispatch.cuh"
#include <cmath>

namespace {
using namespace tau_moe;

// Reset the whole retained capacity so shorter chunks cannot consume old work.
// Slots use the current assignment count as their nonnegative padding sentinel.
__global__ void initialize(int* meta, int* offsets, int* slots, int* tiles,
    int assignments, int capacity, int tile_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PaddedRows + 1) meta[i] = 0;
    if (i <= Experts) offsets[i] = 0;
    if (i < capacity) slots[i] = assignments;
    if (i < capacity / tile_rows) tiles[i] = -1;
}

// Validate a packed router assignment before using it as a device array index.
// Duplicate/invalid routes mark the result invalid instead of reading weights
// out of bounds; the final combination publishes NaNs for inference to reject.
__device__ int expert_for(const float* routes, int slot, int* meta) {
    int token = slot / TopK, rank = slot % TopK;
    float id = routes[token * 2 * TopK + rank];
    float weight = routes[token * 2 * TopK + TopK + rank];
    bool valid = isfinite(id) && id >= 0 && id < Experts && floorf(id) == id
        && isfinite(weight) && weight >= 0.f;
    for (int previous = 0; previous < rank; ++previous)
        valid &= id != routes[token * 2 * TopK + previous];
    if (!valid) { atomicExch(meta + Status, 1); return -1; }
    return (int)id;
}

// Count each valid token/rank assignment once. Counts are never copied to host.
__global__ void count(const float* routes, int* meta, int assignments) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < assignments) {
        int expert = expert_for(routes, slot, meta);
        if (expert >= 0) atomicAdd(meta + expert, 1);
    }
}

// A small device prefix sum assigns disjoint padded expert ranges. Threads then
// publish the tile-to-expert worklist consumed by both projection launches.
__global__ void plan(int* meta, int* offsets, int* tiles, int tile_rows) {
    int expert = threadIdx.x;
    if (!expert) {
        int end = 0;
        for (int e = 0; e < Experts; ++e) {
            offsets[e] = end;
            end += ((meta[e] + tile_rows - 1) / tile_rows) * tile_rows;
        }
        offsets[Experts] = end;
        meta[PaddedRows] = end;
    }
    __syncthreads();
    for (int tile = offsets[expert] / tile_rows; tile < offsets[expert + 1] / tile_rows; ++tile)
        tiles[tile] = expert;
}

// Integer cursor allocation may permute rows within an expert. The slot keeps
// the original token/rank identity, so scatter and ordered combination are exact.
__global__ void scatter(const float* routes, int* meta, const int* offsets,
    int* slots, int assignments) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < assignments) {
        int expert = expert_for(routes, slot, meta);
        if (expert >= 0) {
            int row = atomicAdd(meta + Experts + expert, 1);
            slots[offsets[expert] + row] = slot;
        }
    }
}

// Restore token order after down projections wrote canonical token/rank slots.
// Invalid plans fail closed without consuming uninitialized contribution rows.
__global__ void combine(const __nv_bfloat16* contributions,
    __nv_bfloat16* output, const int* meta, int rows, int width) {
    int column = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (column >= width) return;
    output[(size_t)token * width + column] = meta[Status]
        ? __float2bfloat16_rn(NAN)
        : tau_moe::combine_ranked(contributions + (size_t)token * TopK * width, width, column);
}
}

// Queue fixed-size metadata construction with no allocation or synchronization.
extern "C" int tau_launch_moe_prefill_dispatch(const float* routes, int* meta,
    int* offsets, int* slots, int* tiles, int rows, int capacity, int tile_rows, void* stream) {
    if (!routes || !meta || !offsets || !slots || !tiles || rows <= 0 || rows > 65535
        || (tile_rows != 16 && tile_rows != 32) || capacity <= 0 || capacity % tile_rows) return (int)cudaErrorInvalidValue;
    long long assignments = (long long)rows * tau_moe::TopK;
    long long bound = assignments + min((long long)tau_moe::Experts, assignments) * (tile_rows - 1);
    if (capacity < (bound + tile_rows - 1) / tile_rows * tile_rows)
        return (int)cudaErrorInvalidValue;
    auto s = (cudaStream_t)stream;
    initialize<<<(max(capacity, tau_moe::PaddedRows + 1) + 255) / 256, 256, 0, s>>>(
        meta, offsets, slots, tiles, (int)assignments, capacity, tile_rows);
    count<<<(assignments + 255) / 256, 256, 0, s>>>(routes, meta, (int)assignments);
    plan<<<1, 256, 0, s>>>(meta, offsets, tiles, tile_rows);
    scatter<<<(assignments + 255) / 256, 256, 0, s>>>(routes, meta, offsets, slots, (int)assignments);
    return (int)cudaGetLastError();
}

// The routed prefill owner calls this only after all grouped expert stores.
extern "C" int tau_launch_moe_prefill_combine(const void* contributions,
    void* output, const int* meta, int rows, int width, void* stream) {
    if (!contributions || !output || !meta || rows <= 0 || rows > 65535 || width <= 0)
        return (int)cudaErrorInvalidValue;
    combine<<<dim3((width + 255) / 256, rows), 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)contributions, (__nv_bfloat16*)output, meta, rows, width);
    return (int)cudaGetLastError();
}
