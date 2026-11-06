// Cooperative grouped prefill GEMM. Dispatch, activation and ordered reduction
// remain in their existing owners; decode retains its separate prepared runner.
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdint>
#include "moe_gemm.cuh"
#include "bf16_staging.cuh"
#include "moe_dispatch.cuh"

extern "C" int tau_launch_linear_bf16_fp32_grouped(const void*, const void*,
    const float*, const int*, const int*, void*, int, int, int, int, int, int*, void*);

namespace {
using Bf16 = __nv_bfloat16;

// Borrowed views for one launch. Rust retains the allocations on the supplied
// stream until routing, both projections and ordered combination complete.
struct Arguments {
    const Bf16* input;
    const Bf16* weights;
    const float* routes;
    const int* slots;
    const int* tiles;
    Bf16* output;
    int tokens, capacity, columns, inner;
    int* receipt;
};

using tau_moe::copy_eight;

// All N fragments share A and all M fragments share B within one expert block.
// Stages retain distinct operands, allowing the next load to overlap MMA.
template<int M, int N, int K, int W, bool Down>
__device__ void load_stage(Arguments p, Bf16* a, Bf16* b,
    int row_start, int column_start, int expert, int k_start) {
    constexpr int Stride = K + 8; // Skew shared rows to reduce bank conflicts.
    for (int i = threadIdx.x; i < M * (K / 8); i += 32 * W) {
        int row = row_start + i / (K / 8), k = (i % (K / 8)) * 8;
        bool valid = row < p.capacity;
        if constexpr (!Down) {
            int slot = valid ? p.slots[row] : p.tokens * tau_moe::TopK;
            valid = slot >= 0 && slot < p.tokens * tau_moe::TopK;
            row = valid ? slot / tau_moe::TopK : 0;
        }
        const Bf16* source = p.input;
        if (valid && k_start + k < p.inner)
            source += (size_t)row * p.inner + k_start + k;
        copy_eight(a + (i / (K / 8)) * Stride + k, source,
            valid, max(0, p.inner - k_start - k));
    }
    for (int i = threadIdx.x; i < N * (K / 8); i += 32 * W) {
        int column = column_start + i / (K / 8), k = (i % (K / 8)) * 8;
        const Bf16* source = p.weights;
        if (column < p.columns && k_start + k < p.inner)
            source += ((size_t)expert * p.columns + column) * p.inner + k_start + k;
        copy_eight(b + (i / (K / 8)) * Stride + k, source,
            column < p.columns, max(0, p.inner - k_start - k));
    }
    __pipeline_commit();
}

template<int M, int N, int K, int W, int S, bool Down>
__global__ void grouped_gemm(Arguments p) {
    using namespace nvcuda;
    constexpr int Stride = K + 8, Fragments = (M / 16) * (N / 16) / W;
    __shared__ __align__(32) Bf16 a[S][M * Stride], b[S][N * Stride];
    __shared__ __align__(32) float result[M * N];
    const int row_start = blockIdx.y * M, column_start = blockIdx.x * N;
    const int expert = p.tiles[blockIdx.y];
    if (p.receipt && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        p.receipt[0] = M; p.receipt[1] = N; p.receipt[2] = K;
        p.receipt[3] = W; p.receipt[4] = S;
    }
    if (expert < 0) {
        if constexpr (!Down) {
            for (int i = threadIdx.x; i < M * N; i += W * 32) {
                int row = row_start + i / N, column = column_start + i % N;
                if (row < p.capacity && column < p.columns)
                    p.output[(size_t)row * p.columns + column] = __float2bfloat16_rn(0.f);
            }
        }
        return;
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[Fragments];
    #pragma unroll
    for (int f = 0; f < Fragments; ++f) wmma::fill_fragment(acc[f], 0.f);
    const int chunks = (p.inner + K - 1) / K, warp = threadIdx.x / 32;
    #pragma unroll
    for (int stage = 0; stage < S; ++stage)
        if (stage < chunks)
            load_stage<M,N,K,W,Down>(p, a[stage], b[stage], row_start,
                column_start, expert, stage * K);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk % S;
        if (chunks - chunk >= S) __pipeline_wait_prior(S-1);
        else __pipeline_wait_prior(0);
        __syncthreads();
        // Staged K changes loading only. Each output retains consecutive K=16
        // MMA steps in one FP32 accumulator, including masked final segments.
        #pragma unroll
        for (int k = 0; k < K; k += 16) {
            if (chunk * K + k >= p.inner) break;
            #pragma unroll
            for (int f = 0; f < Fragments; ++f) {
                int tile = warp + f * W;
                wmma::fragment<wmma::matrix_a,16,16,16,Bf16,wmma::row_major> left;
                wmma::fragment<wmma::matrix_b,16,16,16,Bf16,wmma::col_major> right;
                wmma::load_matrix_sync(left, a[stage] + (tile / (N/16)) * 16 * Stride + k, Stride);
                wmma::load_matrix_sync(right, b[stage] + (tile % (N/16)) * 16 * Stride + k, Stride);
                wmma::mma_sync(acc[f], left, right, acc[f]);
            }
        }
        __syncthreads(); // A loader cannot overwrite another warp's operands.
        if (chunk + S < chunks)
            load_stage<M,N,K,W,Down>(p, a[stage], b[stage], row_start,
                column_start, expert, (chunk + S) * K);
    }
    #pragma unroll
    for (int f = 0; f < Fragments; ++f) {
        int tile = warp + f * W;
        wmma::store_matrix_sync(result + (tile / (N/16)) * 16 * N + (tile % (N/16)) * 16,
            acc[f], N, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M * N; i += W * 32) {
        int row = row_start + i / N, column = column_start + i % N;
        if (row >= p.capacity || column >= p.columns) continue;
        if constexpr (Down) {
            int slot = p.slots[row];
            if (slot < p.tokens * tau_moe::TopK) {
                float weight = p.routes[(slot / tau_moe::TopK) * 16 + 8 + slot % 8];
                p.output[(size_t)slot * p.columns + column] = __float2bfloat16_rn(
                    __fmul_rn(result[i], weight));
            }
        } else p.output[(size_t)row * p.columns + column] = __float2bfloat16_rn(result[i]);
    }
}

template<int M, int N, int K, int W, int S>
int launch(Arguments p, bool down, cudaStream_t stream) {
    if constexpr (((M/16)*(N/16)) % W != 0
        || tau_moe::shared_memory_upper_bound<M,N,K,S>() > tau_moe::MaxStaticSharedBytes)
        return (int)cudaErrorInvalidValue;
    else {
        dim3 grid((p.columns + N - 1) / N, p.capacity / M);
        if (down) grouped_gemm<M,N,K,W,S,true><<<grid,W*32,0,stream>>>(p);
        else grouped_gemm<M,N,K,W,S,false><<<grid,W*32,0,stream>>>(p);
        return (int)cudaGetLastError();
    }
}

template<int M, int N, int K, int S>
int warps(Arguments p, bool down, int count, cudaStream_t stream) {
    switch (count) {
        case 1: return launch<M,N,K,1,S>(p,down,stream);
        case 2: return launch<M,N,K,2,S>(p,down,stream);
        case 4: return launch<M,N,K,4,S>(p,down,stream);
        case 8: return launch<M,N,K,8,S>(p,down,stream);
        default: return (int)cudaErrorInvalidValue;
    }
}

template<int M, int N, int K>
int stages(Arguments p, bool down, const tau_moe::GemmTile& c, cudaStream_t stream) {
    if (c.stages == 1) return warps<M,N,K,1>(p,down,c.warps,stream);
    if (c.stages == 2) return warps<M,N,K,2>(p,down,c.warps,stream);
    return (int)cudaErrorInvalidValue;
}

template<int M, int N>
int inner_tile(Arguments p, bool down, const tau_moe::GemmTile& c, cudaStream_t stream) {
    switch(c.k) {
        case 16: return stages<M,N,16>(p,down,c,stream);
        case 32: return stages<M,N,32>(p,down,c,stream);
        case 64: return stages<M,N,64>(p,down,c,stream);
        case 128: return stages<M,N,128>(p,down,c,stream);
        default: return (int)cudaErrorInvalidValue;
    }
}

template<int M>
int columns(Arguments p, bool down, const tau_moe::GemmTile& c, cudaStream_t stream) {
    switch(c.n) {
        case 16: return inner_tile<M,16>(p,down,c,stream);
        case 32: return inner_tile<M,32>(p,down,c,stream);
        case 64: return inner_tile<M,64>(p,down,c,stream);
        default: return (int)cudaErrorInvalidValue;
    }
}
}

extern "C" int tau_launch_moe_prefill_gemm(const void* input, const void* weights,
    const float* routes, const int* slots, const int* tiles, void* output,
    int tokens, int capacity, int n, int k, int down, const tau_moe::GemmTile* c,
    int* receipt, void* stream) {
    if (!c || !input || !weights || !routes || !slots || !tiles || !output
        || tokens <= 0 || capacity <= 0 || n <= 0 || k <= 0 || (down != 0 && down != 1)
        || (c->m != 16 && c->m != 32) || capacity % c->m || capacity/c->m > 65535)
        return (int)cudaErrorInvalidValue;
    if (c->m == 16 && c->n == 16 && c->k == 16 && c->warps == 1 && c->stages == 1)
        return tau_launch_linear_bf16_fp32_grouped(input,weights,routes,slots,tiles,
            output,tokens,capacity,n,k,down,receipt,stream);
    Arguments p{(const Bf16*)input,(const Bf16*)weights,routes,slots,tiles,
        (Bf16*)output,tokens,capacity,n,k,receipt};
    if (c->m == 16) return columns<16>(p,down,*c,(cudaStream_t)stream);
    return columns<32>(p,down,*c,(cudaStream_t)stream);
}
