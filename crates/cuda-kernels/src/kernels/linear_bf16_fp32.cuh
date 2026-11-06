#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace tau_linear {
// Shared K-ordered 16x16 MMA used by ordinary, batched and grouped projections.
// An optional slot map gathers original token rows directly into shared memory;
// sentinel slots zero-pad the tile without allocating a gathered input tensor.
__device__ __forceinline__ void tile(const __nv_bfloat16* input,
    const __nv_bfloat16* weight, __nv_bfloat16* left, __nv_bfloat16* right,
    float* result, int row_start, int column_start, int rows, int columns,
    int inner, const int* slots = nullptr, int assignments = 0, int divisor = 1) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> accumulator;
    wmma::fill_fragment(accumulator, 0.f);
    for (int k_start = 0; k_start < inner; k_start += 16) {
        for (int i = threadIdx.x; i < 256; i += 32) {
            const int r = i / 16, k = i % 16;
            int row = row_start + r;
            bool valid_row = row < rows;
            if (slots && valid_row) {
                int slot = slots[row];
                valid_row = slot >= 0 && slot < assignments;
                row = valid_row ? slot / divisor : -1;
            }
            left[i] = valid_row && k_start + k < inner
                ? input[(size_t)row * inner + k_start + k] : __float2bfloat16(0.f);
            // W[N,K] is the column-major [K,N] operand without a transpose copy.
            right[i] = column_start+r < columns && k_start+k < inner
                ? weight[(size_t)(column_start+r)*inner+k_start+k] : __float2bfloat16(0.f);
        }
        __syncwarp();
        wmma::load_matrix_sync(a, left, 16);
        wmma::load_matrix_sync(b, right, 16);
        wmma::mma_sync(accumulator, a, b, accumulator);
        __syncwarp();
    }
    wmma::store_matrix_sync(result, accumulator, 16, wmma::mem_row_major);
    __syncwarp();
}
}
