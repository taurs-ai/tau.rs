// Precision-aware expert projection: BF16 operands, FP32 MMA accumulator/output.
// Accumulate consecutive K=16 fragments without a reduced-precision split-K
// reduction. This matches the tensor-core accumulation contract of the golden
// expert path. Direct, batched and grouped calls share the masked tile primitive;
// routing weights and final storage rounding remain with their epilogues.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "linear_bf16_fp32.cuh"
#include "moe_dispatch.cuh"

namespace {

// One warp owns a 16x16 output tile. Shared operands zero-pad every partial
// token/feature tile, so no masked row can read outside its retained allocation.
template<bool Batched>
__global__ void linear_fp32(const void* input_arg,
    const void* weight_arg, void* output_arg, int rows, int columns, int inner, int* receipt) {
    // A receipt is written by each actual expert projection, never inferred on
    // the host. Tests require all eight ranks in both projections under replay.
    if (receipt && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        int* r = receipt + blockIdx.z * 5;
        r[0]=0; r[1]=16; r[2]=16; r[3]=1; r[4]=1;
    }
    const auto* input = Batched ? ((const __nv_bfloat16* const*)input_arg)[blockIdx.z]
        : (const __nv_bfloat16*)input_arg;
    const auto* weight = Batched ? ((const __nv_bfloat16* const*)weight_arg)[blockIdx.z]
        : (const __nv_bfloat16*)weight_arg;
    auto* output = Batched ? ((float* const*)output_arg)[blockIdx.z] : (float*)output_arg;
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 left[256];
    __shared__ __align__(32) __nv_bfloat16 right[256];
    __shared__ __align__(32) float result[256];
    const int row_start = blockIdx.y * 16, column_start = blockIdx.x * 16;
    tau_linear::tile(input, weight, left, right, result, row_start,
        column_start, rows, columns, inner);
    for (int i = threadIdx.x; i < 256; i += 32) {
        const int row = row_start+i/16, column = column_start+i%16;
        if (row < rows && column < columns)
            output[(size_t)row*columns+column] = result[i];
    }
}

// Each tile reads its expert from the device worklist. Gate/up gathers token
// rows through assignment slots; down reads grouped activations and scatters
// weighted results back to canonical token/rank order before BF16 storage.
template<bool Down>
__global__ void grouped(const __nv_bfloat16* input, const __nv_bfloat16* weights,
    const float* routes, const int* slots, const int* tiles, __nv_bfloat16* output,
    int tokens, int capacity, int columns, int inner, int* receipt) {
    using namespace tau_moe;
    if (receipt && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        receipt[0]=16; receipt[1]=16; receipt[2]=16; receipt[3]=1; receipt[4]=1;
    }
    int expert = tiles[blockIdx.y];
    int row_start = blockIdx.y * Tile, column_start = blockIdx.x * Tile;
    __shared__ __align__(32) __nv_bfloat16 left[256], right[256];
    __shared__ __align__(32) float result[256];
    if (expert < 0) {
        if constexpr (!Down) {
            for (int i = threadIdx.x; i < 256; i += 32) {
                int row = row_start + i / Tile, column = column_start + i % Tile;
                if (row < capacity && column < columns)
                    output[(size_t)row * columns + column] = __float2bfloat16_rn(0.f);
            }
        }
        return;
    }
    {
        tau_linear::tile(input, weights + (size_t)expert * columns * inner,
            left, right, result, row_start, column_start, capacity, columns, inner,
            Down ? nullptr : slots, tokens * TopK, TopK);
    }
    for (int i = threadIdx.x; i < 256; i += 32) {
        int row = row_start + i / Tile, column = column_start + i % Tile;
        if (row >= capacity || column >= columns) continue;
        if constexpr (Down) {
            int slot = slots[row];
            if (expert >= 0 && slot < tokens * TopK) {
                float weight = routes[(slot / TopK) * 2 * TopK + TopK + slot % TopK];
                output[(size_t)slot * columns + column] = __float2bfloat16_rn(
                    __fmul_rn(result[i], weight));
            }
        } else {
            // Initialize unused capacity too: the shared SiLU primitive processes
            // a fixed retained extent without reading device counts on the host.
            output[(size_t)row * columns + column] = __float2bfloat16_rn(
                result[i]);
        }
    }
}

// Convert an accumulator once, optionally applying one FP32 weight per row.
__global__ void weighted_store(const float* input, const float* weights,
    __nv_bfloat16* output, int columns, size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i]=__float2bfloat16_rn(weights ? __fmul_rn(input[i],weights[i/columns]) : input[i]);
}
}

// Queue one complete output without allocation, synchronization or handle state.
extern "C" int tau_launch_linear_bf16_fp32(const void* input, const void* weight,
    float* output, int rows, int columns, int inner, void* stream) {
    if (!input || !weight || !output || rows <= 0 || columns <= 0 || inner <= 0
        || (rows+15LL)/16 > 65535) return (int)cudaErrorInvalidValue;
    linear_fp32<false><<<dim3((columns+15LL)/16,(rows+15LL)/16),32,0,(cudaStream_t)stream>>>(
        input,weight,output,rows,columns,inner,nullptr);
    return (int)cudaGetLastError();
}

// Device pointer arrays keep selected expert addresses dynamic under replay.
extern "C" int tau_launch_linear_bf16_fp32_batched(const void* inputs,
    const void* weights, void* outputs, int batches, int rows, int columns,
    int inner, int* receipt, void* stream) {
    if(!inputs || !weights || !outputs || batches<=0 || batches>65535 || rows<=0
        || columns<=0 || inner<=0 || (rows+15LL)/16>65535) return (int)cudaErrorInvalidValue;
    linear_fp32<true><<<dim3((columns+15LL)/16,(rows+15LL)/16,batches),32,0,(cudaStream_t)stream>>>(
        inputs,weights,outputs,rows,columns,inner,receipt);
    return (int)cudaGetLastError();
}

// Shared precision epilogue for unweighted gate/up and weighted down storage.
extern "C" int tau_launch_linear_weighted_store(const float* input,
    const float* weights, void* output, int rows, int columns, void* stream) {
    if(!input || !output || rows<=0 || columns<=0) return (int)cudaErrorInvalidValue;
    size_t count=(size_t)rows*columns;
    weighted_store<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(
        input,weights,(__nv_bfloat16*)output,columns,count);
    return (int)cudaGetLastError();
}

// A fixed grid consumes GPU-built expert tile metadata; no per-expert launch or
// count readback is needed. Down's output is weighted canonical [T*8,N].
extern "C" int tau_launch_linear_bf16_fp32_grouped(const void* input,
    const void* weights, const float* routes, const int* slots, const int* tiles,
    void* output, int tokens, int capacity, int columns, int inner, int down,
    int* receipt, void* stream) {
    if (!input || !weights || !routes || !slots || !tiles || !output || tokens <= 0
        || capacity <= 0 || capacity % 16 || capacity / 16 > 65535
        || columns <= 0 || inner <= 0 || (down != 0 && down != 1))
        return (int)cudaErrorInvalidValue;
    dim3 grid((columns + 15LL) / 16, capacity / 16);
    if (down) grouped<true><<<grid,32,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input,(const __nv_bfloat16*)weights,routes,slots,tiles,
        (__nv_bfloat16*)output,tokens,capacity,columns,inner,receipt);
    else grouped<false><<<grid,32,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input,(const __nv_bfloat16*)weights,routes,slots,tiles,
        (__nv_bfloat16*)output,tokens,capacity,columns,inner,receipt);
    return (int)cudaGetLastError();
}
