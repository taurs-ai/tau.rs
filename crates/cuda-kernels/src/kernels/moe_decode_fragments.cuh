#pragma once
#include <mma.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tau_moe {
using DecodeBf16 = __nv_bfloat16;

// Projection arithmetic owners used by the same staged decode pipeline. Each
// warp keeps its own consecutive K16 accumulator, with no split-K reduction.
struct DecodeWmma16 {
    static constexpr int Columns=16, StagedRows=16, Backend=1;
    template<int N,int W> struct Accumulator {
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> values[(N/16)/W];
        __device__ void clear() {
            #pragma unroll
            for(int f=0;f<(N/16)/W;++f) nvcuda::wmma::fill_fragment(values[f],0.f);
        }
        __device__ void step(const DecodeBf16* a,const DecodeBf16* b,int stride,int k,int warp) {
            #pragma unroll
            for(int f=0;f<(N/16)/W;++f) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,DecodeBf16,nvcuda::wmma::row_major> left;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,DecodeBf16,nvcuda::wmma::col_major> right;
                nvcuda::wmma::load_matrix_sync(left,a+k,stride);
                nvcuda::wmma::load_matrix_sync(right,b+(warp+f*W)*16*stride+k,stride);
                nvcuda::wmma::mma_sync(values[f],left,right,values[f]);
            }
        }
        __device__ void store(float* output,int warp) {
            #pragma unroll
            for(int f=0;f<(N/16)/W;++f)
                nvcuda::wmma::store_matrix_sync(output+(warp+f*W)*16,values[f],N,nvcuda::wmma::mem_row_major);
        }
    };
};

// The finer native fragment admits N32/four-warps and N64/eight-warps. Only
// logical row zero exists: other A rows are register zeros, and only that output
// row is staged. The model still receives exactly the same FP32 projection.
struct DecodeMma8 {
    static constexpr int Columns=8, StagedRows=1, Backend=2;
    template<int N,int W> struct Accumulator {
        float values[(N/8)/W][4];
        __device__ void clear() {
            #pragma unroll
            for(int f=0;f<(N/8)/W;++f) {
                #pragma unroll
                for(int i=0;i<4;++i) values[f][i]=0.f;
            }
        }
        __device__ void step(const DecodeBf16* a,const DecodeBf16* b,int stride,int k,int warp) {
            const int lane=threadIdx.x%32, group=lane/4, pair=(lane%4)*2;
            // PTX m16n8k16 BF16: A words 0/2 hold rows 0..7 at K offsets
            // 0/8; words 1/3 hold rows 8..15. Only group zero has a real A row.
            const unsigned a0=group==0?*reinterpret_cast<const unsigned*>(a+k+pair):0;
            const unsigned a2=group==0?*reinterpret_cast<const unsigned*>(a+k+pair+8):0;
            #pragma unroll
            for(int f=0;f<(N/8)/W;++f) {
                int column=(warp+f*W)*8+group;
                const auto* row=b+column*stride+k+pair;
                const unsigned b0=*reinterpret_cast<const unsigned*>(row);
                const unsigned b1=*reinterpret_cast<const unsigned*>(row+8);
                auto& c=values[f];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
            }
        }
        // Logical row zero can go straight to its FP32 slot. No padded result
        // matrix or block-wide publication barrier is needed between warps.
        __device__ void store(float* output,int warp,int valid) {
            const int lane=threadIdx.x%32;
            if(lane<4) {
                #pragma unroll
                for(int f=0;f<(N/8)/W;++f) {
                    int column=(warp+f*W)*8+lane*2;
                    if(column<valid) output[column]=values[f][0];
                    if(column+1<valid) output[column+1]=values[f][1];
                }
            }
        }
    };
};
}
