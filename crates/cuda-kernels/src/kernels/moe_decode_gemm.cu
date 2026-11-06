// Single-token routed projection with cooperative weight loading. Gate/up and
// down share this backend; their precision/storage epilogues stay with the MoE
// composer. Each warp owns complete FP32 accumulators in consecutive K16 order.
#include <mma.h>
#include "bf16_staging.cuh"
#include "moe_decode_gemm.cuh"
#include "moe_decode_fragments.cuh"
#include "moe_gemm.cuh"

extern "C" int tau_launch_linear_bf16_fp32_batched(const void*,const void*,void*,
    int,int,int,int,int*,void*);

namespace {
using Bf16 = __nv_bfloat16;
struct Arguments {
    const Bf16* const* inputs;
    const Bf16* const* weights;
    float* const* outputs;
    int columns, inner;
    int* receipt;
};

// The logical input row uses coalesced 16-byte copies. The old layout stages
// fifteen zero rows; the finer layout supplies them in registers. The shared
// staging helper masks K/output tails and misaligned rows for both layouts.
template<int N, int K, int W, typename Layout>
__device__ void load(Arguments p, Bf16* a, Bf16* b, int k_start) {
    constexpr int Stride=K+8;
    const Bf16* input=p.inputs[blockIdx.z];
    const Bf16* weights=p.weights[blockIdx.z];
    for(int i=threadIdx.x;i<Layout::StagedRows*(K/8);i+=32*W) {
        int row=i/(K/8), k=(i%(K/8))*8;
        const Bf16* source=input;
        if(row==0 && k_start+k<p.inner) source+=k_start+k;
        tau_moe::copy_eight(a+row*Stride+k,source,row==0,max(0,p.inner-k_start-k));
    }
    for(int i=threadIdx.x;i<N*(K/8);i+=32*W) {
        int column=blockIdx.x*N+i/(K/8), k=(i%(K/8))*8;
        const Bf16* source=weights;
        if(column<p.columns && k_start+k<p.inner)
            source+=(size_t)column*p.inner+k_start+k;
        tau_moe::copy_eight(b+(i/(K/8))*Stride+k,source,column<p.columns,max(0,p.inner-k_start-k));
    }
    __pipeline_commit();
}

template<int N,int K,int W,int S,typename Layout>
__global__ void project(Arguments p) {
    using namespace nvcuda;
    constexpr int Stride=K+8;
    __shared__ __align__(32) Bf16 a[S][Layout::StagedRows*Stride],b[S][N*Stride];
    __shared__ __align__(32) float result[Layout::StagedRows*N];
    if(p.receipt && blockIdx.x==0 && threadIdx.x==0) {
        int* r=p.receipt+blockIdx.z*5;
        r[0]=Layout::Backend;r[1]=N;r[2]=K;r[3]=W;r[4]=S;
    }
    typename Layout::template Accumulator<N,W> accumulator;
    accumulator.clear();
    const int chunks=(p.inner+K-1LL)/K,warp=threadIdx.x/32;
    #pragma unroll
    for(int stage=0;stage<S;++stage)
        if(stage<chunks) load<N,K,W,Layout>(p,a[stage],b[stage],stage*K);
    for(int chunk=0;chunk<chunks;++chunk) {
        int stage=chunk%S;
        if(chunks-chunk>=S) __pipeline_wait_prior(S-1);
        else __pipeline_wait_prior(0);
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K;k+=16) {
            if(chunk*K+k>=p.inner) break;
            accumulator.step(a[stage],b[stage],Stride,k,warp);
        }
        __syncthreads(); // Preserve operands until every warp finishes this stage.
        if(chunk+S<chunks) load<N,K,W,Layout>(p,a[stage],b[stage],(chunk+S)*K);
    }
    if constexpr (Layout::Columns==8) {
        accumulator.store(p.outputs[blockIdx.z]+blockIdx.x*N,warp,p.columns-blockIdx.x*N);
    } else {
        accumulator.store(result,warp);
        __syncthreads();
        // Only the logical first row is emitted; no BF16 rounding belongs here.
        for(int i=threadIdx.x;i<N;i+=32*W)
            if(blockIdx.x*N+i<p.columns) p.outputs[blockIdx.z][blockIdx.x*N+i]=result[i];
    }
}

template<int N,int K,int W,int S,typename Layout>
int launch(Arguments p,int batches,cudaStream_t stream) {
    if constexpr ((N/Layout::Columns)%W || tau_moe::shared_memory_upper_bound<Layout::StagedRows,N,K,S>()>tau_moe::MaxStaticSharedBytes)
        return (int)cudaErrorInvalidValue;
    else {
        project<N,K,W,S,Layout><<<dim3((p.columns+N-1LL)/N,1,batches),W*32,0,stream>>>(p);
        return (int)cudaGetLastError();
    }
}
template<int N,int K,int S,typename Layout>
int warps(Arguments p,int batches,int w,cudaStream_t stream) {
    switch(w) {
        case 1:return launch<N,K,1,S,Layout>(p,batches,stream);
        case 2:return launch<N,K,2,S,Layout>(p,batches,stream);
        case 4:return launch<N,K,4,S,Layout>(p,batches,stream);
        case 8:return launch<N,K,8,S,Layout>(p,batches,stream);
        default:return (int)cudaErrorInvalidValue;
    }
}
template<int N,int K,typename Layout>
int stages(Arguments p,int batches,const tau_moe::DecodeProfile& c,cudaStream_t stream) {
    if(c.stages==1) return warps<N,K,1,Layout>(p,batches,c.warps,stream);
    if(c.stages==2) return warps<N,K,2,Layout>(p,batches,c.warps,stream);
    return (int)cudaErrorInvalidValue;
}
template<int N,typename Layout>
int inner(Arguments p,int batches,const tau_moe::DecodeProfile& c,cudaStream_t stream) {
    switch(c.k) {
        case 32:return stages<N,32,Layout>(p,batches,c,stream);
        case 64:return stages<N,64,Layout>(p,batches,c,stream);
        case 128:return stages<N,128,Layout>(p,batches,c,stream);
        default:return (int)cudaErrorInvalidValue;
    }
}
}

// All pointer arrays are device-resident and recomputed by routing on each
// replay. The host profile is immutable and resolved before graph construction.
extern "C" int tau_launch_moe_decode_gemm(const void* inputs,const void* weights,
    void* outputs,int batches,int columns,int k,const tau_moe::DecodeProfile* c,
    int* receipt,void* stream) {
    if(!inputs||!weights||!outputs||!c||batches<=0||batches>65535||columns<=0||k<=0)
        return (int)cudaErrorInvalidValue;
    if(c->backend==0 && c->n==16 && c->k==16 && c->warps==1 && c->stages==1)
        return tau_launch_linear_bf16_fp32_batched(inputs,weights,outputs,batches,1,columns,k,receipt,stream);
    if(c->backend!=1 && c->backend!=2) return (int)cudaErrorInvalidValue;
    Arguments p{(const Bf16* const*)inputs,(const Bf16* const*)weights,
        (float* const*)outputs,columns,k,receipt};
    switch(c->n) {
        case 32:return c->backend==1
            ?inner<32,tau_moe::DecodeWmma16>(p,batches,*c,(cudaStream_t)stream)
            :inner<32,tau_moe::DecodeMma8>(p,batches,*c,(cudaStream_t)stream);
        case 64:return c->backend==1
            ?inner<64,tau_moe::DecodeWmma16>(p,batches,*c,(cudaStream_t)stream)
            :inner<64,tau_moe::DecodeMma8>(p,batches,*c,(cudaStream_t)stream);
        case 128:return c->backend==1
            ?inner<128,tau_moe::DecodeWmma16>(p,batches,*c,(cudaStream_t)stream)
            :inner<128,tau_moe::DecodeMma8>(p,batches,*c,(cudaStream_t)stream);
        default:return (int)cudaErrorInvalidValue;
    }
}
