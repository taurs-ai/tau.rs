// MoE shared-expert gate: FP32 dot/sigmoid and one final BF16 fused-add store.
// Dense MLP projection/activation arithmetic remains separate and unchanged.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "fp32_precision.cuh"

namespace {
// One CTA owns a token. All warps contribute to the dot before any output uses
// its gate. Masked feature tails contribute zero and never read outside storage.
template<int Elements>
__global__ void shared_combine(const __nv_bfloat16* input,const __nv_bfloat16* weight,
    const __nv_bfloat16* shared,const __nv_bfloat16* routed,__nv_bfloat16* output,int width) {
    __shared__ float partial[32], gate;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
    size_t row=(size_t)blockIdx.x*width;
    float p[8]={};
    for(int i=0;i<Elements;++i) {
        int column=threadIdx.x*Elements+i;
        if(column<width) p[i]=__fmul_rn(__bfloat162float(input[row+column]),__bfloat162float(weight[column]));
    }
    float sum=tau_precision::sum_eight(p);
    for(int d=16;d;d/=2) sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,d));
    if(!lane)partial[warp]=sum;
    __syncthreads();
    if(!warp) {
        float total=lane<warps?partial[lane]:0.f;
        for(int d=16;d;d/=2) total=__fadd_rn(total,__shfl_xor_sync(0xffffffff,total,d));
        if(!lane)gate=tau_precision::sigmoid(total);
    }
    __syncthreads();
    for(int i=0;i<Elements;++i) {
        int column=threadIdx.x*Elements+i;
        if(column<width) output[row+column]=__float2bfloat16_rn(__fmaf_rn(gate,
            __bfloat162float(shared[row+column]),__bfloat162float(routed[row+column])));
    }
}
}

// Choose bounded power-of-two geometry before enqueue; no allocation or readback.
extern "C" int tau_launch_moe_shared_combine(const void* input,const void* weight,
    const void* shared,const void* routed,void* output,int rows,int width,void* stream) {
    if(!input||!weight||!shared||!routed||!output||rows<=0||width<=0||width>8192)
        return (int)cudaErrorInvalidValue;
    int tile=1;while(tile<width)tile*=2;
    int warps=1;while(warps<(width+255)/256)warps*=2;
    warps=max(4,min(32,warps));int elements=max(1,tile/(warps*32));
    #define LAUNCH(E) shared_combine<E><<<rows,warps*32,0,(cudaStream_t)stream>>>( \
        (const __nv_bfloat16*)input,(const __nv_bfloat16*)weight,(const __nv_bfloat16*)shared, \
        (const __nv_bfloat16*)routed,(__nv_bfloat16*)output,width)
    switch(elements){case 1:LAUNCH(1);break;case 2:LAUNCH(2);break;case 4:LAUNCH(4);break;case 8:LAUNCH(8);break;default:return (int)cudaErrorInvalidValue;}
    #undef LAUNCH
    return (int)cudaGetLastError();
}
