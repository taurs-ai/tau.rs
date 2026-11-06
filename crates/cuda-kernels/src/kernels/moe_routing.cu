// Batched FP32 softmax/top-8 routing for 256 BF16 expert logits.
// Preserve the golden router's operation order: eight contiguous scores per
// lane, interleaved pair reductions, warp butterfly sum, and full-range division.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "fp32_precision.cuh"

namespace {
using tau_precision::exponential;
using tau_precision::divide;
using tau_precision::sum_eight;

// One warp handles a token and writes packed [eight IDs, eight FP32 weights].
__global__ void route(const __nv_bfloat16* logits, float* output) {
    const int lane = threadIdx.x;
    float scores[8], probabilities[8], selected[8];
    int ids[8];
    bool finite = true;
    float maximum = -INFINITY;
    for (int i=0;i<8;++i) {
        scores[i] = __bfloat162float(logits[(size_t)blockIdx.x*256+lane*8+i]);
        maximum = fmaxf(maximum,scores[i]);
        finite &= isfinite(scores[i]);
    }
    if (!__all_sync(0xffffffff,finite)) {
        if(lane<16) output[(size_t)blockIdx.x*16+lane] = NAN;
        return;
    }
    for(int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    for(int i=0;i<8;++i) probabilities[i]=exponential(__fsub_rn(scores[i],maximum));
    float denominator=sum_eight(probabilities);
    for(int d=16;d;d/=2) denominator=__fadd_rn(denominator,__shfl_xor_sync(0xffffffff,denominator,d));
    for(int i=0;i<8;++i) probabilities[i]=divide(probabilities[i],denominator);
    for(int rank=0;rank<8;++rank) {
        float best=-INFINITY;
        for(int i=0;i<8;++i) best=fmaxf(best,scores[i]);
        for(int d=16;d;d/=2) best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,d));
        int id=256;
        for(int i=0;i<8;++i) if(scores[i]==best) id=min(id,lane*8+i);
        for(int d=16;d;d/=2) id=min(id,__shfl_xor_sync(0xffffffff,id,d));
        float weight=0.f;
        for(int i=0;i<8;++i) if(lane*8+i==id) {weight=probabilities[i];scores[i]=-INFINITY;}
        for(int d=16;d;d/=2) weight=__fadd_rn(weight,__shfl_xor_sync(0xffffffff,weight,d));
        ids[rank]=id;selected[rank]=weight;
    }
    float norm=sum_eight(selected);
    if(lane<8) {
        output[(size_t)blockIdx.x*16+lane]=(float)ids[lane];
        output[(size_t)blockIdx.x*16+8+lane]=divide(selected[lane],norm>0.f?norm:1.f);
    }
}
}

// Validate launch geometry without allocation or synchronization.
extern "C" int tau_launch_moe_routing_fp32(const void* logits, float* output,
    int rows, void* stream) {
    if(!logits || !output || rows<=0) return (int)cudaErrorInvalidValue;
    route<<<rows,32,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)logits,output);
    return (int)cudaGetLastError();
}
