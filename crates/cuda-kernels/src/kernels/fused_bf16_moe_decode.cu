// Device-resident BF16 MoE decode with FP32 routing/projection/combination.
// The same routing and K-ordered MMA primitives serve prefill and decode. All
// scratch and pointer arrays are owned by the model before capture/replay.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include "moe_dispatch.cuh"
#include "moe_decode_gemm.cuh"

extern "C" int tau_launch_moe_routing_fp32(const void*,float*,int,void*);
extern "C" int tau_launch_linear_weighted_store(const float*,const float*,void*,int,int,void*);
extern "C" int tau_launch_silu_and_mul(const void*,void*,int,int,int,void*);

namespace {
constexpr int Experts=256, TopK=8;

// Read the packed routing row before reusing its allocation for pointer arrays.
// A malformed/nonfinite route uses safe expert zero and a NaN contribution;
// inference then rejects the result without issuing an out-of-range weight read.
__global__ void prepare_pointers(const void* input, const void* gate_up,
    const void* down, int* ids, float* weights, std::uintptr_t* pointers,
    float* gate_fp32, __nv_bfloat16* activated, float* down_fp32,
    int hidden, int intermediate) {
    int rank=threadIdx.x;
    float id=0.f, weight=0.f;
    if(rank<TopK) {
        id=((const float*)pointers)[rank];
        weight=((const float*)pointers)[TopK+rank];
    }
    __syncwarp();
    if(rank<TopK) {
        bool valid=isfinite(id)&&id>=0.f&&id<Experts&&isfinite(weight);
        int expert=valid?(int)id:0;
        ids[rank]=expert;weights[rank]=valid?weight:NAN;
        pointers[0*TopK+rank]=(std::uintptr_t)((const __nv_bfloat16*)gate_up+(size_t)expert*2*intermediate*hidden);
        pointers[1*TopK+rank]=(std::uintptr_t)input;
        pointers[2*TopK+rank]=(std::uintptr_t)(gate_fp32+(size_t)rank*2*intermediate);
        pointers[3*TopK+rank]=(std::uintptr_t)((const __nv_bfloat16*)down+(size_t)expert*hidden*intermediate);
        pointers[4*TopK+rank]=(std::uintptr_t)(activated+(size_t)rank*intermediate);
        pointers[5*TopK+rank]=(std::uintptr_t)(down_fp32+(size_t)rank*hidden);
    }
}

// Weighted expert outputs have already rounded once to BF16. Sum them in FP32
// routing-rank order and round only the final token vector.
__global__ void combine(const __nv_bfloat16* experts, __nv_bfloat16* output,int hidden) {
    int column=blockIdx.x*blockDim.x+threadIdx.x;
    if(column>=hidden)return;
    output[column]=tau_moe::combine_ranked(experts,hidden,column);
}
}

// Queue routing, dynamic dispatch, expert projections and deterministic sum on
// one stream. No allocation, host readback or library-handle mutation occurs.
extern "C" int tau_launch_fused_bf16_moe_decode(const void* input,
    const void* logits,const void* gate_up,const void* down,void* selected_ids,
    void* selected_weights,void* pointer_workspace,void* gate_up_output,
    void* activated_output,void* expert_output,void* gate_up_fp32,
    void* expert_fp32,void* output,int hidden,int intermediate,int num_experts,
    int top_k,const tau_moe::DecodeProfile* gate_config,
    const tau_moe::DecodeProfile* down_config,int* receipts,void* stream) {
    if(!input||!logits||!gate_up||!down||!selected_ids||!selected_weights
        ||!pointer_workspace||!gate_up_output||!activated_output||!expert_output
        ||!gate_up_fp32||!expert_fp32||!output||hidden<=0||intermediate<=0
        ||!gate_config||!down_config||num_experts!=Experts||top_k!=TopK)return -1;
    int code=tau_launch_moe_routing_fp32(logits,(float*)pointer_workspace,1,stream);
    if(code)return code;
    auto* pointers=(std::uintptr_t*)pointer_workspace;
    prepare_pointers<<<1,32,0,(cudaStream_t)stream>>>(input,gate_up,down,
        (int*)selected_ids,(float*)selected_weights,pointers,(float*)gate_up_fp32,
        (__nv_bfloat16*)activated_output,(float*)expert_fp32,hidden,intermediate);
    if((code=(int)cudaGetLastError()))return code;
    code=tau_launch_moe_decode_gemm(pointers+TopK,pointers,pointers+2*TopK,
        TopK,2*intermediate,hidden,gate_config,receipts,stream);
    if(code)return code;
    code=tau_launch_linear_weighted_store((float*)gate_up_fp32,nullptr,gate_up_output,
        TopK,2*intermediate,stream);
    if(code)return code;
    code=tau_launch_silu_and_mul(gate_up_output,activated_output,TopK,intermediate,1,stream);
    if(code)return code;
    code=tau_launch_moe_decode_gemm(pointers+4*TopK,pointers+3*TopK,
        pointers+5*TopK,TopK,hidden,intermediate,down_config,receipts ? receipts+TopK*5 : nullptr,stream);
    if(code)return code;
    code=tau_launch_linear_weighted_store((float*)expert_fp32,(float*)selected_weights,
        expert_output,TopK,hidden,stream);
    if(code)return code;
    combine<<<(hidden+255)/256,256,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)expert_output,(__nv_bfloat16*)output,hidden);
    return (int)cudaGetLastError();
}
