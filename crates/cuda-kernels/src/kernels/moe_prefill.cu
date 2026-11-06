// Stateless prefill launch composition. The Rust owner retains every buffer;
// routing, grouping, projections and reduction never read counts back to CPU.
#include <cuda_runtime.h>
#include <climits>
#include "moe_dispatch.cuh"
#include "moe_gemm.cuh"

extern "C" int tau_launch_moe_routing_fp32(const void*, float*, int, void*);
extern "C" int tau_launch_moe_prefill_dispatch(const float*,int*,int*,int*,int*,int,int,int,void*);
extern "C" int tau_launch_silu_and_mul(const void*,void*,int,int,int,void*);
extern "C" int tau_launch_moe_prefill_combine(const void*,void*,const int*,int,int,void*);

// Queue the same FP32 routing/MMA and BF16 storage boundaries as the reference
// expert loop, with a fixed number of launches independent of active experts.
extern "C" int tau_launch_moe_prefill(const void* input, const void* logits,
    const void* gate_up, const void* down, float* routes, int* meta, int* offsets,
    int* slots, int* tiles, void* projected, void* activated, void* contributions,
    void* output, int tokens, int hidden, int intermediate, int capacity,
    const tau_moe::GemmTile* up_config, const tau_moe::GemmTile* down_config,
    int* receipts, void* stream) {
    if (!up_config || !down_config || up_config->m != down_config->m || !input || !logits || !gate_up || !down || !routes || !meta || !offsets
        || !slots || !tiles || !projected || !activated || !contributions || !output
        || tokens <= 0 || hidden <= 0 || intermediate <= 0 || intermediate > INT_MAX / 2)
        return (int)cudaErrorInvalidValue;
    int code = tau_launch_moe_routing_fp32(logits,routes,tokens,stream);
    if (code) return code;
    code = tau_launch_moe_prefill_dispatch(routes,meta,offsets,slots,tiles,tokens,capacity,up_config->m,stream);
    if (code) return code;
    code = tau_launch_moe_prefill_gemm(input,gate_up,routes,slots,tiles,
        projected,tokens,capacity,2*intermediate,hidden,0,up_config,receipts,stream);
    if (code) return code;
    code = tau_launch_silu_and_mul(projected,activated,capacity,intermediate,1,stream);
    if (code) return code;
    code = tau_launch_moe_prefill_gemm(activated,down,routes,slots,tiles,
        contributions,tokens,capacity,hidden,intermediate,1,down_config,receipts ? receipts + 5 : nullptr,stream);
    if (code) return code;
    return tau_launch_moe_prefill_combine(contributions,output,meta,tokens,hidden,stream);
}
