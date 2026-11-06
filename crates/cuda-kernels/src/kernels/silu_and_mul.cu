#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>

// Fused SwiGLU elementwise stage for a packed gate/up projection. Candle owns
// the input and output allocations and supplies its active CUDA stream.

namespace {

// A conventional elementwise block width gives one-token decode enough blocks
// to cover intermediate dimensions larger than 256 while retaining coalesced
// accesses within each packed half.
constexpr int kThreads = 256;

// Promotes a stored input value to FP32 for the sigmoid and multiplication.
template <typename T>
__device__ float to_float(T value);

// F32 input already uses the arithmetic precision needed by this kernel.
template <>
__device__ float to_float<float>(float value) {
    return value;
}

// BF16 input is widened before evaluating SiLU and the gate/up product.
template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

// Converts the FP32 gated value back to the tensor's storage dtype.
template <typename T>
__device__ T from_float(float value);

// F32 output preserves the computed value directly.
template <>
__device__ float from_float<float>(float value) {
    return value;
}

// BF16 output uses round-to-nearest conversion after the FP32 calculation.
template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

// Evaluates one or more output elements per thread. Each input row stores
// `[gate..., up...]`; the output row is half as wide and contains
// `gate * sigmoid(gate) * up`.
template <typename T>
__global__ void silu_and_mul_kernel(
    const T* input,
    T* output,
    int rows,
    int columns) {
    const long long output_count =
        static_cast<long long>(rows) * static_cast<long long>(columns);
    const long long thread_index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long thread_stride =
        static_cast<long long>(gridDim.x) * blockDim.x;

    // A grid-stride loop also covers unusually large tensors whose output has
    // more elements than a single maximum-width one-dimensional grid.
    for (long long output_index = thread_index; output_index < output_count;
         output_index += thread_stride) {
        const long long row = output_index / columns;
        const int column = static_cast<int>(output_index % columns);
        const long long input_row = row * static_cast<long long>(columns) * 2;
        const float gate = to_float(input[input_row + column]);
        const float up = to_float(input[input_row + columns + column]);
        const float silu = gate / (1.0f + expf(-gate));
        output[output_index] = from_float<T>(silu * up);
    }
}

}  // namespace

// Validates the language-neutral ABI and enqueues the typed elementwise kernel.
// Dtype 0 is F32 and dtype 1 is BF16, matching Rust's `SiluAndMulDType`.
// Execution remains asynchronous on the caller-provided stream.
extern "C" int tau_launch_silu_and_mul(
    const void* input,
    void* output,
    int rows,
    int columns,
    int dtype,
    void* stream) {
    // Tensor pointers and logical dimensions must describe a non-empty packed
    // gate/up tensor. A null stream remains CUDA's valid default stream.
    if (input == nullptr || output == nullptr || rows <= 0 || columns <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const long long output_count =
        static_cast<long long>(rows) * static_cast<long long>(columns);
    const long long required_blocks =
        (output_count + kThreads - 1) / kThreads;
    // CUDA one-dimensional grids use a signed-31-bit practical limit on the
    // supported NVIDIA targets. Capping the grid still lets the grid-stride
    // loop process every output element.
    const int blocks = static_cast<int>(
        required_blocks > INT_MAX ? INT_MAX : required_blocks);
    const dim3 grid(blocks);
    const dim3 block(kThreads);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // Cast opaque ABI pointers only after the dtype tag selects their concrete
    // representation.
    switch (dtype) {
        case 0:
            silu_and_mul_kernel<<<grid, block, 0, cuda_stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                rows,
                columns);
            break;
        case 1:
            silu_and_mul_kernel<<<grid, block, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<__nv_bfloat16*>(output),
                rows,
                columns);
            break;
        default:
            return static_cast<int>(cudaErrorInvalidValue);
    }

    // Surface configuration and enqueue failures without synchronizing.
    return static_cast<int>(cudaGetLastError());
}
