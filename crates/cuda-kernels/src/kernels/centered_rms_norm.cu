#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Centered RMSNorm kernels for the hybrid Candle/native boundary. The caller owns all
// device allocations and supplies Candle's active stream; this file does not
// implement a separate tensor runtime.

namespace {

// Keep complete warps even for narrow rows: lanes without input contribute zero
// and still participate in every shuffle. Wider rows use strided accumulation.
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
static_assert(kThreads % kWarpSize == 0 && kWarps <= kWarpSize);

// All 32 lanes must call this together; lane zero receives the complete sum.
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Promotes an input or weight value to FP32 so both supported storage dtypes use
// the same accumulation precision.
template <typename T>
__device__ float to_float(T value);

// F32 storage already has the precision used by the reduction.
template <>
__device__ float to_float<float>(float value) {
    return value;
}

// BF16 values are widened before multiplication and summation.
template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

// Converts the normalized FP32 result back to the tensor's storage dtype.
template <typename T>
__device__ T from_float(float value);

// Preserve the normalized value directly for F32 output tensors.
template <>
__device__ float from_float<float>(float value) {
    return value;
}

// Round the normalized value to the nearest representable BF16 value.
template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

// Normalizes one row per block. Every thread accumulates a strided subset of the
// row in FP32, the block reduces those partial sums, and all threads then write
// their strided output elements. `weight` is centered, so its effective scale is
// `1 + weight[column]` rather than `weight[column]`.
template <typename T>
__global__ void centered_rms_norm_kernel(
    const T* input,
    const T* weight,
    T* output,
    int columns,
    float epsilon) {
    // The launcher creates exactly one block for each logical input row.
    const int row = blockIdx.x;
    const int row_offset = row * columns;

    // Accumulate this thread's portion of the sum of squares. The strided loop
    // supports arbitrary positive hidden widths with the fixed-size block.
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = to_float(input[row_offset + column]);
        square_sum += value * value;
    }

    // Reduce within each warp in registers, then publish one sum per warp.
    // All lanes participate, including those beyond the end of a short row.
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    __shared__ float warp_sums[kWarps];
    __shared__ float row_sum;
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();

    // The first warp combines the eight warp sums. Its unused lanes contribute
    // zero, so the same full-warp shuffle mask remains valid. A separate result
    // slot avoids overwriting the warp sums while they are being read.
    if (warp == 0) {
        const float total = warp_sum(lane < kWarps ? warp_sums[lane] : 0.0f);
        if (lane == 0) {
            row_sum = total;
        }
    }
    __syncthreads();

    // The shared result is visible to every thread after the final barrier.
    const float inverse_rms = rsqrtf(row_sum / columns + epsilon);

    // Apply normalization and the centered learned scale, then convert back to
    // the original storage dtype.
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = to_float(input[row_offset + column]);
        const float scale = 1.0f + to_float(weight[column]);
        output[row_offset + column] = from_float<T>(value * inverse_rms * scale);
    }
}

// Compile fixed-width memory operations, then select among them at launch.
// The preferred width is a policy ceiling; actual pointers may require less.
template <typename T, int VectorBytes>
struct alignas(VectorBytes) NormVector {
    T values[VectorBytes / sizeof(T)];
};

// Specialized widths bound the register array and let nvcc unroll its accesses.
// Input values survive the reduction in registers, avoiding a second global read.
template <typename T, int Columns, int Threads, int VectorBytes>
__global__ void vectorized_centered_rms_norm_kernel(
    const T* input, const T* weight, T* output, float epsilon) {
    constexpr int kElements = VectorBytes / sizeof(T);
    constexpr int kVectors = Columns / kElements;
    constexpr int kIterations = (kVectors + Threads - 1) / Threads;
    constexpr int kBlockWarps = Threads / kWarpSize;
    static_assert(Columns % kElements == 0 && Threads % kWarpSize == 0);
    static_assert(VectorBytes == 4 || VectorBytes == 8 || VectorBytes == 16);
    static_assert(sizeof(NormVector<T, VectorBytes>) == VectorBytes);
    const size_t row_offset = size_t(blockIdx.x) * kVectors;
    const auto* row =
        reinterpret_cast<const NormVector<T, VectorBytes>*>(input) + row_offset;
    const auto* scales = reinterpret_cast<const NormVector<T, VectorBytes>*>(weight);
    auto* result =
        reinterpret_cast<NormVector<T, VectorBytes>*>(output) + row_offset;
    float values[kIterations][kElements];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int vector = threadIdx.x + i * Threads;
        if (vector < kVectors) {
            const NormVector<T, VectorBytes> loaded = row[vector];
#pragma unroll
            for (int element = 0; element < kElements; ++element) {
                const float value = to_float(loaded.values[element]);
                values[i][element] = value;
                sum += value * value;
            }
        }
    }

    // Every launched lane participates, including lanes with no vector to load.
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    sum = warp_sum(sum);
    if constexpr (Threads == kWarpSize) {
        sum = __shfl_sync(0xffffffffu, sum, 0);
    } else {
        __shared__ float partial[kBlockWarps];
        __shared__ float total;
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            const float reduced = warp_sum(lane < kBlockWarps ? partial[lane] : 0.0f);
            if (lane == 0) total = reduced;
        }
        __syncthreads();
        sum = total;
    }
    const float inverse_rms = rsqrtf(sum / Columns + epsilon);
#pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int vector = threadIdx.x + i * Threads;
        if (vector < kVectors) {
            const NormVector<T, VectorBytes> scale = scales[vector];
            NormVector<T, VectorBytes> normalized;
#pragma unroll
            for (int element = 0; element < kElements; ++element) {
                normalized.values[element] = from_float<T>(
                    values[i][element] * inverse_rms *
                    (1.0f + to_float(scale.values[element])));
            }
            result[vector] = normalized;
        }
    }
}

template <typename T, int Threads, int VectorBytes>
bool launch_vectorized_width(
    const T* input, const T* weight, T* output, int rows, int columns,
    float epsilon, cudaStream_t stream) {
    switch (columns) {
        case 256:
            vectorized_centered_rms_norm_kernel<T, 256, Threads, VectorBytes>
                <<<rows, Threads, 0, stream>>>(input, weight, output, epsilon);
            return true;
        case 2560:
            vectorized_centered_rms_norm_kernel<T, 2560, Threads, VectorBytes>
                <<<rows, Threads, 0, stream>>>(input, weight, output, epsilon);
            return true;
        case 5120:
            vectorized_centered_rms_norm_kernel<T, 5120, Threads, VectorBytes>
                <<<rows, Threads, 0, stream>>>(input, weight, output, epsilon);
            return true;
        default:
            return false;
    }
}

// Thread layout remains the measured narrow/wide policy, independent of the
// memory vector width. The benchmark can still compare alternate layouts.
template <typename T, int VectorBytes>
bool launch_vectorized_threads(
    const T* input, const T* weight, T* output, int rows, int columns,
    float epsilon, cudaStream_t stream, int threads) {
    switch (threads) {
        case 32:
            return launch_vectorized_width<T, 32, VectorBytes>(
                input, weight, output, rows, columns, epsilon, stream);
        case 64:
            return launch_vectorized_width<T, 64, VectorBytes>(
                input, weight, output, rows, columns, epsilon, stream);
        case 128:
            return launch_vectorized_width<T, 128, VectorBytes>(
                input, weight, output, rows, columns, epsilon, stream);
        case 256:
            return launch_vectorized_width<T, 256, VectorBytes>(
                input, weight, output, rows, columns, epsilon, stream);
        default:
            return false;
    }
}

// Contiguous slices may have nonzero offsets. Prefer the widest allowed access
// aligned for all three pointers AND every row, then fall back to scalar loads.
// No device queries, allocations, profiling, or mutable global policy occur here.
template <typename T>
bool launch_vectorized(
    const T* input, const T* weight, T* output, int rows, int columns,
    float epsilon, cudaStream_t stream, int threads, int preferred_bytes = 16) {
    const auto alignment = reinterpret_cast<std::uintptr_t>(input) |
                           reinterpret_cast<std::uintptr_t>(weight) |
                           reinterpret_cast<std::uintptr_t>(output) |
                           (size_t(columns) * sizeof(T));
    if (preferred_bytes >= 16 && alignment % 16 == 0) {
        return launch_vectorized_threads<T, 16>(
            input, weight, output, rows, columns, epsilon, stream, threads);
    }
    if (preferred_bytes >= 8 && alignment % 8 == 0) {
        return launch_vectorized_threads<T, 8>(
            input, weight, output, rows, columns, epsilon, stream, threads);
    }
    if (preferred_bytes >= 4 && alignment % 4 == 0) {
        return launch_vectorized_threads<T, 4>(
            input, weight, output, rows, columns, epsilon, stream, threads);
    }
    return false;
}

// Match Candle's storage-dtype addition before normalization. In particular,
// BF16 addition rounds here, not after an FP32 sum has already been normalized.
__device__ __forceinline__ float residual_add(float a, float b) {
    return __fadd_rn(a, b);
}
__device__ __forceinline__ __nv_bfloat16 residual_add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

// Generic fused path for irregular widths and naturally aligned scalar slices.
// The block barrier publishes the rounded residual before the second pass reads it.
template <typename T>
__global__ void scalar_fused_residual_rms_norm_kernel(
    const T* input, const T* branch, const T* weight, T* residual_output,
    T* output, int columns, float epsilon) {
    const size_t row_offset = size_t(blockIdx.x) * columns;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const T combined = residual_add(input[row_offset + column], branch[row_offset + column]);
        residual_output[row_offset + column] = combined;
        const float value = to_float(combined);
        sum += value * value;
    }
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    __shared__ float partial[kWarps];
    __shared__ float total;
    sum = warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        const float reduced = warp_sum(lane < kWarps ? partial[lane] : 0.0f);
        if (lane == 0) total = reduced;
    }
    __syncthreads();
    const float inverse_rms = rsqrtf(total / columns + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = to_float(residual_output[row_offset + column]);
        output[row_offset + column] = from_float<T>(
            value * inverse_rms * (1.0f + to_float(weight[column])));
    }
}

// Same vector partition and reduction as standalone RMSNorm. Both outputs are
// written once; rounded residual values stay in registers through normalization.
template <typename T, int Columns, int Threads, int VectorBytes>
__global__ void fused_residual_rms_norm_kernel(
    const T* input, const T* branch, const T* weight,
    T* residual_output, T* output, float epsilon) {
    constexpr int kElements = VectorBytes / sizeof(T);
    constexpr int kVectors = Columns / kElements;
    constexpr int kIterations = (kVectors + Threads - 1) / Threads;
    constexpr int kBlockWarps = Threads / kWarpSize;
    static_assert(Columns % kElements == 0 && Threads % kWarpSize == 0);
    static_assert(VectorBytes == 4 || VectorBytes == 8 || VectorBytes == 16);
    static_assert(sizeof(NormVector<T, VectorBytes>) == VectorBytes);
    const size_t row_offset = size_t(blockIdx.x) * kVectors;
    const auto* row =
        reinterpret_cast<const NormVector<T, VectorBytes>*>(input) + row_offset;
    const auto* branches =
        reinterpret_cast<const NormVector<T, VectorBytes>*>(branch) + row_offset;
    auto* residuals =
        reinterpret_cast<NormVector<T, VectorBytes>*>(residual_output) + row_offset;
    const auto* scales = reinterpret_cast<const NormVector<T, VectorBytes>*>(weight);
    auto* result =
        reinterpret_cast<NormVector<T, VectorBytes>*>(output) + row_offset;
    float values[kIterations][kElements];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int vector = threadIdx.x + i * Threads;
        if (vector < kVectors) {
            const NormVector<T, VectorBytes> loaded = row[vector];
            const NormVector<T, VectorBytes> added = branches[vector];
            NormVector<T, VectorBytes> rounded;
#pragma unroll
            for (int element = 0; element < kElements; ++element) {
                const T combined = residual_add(loaded.values[element], added.values[element]);
                rounded.values[element] = combined;
                const float value = to_float(combined);
                values[i][element] = value;
                sum += value * value;
            }
            residuals[vector] = rounded;
        }
    }

    // Every launched lane participates, including lanes with no vector to load.
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    sum = warp_sum(sum);
    if constexpr (Threads == kWarpSize) {
        sum = __shfl_sync(0xffffffffu, sum, 0);
    } else {
        __shared__ float partial[kBlockWarps];
        __shared__ float total;
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            const float reduced = warp_sum(lane < kBlockWarps ? partial[lane] : 0.0f);
            if (lane == 0) total = reduced;
        }
        __syncthreads();
        sum = total;
    }
    const float inverse_rms = rsqrtf(sum / Columns + epsilon);
#pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int vector = threadIdx.x + i * Threads;
        if (vector < kVectors) {
            const NormVector<T, VectorBytes> scale = scales[vector];
            NormVector<T, VectorBytes> normalized;
#pragma unroll
            for (int element = 0; element < kElements; ++element) {
                normalized.values[element] = from_float<T>(
                    values[i][element] * inverse_rms *
                    (1.0f + to_float(scale.values[element])));
            }
            result[vector] = normalized;
        }
    }
}

template <typename T, int VectorBytes>
bool launch_fused_width(
    const T* input, const T* branch, const T* weight, T* residual_output, T* output,
    int rows, int columns, float epsilon, cudaStream_t stream) {
    switch (columns) {
        case 256:
            fused_residual_rms_norm_kernel<T, 256, 32, VectorBytes>
                <<<rows, 32, 0, stream>>>(input, branch, weight, residual_output, output, epsilon);
            return true;
        case 2560:
            fused_residual_rms_norm_kernel<T, 2560, 256, VectorBytes>
                <<<rows, 256, 0, stream>>>(input, branch, weight, residual_output, output, epsilon);
            return true;
        case 5120:
            fused_residual_rms_norm_kernel<T, 5120, 256, VectorBytes>
                <<<rows, 256, 0, stream>>>(input, branch, weight, residual_output, output, epsilon);
            return true;
        default:
            return false;
    }
}

template <typename T>
void launch_fused(
    const void* input, const void* branch, const void* weight, void* residual_output,
    void* output, int rows, int columns, float epsilon, cudaStream_t stream, int preferred_bytes) {
    const auto* x = static_cast<const T*>(input);
    const auto* b = static_cast<const T*>(branch);
    const auto* w = static_cast<const T*>(weight);
    auto* r = static_cast<T*>(residual_output);
    auto* y = static_cast<T*>(output);
    const auto alignment = reinterpret_cast<std::uintptr_t>(x) |
        reinterpret_cast<std::uintptr_t>(b) | reinterpret_cast<std::uintptr_t>(w) |
        reinterpret_cast<std::uintptr_t>(r) | reinterpret_cast<std::uintptr_t>(y) |
        (size_t(columns) * sizeof(T));
    bool launched = false;
    if (preferred_bytes >= 16 && alignment % 16 == 0) {
        launched = launch_fused_width<T, 16>(x, b, w, r, y, rows, columns, epsilon, stream);
    } else if (preferred_bytes >= 8 && alignment % 8 == 0) {
        launched = launch_fused_width<T, 8>(x, b, w, r, y, rows, columns, epsilon, stream);
    } else if (preferred_bytes >= 4 && alignment % 4 == 0) {
        launched = launch_fused_width<T, 4>(x, b, w, r, y, rows, columns, epsilon, stream);
    }
    if (!launched) {
        scalar_fused_residual_rms_norm_kernel<<<rows, kThreads, 0, stream>>>(
            x, b, w, r, y, columns, epsilon);
    }
}

}  // namespace

// Validates the language-neutral ABI arguments and enqueues the matching typed
// kernel on the caller-owned CUDA stream. Dtype 0 is F32 and dtype 1 is BF16,
// matching Rust's `RmsNormDType`. The function returns a CUDA runtime status and
// deliberately does not synchronize, so execution remains asynchronous.
extern "C" int tau_launch_centered_rms_norm_configured(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int columns,
    float epsilon,
    int dtype,
    void* stream,
    int preferred_bytes) {
    // A null stream selects CUDA's valid default stream, but tensor pointers,
    // shapes, and epsilon must describe a usable RMSNorm operation.
    if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 ||
        columns <= 0 || !(epsilon > 0.0f) ||
        (preferred_bytes != 0 && preferred_bytes != 4 &&
         preferred_bytes != 8 && preferred_bytes != 16)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // Scalar fallback handles other row widths or pointers lacking 4-byte alignment.
    const dim3 grid(rows);
    const dim3 block(kThreads);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    // Local SM120 measurements favor 256 threads for wide decode/prefill rows.
    // Width 256 fits one warp and avoids all block-wide barriers. Other GPUs
    // retain this portable layout until their own benchmark justifies tuning.
    const int vector_threads = columns == 256 ? 32 : 256;
    // Cast the opaque ABI pointers only after the dtype tag has selected their
    // concrete element representation.
    switch (dtype) {
        case 0:
            if (launch_vectorized(static_cast<const float*>(input),
                    static_cast<const float*>(weight), static_cast<float*>(output),
                    rows, columns, epsilon, cuda_stream,
                    vector_threads, preferred_bytes)) {
                break;
            }
            centered_rms_norm_kernel<<<grid, block, 0, cuda_stream>>>(
                static_cast<const float*>(input),
                static_cast<const float*>(weight),
                static_cast<float*>(output),
                columns,
                epsilon);
            break;
        case 1:
            if (launch_vectorized(static_cast<const __nv_bfloat16*>(input),
                    static_cast<const __nv_bfloat16*>(weight),
                    static_cast<__nv_bfloat16*>(output), rows, columns,
                    epsilon, cuda_stream, vector_threads, preferred_bytes)) {
                break;
            }
            centered_rms_norm_kernel<<<grid, block, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<__nv_bfloat16*>(output),
                columns,
                epsilon);
            break;
        default:
            return static_cast<int>(cudaErrorInvalidValue);
    }
    // Report configuration or enqueue failures without waiting for completion.
    return static_cast<int>(cudaGetLastError());
}

// Preserve the original ABI for standalone callers using the portable heuristic.
extern "C" int tau_launch_centered_rms_norm(
    const void* input, const void* weight, void* output, int rows, int columns,
    float epsilon, int dtype, void* stream) {
    return tau_launch_centered_rms_norm_configured(
        input, weight, output, rows, columns, epsilon, dtype, stream, 16);
}

// Standalone benchmark only: compare thread layouts without a serving option or
// duplicated candidate math. This symbol is absent from normal Cargo builds.
#ifdef TAU_RMSNORM_BENCHMARK
extern "C" int tau_launch_centered_rms_norm_tuned(
    const void* input, const void* weight, void* output, int rows, int columns,
    float epsilon, int dtype, void* stream, int threads) {
    if (!input || !weight || !output || rows <= 0 || columns <= 0 || !(epsilon > 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    bool launched = false;
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dtype == 0) {
        launched = launch_vectorized(static_cast<const float*>(input),
            static_cast<const float*>(weight), static_cast<float*>(output),
            rows, columns, epsilon, cuda_stream, threads);
    } else if (dtype == 1) {
        launched = launch_vectorized(static_cast<const __nv_bfloat16*>(input),
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<__nv_bfloat16*>(output),
            rows, columns, epsilon, cuda_stream, threads);
    }
    return static_cast<int>(launched ? cudaGetLastError() : cudaErrorInvalidValue);
}
#endif

// Both outputs are distinct, caller-owned buffers. Neither input may be mutated;
// the caller retains the rounded residual for the layer's later MLP addition.
extern "C" int tau_launch_fused_residual_rms_norm(
    const void* input, const void* branch, const void* weight, void* residual_output,
    void* output, int rows, int columns, float epsilon, int dtype, void* stream,
    int preferred_bytes) {
    if (!input || !branch || !weight || !residual_output || !output ||
        rows <= 0 || columns <= 0 || !(epsilon > 0.0f) || !std::isfinite(epsilon) ||
        (preferred_bytes != 0 && preferred_bytes != 4 && preferred_bytes != 8 && preferred_bytes != 16)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dtype == 0) {
        launch_fused<float>(input, branch, weight, residual_output, output,
            rows, columns, epsilon, cuda_stream, preferred_bytes);
    } else if (dtype == 1) {
        launch_fused<__nv_bfloat16>(input, branch, weight, residual_output, output,
            rows, columns, epsilon, cuda_stream, preferred_bytes);
    } else {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}
