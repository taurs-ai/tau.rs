//! Allocation-free BF16 decode launchers; callers own all buffers and ordering.
use crate::{LaunchError, cuda_ffi};
use core::ffi::c_void;

/// Converts the native launch status into the shared error type; zero is success.
fn check(code: i32) -> Result<(), LaunchError> {
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError {
            code,
            kernel: "prepared decode",
        })
    }
}

/// Writes an embedding, residual sum, packed query, gated context or KV append.
/// The kind selects one indexing rule in a shared kernel; token and KV position
/// are read from device metadata so the same launch can be captured once.
///
/// # Safety
/// All buffers are BF16 on stream's device, disjoint except read-only inputs.
/// Metadata is two U32 values [token,position], with token < vocabulary and
/// position < capacity. kind=0: input [vocabulary,width], output [width].
/// kind=1: inputs/output [elements]. kind=2: input [heads,2*width], output
/// [heads,width]. kind=3: input [heads,width], rhs [heads,2*width]. kind=5:
/// input/output [n], rhs [1], applying a BF16-rounded scalar sigmoid. kind=4:
/// input [heads,width], output [heads,capacity,width], optionally transposed K.
/// All products/strides must fit allocation bounds. Inputs stay immutable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_decode_elementwise(
    kind: i32,
    input: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: *const u32,
    elements: i32,
    width: i32,
    capacity: i32,
    transposed: i32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: Caller upholds the documented storage/metadata contract.
    check(unsafe {
        cuda_ffi::tau_launch_decode_elementwise(
            kind, input, rhs, output, metadata, elements, width, capacity, transposed,
            stream,
        )
    })
}

/// Rotates one token's leading head dimensions using retained position metadata.
/// Reads the corresponding sine/cosine row and preserves the unrotated dimensions.
///
/// # Safety
/// Input/output are disjoint BF16 [heads,dim]. cos/sin are BF16 [positions,R/2].
/// Metadata [token,position] has position < positions. 0<R<=dim and R is even.
/// All dimensions and allocations must cover every access on stream's device.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_decode_rope(
    input: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    output: *mut c_void,
    metadata: *const u32,
    elements: i32,
    dim: i32,
    rotary_dim: i32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: Caller upholds the documented storage/metadata contract.
    check(unsafe {
        cuda_ffi::tau_launch_decode_rope(
            input, cos, sin, output, metadata, elements, dim, rotary_dim, stream,
        )
    })
}

/// Extracts Q, normalizes Q/K, rotates their leading dimensions and appends KV.
/// Normalization is FP32 with a BF16 boundary before the existing BF16 RoPE math.
///
/// # Safety
/// All tensors are BF16 on stream's device and retained through execution.
/// Inputs query_gate [heads,2*dim], key/value [kv_heads,dim], weights [dim], and
/// cos/sin [positions,rotary_dim/2] are immutable. Writable query [heads,dim] and
/// K/V caches [kv_heads,capacity,dim] are disjoint from inputs and each other.
/// K storage may be transposed. Metadata is U32 [token,position] with position
/// below both capacity and positions. Dimensions satisfy heads%kv_heads=0,
/// heads,kv_heads>0, 0<dim<=256, 0<capacity<=8192, and 0<rotary_dim<=dim even.
/// Epsilons are finite and positive. Every access fits its allocation and all
/// buffer use is serialized on the owning private stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_decode_attention_prepare(
    query_gate: *const c_void,
    key: *const c_void,
    value: *const c_void,
    query_weight: *const c_void,
    key_weight: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    query_output: *mut c_void,
    cached_key: *mut c_void,
    cached_value: *mut c_void,
    metadata: *const u32,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    capacity: i32,
    rotary_dim: i32,
    transposed: i32,
    fp32_rope: i32,
    query_epsilon: f32,
    key_epsilon: f32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: Caller upholds the documented ownership and shape contracts.
    check(unsafe {
        cuda_ffi::tau_launch_decode_attention_prepare(
            query_gate,
            key,
            value,
            query_weight,
            key_weight,
            cos,
            sin,
            query_output,
            cached_key,
            cached_value,
            metadata,
            heads,
            kv_heads,
            dim,
            capacity,
            rotary_dim,
            transposed,
            fp32_rope,
            query_epsilon,
            key_epsilon,
            stream,
        )
    })
}

/// Computes grouped attention, reading only live KV slots through device position.
/// Query heads share KV heads directly. The native launcher computes selected
/// tiles with FP32 softmax/context scratch, then combines their normalized results
/// when capacity spans multiple tiles. Unused tiles contribute zero after reset.
///
/// # Safety
/// Q/output are disjoint BF16 [heads,dim], K/V [kv_heads,capacity,dim], K may
/// have transposed storage. Metadata is [token,position], position < capacity.
/// heads>0 is divisible by kv_heads>0, 0<dim<=256, 0<capacity<=8192. Buffers
/// cover all shapes on stream's device and remain alive through execution.
/// block_size is 32, 64, 128 or 256 context tokens; CUDA blocks use 256 threads.
/// Workspace is private mutable FP32 [heads,ceil(capacity/block_size),dim+2], disjoint
/// from inputs/output and fully written before each use on the same stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_decode_attention(
    query: *const c_void,
    key: *const c_void,
    value: *const c_void,
    output: *mut c_void,
    metadata: *const u32,
    workspace: *mut f32,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    capacity: i32,
    transposed: i32,
    block_size: i32,
    bf16_products: i32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: Caller upholds the documented storage/metadata contract.
    check(unsafe {
        cuda_ffi::tau_launch_decode_attention(
            query,
            key,
            value,
            output,
            metadata,
            workspace,
            heads,
            kv_heads,
            dim,
            capacity,
            transposed,
            block_size,
            bf16_products,
            stream,
        )
    })
}
