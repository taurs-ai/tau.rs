//! Foreign function interface (FFI) between Rust and the native CUDA launchers.
//!
//! These declarations tell Rust how to call the `extern "C"` functions compiled
//! from `src/kernels/*.cu`. Each function is a C++ host launcher that queues one
//! or more GPU operations; the GPU computation itself lives in the CUDA sources.
//! All native CUDA launcher declarations belong here. Rust wrappers in `lib.rs`
//! and kernel-family modules such as `gdn_prefill.rs` call these declarations,
//! document their buffer contracts and convert integer codes to Rust `Result`.
//!
//! Callers supply GPU buffers and the CUDA stream that orders their execution.
//! Raw pointers carry no shape, dtype or ownership information, so the Candle
//! bridges must validate tensors and keep their storage alive while work runs.
//! A return value of zero means launch submission succeeded; nonzero means an
//! argument, CUDA or cuBLAS error. Submission is asynchronous: returning does not
//! mean the GPU has finished, and execution errors can surface at a later sync.

use core::ffi::{c_float, c_int, c_void};

unsafe extern "C" {
    pub(super) fn tau_launch_moe_decode_gemm(
        inputs: *const c_void,
        weights: *const c_void,
        outputs: *mut c_void,
        batches: i32,
        columns: i32,
        inner: i32,
        config: &crate::MoeDecodeProfile,
        receipts: *mut i32,
        stream: *mut c_void,
    ) -> i32;

    /// Builds device-only grouped assignment metadata for native MoE prefill.
    pub(super) fn tau_launch_moe_prefill_dispatch(
        routes: *const f32,
        meta: *mut i32,
        offsets: *mut i32,
        slots: *mut i32,
        tiles: *mut i32,
        tokens: i32,
        capacity: i32,
        tile_rows: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Runs routing, dispatch, grouped projections and ordered accumulation.
    pub(super) fn tau_launch_moe_prefill(
        input: *const c_void,
        logits: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        routes: *mut f32,
        meta: *mut i32,
        offsets: *mut i32,
        slots: *mut i32,
        tiles: *mut i32,
        projected: *mut c_void,
        activated: *mut c_void,
        contributions: *mut c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        capacity: i32,
        gate_up_config: *const crate::MoePrefillTile,
        down_config: *const crate::MoePrefillTile,
        receipts: *mut i32,
        stream: *mut c_void,
    ) -> i32;

    /// FP32 scalar gate dot/sigmoid followed by fused shared/routed BF16 output.
    pub(super) fn tau_launch_moe_shared_combine(
        input: *const c_void,
        weight: *const c_void,
        shared: *const c_void,
        routed: *const c_void,
        output: *mut c_void,
        rows: c_int,
        width: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Packs 256-expert BF16 router rows into eight IDs/eight FP32 weights.
    pub(super) fn tau_launch_moe_routing_fp32(
        logits: *const c_void,
        output: *mut c_float,
        rows: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Projects row-major BF16 matrices into an FP32 output with K-ordered MMA.
    pub(super) fn tau_launch_linear_bf16_fp32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_float,
        rows: c_int,
        columns: c_int,
        inner: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Pure size query shared by the allocator and native launch validation.
    pub(super) fn tau_greedy_argmax_workspace_words(vocabulary: u32) -> usize;
    /// Reduces FP32/BF16 vocabulary logits and writes [token_id, valid].
    pub(super) fn tau_launch_greedy_argmax(
        logits: *const c_void,
        dtype: c_int,
        vocabulary: u32,
        scratch: *mut u32,
        scratch_words: usize,
        result: *mut u32,
        stream: *mut c_void,
    ) -> c_int;
    /// Queues one decode embedding/residual/query-pack/gate/KV-write kernel.
    /// The Rust wrapper documents the buffer layout selected by `kind`.
    pub(super) fn tau_launch_decode_elementwise(
        kind: c_int,
        input: *const c_void,
        rhs: *const c_void,
        output: *mut c_void,
        metadata: *const u32,
        elements: c_int,
        width: c_int,
        capacity: c_int,
        transposed: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Applies partial RoPE using the position in retained device metadata,
    /// leaving the remaining head dimensions unchanged.
    pub(super) fn tau_launch_decode_rope(
        input: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        output: *mut c_void,
        metadata: *const u32,
        elements: c_int,
        dim: c_int,
        rotary_dim: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Fuses Q extraction, centered Q/K normalization, partial RoPE and KV append.
    pub(super) fn tau_launch_decode_attention_prepare(
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
        heads: c_int,
        kv_heads: c_int,
        dim: c_int,
        capacity: c_int,
        rotary_dim: c_int,
        transposed: c_int,
        fp32_rope: c_int,
        query_epsilon: f32,
        key_epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    /// Runs grouped attention over live KV, with FP32 tile results in caller-owned
    /// scratch and a combine launch when more than one context tile is needed.
    pub(super) fn tau_launch_decode_attention(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        metadata: *const u32,
        workspace: *mut f32,
        heads: c_int,
        kv_heads: c_int,
        dim: c_int,
        capacity: c_int,
        transposed: c_int,
        block_size: c_int,
        bf16_products: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Filters projected Q/K/V features across a prompt chunk before GDN memory
    /// updates. Each channel has its own learned four-token causal filter, so
    /// the current token reads its three predecessors without mixing channels.
    ///
    /// Reads BF16 input, weights and previous raw-token history, honoring the
    /// input/history strides. Writes BF16 filtered values to `output`; activation
    /// and publication of the next history remain in the model's convolution
    /// component. This launcher does not modify input or history buffers.
    pub(super) fn tau_launch_bf16_gdn_prefill_convolution(
        input: *const c_void,
        history: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        channels: c_int,
        tokens: c_int,
        input_channel_stride: i64,
        input_token_stride: i64,
        history_channel_stride: i64,
        history_token_stride: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Adds a decoder branch back into the residual stream and normalizes that
    /// sum in one GPU kernel. Returns both the sum (`residual_output`) for later
    /// residual connections and its normalized values (`output`) for the next
    /// sublayer. Vectorized paths retain the sum in registers, avoiding another
    /// kernel launch and a separate normalization read of the residual buffer.
    ///
    /// The sum rounds to the selected FP32/BF16 storage dtype before RMSNorm;
    /// normalization uses FP32 arithmetic and the centered scale `1 + weight`.
    /// `preferred_bytes` limits vectorized accesses to suitably aligned buffers.
    pub(super) fn tau_launch_fused_residual_rms_norm(
        input: *const c_void,
        branch: *const c_void,
        weight: *const c_void,
        residual_output: *mut c_void,
        output: *mut c_void,
        rows: c_int,
        columns: c_int,
        epsilon: c_float,
        dtype: c_int,
        stream: *mut c_void,
        preferred_bytes: c_int,
    ) -> c_int;

    /// Normalizes each activation row by its root-mean-square magnitude and
    /// applies the learned centered scale: `x / sqrt(mean(x*x) + eps) * (1 + w)`.
    /// This prepares decoder activations for attention, GDN or MLP computation.
    ///
    /// Input, weight and output use the selected FP32/BF16 storage dtype, with
    /// FP32 normalization arithmetic. `preferred_bytes` limits memory access
    /// width; alignment and row shape determine the vectorized or scalar path.
    pub(super) fn tau_launch_centered_rms_norm_configured(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: c_int,
        columns: c_int,
        epsilon: c_float,
        dtype: c_int,
        stream: *mut c_void,
        preferred_bytes: c_int,
    ) -> c_int;

    /// Computes the gated activation in a SwiGLU MLP: `SiLU(gate) * up`.
    /// Each packed input row contains the gate projection followed by the up
    /// projection; the output has half as many columns. The down projection runs
    /// afterward in the caller. Supports FP32/BF16 storage with FP32 arithmetic.
    pub(super) fn tau_launch_silu_and_mul(
        input: *const c_void,
        output: *mut c_void,
        rows: c_int,
        columns: c_int,
        dtype: c_int,
        stream: *mut c_void,
    ) -> c_int;

    /// Computes the routed-expert contribution for one token in a sparse MoE.
    /// Selects eight of 256 experts from router scores, normalizes their routing
    /// weights, and prepares GPU pointer arrays for the selected expert weights.
    /// Tensor-core projections retain FP32 accumulators through weighting, with
    /// a native SiLU kernel between them. A final GPU reduction sums weighted
    /// BF16 expert outputs in FP32 routing-rank order.
    ///
    /// Activations and expert weights are BF16; routing weights and output
    /// accumulation are FP32. The caller provides BF16 and FP32 intermediate
    /// buffers. The model adds the shared-expert contribution separately.
    pub(super) fn tau_launch_fused_bf16_moe_decode(
        input: *const c_void,
        router_logits: *const c_void,
        gate_up_weights: *const c_void,
        down_weights: *const c_void,
        selected_ids: *mut c_void,
        selected_weights: *mut c_void,
        pointer_workspace: *mut c_void,
        gate_up_output: *mut c_void,
        activated_output: *mut c_void,
        expert_output: *mut c_void,
        gate_up_fp32: *mut c_void,
        expert_fp32: *mut c_void,
        output: *mut c_void,
        hidden_size: c_int,
        intermediate_size: c_int,
        num_experts: c_int,
        experts_per_token: c_int,
        gate_up_config: &crate::MoeDecodeProfile,
        down_config: &crate::MoeDecodeProfile,
        receipts: *mut i32,
        stream: *mut c_void,
    ) -> c_int;

    /// Advances Gated DeltaNet by one generated token. Updates the short causal
    /// convolution history, normalizes Q/K, applies learned decay and correction
    /// to the long-term memory matrix, and produces a gated RMS-normalized value
    /// vector for the caller's output projection.
    ///
    /// `qkvz` packs projected queries, keys, values and the output gate; `ba`
    /// packs the update/decay projections. Both input projections and the final
    /// output projection are performed by Candle outside this launcher.
    /// Convolution history is BF16 and recurrent state is FP32; both are updated
    /// in place. Production 128x128 heads use separate recurrence and norm
    /// launches with `recurrence_output` scratch; other sizes fuse those two steps.
    pub(super) fn tau_launch_fused_bf16_gated_delta_net_decode(
        qkvz: *const c_void,
        ba: *const c_void,
        conv_weight: *const c_void,
        dt_bias: *const c_void,
        negative_a: *const c_void,
        norm_weight: *const f32,
        conv_state: *mut c_void,
        recurrent_state: *mut c_void,
        convolved_qkv: *mut c_void,
        recurrence_output: *mut c_void,
        output: *mut c_void,
        conv_width: c_int,
        conv_kernel_size: c_int,
        num_key_heads: c_int,
        num_value_heads: c_int,
        key_head_dim: c_int,
        value_head_dim: c_int,
        epsilon: c_float,
        reference_reduction: c_int,
        stream: *mut c_void,
    ) -> c_int;

    /// Converts each FP32 recurrence readout through BF16, computes its RMSNorm
    /// in FP32, applies the direct learned norm weight and `SiLU(gate)`, and
    /// writes BF16 activations for the output projection. Processes all prompt
    /// rows together; each row contains one head's 128 value features.
    pub(super) fn tau_launch_gdn_prefill_norm(
        input: *const f32,
        gate: *const c_void,
        weight: *const f32,
        output: *mut c_void,
        rows: i32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Prepares filtered Q/K/V features for recurrence without separate tensor
    /// expansion and normalization passes. Reads strided BF16 convolution output,
    /// maps shared Q/K heads to their value heads, L2-normalizes Q/K, scales Q,
    /// and packs FP32 Q, K, V, decay multiplier (`alpha`) and update gate (`beta`).
    pub(super) fn tau_launch_gdn_prefill_prepare(
        mixed: *const c_void,
        alpha: *const f32,
        beta: *const c_void,
        output: *mut f32,
        tokens: i32,
        key_heads: i32,
        value_heads: i32,
        token_stride: i64,
        channel_stride: i64,
        round_normalized: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Processes every token in a prompt chunk, applying decay, memory prediction,
    /// delta correction and query readout to the recurrent FP32 memory matrix.
    /// State-column tiles stay in GPU registers across tokens to avoid repeatedly
    /// reading/writing the entire matrix. Reads `initial` without modifying it;
    /// `output` contains each token's readout followed by the final state used
    /// by the next chunk or decode. Separate state buffers also support capture.
    pub(super) fn tau_launch_gdn_prefill_recurrence(
        input: *const f32,
        initial: *const f32,
        output: *mut f32,
        tokens: i32,
        heads: i32,
        stream: *mut c_void,
    ) -> i32;
}
