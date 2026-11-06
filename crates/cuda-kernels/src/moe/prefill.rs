//! Native prefill dispatch contracts shared by the buffer owner and its tests.
use crate::{LaunchError, MoePrefillConfig, MoePrefillTile, cuda_ffi};
use core::ffi::c_void;

pub const MOE_PREFILL_META_WORDS: usize = 514;
pub const MOE_PREFILL_EXPERTS: usize = 256;
pub const MOE_PREFILL_TILE: usize = 16;

/// Bounds all assignments and worst-case padding, rather than average expert
/// occupancy. Buffer owners use this before allocation; no routing is read back.
pub fn moe_prefill_capacity(tokens: usize) -> Option<usize> {
    moe_prefill_capacity_for_tile(tokens, MoePrefillTile::default())
}

/// Bounds scratch using the same M as native dispatch, including every expert
/// padding tail. Allocation never assumes average routing occupancy.
pub fn moe_prefill_capacity_for_tile(
    tokens: usize,
    tile: MoePrefillTile,
) -> Option<usize> {
    if tokens == 0 || tokens > 65535 {
        return None;
    }
    let m = tile.rows();
    let assignments = tokens.checked_mul(8)?;
    assignments
        .checked_add(assignments.min(MOE_PREFILL_EXPERTS).checked_mul(m - 1)?)?
        .checked_add(m - 1)
        .map(|n| n / m * m)
}

/// Build counts, padded offsets, permutation and expert worklist on one stream.
///
/// # Safety
/// Routes cover F32 [tokens,16], metadata 514 i32, offsets 257 i32, slots
/// `capacity` i32 and tiles `capacity/tile.rows()` i32. Capacity meets the size query.
/// Tile is the same startup geometry used by both projections. All buffers are
/// disjoint on the stream device and survive queued execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_prefill_dispatch(
    routes: *const f32,
    meta: *mut i32,
    offsets: *mut i32,
    slots: *mut i32,
    tiles: *mut i32,
    tokens: i32,
    capacity: i32,
    tile: MoePrefillTile,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The caller owns the documented stream and buffer contracts.
    let code = unsafe {
        cuda_ffi::tau_launch_moe_prefill_dispatch(
            routes,
            meta,
            offsets,
            slots,
            tiles,
            tokens,
            capacity,
            tile.rows() as i32,
            stream,
        )
    };
    status(code, "MoE prefill dispatch")
}

/// Queue both grouped projections and the ordered routed sum without host work.
///
/// # Safety
/// Input is BF16 [T,H], logits BF16 [T,256], gate_up BF16 [256,2I,H], down
/// BF16 [256,H,I]. Routes/metadata match dispatch's contract. Gate/up scratch is
/// BF16 [capacity,2I], activation [capacity,I], contributions [T*8,H], output
/// [T,H]. Receipts is null in production or a disjoint writable ten-i32 test
/// allocation stamped by the two executed projection kernels. Config is read
/// on the host during submission; kernels receive its compiled specialization.
/// Every pointer is disjoint, typed as stated, retained and stream ordered.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_prefill(
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
    config: MoePrefillConfig,
    receipts: *mut i32,
    stream: *mut c_void,
) -> Result<(), LaunchError> {
    // SAFETY: The buffer owner validates all dimensions, views and lifetimes.
    let code = unsafe {
        cuda_ffi::tau_launch_moe_prefill(
            input,
            logits,
            gate_up,
            down,
            routes,
            meta,
            offsets,
            slots,
            tiles,
            projected,
            activated,
            contributions,
            output,
            tokens,
            hidden,
            intermediate,
            capacity,
            &config.gate_up(),
            &config.down(),
            receipts,
            stream,
        )
    };
    status(code, "MoE grouped prefill")
}

/// Preserve native diagnostics while keeping Rust callers on Result-based errors.
fn status(code: i32, kernel: &'static str) -> Result<(), LaunchError> {
    if code == 0 {
        Ok(())
    } else {
        Err(LaunchError { code, kernel })
    }
}
