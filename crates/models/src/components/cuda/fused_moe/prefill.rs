//! Device-only prefill execution over the existing packed expert weights.
//!
//! The sparse block selects this runner for supported prompt chunks. It retains
//! dispatch/projection scratch across calls; each call returns an independently
//! owned output so later chunks cannot overwrite a retained model result.

#[cfg(test)]
mod observation;
#[cfg(test)]
mod tests;

use super::{FusedMoeDimensions, FusedMoeRunner};
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Storage, Tensor,
};
use half::bf16;
use std::sync::Mutex;

/// Select the measured preset only for its validated architecture and
/// expert matrix widths. Other devices/shapes retain the control until measured;
/// explicit startup profiles bypass this policy and are validated separately.
fn device_default(
    dimensions: FusedMoeDimensions,
    capability: Option<(i32, i32)>,
) -> tau_cuda_kernels::MoePrefillConfig {
    if matches!(capability, Some((12, 0)) | Some((10, 3)))
        && dimensions.hidden_size == 2048
        && dimensions.intermediate_size == 512
    {
        let tile = tau_cuda_kernels::MoePrefillTile::new([16, 64, 64, 4, 1])
            .expect("validated SM120/SM103 MoE prefill preset");
        tau_cuda_kernels::MoePrefillConfig::new(tile, tile)
            .expect("matching default dispatch padding")
    } else {
        tau_cuda_kernels::MoePrefillConfig::default()
    }
}

/// Reusable dispatch and expert intermediates, independent of decode workspace.
/// Capacity grows before enqueue and every active extent is rewritten per chunk.
struct Workspace {
    tokens: usize,
    padded: usize,
    routes: CudaSlice<f32>,
    meta: CudaSlice<i32>,
    offsets: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    tiles: CudaSlice<i32>,
    projected: CudaSlice<bf16>,
    activated: CudaSlice<bf16>,
    contributions: CudaSlice<bf16>,
    #[cfg(test)]
    receipts: CudaSlice<i32>,
}

/// Layer-owned prefill scratch plus shared immutable expert tensor views.
/// Clones retain weights but start without scratch, preventing graph/test clones
/// from accidentally sharing mutable prefill storage with their source layer.
pub(in crate::components) struct MoePrefillRunner {
    gate_up: Tensor,
    down: Tensor,
    dimensions: FusedMoeDimensions,
    config: tau_cuda_kernels::MoePrefillConfig,
    workspace: Mutex<Option<Workspace>>,
    #[cfg(test)]
    calls: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for MoePrefillRunner {
    /// Report geometry without exposing live GPU allocations or mutex contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoePrefillRunner")
            .field("dimensions", &self.dimensions)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
impl Clone for MoePrefillRunner {
    /// Share only immutable parameters; scratch belongs to the new execution owner.
    fn clone(&self) -> Self {
        Self {
            gate_up: self.gate_up.clone(),
            down: self.down.clone(),
            dimensions: self.dimensions,
            config: self.config,
            workspace: Mutex::new(None),
            #[cfg(test)]
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl FusedMoeRunner {
    /// Reuse validated checkpoint views to create an independent prefill owner.
    pub(in crate::components) fn prefill_runner(&self) -> MoePrefillRunner {
        use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };
        let capability = self
            .gate_up_weights
            .device()
            .as_cuda_device()
            .ok()
            .and_then(|device| {
                let stream = device.cuda_stream();
                let context = stream.context();
                context
                    .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                    .ok()
                    .zip(
                        context
                            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                            .ok(),
                    )
            });
        MoePrefillRunner {
            gate_up: self.gate_up_weights.clone(),
            down: self.down_weights.clone(),
            dimensions: self.dimensions,
            config: device_default(self.dimensions, capability),
            workspace: Mutex::new(None),
            #[cfg(test)]
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl MoePrefillRunner {
    pub(in crate::components) fn config(&self) -> tau_cuda_kernels::MoePrefillConfig {
        self.config
    }

    /// Fix dispatch and both projection geometries before any scratch exists.
    pub(in crate::components) fn configure(
        &mut self,
        config: tau_cuda_kernels::MoePrefillConfig,
    ) {
        assert!(self.workspace.get_mut().unwrap().is_none());
        self.config = config;
    }

    /// Select supported contiguous token rows before entering the custom operator.
    pub(in crate::components) fn supports_input(
        &self,
        input: &Tensor,
        logits: &Tensor,
    ) -> bool {
        let Ok((tokens, hidden)) = input.dims2() else {
            return false;
        };
        hidden == self.dimensions.hidden_size
            && tokens > 0
            && tokens <= 65535
            && tau_cuda_kernels::moe_prefill_capacity_for_tile(
                tokens,
                self.config.gate_up(),
            )
            .is_some_and(|n| n / self.config.gate_up().rows() <= 65535)
            && logits.dims() == [tokens, self.dimensions.num_experts]
            && [input, logits].iter().all(|t| {
                t.dtype() == DType::BF16
                    && t.is_contiguous()
                    && t.device().is_cuda()
                    && t.device().same_device(self.gate_up.device())
            })
    }

    /// Queue routing, GPU dispatch, two grouped projections and ordered combine.
    /// Observations read actual retained buffers only in an active test capture.
    pub(in crate::components) fn forward(
        &self,
        input: &Tensor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        if !self.supports_input(input, logits) {
            candle::bail!("unsupported grouped MoE prefill input")
        }
        let output = input.apply_op2_no_bwd(logits, &Operation(self))?;
        #[cfg(test)]
        {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if crate::test_support::tensor_capture::active() {
                self.observe(input.dim(0)?)?;
            }
        }
        Ok(output)
    }

    /// Read specialization stamps written by the actual GPU projections.
    #[cfg(test)]
    pub(in crate::components) fn executed_tiles(&self) -> [[i32; 5]; 2] {
        let owner = self.workspace.lock().unwrap();
        let stream = self
            .gate_up
            .device()
            .as_cuda_device()
            .unwrap()
            .cuda_stream();
        let words = stream
            .clone_dtoh(&owner.as_ref().unwrap().receipts)
            .unwrap();
        [
            words[..5].try_into().unwrap(),
            words[5..].try_into().unwrap(),
        ]
    }

    /// Count successful native calls so correctness tests cannot pass via fallback.
    #[cfg(test)]
    pub(in crate::components) fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Expose retained capacity/addresses for reuse tests and memory accounting.
    /// This inspects allocation metadata without copying device routing data.
    #[cfg(test)]
    pub(in crate::components) fn workspace_info(&self) -> Option<(usize, usize, u64)> {
        let owner = self.workspace.lock().unwrap();
        owner.as_ref().map(|w| {
            let stream = self
                .gate_up
                .device()
                .as_cuda_device()
                .unwrap()
                .cuda_stream();
            let (pointer, guard) = w.projected.device_ptr(&stream);
            let bytes = w.routes.len() * 4
                + (w.meta.len() + w.offsets.len() + w.slots.len() + w.tiles.len()) * 4
                + (w.projected.len() + w.activated.len() + w.contributions.len()) * 2;
            drop(guard);
            (w.tokens, bytes, pointer)
        })
    }

    /// Allocate all scratch for a declared capacity, checking products before GPU
    /// allocation. The first use/growth is outside any supported prefill capture.
    fn allocate(
        &self,
        device: &candle::CudaDevice,
        tokens: usize,
    ) -> Result<Workspace> {
        let padded = tau_cuda_kernels::moe_prefill_capacity_for_tile(
            tokens,
            self.config.gate_up(),
        )
        .ok_or_else(|| {
            candle::Error::Msg("MoE prefill capacity overflow".to_owned())
        })?;
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .filter(|n| i32::try_from(*n).is_ok())
                .ok_or_else(|| {
                    candle::Error::Msg("MoE prefill workspace overflow".to_owned())
                })
        };
        let projected = product(padded, self.dimensions.intermediate_size * 2)?;
        let activated = product(padded, self.dimensions.intermediate_size)?;
        let contributions = product(product(tokens, 8)?, self.dimensions.hidden_size)?;
        // SAFETY: Native initialization/projections write every read element;
        // padding and unused tiles are initialized before the shared activation.
        unsafe {
            Ok(Workspace {
                tokens,
                padded,
                routes: device.alloc::<f32>(product(tokens, 16)?)?,
                meta: device.alloc::<i32>(tau_cuda_kernels::MOE_PREFILL_META_WORDS)?,
                offsets: device.alloc::<i32>(257)?,
                slots: device.alloc::<i32>(padded)?,
                tiles: device.alloc::<i32>(padded / self.config.gate_up().rows())?,
                projected: device.alloc::<bf16>(projected)?,
                activated: device.alloc::<bf16>(activated)?,
                contributions: device.alloc::<bf16>(contributions)?,
                #[cfg(test)]
                receipts: device.alloc_zeros::<i32>(10)?,
            })
        }
    }

    /// Validate storage views, retain pointer guards and enqueue onto the input's
    /// CUDA stream. There is no route/count readback or per-expert Rust dispatch.
    fn launch(
        &self,
        input: &CudaStorage,
        il: &Layout,
        logits: &CudaStorage,
        ll: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, hidden) = il.shape().dims2()?;
        if input.device.id() != logits.device.id()
            || hidden != self.dimensions.hidden_size
            || ll.dims() != [tokens, 256]
            || input.dtype() != DType::BF16
            || logits.dtype() != DType::BF16
        {
            candle::bail!("invalid grouped MoE prefill storage")
        }
        let range = |l: &Layout| {
            l.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg("MoE prefill requires contiguous views".to_owned())
            })
        };
        let (start, end) = range(il)?;
        let x = input.as_cuda_slice::<bf16>()?.slice(start..end);
        let (start, end) = range(ll)?;
        let scores = logits.as_cuda_slice::<bf16>()?.slice(start..end);
        let retained = [&self.gate_up, &self.down].map(Tensor::storage_and_layout);
        let weights = retained
            .iter()
            .map(|(s, l)| {
                let Storage::Cuda(s) = &**s else {
                    candle::bail!("expected CUDA experts")
                };
                if s.device.id() != input.device.id() {
                    candle::bail!("expert device mismatch")
                }
                let (start, end) = range(l)?;
                Ok(s.as_cuda_slice::<bf16>()?.slice(start..end))
            })
            .collect::<Result<Vec<_>>>()?;
        let device = input.device.clone();
        let stream = device.cuda_stream();
        let mut owner = self
            .workspace
            .lock()
            .map_err(|_| candle::Error::Msg("prefill workspace poisoned".to_owned()))?;
        if owner.as_ref().is_none_or(|w| w.tokens < tokens) {
            *owner = Some(
                self.allocate(
                    &device,
                    tokens
                        .checked_next_power_of_two()
                        .filter(|n| *n <= 65535)
                        .unwrap_or(tokens),
                )?,
            );
        }
        let w = owner.as_mut().unwrap();
        // SAFETY: The final combine kernel initializes all T*H output elements.
        let mut output = unsafe { device.alloc::<bf16>(tokens * hidden)? };
        let (xp, xg) = x.device_ptr(&stream);
        let (sp, sg) = scores.device_ptr(&stream);
        let wp = weights
            .iter()
            .map(|w| w.device_ptr(&stream))
            .collect::<Vec<_>>();
        let (rp, rg) = w.routes.device_ptr_mut(&stream);
        let (mp, mg) = w.meta.device_ptr_mut(&stream);
        let (op, og) = w.offsets.device_ptr_mut(&stream);
        let (pp, pg) = w.slots.device_ptr_mut(&stream);
        let (tp, tg) = w.tiles.device_ptr_mut(&stream);
        let (gp, gg) = w.projected.device_ptr_mut(&stream);
        let (ap, ag) = w.activated.device_ptr_mut(&stream);
        let (cp, cg) = w.contributions.device_ptr_mut(&stream);
        let (yp, yg) = output.device_ptr_mut(&stream);
        #[cfg(test)]
        let (receipt_pointer, receipt_guard) = w.receipts.device_ptr_mut(&stream);
        #[cfg(not(test))]
        let receipt_pointer = 0u64;
        // SAFETY: Supported shapes/products and all typed view ranges are checked;
        // allocations, storage owners and pointer guards survive stream enqueue.
        unsafe {
            tau_cuda_kernels::launch_moe_prefill(
                xp as *const _,
                sp as *const _,
                wp[0].0 as *const _,
                wp[1].0 as *const _,
                rp as *mut _,
                mp as *mut _,
                op as *mut _,
                pp as *mut _,
                tp as *mut _,
                gp as *mut _,
                ap as *mut _,
                cp as *mut _,
                yp as *mut _,
                tokens as i32,
                hidden as i32,
                self.dimensions.intermediate_size as i32,
                w.padded as i32,
                self.config,
                receipt_pointer as *mut _,
                stream.cu_stream().cast(),
            )
        }
        .map_err(candle::Error::wrap)?;
        drop((yg, cg, ag, gg, tg, pg, og, mg, rg, wp, sg, xg));
        #[cfg(test)]
        drop(receipt_guard);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            il.shape().clone(),
        ))
    }
}

/// Borrow the layer owner during ordinary tensor execution, retaining its scratch.
struct Operation<'a>(&'a MoePrefillRunner);
impl CustomOp2 for Operation<'_> {
    /// Identify errors at the grouped prefill execution boundary.
    fn name(&self) -> &'static str {
        "tau-grouped-moe-prefill"
    }
    /// Unsupported CPU execution stays with the sparse block's reference path.
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("CUDA-only grouped prefill")
    }
    /// Use the same layer-owned workspace for every supported prompt chunk.
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        r: &CudaStorage,
        rl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.0.launch(x, xl, r, rl)
    }
}
