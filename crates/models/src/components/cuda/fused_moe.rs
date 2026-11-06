//! CUDA MoE execution over shared immutable packed expert weights.
//!
//! This file owns the single-token decode runner. The prefill child owns device
//! grouping and chunk scratch independently; both reuse the projection primitives.
//!
//! The runner retains the checkpoint's packed expert tensors and allocates one
//! reusable workspace when the model is constructed. Each forward call passes
//! the token and router logits through a binary Candle custom operator so the
//! native pipeline uses Candle's active stream and returns Candle-owned output.

mod configuration;
mod prefill;
mod prepared;
#[cfg(test)]
mod validation;
pub(in crate::components) use prefill::MoePrefillRunner;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape,
    Storage, Tensor,
};
use core::ffi::c_void;
use half::bf16;
use std::sync::{Arc, Mutex};

const SUPPORTED_EXPERTS: usize = 256;
const SUPPORTED_TOP_K: usize = 8;
const POINTER_ARRAY_COUNT: usize = 6;

/// Packed routed-expert dimensions accepted by the native development backend.
#[derive(Debug, Clone, Copy)]
struct FusedMoeDimensions {
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    experts_per_token: usize,
}

/// Reusable device buffers needed between the asynchronous MoE pipeline stages.
#[derive(Debug)]
struct FusedMoeWorkspace {
    selected_ids: CudaSlice<i32>,
    selected_weights: CudaSlice<f32>,
    pointer_arrays: CudaSlice<u64>,
    gate_up_output: CudaSlice<bf16>,
    activated_output: CudaSlice<bf16>,
    expert_output: CudaSlice<bf16>,
    gate_up_fp32: CudaSlice<f32>,
    expert_fp32: CudaSlice<f32>,
    #[cfg(test)]
    receipts: CudaSlice<i32>,
}

/// Cloneable model-owned runner for the narrow BF16 CUDA decode contract.
#[derive(Debug, Clone)]
pub(in crate::components) struct FusedMoeRunner {
    gate_up_weights: Tensor,
    down_weights: Tensor,
    dimensions: FusedMoeDimensions,
    workspace: Arc<Mutex<FusedMoeWorkspace>>,
    config: tau_cuda_kernels::MoeDecodeConfig,
    resolution: Option<[tau_cuda_kernels::MoeDecodeResolution; 2]>,
}

impl FusedMoeRunner {
    /// Reads actual selected routes for isolated numerical diagnostics.
    #[cfg(test)]
    pub(in crate::components) fn selected_routing(
        &self,
    ) -> CandleResult<(Vec<i32>, Vec<f32>)> {
        let workspace = self.workspace.lock().unwrap();
        let stream = self
            .gate_up_weights
            .device()
            .as_cuda_device()?
            .cuda_stream();
        Ok((
            stream
                .clone_dtoh(&workspace.selected_ids)
                .map_err(candle::Error::wrap)?,
            stream
                .clone_dtoh(&workspace.selected_weights)
                .map_err(candle::Error::wrap)?,
        ))
    }

    /// Builds a runner when packed tensors satisfy the portable decode contract.
    ///
    /// Unsupported weight dtypes, layouts, devices, expert counts, and top-k
    /// counts return `None` before allocating workspace. Allocation failures for
    /// an otherwise supported model remain visible to the caller.
    pub(in crate::components) fn new(
        gate_up_weights: Tensor,
        down_weights: Tensor,
        experts_per_token: usize,
    ) -> CandleResult<Option<Self>> {
        let Ok((num_experts, packed_intermediate_size, hidden_size)) =
            gate_up_weights.dims3()
        else {
            return Ok(None);
        };
        if packed_intermediate_size == 0 || packed_intermediate_size % 2 != 0 {
            return Ok(None);
        }
        let intermediate_size = packed_intermediate_size / 2;
        let expected_down_shape = [num_experts, hidden_size, intermediate_size];
        let supported = num_experts == SUPPORTED_EXPERTS
            && experts_per_token == SUPPORTED_TOP_K
            && hidden_size > 0
            && intermediate_size > 0
            && gate_up_weights.device().is_cuda()
            && gate_up_weights.device().same_device(down_weights.device())
            && gate_up_weights.dtype() == DType::BF16
            && down_weights.dtype() == DType::BF16
            && gate_up_weights.is_contiguous()
            && down_weights.is_contiguous()
            && down_weights.dims() == expected_down_shape
            && i32::try_from(hidden_size).is_ok()
            && intermediate_size <= (i32::MAX / 2) as usize;
        if !supported {
            return Ok(None);
        }

        let dimensions = FusedMoeDimensions {
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        };
        let workspace = Self::allocate_workspace(&gate_up_weights, dimensions)?;
        Ok(Some(Self {
            gate_up_weights,
            down_weights,
            dimensions,
            workspace: Arc::new(Mutex::new(workspace)),
            config: tau_cuda_kernels::MoeDecodeConfig::default(),
            resolution: None,
        }))
    }

    /// Allocates every intermediate buffer once on the packed weights' device.
    fn allocate_workspace(
        gate_up_weights: &Tensor,
        dimensions: FusedMoeDimensions,
    ) -> CandleResult<FusedMoeWorkspace> {
        let device = gate_up_weights.device().as_cuda_device()?;
        let gate_up_elements = dimensions
            .experts_per_token
            .checked_mul(dimensions.intermediate_size)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE gate/up workspace size overflow".to_owned(),
                )
            })?;
        let activated_elements = dimensions
            .experts_per_token
            .checked_mul(dimensions.intermediate_size)
            .ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE activation workspace size overflow".to_owned(),
                )
            })?;
        let expert_output_elements = dimensions
            .experts_per_token
            .checked_mul(dimensions.hidden_size)
            .ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE output workspace size overflow".to_owned(),
                )
            })?;
        let pointer_elements = dimensions
            .experts_per_token
            .checked_mul(POINTER_ARRAY_COUNT)
            .ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE pointer workspace size overflow".to_owned(),
                )
            })?;

        // SAFETY: Every allocation is fully initialized by native kernels before
        // any corresponding workspace contents are read.
        let workspace = unsafe {
            FusedMoeWorkspace {
                selected_ids: device.alloc::<i32>(dimensions.experts_per_token)?,
                selected_weights: device.alloc::<f32>(dimensions.experts_per_token)?,
                pointer_arrays: device.alloc::<u64>(pointer_elements)?,
                gate_up_output: device.alloc::<bf16>(gate_up_elements)?,
                activated_output: device.alloc::<bf16>(activated_elements)?,
                expert_output: device.alloc::<bf16>(expert_output_elements)?,
                gate_up_fp32: device.alloc::<f32>(gate_up_elements)?,
                expert_fp32: device.alloc::<f32>(expert_output_elements)?,
                #[cfg(test)]
                receipts: device.alloc::<i32>(80)?,
            }
        };
        Ok(workspace)
    }

    /// Reports whether dynamic input and router tensors can use this runner.
    pub(in crate::components) fn supports_input(
        &self,
        input: &Tensor,
        router_logits: &Tensor,
    ) -> bool {
        input.device().same_device(self.gate_up_weights.device())
            && input.device().same_device(router_logits.device())
            && input.dtype() == DType::BF16
            && router_logits.dtype() == DType::BF16
            && input.is_contiguous()
            && router_logits.is_contiguous()
            && input.dims() == [1, self.dimensions.hidden_size]
            && router_logits.dims() == [1, self.dimensions.num_experts]
    }

    /// Applies the native pipeline through Candle's no-backward operator boundary.
    pub(in crate::components) fn forward(
        &self,
        input: &Tensor,
        router_logits: &Tensor,
    ) -> CandleResult<Tensor> {
        input.apply_op2_no_bwd(router_logits, &FusedMoeDecodeCuda { runner: self })
    }

    /// Acquires tensor storage and workspace pointers and enqueues native work.
    fn launch(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        router_storage: &CudaStorage,
        router_layout: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let device = input_storage.device.clone();
        // SAFETY: The shared launch initializes the whole single-token output.
        let output = unsafe { device.alloc::<bf16>(self.dimensions.hidden_size)? };
        let mut output = CudaStorage::wrap_cuda_slice(output, device);
        self.launch_into(
            input_storage,
            input_layout,
            router_storage,
            router_layout,
            &mut output,
        )?;
        Ok((output, input_layout.shape().clone()))
    }

    /// Queues the same native pipeline into either an ordinary output or a
    /// prepared slot. Only the caller chooses allocation ownership;
    /// expert selection, pointer preparation and all arithmetic remain shared.
    fn launch_into(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        router_storage: &CudaStorage,
        router_layout: &Layout,
        output: &mut CudaStorage,
    ) -> CandleResult<()> {
        if input_storage.device.id() != router_storage.device.id()
            || input_storage.device.id()
                != self.gate_up_weights.device().as_cuda_device()?.id()
            || output.device.id() != input_storage.device.id()
        {
            candle::bail!("fused MoE tensors must use one CUDA device")
        }
        if input_storage.dtype() != DType::BF16
            || router_storage.dtype() != DType::BF16
            || output.dtype() != DType::BF16
            || output.as_cuda_slice::<bf16>()?.len() != self.dimensions.hidden_size
        {
            candle::bail!("fused MoE dynamic tensors must use BF16 storage")
        }

        let input_shape = input_layout.shape();
        if input_shape.dims() != [1, self.dimensions.hidden_size]
            || router_layout.shape().dims() != [1, self.dimensions.num_experts]
        {
            candle::bail!("fused MoE dynamic tensor shapes changed after dispatch")
        }
        let (input_start, input_end) =
            input_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg("fused MoE input must be contiguous".to_owned())
            })?;
        let (router_start, router_end) =
            router_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE router logits must be contiguous".to_owned(),
                )
            })?;
        let input = input_storage
            .as_cuda_slice::<bf16>()?
            .slice(input_start..input_end);
        let router_logits = router_storage
            .as_cuda_slice::<bf16>()?
            .slice(router_start..router_end);

        let (gate_up_guard, gate_up_layout) = self.gate_up_weights.storage_and_layout();
        let gate_up_storage = match &*gate_up_guard {
            Storage::Cuda(storage) => storage,
            _ => candle::bail!("fused MoE gate/up weights must use CUDA storage"),
        };
        let (gate_up_start, gate_up_end) =
            gate_up_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE gate/up weights must be contiguous".to_owned(),
                )
            })?;
        let gate_up_weights = gate_up_storage
            .as_cuda_slice::<bf16>()?
            .slice(gate_up_start..gate_up_end);

        let (down_guard, down_layout) = self.down_weights.storage_and_layout();
        let down_storage = match &*down_guard {
            Storage::Cuda(storage) => storage,
            _ => candle::bail!("fused MoE down weights must use CUDA storage"),
        };
        let (down_start, down_end) =
            down_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "fused MoE down weights must be contiguous".to_owned(),
                )
            })?;
        let down_weights = down_storage
            .as_cuda_slice::<bf16>()?
            .slice(down_start..down_end);

        let device = input_storage.device.clone();
        let mut workspace = self.workspace.lock().map_err(|_| {
            candle::Error::Msg("fused MoE workspace lock is poisoned".to_owned())
        })?;
        let FusedMoeWorkspace {
            selected_ids,
            selected_weights,
            pointer_arrays,
            gate_up_output,
            activated_output,
            expert_output,
            gate_up_fp32,
            expert_fp32,
            #[cfg(test)]
            receipts,
        } = &mut *workspace;
        let stream = device.cuda_stream();

        let (input_ptr, input_pointer_guard) = input.device_ptr(&stream);
        let (router_ptr, router_pointer_guard) = router_logits.device_ptr(&stream);
        let (gate_up_ptr, gate_up_pointer_guard) = gate_up_weights.device_ptr(&stream);
        let (down_ptr, down_pointer_guard) = down_weights.device_ptr(&stream);
        let (selected_ids_ptr, selected_ids_guard) =
            selected_ids.device_ptr_mut(&stream);
        let (selected_weights_ptr, selected_weights_guard) =
            selected_weights.device_ptr_mut(&stream);
        let (pointer_arrays_ptr, pointer_arrays_guard) =
            pointer_arrays.device_ptr_mut(&stream);
        let (gate_up_output_ptr, gate_up_output_guard) =
            gate_up_output.device_ptr_mut(&stream);
        let (activated_output_ptr, activated_output_guard) =
            activated_output.device_ptr_mut(&stream);
        let (expert_output_ptr, expert_output_guard) =
            expert_output.device_ptr_mut(&stream);
        let (gate_fp32_ptr, gate_fp32_guard) = gate_up_fp32.device_ptr_mut(&stream);
        let (expert_fp32_ptr, expert_fp32_guard) = expert_fp32.device_ptr_mut(&stream);
        let (output_ptr, output_guard) =
            output.as_cuda_slice_mut::<bf16>()?.device_ptr_mut(&stream);

        #[cfg(test)]
        let (receipts_ptr, receipts_guard) = receipts.device_ptr_mut(&stream);
        #[cfg(not(test))]
        let receipts_ptr = 0;
        // SAFETY: Construction and dispatch validate all shapes, dtypes, device
        // identities, contiguous offsets, allocation lengths, and integer
        // conversions. Pointer guards and tensor storage guards remain alive
        // until every asynchronous operation has been enqueued on this stream.
        unsafe {
            tau_cuda_kernels::launch_fused_bf16_moe_decode(
                input_ptr as *const c_void,
                router_ptr as *const c_void,
                gate_up_ptr as *const c_void,
                down_ptr as *const c_void,
                selected_ids_ptr as *mut c_void,
                selected_weights_ptr as *mut c_void,
                pointer_arrays_ptr as *mut c_void,
                gate_up_output_ptr as *mut c_void,
                activated_output_ptr as *mut c_void,
                expert_output_ptr as *mut c_void,
                gate_fp32_ptr as *mut c_void,
                expert_fp32_ptr as *mut c_void,
                output_ptr as *mut c_void,
                i32::try_from(self.dimensions.hidden_size)
                    .expect("validated hidden size"),
                i32::try_from(self.dimensions.intermediate_size)
                    .expect("validated intermediate size"),
                i32::try_from(self.dimensions.num_experts)
                    .expect("validated expert count"),
                i32::try_from(self.dimensions.experts_per_token)
                    .expect("validated top-k count"),
                &self.config.gate_up,
                &self.config.down,
                receipts_ptr as *mut i32,
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(|error| {
            candle::Error::Cuda(Box::<dyn std::error::Error + Send + Sync>::from(
                error.to_string(),
            ))
        })?;

        #[cfg(test)]
        drop(receipts_guard);
        drop(output_guard);
        drop((expert_fp32_guard, gate_fp32_guard));
        drop(expert_output_guard);
        drop(activated_output_guard);
        drop(gate_up_output_guard);
        drop(pointer_arrays_guard);
        drop(selected_weights_guard);
        drop(selected_ids_guard);
        drop(down_pointer_guard);
        drop(gate_up_pointer_guard);
        drop(router_pointer_guard);
        drop(input_pointer_guard);
        drop(workspace);
        drop(down_guard);
        drop(gate_up_guard);

        Ok(())
    }
}

/// Binary custom operator carrying immutable expert tensors and reusable state.
#[derive(Debug, Clone, Copy)]
struct FusedMoeDecodeCuda<'a> {
    runner: &'a FusedMoeRunner,
}

impl CustomOp2 for FusedMoeDecodeCuda<'_> {
    /// Returns the operation name used by Candle diagnostics.
    fn name(&self) -> &'static str {
        "tau-fused-bf16-moe-decode-cuda"
    }

    /// Rejects accidental CPU use because contract selection is CUDA-only.
    fn cpu_fwd(
        &self,
        _input_storage: &CpuStorage,
        _input_layout: &Layout,
        _router_storage: &CpuStorage,
        _router_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("{} is a CUDA-only operator", self.name())
    }

    /// Launches the BF16 implementation for already-validated CUDA tensors.
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        router_storage: &CudaStorage,
        router_layout: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        self.runner
            .launch(input_storage, input_layout, router_storage, router_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Rejects CPU tensors before allocating any native runner state.
    #[test]
    fn unsupported_cpu_weights_do_not_create_runner() {
        let device = Device::Cpu;
        let gate_up =
            Tensor::zeros((SUPPORTED_EXPERTS, 8, 16), DType::BF16, &device).unwrap();
        let down =
            Tensor::zeros((SUPPORTED_EXPERTS, 16, 4), DType::BF16, &device).unwrap();

        assert!(
            FusedMoeRunner::new(gate_up, down, SUPPORTED_TOP_K)
                .unwrap()
                .is_none()
        );
    }
}
