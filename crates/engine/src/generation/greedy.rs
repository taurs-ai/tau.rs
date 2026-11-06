//! GPU greedy selection with engine-owned reusable storage and scalar readback.
//!
//! Model graphs still finish at logits. This sampler uses the output's actual
//! device stream, then waits only for its small result copy outside capture.

use super::candle_inference_error;
use crate::InferenceError;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PinnedHostSlice, sys,
};
use candle::{CudaDevice, DType, Device, Storage, Tensor};
use core::ffi::c_void;
use half::bf16;
use std::sync::Arc;
use tau_cuda_kernels::{
    GreedyDType, greedy_argmax_workspace_words, launch_greedy_argmax,
};

/// Selects greedy IDs on CUDA and caches the resources needed between calls.
/// Unsupported contracts return None so the caller can use the CPU selector.
#[derive(Default)]
pub(crate) struct GpuGreedy {
    workspace: Option<CudaGreedyWorkspace>,
}

/// Reusable CUDA resources for vocabulary reduction and selected-token readback.
///
/// `scratch` holds one candidate per reduction block; `result` holds the final
/// `[token_id, valid]` pair, copied into pinned `host` storage. The retained device,
/// stream and vocabulary identify when these allocations can be reused across
/// tokens and requests. Both reduction kernels and the result copy use this stream.
struct CudaGreedyWorkspace {
    device: CudaDevice,
    stream: Arc<CudaStream>,
    vocabulary: u32,
    scratch: CudaSlice<u32>,
    result: CudaSlice<u32>,
    host: PinnedHostSlice<u32>,
}

impl GpuGreedy {
    /// Returns a GPU-selected ID, a visible backend error, or None for fallback.
    /// NaNs/ordering/ties match `next_token_from_values` exactly. Empty vectors
    /// and unsupported layouts/dtypes retain that function's existing behavior.
    pub(crate) fn select(
        &mut self,
        logits: &Tensor,
    ) -> Result<Option<u32>, InferenceError> {
        let Device::Cuda(device) = logits.device() else {
            return Ok(None);
        };
        if logits.dims().len() != 3
            || logits.dims()[0] != 1
            || logits.dims()[1] != 1
            || !logits.is_contiguous()
            || !matches!(logits.dtype(), DType::BF16 | DType::F32)
        {
            return Ok(None);
        }
        let Ok(vocabulary) = u32::try_from(logits.elem_count()) else {
            return Ok(None);
        };
        if vocabulary == 0 {
            return Ok(None);
        }
        let stream = device.cuda_stream();
        if stream.capture_status().map_err(backend_error)?
            != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(InferenceError::Backend(
                "greedy scalar readback must remain outside CUDA capture".to_owned(),
            ));
        }
        let reusable = self.workspace.as_ref().is_some_and(|w| {
            w.vocabulary == vocabulary
                && w.device.id() == device.id()
                && w.stream.cu_stream() == stream.cu_stream()
        });
        if !reusable {
            self.workspace = Some(CudaGreedyWorkspace::new(device, vocabulary)?);
        }
        let workspace = self.workspace.as_mut().expect("prepared greedy workspace");
        let (storage, layout) = logits.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            return Err(InferenceError::Backend(
                "expected CUDA logits storage".to_owned(),
            ));
        };
        let selected = match storage.dtype() {
            DType::F32 => workspace.launch(
                storage
                    .as_cuda_slice::<f32>()
                    .map_err(candle_inference_error)?,
                layout.start_offset(),
                GreedyDType::F32,
            ),
            DType::BF16 => workspace.launch(
                storage
                    .as_cuda_slice::<bf16>()
                    .map_err(candle_inference_error)?,
                layout.start_offset(),
                GreedyDType::BF16,
            ),
            _ => unreachable!("supported dtype checked before workspace preparation"),
        }?;
        Ok(Some(selected))
    }
}

impl CudaGreedyWorkspace {
    /// Allocates before the first selection or a device/vocabulary change. The
    /// pinned host buffer owns its copy-completion event and is reused thereafter.
    fn new(device: &CudaDevice, vocabulary: u32) -> Result<Self, InferenceError> {
        let stream = device.cuda_stream();
        let scratch = stream
            .alloc_zeros(greedy_argmax_workspace_words(vocabulary))
            .map_err(backend_error)?;
        let result = stream.alloc_zeros(2).map_err(backend_error)?;
        // SAFETY: U32 has no invalid bit patterns. Initialize both host words
        // before any typed slice access, then retain the allocation through copies.
        let mut host = unsafe { stream.context().alloc_pinned::<u32>(2) }
            .map_err(backend_error)?;
        // SAFETY: The fresh allocation holds two writable U32 words, and no
        // asynchronous copy can reference it before construction finishes.
        unsafe {
            std::ptr::write_bytes(host.as_mut_ptr().map_err(backend_error)?, 0, 2)
        };
        Ok(Self {
            device: device.clone(),
            stream,
            vocabulary,
            scratch,
            result,
            host,
        })
    }

    /// Resolves the offset view while keeping all dependency guards alive through
    /// enqueue. The final host access waits for its copy event, which is ordered
    /// after the model forward and both reductions on this same stream.
    fn launch<T: DeviceRepr>(
        &mut self,
        input: &CudaSlice<T>,
        offset: usize,
        dtype: GreedyDType,
    ) -> Result<u32, InferenceError> {
        let input = input.slice(offset..offset + self.vocabulary as usize);
        let scratch_words = self.scratch.len();
        {
            let (source, _input_guard) = input.device_ptr(&self.stream);
            let (scratch, _scratch_guard) = self.scratch.device_ptr_mut(&self.stream);
            let (result, _result_guard) = self.result.device_ptr_mut(&self.stream);
            // SAFETY: The contiguous input view covers the complete vocabulary.
            // Scratch uses the native size query; result owns two U32 words.
            // Guards track prior writes and retain enqueue dependencies. All
            // allocations are disjoint and survive the ordered scalar readback.
            unsafe {
                launch_greedy_argmax(
                    source as *const c_void,
                    dtype,
                    self.vocabulary,
                    scratch as *mut u32,
                    scratch_words,
                    result as *mut u32,
                    self.stream.cu_stream() as *mut c_void,
                )
            }
            .map_err(backend_error)?;
        }
        self.stream
            .memcpy_dtoh(&self.result, &mut self.host)
            .map_err(backend_error)?;
        // Match ordinary logit readback's completion boundary on this inference
        // stream. Waiting first on the pinned buffer's blocking event can delay
        // the next enqueue when the host sleeps through a long model forward.
        // The copy event still owns the pinned memory's asynchronous lifetime.
        self.stream.synchronize().map_err(backend_error)?;
        let selected = self.host.as_slice().map_err(backend_error)?;
        if selected[1] == 0 {
            return Err(InferenceError::Backend(
                "model logits are all NaN".to_owned(),
            ));
        }
        Ok(selected[0])
    }
}

fn backend_error(error: impl std::fmt::Display) -> InferenceError {
    InferenceError::Backend(format!("GPU greedy selection failed: {error}"))
}

#[cfg(test)]
mod tests;
