//! Shared prepared cuBLAS handle and owned scratch lifetime.
//!
//! Bind the stream before assigning workspace: later stream changes reset that
//! assignment. Prepared dense projections borrow this owner; routed experts
//! use their separate persistent tensor-core workspace.

use super::error;
use candle::cuda_backend::cudarc::{
    cublas,
    driver::{CudaSlice, DevicePtrMut},
};
use candle::{Device, Result};
use std::sync::Arc;

/// One workspace serves all prepared projections on the private serialized
/// stream. This fixed graph cost belongs to the existing workspace headroom.
const BLAS_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

pub(in crate::components::cuda) struct PreparedBlas {
    // Destroy the handle before releasing its workspace. The graph owner waits
    // for replay before dropping the program and these retained resources.
    pub(super) handle: cublas::CudaBlas,
    _workspace: CudaSlice<u8>,
}

impl PreparedBlas {
    /// Assigns owned scratch after selecting the stream and before any GEMM.
    /// A dedicated handle keeps prefill and ordinary eager handle settings local.
    pub(super) fn new(device: &Device) -> Result<Self> {
        let stream = device.as_cuda_device()?.cuda_stream();
        let handle = cublas::CudaBlas::new(Arc::clone(&stream)).map_err(error)?;
        let mut workspace = stream
            .alloc_zeros::<u8>(BLAS_WORKSPACE_BYTES)
            .map_err(error)?;
        let (pointer, guard) = workspace.device_ptr_mut(&stream);
        // SAFETY: CUDA allocation supplies the required 256-byte alignment. The
        // handle retains this stream; its workspace remains owned until handle
        // teardown. All projections and graph replay use this one ordered stream.
        unsafe {
            cublas::sys::cublasSetWorkspace_v2(
                *handle.handle(),
                pointer as *mut core::ffi::c_void,
                BLAS_WORKSPACE_BYTES,
            )
            .result()
            .map_err(error)?;
        }
        drop(guard);
        Ok(Self {
            handle,
            _workspace: workspace,
        })
    }
}
