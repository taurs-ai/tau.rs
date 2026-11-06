//! Decode workspace allocation and explicit state import/export.

use super::*;

impl GatedDeltaNetRunner {
    /// Allocates state and scratch once, before capture or steady-state decode.
    ///
    /// Convolution history and recurrent memory start at zero. Scratch buffers
    /// start uninitialized because kernels overwrite them before reading them.
    /// The 128x128 recurrence also needs an intermediate output for its separate
    /// normalization kernel. Reusing these buffers preserves addresses for graphs.
    pub(super) fn allocate_workspace(
        conv_weight: &Tensor,
        dimensions: GatedDeltaNetDimensions,
    ) -> CandleResult<GatedDeltaNetWorkspace> {
        let stream = conv_weight.device().as_cuda_device()?.cuda_stream();
        let conv_state_elements = dimensions
            .conv_width
            .checked_mul(dimensions.conv_kernel_size)
            .ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet convolution-state size overflow".to_owned(),
                )
            })?;
        let recurrent_state_elements = dimensions
            .num_value_heads
            .checked_mul(dimensions.key_head_dim)
            .and_then(|count| count.checked_mul(dimensions.value_head_dim))
            .ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet recurrent-state size overflow".to_owned(),
                )
            })?;
        Ok(GatedDeltaNetWorkspace {
            conv_state: stream
                .alloc_zeros::<bf16>(conv_state_elements)
                .map_err(cuda_driver_error)?,
            recurrent_state: stream
                .alloc_zeros::<f32>(recurrent_state_elements)
                .map_err(cuda_driver_error)?,
            // SAFETY: The convolution kernel initializes every element before use.
            convolved_qkv: unsafe {
                stream
                    .alloc::<bf16>(dimensions.conv_width)
                    .map_err(cuda_driver_error)?
            },
            recurrence_output: if dimensions.key_head_dim == 128
                && dimensions.value_head_dim == 128
            {
                // SAFETY: All four tiles initialize their columns before norm
                // reads them on the same stream. Scratch needs no reset/reseed.
                Some(unsafe {
                    stream
                        .alloc::<bf16>(dimensions.value_width)
                        .map_err(cuda_driver_error)?
                })
            } else {
                None
            },
        })
    }

    /// Seeds native state from Candle prefill state or resets it for a new request.
    ///
    /// Writes into the existing allocations on Candle's stream. Call outside capture
    /// so the copies/zeros do not become part of every decode replay.
    /// Supply both tensors to continue from prefill, or neither to start from zero.
    /// Supplying only one is an error because the two states describe one sequence.
    pub(in crate::components) fn seed_state(
        &self,
        conv_state: Option<&Tensor>,
        recurrent_state: Option<&Tensor>,
    ) -> CandleResult<()> {
        match (conv_state, recurrent_state) {
            (None, None) => {
                let stream = self.conv_weight.device().as_cuda_device()?.cuda_stream();
                let mut workspace = self.lock_workspace()?;
                stream
                    .memset_zeros(&mut workspace.conv_state)
                    .map_err(cuda_driver_error)?;
                stream
                    .memset_zeros(&mut workspace.recurrent_state)
                    .map_err(cuda_driver_error)?;
                Ok(())
            }
            (Some(conv_state), Some(recurrent_state)) => {
                self.copy_candle_state(conv_state, recurrent_state)
            }
            _ => candle::bail!(
                "Gated DeltaNet convolution and recurrent state must coexist"
            ),
        }
    }

    /// Imports Candle history and recurrent memory into the existing decode buffers.
    ///
    /// Validate both states before copying, materialize contiguous sources when
    /// needed, then enqueue device-to-device copies. Destination addresses remain
    /// fixed, so previously captured decode graphs keep using the same buffers.
    fn copy_candle_state(
        &self,
        conv_state: &Tensor,
        recurrent_state: &Tensor,
    ) -> CandleResult<()> {
        let dimensions = self.dimensions;
        let supported = conv_state.device().same_device(self.conv_weight.device())
            && recurrent_state
                .device()
                .same_device(self.conv_weight.device())
            && conv_state.dtype() == DType::BF16
            && recurrent_state.dtype() == DType::F32
            && conv_state.dims()
                == [1, dimensions.conv_width, dimensions.conv_kernel_size]
            && recurrent_state.dims()
                == [
                    1,
                    dimensions.num_value_heads,
                    dimensions.key_head_dim,
                    dimensions.value_head_dim,
                ];
        if !supported {
            candle::bail!("reference Gated DeltaNet state cannot seed native decode")
        }
        // Reference convolution can leave a strided trailing-window view; native
        // prefill already produces compact history. Materialize only when needed
        // at this boundary; steady-state decode updates its buffers directly.
        let conv_state = conv_state.contiguous()?;
        let recurrent_state = recurrent_state.contiguous()?;

        let (conv_guard, conv_layout) = conv_state.storage_and_layout();
        let conv_storage = match &*conv_guard {
            Storage::Cuda(storage) => storage,
            _ => candle::bail!("Gated DeltaNet convolution state must use CUDA"),
        };
        let (conv_start, conv_end) =
            conv_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet convolution state must be contiguous".to_owned(),
                )
            })?;
        let conv_source = conv_storage
            .as_cuda_slice::<bf16>()?
            .slice(conv_start..conv_end);

        let (recurrent_guard, recurrent_layout) = recurrent_state.storage_and_layout();
        let recurrent_storage = match &*recurrent_guard {
            Storage::Cuda(storage) => storage,
            _ => candle::bail!("Gated DeltaNet recurrent state must use CUDA"),
        };
        let (recurrent_start, recurrent_end) =
            recurrent_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet recurrent state must be contiguous".to_owned(),
                )
            })?;
        let recurrent_source = recurrent_storage
            .as_cuda_slice::<f32>()?
            .slice(recurrent_start..recurrent_end);

        let stream = conv_storage.device.cuda_stream();
        let mut workspace = self.lock_workspace()?;
        stream
            .memcpy_dtod(&conv_source, &mut workspace.conv_state)
            .map_err(cuda_driver_error)?;
        stream
            .memcpy_dtod(&recurrent_source, &mut workspace.recurrent_state)
            .map_err(cuda_driver_error)?;
        drop(workspace);
        drop(recurrent_guard);
        drop(conv_guard);
        Ok(())
    }

    /// Returns independent state snapshots for reference fallback or validation.
    ///
    /// Copies are ordered on Candle's stream and do not change native state ownership.
    /// The returned tensors own independent GPU allocations, so later decode
    /// steps cannot overwrite a saved snapshot. No state is copied to the CPU.
    pub(in crate::components) fn export_state(&self) -> CandleResult<(Tensor, Tensor)> {
        let device = self.conv_weight.device().as_cuda_device()?.clone();
        let stream = device.cuda_stream();
        let workspace = self.lock_workspace()?;
        let conv_state = stream
            .clone_dtod(&workspace.conv_state)
            .map_err(cuda_driver_error)?;
        let recurrent_state = stream
            .clone_dtod(&workspace.recurrent_state)
            .map_err(cuda_driver_error)?;
        drop(workspace);

        let dimensions = self.dimensions;
        let conv_state = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(conv_state, device.clone())),
            (1, dimensions.conv_width, dimensions.conv_kernel_size),
            BackpropOp::none(),
            false,
        );
        let recurrent_state = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(recurrent_state, device)),
            (
                1,
                dimensions.num_value_heads,
                dimensions.key_head_dim,
                dimensions.value_head_dim,
            ),
            BackpropOp::none(),
            false,
        );
        Ok((conv_state, recurrent_state))
    }

    /// Exposes allocation identity to lifecycle tests without exporting mutable
    /// storage.
    /// Tests use these two device addresses to confirm resets and reseeding reuse
    /// the convolution/recurrent buffers rather than replacing them.
    #[cfg(test)]
    pub(in crate::components) fn state_addresses(&self) -> CandleResult<(u64, u64)> {
        let stream = self.conv_weight.device().as_cuda_device()?.cuda_stream();
        let workspace = self.lock_workspace()?;
        let (conv, _conv_guard) = workspace.conv_state.device_ptr(&stream);
        let (recurrent, _recurrent_guard) =
            workspace.recurrent_state.device_ptr(&stream);
        Ok((conv, recurrent))
    }
}
