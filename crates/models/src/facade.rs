//! Opaque family façades keep decoder/components private across the crate boundary.

use crate::{CausalLanguageModel, CudaGraphMode, CudaGraphStats};
use candle::{Device, Result, Tensor};
use tau_metrics::timing::DecodeTimer;

// Both wrappers contain exactly the existing model. Delegation is static: no
// boxing, per-token dispatch, extra buffers or changes to graph/state ownership.
macro_rules! hybrid_facade {
    ($name:ident, $inner:ty) => {
        /// Loaded family model with private architecture/component implementation.
        pub struct $name {
            inner: $inner,
        }

        impl $name {
            /// Wraps existing ownership after validated construction.
            pub(crate) fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }
        }

        impl CausalLanguageModel for $name {
            /// Reserves state using the model's existing capacity/lifetime rules.
            fn reserve_kv_cache(&mut self, capacity: usize) -> Result<()> {
                self.inner.reserve_kv_cache(capacity)
            }
            /// Preserves ordinary eager arithmetic for the supplied token chunk.
            fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
                CausalLanguageModel::forward(&mut self.inner, input, offset)
            }
            /// Preserves chunk/tail dispatch and final-chunk logit selection.
            fn forward_prefill_chunk(
                &mut self,
                input: &Tensor,
                offset: usize,
                full_chunk: bool,
                last_chunk: bool,
            ) -> Result<Option<Tensor>> {
                CausalLanguageModel::forward_prefill_chunk(
                    &mut self.inner,
                    input,
                    offset,
                    full_chunk,
                    last_chunk,
                )
            }
            /// Delegates decode without treating one-token prefill as decode.
            fn forward_decode(
                &mut self,
                input: &Tensor,
                offset: usize,
            ) -> Result<Tensor> {
                CausalLanguageModel::forward_decode(&mut self.inner, input, offset)
            }
            /// Retains the allocation-free host-token graph replay entry point.
            fn forward_decode_token(
                &mut self,
                token: u32,
                offset: usize,
                device: &Device,
            ) -> Result<Tensor> {
                CausalLanguageModel::forward_decode_token(
                    &mut self.inner,
                    token,
                    offset,
                    device,
                )
            }
            /// Keeps event placement outside capture and after metadata preparation.
            fn forward_decode_token_timed(
                &mut self,
                token: u32,
                offset: usize,
                device: &Device,
                timing: &mut DecodeTimer,
            ) -> Result<Tensor> {
                CausalLanguageModel::forward_decode_token_timed(
                    &mut self.inner,
                    token,
                    offset,
                    device,
                    timing,
                )
            }
            /// Reports the model's actual graph enablement.
            fn cuda_graph_enabled(&self) -> bool {
                self.inner.cuda_graph_enabled()
            }
            /// Returns cumulative counters without changing them.
            fn cuda_graph_counts(&self) -> CudaGraphStats {
                self.inner.cuda_graph_counts()
            }
            /// Clears sequence state while preserving existing graph/storage lifetimes.
            fn clear_kv_cache(&mut self) {
                self.inner.clear_cache();
            }
            /// Reports family geometry used by engine-level memory budgeting.
            fn kv_memory_per_token(&self, chunk: usize) -> Result<(usize, usize)> {
                self.inner.kv_memory_per_token(chunk)
            }
            /// Applies a validated engine budget before request storage is reserved.
            fn set_kv_cache_capacity(&mut self, capacity: usize) -> Result<()> {
                self.inner.set_kv_cache_capacity(capacity)
            }
            /// Selects requested coverage; model dispatch retains capability checks.
            fn set_cuda_graph(&mut self, mode: CudaGraphMode) {
                self.inner.set_cuda_graph(mode);
            }
        }
    };
}

hybrid_facade!(Qwen3_5Model, crate::qwen3_5::Qwen3_5Model);
hybrid_facade!(Qwen3_5MoeModel, crate::qwen3_5_moe::Qwen3_5MoeModel);
