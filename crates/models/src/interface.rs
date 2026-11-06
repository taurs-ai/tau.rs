//! Minimal generation interface and bridges to existing causal model types.
//! Tokenization, chat formatting and checkpoint loading remain outside it.

use crate::CudaGraphStats;
use crate::qwen2::Qwen2ForCausalLM;
use crate::qwen3::Qwen3ForCausalLM;
use crate::qwen3_5_common::HybridCausalLm;
use candle::Tensor as CandleTensor;
use tau_metrics::timing::DecodeTimer;

/// Minimal model operations required by the shared autoregressive runtime.
///
/// The trait is local to tau.rs, so Candle's family-specific model types can
/// implement it without wrappers. Chat formatting and model loading remain outside
/// this interface because those behaviors differ by model family.
pub trait CausalLanguageModel {
    /// Reserve the request's prompt/output budget where the model supports it.
    /// Candle-owned model families keep their existing allocation policy.
    fn reserve_kv_cache(&mut self, _capacity: usize) -> candle::Result<()> {
        Ok(())
    }

    /// Evaluates one token chunk at the supplied cache offset.
    fn forward(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor>;

    /// Evaluates a prompt chunk, retaining its state and returning logits only
    /// for the last chunk. Families without prefill graphs use ordinary forward.
    fn forward_prefill_chunk(
        &mut self,
        input: &CandleTensor,
        offset: usize,
        _full_chunk: bool,
        last_chunk: bool,
    ) -> candle::Result<Option<CandleTensor>> {
        let logits = self.forward(input, offset)?;
        Ok(last_chunk.then_some(logits))
    }

    /// Decode is explicit: shape alone cannot distinguish a one-token prompt.
    fn forward_decode(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        self.forward(input, offset)
    }

    /// Accepts a host token at the generation boundary. The default builds the
    /// ordinary one-token input; models with retained graph metadata can override
    /// it to avoid that per-step device allocation.
    fn forward_decode_token(
        &mut self,
        token: u32,
        offset: usize,
        device: &candle::Device,
    ) -> candle::Result<CandleTensor> {
        let input = CandleTensor::new(&[[token]], device)?;
        self.forward_decode(&input, offset)
    }

    /// Times an ordinary decode with input allocation outside the GPU interval.
    /// Models with retained metadata override this to place events after their
    /// state preparation and any complete-graph capture.
    fn forward_decode_token_timed(
        &mut self,
        token: u32,
        offset: usize,
        device: &candle::Device,
        timing: &mut DecodeTimer,
    ) -> candle::Result<CandleTensor> {
        let input = CandleTensor::new(&[[token]], device)?;
        timing.begin_forward("model_forward")?;
        let logits = self.forward_decode(&input, offset)?;
        timing.end_forward()?;
        Ok(logits)
    }

    fn cuda_graph_enabled(&self) -> bool {
        false
    }

    /// Snapshots capture/replay counters for per-step activity checks. Models
    /// without graph support return zero; generation subtracts the before-step
    /// snapshot from the after-step snapshot to identify actual graph execution.
    fn cuda_graph_counts(&self) -> CudaGraphStats {
        CudaGraphStats::default()
    }

    /// Clears sequence state before processing an unrelated request.
    fn clear_kv_cache(&mut self);

    /// Returns per-token KV/workspace bytes; legacy families retain their policy.
    fn kv_memory_per_token(
        &self,
        _prefill_chunk: usize,
    ) -> candle::Result<(usize, usize)> {
        Ok((0, 0))
    }

    /// Applies a capacity budget where supported, before request reservation.
    fn set_kv_cache_capacity(&mut self, _capacity: usize) -> candle::Result<()> {
        Ok(())
    }

    /// Selects requested graph coverage; unsupported families stay eager.
    fn set_cuda_graph(&mut self, _mode: crate::CudaGraphMode) {}
}

impl CausalLanguageModel for Qwen2ForCausalLM {
    /// Delegates shared-runtime evaluation to Candle's Qwen2 model.
    fn forward(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        Qwen2ForCausalLM::forward(self, input, offset)
    }

    /// Clears Candle's Qwen2 key/value cache.
    fn clear_kv_cache(&mut self) {
        Qwen2ForCausalLM::clear_kv_cache(self);
    }
}

impl CausalLanguageModel for Qwen3ForCausalLM {
    /// Delegates shared-runtime evaluation to Candle's Qwen3 model.
    fn forward(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        Qwen3ForCausalLM::forward(self, input, offset)
    }

    /// Clears Candle's Qwen3 key/value cache.
    fn clear_kv_cache(&mut self) {
        Qwen3ForCausalLM::clear_kv_cache(self);
    }
}

impl<F> CausalLanguageModel for HybridCausalLm<F>
where
    F: crate::components::GraphFeedForward,
{
    /// Supplies the existing family geometry to the runtime's budget policy.
    fn kv_memory_per_token(&self, chunk: usize) -> candle::Result<(usize, usize)> {
        HybridCausalLm::kv_memory_per_token(self, chunk)
    }

    /// Preserves the model's pre-allocation capacity validation.
    fn set_kv_cache_capacity(&mut self, capacity: usize) -> candle::Result<()> {
        HybridCausalLm::set_kv_cache_capacity(self, capacity)
    }

    /// Retains the current model-owned graph invalidation and selection rules.
    fn set_cuda_graph(&mut self, mode: crate::CudaGraphMode) {
        HybridCausalLm::set_cuda_graph(self, mode);
    }

    /// Delegates the exact replay boundary to the hybrid model's graph owner.
    fn forward_decode_token_timed(
        &mut self,
        token: u32,
        offset: usize,
        _device: &candle::Device,
        timing: &mut DecodeTimer,
    ) -> candle::Result<CandleTensor> {
        HybridCausalLm::forward_decode_token_timed(self, token, offset, timing)
    }

    /// Preserves the host-token entry point so eligible hybrid models update
    /// retained metadata and replay without first allocating an input tensor.
    fn forward_decode_token(
        &mut self,
        token: u32,
        offset: usize,
        _device: &candle::Device,
    ) -> candle::Result<CandleTensor> {
        HybridCausalLm::forward_decode_token(self, token, offset)
    }
    fn reserve_kv_cache(&mut self, capacity: usize) -> candle::Result<()> {
        HybridCausalLm::reserve_kv_cache(self, capacity)
    }

    /// Delegates eager evaluation to tau.rs's shared Qwen3.5 hybrid model.
    fn forward(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        HybridCausalLm::forward_eager(self, input, offset)
    }

    /// Clears attention and DeltaNet state in the hybrid model.
    fn clear_kv_cache(&mut self) {
        HybridCausalLm::clear_cache(self);
    }

    fn forward_prefill_chunk(
        &mut self,
        input: &CandleTensor,
        offset: usize,
        full_chunk: bool,
        last_chunk: bool,
    ) -> candle::Result<Option<CandleTensor>> {
        HybridCausalLm::forward_prefill_chunk(
            self, input, offset, full_chunk, last_chunk,
        )
    }

    fn forward_decode(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        // Delegate to HybridCausalLm's decode dispatch.
        HybridCausalLm::forward_decode(self, input, offset)
    }

    fn cuda_graph_enabled(&self) -> bool {
        HybridCausalLm::cuda_graph_enabled(self)
    }

    /// Exposes the hybrid model's combined complete-forward and per-layer graph
    /// counters to the shared generation loop without changing their scope.
    fn cuda_graph_counts(&self) -> CudaGraphStats {
        HybridCausalLm::cuda_graph_counts(self)
    }
}
