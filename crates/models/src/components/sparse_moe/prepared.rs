//! Sparse feed-forward contribution to the shared complete decode program.
//!
//! This owner composes operations; the CUDA runner owns routed execution and
//! the model owns capture, replay and recurrent/cache state. Prefill stays eager.

use super::{Execution, SparseMoeBlock};
use crate::components::{DecodeProgram, GraphFeedForward};
use candle::{Result, Tensor};

impl GraphFeedForward for SparseMoeBlock {
    /// Keeps eager decode on its existing native path, separate from prefill.
    fn forward_decode(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_phase(input, Execution::Decode)
    }
    /// Checks the entire single-token block before preparing or advancing state.
    fn supports_prepared_decode(&self, input: &Tensor) -> bool {
        self.fused_moe_runner
            .as_ref()
            .is_some_and(|runner| runner.supports_prepared_input(input))
            && self.router.supports_prepared_decode(input)
            && self
                .shared_expert
                .as_ref()
                .is_none_or(|shared| shared.supports_prepared_decode(input))
    }

    /// Records routing, routed experts and the independent shared branch.
    /// Selected experts are device values recomputed on every replay, never
    /// host decisions baked into the graph's fixed topology or addresses.
    fn prepare_decode_program(
        &self,
        program: &mut DecodeProgram,
        input: &Tensor,
    ) -> Result<Tensor> {
        if !self.supports_prepared_decode(input) {
            candle::bail!("unsupported prepared sparse MoE block")
        }
        let logits = self.router.prepare_decode_program(program, input)?;
        let routed = self
            .fused_moe_runner
            .as_ref()
            .expect("validated routed runner")
            .prepare_decode_program(program, input, &logits)?;
        match &self.shared_expert {
            Some(shared) => shared.prepare_decode_combine(program, input, &routed),
            None => Ok(routed),
        }
    }
}
