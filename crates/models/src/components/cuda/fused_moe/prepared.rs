//! Fixed output-slot adapter for the existing routed expert launcher.
//!
//! The program retains weights and the same workspace as ordinary execution. This
//! adapter changes storage ownership only; native routing and expert math are
//! shared with ordinary execution.

use super::FusedMoeRunner;
use crate::components::cuda::decode::DecodeProgram;
use candle::{CpuStorage, CudaStorage, DType, InplaceOp1, Layout, Result, Tensor};

impl FusedMoeRunner {
    /// Selects the validated BF16 single-token contract without allocating slots.
    pub(in crate::components) fn supports_prepared_input(
        &self,
        input: &Tensor,
    ) -> bool {
        input.dtype() == DType::BF16
            && input.device().same_device(self.gate_up_weights.device())
            && input.is_contiguous()
            && input.dims() == [1, 1, self.dimensions.hidden_size]
    }

    /// Retains a launch into a preallocated slot with no nested capture/allocations.
    pub(in crate::components) fn prepare_decode_program(
        &self,
        program: &mut DecodeProgram,
        input: &Tensor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        if !self.supports_prepared_input(input) {
            candle::bail!("unsupported prepared routed MoE input")
        }
        let input = input.reshape((1, self.dimensions.hidden_size))?;
        let logits = logits.reshape((1, self.dimensions.num_experts))?;
        if !self.supports_input(&input, &logits) {
            candle::bail!("unsupported prepared routed MoE logits")
        }
        let output = program.buffer((1, 1, self.dimensions.hidden_size))?;
        #[cfg(test)]
        {
            use crate::test_support::tensor_capture as capture;
            capture::tensor("moe-input", &input);
            capture::tensor("moe-logits", &logits);
            capture::tensor("moe-routed", &output);
            let runner = self.clone();
            capture::defer("moe", move || runner.validation_snapshot());
        }
        let operation = PreparedMoe {
            runner: self.clone(),
            input,
            logits,
        };
        let target = output.clone();
        program.push(Box::new(move || target.inplace_op1(&operation)));
        Ok(output)
    }
}

struct PreparedMoe {
    runner: FusedMoeRunner,
    input: Tensor,
    logits: Tensor,
}

impl InplaceOp1 for PreparedMoe {
    /// Identifies failures at the prepared routed-expert boundary.
    fn name(&self) -> &'static str {
        "tau-prepared-moe-decode"
    }

    /// Rejects CPU execution; eligibility is decided before preparing the model.
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        candle::bail!("prepared MoE decode is CUDA-only")
    }

    /// Borrows retained inputs and queues the shared native launch. Capture and
    /// replay use this stream exclusively; every route/scratch value is rewritten
    /// before being consumed, even when the selected expert set changes entirely.
    fn cuda_fwd(&self, output: &mut CudaStorage, layout: &Layout) -> Result<()> {
        if layout.start_offset() != 0
            || layout.shape().dims() != [1, 1, self.runner.dimensions.hidden_size]
        {
            candle::bail!("prepared MoE output slot changed")
        }
        let (input_storage, input_layout) = self.input.storage_and_layout();
        let (logit_storage, logit_layout) = self.logits.storage_and_layout();
        let (candle::Storage::Cuda(input), candle::Storage::Cuda(logits)) =
            (&*input_storage, &*logit_storage)
        else {
            candle::bail!("prepared MoE requires retained CUDA inputs")
        };
        self.runner
            .launch_into(input, input_layout, logits, logit_layout, output)
    }
}
