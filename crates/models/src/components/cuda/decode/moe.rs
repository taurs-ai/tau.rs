//! MoE-specific precision composition inside the shared prepared program.
use super::{DecodeProgram, error, operation::Kind};
use candle::{Result, Tensor};

impl DecodeProgram {
    /// Keeps the shared MLP unchanged while fusing FP32 scalar gating and add.
    /// All slots/weights are retained before capture; no dense defaults change.
    pub(crate) fn moe_shared_combine(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        shared: &Tensor,
        routed: &Tensor,
    ) -> Result<Tensor> {
        let width = input.elem_count();
        if width == 0
            || width > 8192
            || weight.elem_count() != width
            || shared.elem_count() != width
            || routed.elem_count() != width
        {
            candle::bail!("invalid prepared MoE shared-combine geometry")
        }
        let output = self.buffer(shared.shape().clone())?;
        self.operation(
            output,
            &[input, weight, shared, routed],
            Kind::MoeShared {
                rows: 1,
                width: i32::try_from(width).map_err(error)?,
            },
        )
    }
}
