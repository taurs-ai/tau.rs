//! Test-only snapshots and isolated projection enqueue over persistent decode
//! scratch. Prepared graph observers and the artifact replay share this owner;
//! all readbacks happen after execution and outside graph capture.
use super::*;

impl FusedMoeRunner {
    /// Queue only one projection using the pipeline's seeded device worklist.
    /// Validation retains the runner and every operand through graph teardown;
    /// callers seed routing/activation first and serialize on this private stream.
    pub(super) fn validation_project(&self, down: bool) -> CandleResult<()> {
        let mut workspace = self.workspace.lock().unwrap();
        let stream = self
            .gate_up_weights
            .device()
            .as_cuda_device()?
            .cuda_stream();
        let FusedMoeWorkspace {
            pointer_arrays,
            receipts,
            ..
        } = &mut *workspace;
        let (pointer, guard) = pointer_arrays.device_ptr(&stream);
        let (receipt, receipt_guard) = receipts.device_ptr_mut(&stream);
        let (input, weight, out, columns, inner, profile, offset) = if down {
            (
                4,
                3,
                5,
                self.dimensions.hidden_size,
                self.dimensions.intermediate_size,
                self.config.down,
                40,
            )
        } else {
            (
                1,
                0,
                2,
                2 * self.dimensions.intermediate_size,
                self.dimensions.hidden_size,
                self.config.gate_up,
                0,
            )
        };
        // SAFETY: Seeded arrays contain eight complete/disjoint operand views.
        // The graph/program retain this runner, buffers and immutable banks; all
        // subsequent work and readbacks use this event-tracking-disabled stream.
        unsafe {
            tau_cuda_kernels::launch_moe_decode_projection(
                (pointer + input * 8 * 8) as *const c_void,
                (pointer + weight * 8 * 8) as *const c_void,
                (pointer + out * 8 * 8) as *mut c_void,
                8,
                columns as i32,
                inner as i32,
                &profile,
                (receipt + offset * 4) as *mut i32,
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
        drop((guard, receipt_guard));
        Ok(())
    }

    pub(in crate::components::cuda::fused_moe) fn validation_snapshot(
        &self,
    ) -> BTreeMap<String, Value> {
        let workspace = self.workspace.lock().unwrap();
        let stream = self
            .gate_up_weights
            .device()
            .as_cuda_device()
            .unwrap()
            .cuda_stream();
        let d = self.dimensions;
        let host = |shape, values| Value::Host { shape, values };
        let bf = |values: Vec<bf16>| {
            values.into_iter().map(bf16::to_f32).collect::<Vec<_>>()
        };
        BTreeMap::from([
            (
                "ids".to_owned(),
                host(
                    vec![8],
                    stream
                        .clone_dtoh(&workspace.selected_ids)
                        .unwrap()
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                ),
            ),
            (
                "weights".to_owned(),
                host(
                    vec![8],
                    stream.clone_dtoh(&workspace.selected_weights).unwrap(),
                ),
            ),
            (
                "gate-up-fp32".to_owned(),
                host(
                    vec![8, 2 * d.intermediate_size],
                    stream.clone_dtoh(&workspace.gate_up_fp32).unwrap(),
                ),
            ),
            (
                "gate-up-bf16".to_owned(),
                host(
                    vec![8, 2 * d.intermediate_size],
                    bf(stream.clone_dtoh(&workspace.gate_up_output).unwrap()),
                ),
            ),
            (
                "activated".to_owned(),
                host(
                    vec![8, d.intermediate_size],
                    bf(stream.clone_dtoh(&workspace.activated_output).unwrap()),
                ),
            ),
            (
                "down-fp32".to_owned(),
                host(
                    vec![8, d.hidden_size],
                    stream.clone_dtoh(&workspace.expert_fp32).unwrap(),
                ),
            ),
            (
                "weighted-bf16".to_owned(),
                host(
                    vec![8, d.hidden_size],
                    bf(stream.clone_dtoh(&workspace.expert_output).unwrap()),
                ),
            ),
        ])
    }
}
