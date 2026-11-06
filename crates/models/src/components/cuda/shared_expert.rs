//! Precision-aware MoE shared gate/add; shared dense MLP math stays unchanged.

use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Storage, Tensor,
};
use half::bf16;

struct Combine {
    weight: Tensor,
    shared: Tensor,
    routed: Tensor,
}
impl CustomOp1 for Combine {
    /// Identifies errors at the shared-gate output boundary.
    fn name(&self) -> &'static str {
        "tau-moe-shared-combine"
    }
    /// Leaves unsupported CPU configurations on the portable component path.
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("CUDA-only shared combine")
    }
    /// Retains every tensor view/guard until the fused operation is enqueued.
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, width) = layout.shape().dims2()?;
        let rows_i32 = i32::try_from(rows).map_err(candle::Error::wrap)?;
        let width_i32 = i32::try_from(width).map_err(candle::Error::wrap)?;
        let count = rows.checked_mul(width).ok_or_else(|| {
            candle::Error::Msg("shared output size overflow".to_owned())
        })?;
        let device = input.device.clone();
        let stream = device.cuda_stream();
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle::Error::Msg("noncontiguous shared input".to_owned())
        })?;
        let x = input.as_cuda_slice::<bf16>()?.slice(start..end);
        let retained =
            [&self.weight, &self.shared, &self.routed].map(Tensor::storage_and_layout);
        let views = retained
            .iter()
            .map(|(storage, layout)| {
                let Storage::Cuda(storage) = &**storage else {
                    candle::bail!("expected CUDA shared input")
                };
                if storage.device.id() != device.id() {
                    candle::bail!("shared device mismatch")
                }
                let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                    candle::Error::Msg("noncontiguous shared operand".to_owned())
                })?;
                Ok(storage.as_cuda_slice::<bf16>()?.slice(start..end))
            })
            .collect::<Result<Vec<_>>>()?;
        let pointers = views
            .iter()
            .map(|view| view.device_ptr(&stream))
            .collect::<Vec<_>>();
        let (xp, x_guard) = x.device_ptr(&stream);
        // SAFETY: The kernel writes every output element before it is observable.
        let mut output = unsafe { device.alloc::<bf16>(count)? };
        let (out, out_guard) = output.device_ptr_mut(&stream);
        // SAFETY: Wrapper geometry/dtype/device checks and retained guards cover
        // every buffer; all work is ordered on this device's CUDA stream.
        unsafe {
            tau_cuda_kernels::launch_moe_shared_combine(
                xp as *const _,
                pointers[0].0 as *const _,
                pointers[1].0 as *const _,
                pointers[2].0 as *const _,
                out as *mut _,
                rows_i32,
                width_i32,
                stream.cu_stream().cast(),
            )
        }
        .map_err(candle::Error::wrap)?;
        drop((out_guard, x_guard, pointers));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

/// Combines independently rounded expert outputs with the FP32 learned gate.
pub(in crate::components) fn moe_shared_combine(
    input: &Tensor,
    weight: &Tensor,
    shared: &Tensor,
    routed: &Tensor,
) -> Result<Tensor> {
    let (rows, width) = input.dims2()?;
    if rows == 0
        || width == 0
        || width > 8192
        || weight.elem_count() != width
        || shared.shape() != input.shape()
        || routed.shape() != input.shape()
        || [input, weight, shared, routed].iter().any(|tensor| {
            !tensor.is_contiguous()
                || tensor.dtype() != DType::BF16
                || !tensor.device().is_cuda()
                || !tensor.device().same_device(input.device())
        })
    {
        candle::bail!("unsupported shared combine contract")
    }
    input.apply_op1_no_bwd(&Combine {
        weight: weight.clone(),
        shared: shared.clone(),
        routed: routed.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Compare independent scalar-gate/FMA math on masked feature tails and
    /// saturated/zero gates. Inputs are stored BF16 before the reference dot.
    #[test]
    fn shared_fp32_gate_matches_scalar_reference_with_partial_tiles() {
        let device = Device::new_cuda(0).unwrap();
        for width in [2, 17, 2048] {
            let mut input = vec![0f32; 3 * width];
            input[0] = 1.;
            input[1] = 1. / 256.;
            input[2 * width] = -4.;
            let mut weight = vec![0f32; width];
            weight[0] = 1.;
            weight[1] = 1.;
            let shared = (0..3 * width)
                .map(|i| bf16::from_f32(((i % 13) as f32 - 6.) / 16.).to_f32())
                .collect::<Vec<_>>();
            let routed = (0..3 * width)
                .map(|i| bf16::from_f32(((i % 17) as f32 - 8.) / 32.).to_f32())
                .collect::<Vec<_>>();
            let tensor = |values: &[f32], shape: &[usize]| {
                Tensor::from_vec(values.to_vec(), shape, &device)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
            };
            let output = moe_shared_combine(
                &tensor(&input, &[3, width]),
                &tensor(&weight, &[width]),
                &tensor(&shared, &[3, width]),
                &tensor(&routed, &[3, width]),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            for row in 0..3 {
                let dot = (0..width)
                    .map(|i| f64::from(input[row * width + i]) * f64::from(weight[i]))
                    .sum::<f64>();
                let gate = (1. / (1. + (-dot).exp())) as f32;
                for column in 0..width {
                    let index = row * width + column;
                    let expected =
                        bf16::from_f32(gate.mul_add(shared[index], routed[index]))
                            .to_f32();
                    assert_eq!(output[index], expected, "{width}: {row}/{column}");
                }
            }
        }
    }
}
