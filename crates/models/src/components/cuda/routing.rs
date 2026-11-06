//! BF16 router rows to FP32 selected weights, with no global math-mode changes.

use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor,
};
use half::bf16;

struct Route;
impl CustomOp1 for Route {
    /// Identifies normalization failures at the routing boundary.
    fn name(&self) -> &'static str {
        "tau-moe-routing-fp32"
    }
    /// Keeps unsupported/CPU routing on its existing portable implementation.
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("CUDA-only routing")
    }
    /// Allocates the packed output and enqueues the validated row-wise router.
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, experts) = layout.shape().dims2()?;
        if rows == 0 || experts != 256 {
            candle::bail!("unsupported native routing shape")
        }
        let count = rows
            .checked_mul(16)
            .ok_or_else(|| candle::Error::Msg("routing size overflow".to_owned()))?;
        let rows_i32 = i32::try_from(rows).map_err(candle::Error::wrap)?;
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle::Error::Msg("routing requires contiguous rows".to_owned())
        })?;
        let values = input.as_cuda_slice::<bf16>()?.slice(start..end);
        let device = input.device.clone();
        let stream = device.cuda_stream();
        // SAFETY: The kernel initializes every output slot, including nonfinite errors.
        let mut output = unsafe { device.alloc::<f32>(count)? };
        let (p, guard) = values.device_ptr(&stream);
        let (out, out_guard) = output.device_ptr_mut(&stream);
        // SAFETY: Dtype, shape, contiguous offsets, allocation and row bounds are
        // checked; owners/guards remain alive through enqueue on this stream.
        unsafe {
            tau_cuda_kernels::launch_moe_routing_fp32(
                p as *const _,
                out as *mut _,
                rows_i32,
                stream.cu_stream().cast(),
            )
        }
        .map_err(candle::Error::wrap)?;
        drop((out_guard, guard));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from((rows, 16)),
        ))
    }
}

/// Selects a narrow native contract; the component retains fallback policy.
pub(in crate::components) fn moe_routes_fp32(logits: &Tensor) -> Result<Tensor> {
    if !logits.device().is_cuda()
        || logits.dtype() != DType::BF16
        || !logits.is_contiguous()
    {
        candle::bail!("native routing requires contiguous CUDA BF16 logits")
    }
    logits.apply_op1_no_bwd(&Route)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// All ties choose lower IDs, while a nonfinite score marks the complete
    /// row invalid rather than allowing a partial assignment to reach experts.
    #[test]
    fn fp32_router_handles_uniform_ties_and_nonfinite_rows() {
        let device = Device::new_cuda(0).unwrap();
        let mut values = vec![0f32; 3 * 256];
        values[256 + 7] = f32::NAN;
        values[512 + 5] = f32::INFINITY;
        let input = Tensor::from_vec(values, (3, 256), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let output = moe_routes_fp32(&input).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(&output[0][..8], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        assert_eq!(&output[0][8..], &[0.125; 8]);
        assert!(output[1..].iter().flatten().all(|value| value.is_nan()));
    }

    /// Replays actual reference scores independently of embeddings, GDN and
    /// expert weights. The small frozen corpus covers boundary ties and late
    /// layers; an explicit larger corpus supports full diagnostic replay.
    #[test]
    fn fp32_router_matches_frozen_sglang_rows() {
        let device = Device::new_cuda(0).unwrap();
        let values = match std::env::var("TAU_MOE_ROUTER_CORPUS") {
            Ok(path) => candle::safetensors::load(path, &device).unwrap(),
            Err(_) => candle::safetensors::load_buffer(
                include_bytes!("fixtures/moe_router_sm90.safetensors"),
                &device,
            )
            .unwrap(),
        };
        let input = values["scores"].to_dtype(DType::BF16).unwrap();
        let output = moe_routes_fp32(&input).unwrap().to_vec2::<f32>().unwrap();
        let ids = values["ids"].to_vec2::<u32>().unwrap();
        let weights = values["weights"].to_vec2::<f32>().unwrap();
        let scores = values["scores"].to_vec2::<f32>().unwrap();
        assert_eq!(output.len(), ids.len());
        assert_eq!(output.len(), weights.len());
        let mut maximum = 0f32;
        let mut exact_weights = 0;
        let mut native_fp64_error = 0f64;
        let mut reference_fp64_error = 0f64;
        for (row, ((actual, ids), weights)) in
            output.iter().zip(&ids).zip(&weights).enumerate()
        {
            assert_eq!(ids.len(), 8);
            assert_eq!(weights.len(), 8);
            let maximum_score = f64::from(scores[row][ids[0] as usize]);
            let exponentials = ids
                .iter()
                .map(|id| (f64::from(scores[row][*id as usize]) - maximum_score).exp())
                .collect::<Vec<_>>();
            let denominator: f64 = exponentials.iter().sum();
            for rank in 0..8 {
                assert_eq!(actual[rank], ids[rank] as f32, "row {row}, rank {rank}");
                let error = (actual[8 + rank] - weights[rank]).abs();
                assert!(
                    // Keep the frozen prefill contract's existing FP32 weight
                    // floor and relative budget; ID agreement remains exact.
                    error <= 1e-7 + 1e-5 * weights[rank].abs(),
                    "routing weight row {row}, rank {rank}: {error:e}"
                );
                maximum = maximum.max(error);
                let oracle = exponentials[rank] / denominator;
                let fp32_budget = 1e-7 + 1e-5 * oracle.abs();
                assert!((f64::from(actual[8 + rank]) - oracle).abs() <= fp32_budget);
                assert!((f64::from(weights[rank]) - oracle).abs() <= fp32_budget);
                native_fp64_error =
                    native_fp64_error.max((f64::from(actual[8 + rank]) - oracle).abs());
                reference_fp64_error =
                    reference_fp64_error.max((f64::from(weights[rank]) - oracle).abs());
                exact_weights +=
                    usize::from(actual[8 + rank].to_bits() == weights[rank].to_bits());
            }
        }
        eprintln!(
            "SGLANG_ROUTING rows={} ids_exact=true weights_exact={}/{} max_abs={maximum:e} native_vs_fp64={native_fp64_error:e} sglang_vs_fp64={reference_fp64_error:e}",
            output.len(),
            exact_weights,
            output.len() * 8
        );
    }
}
