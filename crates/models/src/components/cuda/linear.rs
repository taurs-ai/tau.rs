//! Bias-free BF16 projections with FP32 output and accumulation.
//!
//! Returning FP32 prevents an intermediate BF16 rounding before a caller applies
//! row weights. The caller chooses the final storage dtype; ordinary dense MLPs
//! keep their existing projection path and cuBLAS handle settings.

use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor,
};
use half::bf16;

struct LinearFp32;

impl CustomOp2 for LinearFp32 {
    /// Identifies errors at the precision-aware projection boundary.
    fn name(&self) -> &'static str {
        "tau-bf16-linear-fp32"
    }

    /// Keeps CPU reference arithmetic with its existing tensor implementation.
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("FP32-output BF16 projection requires CUDA")
    }

    /// Queues a tensor-core GEMM into FP32 storage without changing global precision
    /// or handle math modes. Both input views remain guarded through enqueue.
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, k) = il.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if rows == 0
            || k == 0
            || n == 0
            || wk != k
            || input.device.id() != weight.device.id()
        {
            candle::bail!("invalid FP32-output projection geometry/device")
        }
        let integer = |value| i32::try_from(value).map_err(candle::Error::wrap);
        let (m, width, inner) = (integer(rows)?, integer(n)?, integer(k)?);
        let count = rows
            .checked_mul(n)
            .ok_or_else(|| candle::Error::Msg("projection size overflow".to_owned()))?;
        let (a, ae) = il.contiguous_offsets().ok_or_else(|| {
            candle::Error::Msg("projection input must be contiguous".to_owned())
        })?;
        let (b, be) = wl.contiguous_offsets().ok_or_else(|| {
            candle::Error::Msg("projection weights must be contiguous".to_owned())
        })?;
        let x = input.as_cuda_slice::<bf16>()?.slice(a..ae);
        let w = weight.as_cuda_slice::<bf16>()?.slice(b..be);
        let device = input.device.clone();
        let stream = device.cuda_stream();
        // SAFETY: beta=0 writes every output element before storage is returned.
        let mut output = unsafe { device.alloc::<f32>(count)? };
        let (xp, x_guard) = x.device_ptr(&stream);
        let (wp, w_guard) = w.device_ptr(&stream);
        let (out, out_guard) = output.device_ptr_mut(&stream);
        // SAFETY: Shapes, storage types, offsets and integer bounds are checked
        // above. Owners/guards survive enqueue on this allocation's CUDA stream.
        unsafe {
            tau_cuda_kernels::launch_linear_bf16_fp32(
                xp as *const _,
                wp as *const _,
                out as *mut f32,
                m,
                width,
                inner,
                stream.cu_stream().cast(),
            )
            .map_err(candle::Error::wrap)?;
        }
        drop((out_guard, w_guard, x_guard));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from((rows, n)),
        ))
    }
}

/// Projects contiguous BF16 token rows while preserving FP32 accumulation/output.
pub(in crate::components) fn linear_bf16_fp32(
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    if !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.is_contiguous()
        || !weight.is_contiguous()
    {
        candle::bail!("FP32-output projection requires contiguous CUDA/BF16 tensors")
    }
    input.apply_op2_no_bwd(weight, &LinearFp32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Checks partial MMA tiles and nonzero tensor-view offsets against an
    /// independent FP64 dot product. Dyadic values make the FP32 answer exact.
    #[test]
    fn fp32_projection_matches_exact_dots_with_masks_and_offsets() {
        let device = Device::new_cuda(0).unwrap();
        for (rows, columns, inner) in
            [(1, 1, 7), (3, 11, 16), (17, 19, 33), (5, 32, 64)]
        {
            let data = |count, seed| {
                (0..count)
                    .map(|i| (((i * 7 + seed) % 15) as f32 - 7.) / 16.)
                    .collect::<Vec<_>>()
            };
            let x = data(rows * inner, 3);
            let w = data(columns * inner, 5);
            let input = Tensor::from_vec(
                [vec![0.; inner], x.clone()].concat(),
                (rows + 1, inner),
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .narrow(0, 1, rows)
            .unwrap();
            let weight = Tensor::from_vec(
                [vec![0.; inner], w.clone()].concat(),
                (columns + 1, inner),
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .narrow(0, 1, columns)
            .unwrap();
            let got = linear_bf16_fp32(&input, &weight)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for row in 0..rows {
                for column in 0..columns {
                    let expected = (0..inner)
                        .map(|k| {
                            f64::from(x[row * inner + k])
                                * f64::from(w[column * inner + k])
                        })
                        .sum::<f64>() as f32;
                    assert_eq!(
                        got[row * columns + column].to_bits(),
                        expected.to_bits(),
                        "{rows}/{columns}/{inner}: {row}/{column}"
                    );
                }
            }
        }
    }

    /// Weighting must precede BF16 storage, including a dot exactly halfway
    /// between BF16 values. Early output/weight rounding gives a different result.
    #[test]
    fn fp32_projection_retains_bits_until_weight_application() {
        let device = Device::new_cuda(0).unwrap();
        let x = Tensor::new(&[[1f32, 1.]], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::new(&[[1f32, 1. / 256.]], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let raw = linear_bf16_fp32(&x, &w).unwrap();
        assert_eq!(raw.to_vec2::<f32>().unwrap(), vec![vec![1. + 1. / 256.]]);
        let scale = Tensor::new(&[[0.9f32]], &device).unwrap();
        let actual = raw
            .broadcast_mul(&scale)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        let expected = bf16::from_f32((1. + 1. / 256.) * 0.9).to_f32();
        let early = bf16::from_f32(
            bf16::from_f32(1. + 1. / 256.).to_f32() * bf16::from_f32(0.9).to_f32(),
        )
        .to_f32();
        assert_eq!(actual, expected);
        assert_ne!(actual, early);
    }
}
