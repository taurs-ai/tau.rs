//! Fused grouped-head preparation from strided convolution output.

use super::cuda_driver_error;
use super::recurrence::prefill_packed_cuda;
use candle::Result as CandleResult;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Shape, Tensor};
use core::ffi::c_void;
use half::bf16;

fn storage_end(layout: &Layout) -> Option<usize> {
    layout
        .dims()
        .iter()
        .zip(layout.stride())
        .try_fold(layout.start_offset().checked_add(1)?, |end, (&n, &s)| {
            end.checked_add(n.checked_sub(1)?.checked_mul(s)?)
        })
        .filter(|&n| i64::try_from(n).is_ok())
}

pub(in crate::components) fn supports_mixed_prefill(
    mixed: &Tensor,
    key_heads: usize,
    value_heads: usize,
) -> bool {
    mixed.device().is_cuda()
        && mixed.dtype() == DType::BF16
        && key_heads > 0
        && value_heads > 0
        && value_heads.is_multiple_of(key_heads)
        && mixed.dims3().is_ok_and(|(b, t, c)| {
            b == 1
                && t > 1
                && key_heads
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(value_heads))
                    .and_then(|n| n.checked_mul(128))
                    == Some(c)
                && t.checked_mul(value_heads)
                    .and_then(|n| n.checked_mul(386))
                    .is_some_and(|n| i32::try_from(n).is_ok())
        })
        && storage_end(mixed.layout()).is_some()
}

pub(in crate::components) fn mixed_prefill_cuda(
    mixed: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
    key_heads: usize,
    value_heads: usize,
    round_normalized: bool,
) -> CandleResult<(Tensor, Tensor)> {
    if !supports_mixed_prefill(mixed, key_heads, value_heads) {
        candle::bail!("unsupported GDN mixed prefill contract")
    }
    let packed = mixed.apply_op3_no_bwd(
        alpha,
        beta,
        &Prepare {
            key_heads,
            value_heads,
            round_normalized,
        },
    )?;
    prefill_packed_cuda(&packed, initial)
}

struct Prepare {
    key_heads: usize,
    value_heads: usize,
    round_normalized: bool,
}

impl CustomOp3 for Prepare {
    fn name(&self) -> &'static str {
        "tau-gdn-prefill-prepare"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("GDN native preparation requires CUDA")
    }
    fn cuda_fwd(
        &self,
        mixed: &CudaStorage,
        ml: &Layout,
        alpha: &CudaStorage,
        al: &Layout,
        beta: &CudaStorage,
        bl: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let (batch, tokens, channels) = ml.shape().dims3()?;
        let heads = self.value_heads;
        let count = tokens
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(386))
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or_else(|| candle::Error::Msg("GDN preparation overflow".to_owned()))?;
        if batch != 1
            || tokens <= 1
            || self.key_heads == 0
            || heads == 0
            || !heads.is_multiple_of(self.key_heads)
            || channels != (2 * self.key_heads + heads) * 128
            || al.dims() != [1, tokens, heads]
            || bl.dims() != al.dims()
            || mixed.device.id() != alpha.device.id()
            || mixed.device.id() != beta.device.id()
        {
            candle::bail!("invalid GDN preparation layout/device")
        }
        let m = mixed.as_cuda_slice::<bf16>()?;
        let a = alpha.as_cuda_slice::<f32>()?;
        let b = beta.as_cuda_slice::<bf16>()?;
        for (layout, size) in [(ml, m.len()), (al, a.len()), (bl, b.len())] {
            if storage_end(layout).is_none_or(|n| n > size) {
                candle::bail!("GDN preparation view exceeds storage")
            }
        }
        if !al.is_contiguous() || !bl.is_contiguous() {
            candle::bail!("GDN preparation requires packed gates")
        }
        let m = m.slice(ml.start_offset()..);
        let a = a.slice(al.start_offset()..);
        let b = b.slice(bl.start_offset()..);
        let device = mixed.device.clone();
        // SAFETY: Preparation initializes all packed elements.
        let mut output = unsafe { device.alloc::<f32>(count)? };
        let stream = device.cuda_stream();
        let (mp, mg) = m.device_ptr(&stream);
        let (ap, ag) = a.device_ptr(&stream);
        let (bp, bg) = b.device_ptr(&stream);
        let (op, og) = output.device_ptr_mut(&stream);
        // SAFETY: Validated typed storage, views, dimensions and devices. Fresh
        // output and all pointer guards remain live through asynchronous enqueue.
        unsafe {
            tau_cuda_kernels::launch_gdn_prefill_prepare(
                mp as *const c_void,
                ap as *const f32,
                bp as *const c_void,
                op as *mut f32,
                tokens as i32,
                self.key_heads as i32,
                heads as i32,
                ml.stride()[1] as i64,
                ml.stride()[2] as i64,
                i32::from(self.round_normalized),
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(cuda_driver_error)?;
        drop((mg, ag, bg, og));
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from((1, tokens, heads, 386)),
        ))
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;
    use candle::Device;

    /// Inspect actual packed Q/K: the MoE preset must round normalization to
    /// BF16 before query scaling, while the dense default retains FP32 values.
    #[test]
    fn moe_prefill_normalization_retains_bf16_storage_boundary() {
        let device = Device::new_cuda(0).unwrap();
        let data = (0..2 * 384)
            .map(|i| bf16::from_f32(((i * 17 % 97) as f32 - 48.) / 32.).to_f32())
            .collect::<Vec<_>>();
        let mixed = Tensor::from_vec(data.clone(), (1, 2, 384), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let alpha = Tensor::ones((1, 2, 1), DType::F32, &device).unwrap();
        let beta = Tensor::ones((1, 2, 1), DType::BF16, &device).unwrap();
        let packed = |rounded| {
            mixed
                .apply_op3_no_bwd(
                    &alpha,
                    &beta,
                    &Prepare {
                        key_heads: 1,
                        value_heads: 1,
                        round_normalized: rounded,
                    },
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (dense, moe) = (packed(false), packed(true));
        assert_ne!(dense, moe);
        for token in 0..2 {
            for feature in 0..256 {
                let vector_start = token * 384 + (feature / 128) * 128;
                let sum = data[vector_start..vector_start + 128]
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>();
                let normalized = (f64::from(data[token * 384 + feature])
                    / (sum + 1e-6).sqrt()) as f32;
                let scale = if feature < 128 {
                    (1.0 / 128f64.sqrt()) as f32
                } else {
                    1.
                };
                let expected = bf16::from_f32(normalized).to_f32() * scale;
                assert_eq!(moe[token * 386 + feature], expected);
            }
        }
    }
}
