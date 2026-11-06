//! Stateful single-token GDN decode: parameter validation, retained workspace,
//! state import/export and native convolution/recurrence/normalization launches.
//! Keeping the runner and its launch together makes pointer ownership explicit.
//!
//! Projected inputs, convolution weights/history, `dt_bias`, convolved Q/K/V and
//! output use BF16. `negative_a = -exp(A_log)`, normalization weights and recurrent
//! state use FP32. Recurrence and gated normalization compute in FP32, retaining
//! BF16 rounding before normalization and at output.
//!
//! Candle performs the packed input projections and output projection outside
//! this runner. Its binary custom operator launches the native core on Candle's
//! active stream. Production 128x128 heads use tiled recurrence and a separate
//! normalization pass; other head sizes keep the original fused kernel.

use super::cuda_driver_error;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle::op::BackpropOp;
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape,
    Storage, Tensor,
};
use core::ffi::c_void;
use half::bf16;
use std::sync::{Arc, Mutex};

mod state;

const MAX_HEAD_DIMENSION: usize = 256;

#[cfg(test)]
mod recurrence_tests;

/// Static dimensions accepted by one native Gated DeltaNet runner.
#[derive(Debug, Clone, Copy)]
struct GatedDeltaNetDimensions {
    hidden_size: usize,
    conv_kernel_size: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    value_width: usize,
    conv_width: usize,
    qkvz_width: usize,
}

/// BF16 convolution buffers and FP32 recurrent state retained across decode calls.
#[derive(Debug)]
struct GatedDeltaNetWorkspace {
    conv_state: CudaSlice<bf16>,
    recurrent_state: CudaSlice<f32>,
    convolved_qkv: CudaSlice<bf16>,
    // Every tiled recurrence block writes disjoint columns here. Retaining this
    // buffer makes the following full-head norm safe across graph replays.
    recurrence_output: Option<CudaSlice<bf16>>,
}

/// Cloneable layer-owned state and immutable parameters for native decode.
#[derive(Debug, Clone)]
pub(in crate::components) struct GatedDeltaNetRunner {
    conv_weight: Tensor,
    dt_bias: Tensor,
    negative_a: Tensor,
    norm_weight: Tensor,
    dimensions: GatedDeltaNetDimensions,
    epsilon: f32,
    reference_reduction: bool,
    workspace: Arc<Mutex<GatedDeltaNetWorkspace>>,
}

impl GatedDeltaNetRunner {
    /// Selects the MoE compatibility reduction without changing dense defaults.
    pub(in crate::components) fn set_reference_reduction(&mut self, enabled: bool) {
        self.reference_reduction = enabled;
    }

    /// Constructs a runner when parameters satisfy the BF16 CUDA decode contract.
    ///
    /// Requires BF16 `conv_weight`/`dt_bias` and FP32 `negative_a`/`norm_weight`.
    /// Checks fixed dimensions, packed parameter layouts, devices and integer
    /// limits before allocating the reusable state. Unsupported parameters return
    /// `Ok(None)` so the model can use Candle; allocation failures remain errors.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::components) fn new(
        hidden_size: usize,
        conv_kernel_size: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        epsilon: f64,
        conv_weight: Tensor,
        dt_bias: Tensor,
        negative_a: Tensor,
        norm_weight: Tensor,
    ) -> CandleResult<Option<Self>> {
        let Some(key_width) = num_key_heads.checked_mul(key_head_dim) else {
            return Ok(None);
        };
        let Some(value_width) = num_value_heads.checked_mul(value_head_dim) else {
            return Ok(None);
        };
        let Some(conv_width) = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
        else {
            return Ok(None);
        };
        let Some(qkvz_width) = conv_width.checked_add(value_width) else {
            return Ok(None);
        };
        let same_device = conv_weight.device().same_device(dt_bias.device())
            && conv_weight.device().same_device(negative_a.device())
            && conv_weight.device().same_device(norm_weight.device());
        let supported = hidden_size > 0
            && conv_kernel_size > 0
            && key_head_dim > 0
            && value_head_dim > 0
            && key_head_dim <= MAX_HEAD_DIMENSION
            && value_head_dim <= MAX_HEAD_DIMENSION
            && num_key_heads > 0
            && num_value_heads > 0
            && num_value_heads.is_multiple_of(num_key_heads)
            && epsilon.is_finite()
            && epsilon > 0.0
            && epsilon <= f32::MAX as f64
            && conv_weight.device().is_cuda()
            && same_device
            && conv_weight.dtype() == DType::BF16
            && dt_bias.dtype() == DType::BF16
            && negative_a.dtype() == DType::F32
            && norm_weight.dtype() == DType::F32
            && conv_weight.is_contiguous()
            && dt_bias.is_contiguous()
            && negative_a.is_contiguous()
            && norm_weight.is_contiguous()
            && conv_weight.dims() == [conv_width, 1, conv_kernel_size]
            && dt_bias.dims() == [num_value_heads]
            && negative_a.dims() == [num_value_heads]
            && norm_weight.dims() == [value_head_dim]
            && [
                hidden_size,
                conv_kernel_size,
                key_head_dim,
                value_head_dim,
                num_key_heads,
                num_value_heads,
                key_width,
                value_width,
                conv_width,
                qkvz_width,
            ]
            .into_iter()
            .all(|dimension| i32::try_from(dimension).is_ok());
        if !supported {
            return Ok(None);
        }

        let dimensions = GatedDeltaNetDimensions {
            hidden_size,
            conv_kernel_size,
            key_head_dim,
            value_head_dim,
            num_key_heads,
            num_value_heads,
            value_width,
            conv_width,
            qkvz_width,
        };
        let workspace = Self::allocate_workspace(&conv_weight, dimensions)?;
        Ok(Some(Self {
            conv_weight,
            dt_bias,
            negative_a,
            norm_weight,
            dimensions,
            epsilon: epsilon as f32,
            reference_reduction: false,
            workspace: Arc::new(Mutex::new(workspace)),
        }))
    }

    /// Checks whether a hidden-state tensor selects the native decode contract.
    ///
    /// Decode accepts one contiguous BF16 token from one sequence, with the
    /// configured hidden width on the runner's CUDA device. Fixed weights were
    /// already checked by `new`; this checks the input that changes on each call.
    pub(in crate::components) fn supports_input(&self, xs: &Tensor) -> bool {
        xs.device().same_device(self.conv_weight.device())
            && xs.dtype() == DType::BF16
            && xs.is_contiguous()
            && xs.dims() == [1, 1, self.dimensions.hidden_size]
    }

    /// Applies the native recurrent core through Candle's binary operator boundary.
    ///
    /// `qkvz` and `ba` are already-projected one-token tensors. The adapter borrows
    /// this runner, advances its retained state, and returns the normalized BF16
    /// value tensor for the outer GDN's output projection. Seed state before the
    /// first call of a sequence or after handing off from prefill.
    pub(in crate::components) fn forward(
        &self,
        qkvz: &Tensor,
        ba: &Tensor,
    ) -> CandleResult<Tensor> {
        qkvz.apply_op2_no_bwd(ba, &GatedDeltaNetDecodeCuda { runner: self })
    }

    /// Acquires every parameter and state pointer and enqueues the native kernels.
    ///
    /// This is the decode adapter's implementation: validate projected inputs,
    /// borrow packed weights, lock reusable workspace and allocate this token's
    /// output. Enqueue convolution, recurrence and gated normalization on Candle's
    /// stream, then return output storage while preserving state for the next call.
    fn launch(
        &self,
        qkvz: &CudaStorage,
        ql: &Layout,
        ba: &CudaStorage,
        bl: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        let device = qkvz.device.clone();
        // SAFETY: launch_into initializes every output element.
        let mut output = unsafe { device.alloc::<bf16>(self.dimensions.value_width)? };
        self.launch_into(qkvz, ql, ba, bl, &mut output)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from((1, 1, self.dimensions.value_width)),
        ))
    }

    /// Advances the native GDN state and writes this token into a retained output.
    /// The in-place adapter shares the ordinary decode launcher, so prepared
    /// graphs change output ownership without changing recurrence arithmetic.
    pub(in crate::components) fn forward_into(
        &self,
        qkvz: &Tensor,
        ba: &Tensor,
        output: &Tensor,
    ) -> CandleResult<()> {
        output.inplace_op3(qkvz, ba, &DecodeInto(self))
    }

    /// Validates projected inputs, borrows parameters and locks the runner's
    /// retained workspace before queuing native decode into the supplied buffer.
    /// Convolution/recurrent state advances in place on the same CUDA stream;
    /// this shared launcher allocates no output and does not synchronize the CPU.
    fn launch_into(
        &self,
        qkvz_storage: &CudaStorage,
        qkvz_layout: &Layout,
        ba_storage: &CudaStorage,
        ba_layout: &Layout,
        output: &mut CudaSlice<bf16>,
    ) -> CandleResult<()> {
        let dimensions = self.dimensions;
        if qkvz_storage.device.id() != ba_storage.device.id()
            || qkvz_storage.device.id()
                != self.conv_weight.device().as_cuda_device()?.id()
        {
            candle::bail!("Gated DeltaNet decode tensors must use one CUDA device")
        }
        if qkvz_storage.dtype() != DType::BF16 || ba_storage.dtype() != DType::BF16 {
            candle::bail!("Gated DeltaNet projected inputs must use BF16")
        }
        if qkvz_layout.shape().dims() != [1, 1, dimensions.qkvz_width]
            || ba_layout.shape().dims() != [1, 1, 2 * dimensions.num_value_heads]
        {
            candle::bail!(
                "Gated DeltaNet projected input shapes changed after dispatch"
            )
        }
        let qkvz = contiguous_bf16_view(qkvz_storage, qkvz_layout, "QKVZ")?;
        let ba = contiguous_bf16_view(ba_storage, ba_layout, "BA")?;

        let (conv_guard, conv_layout) = self.conv_weight.storage_and_layout();
        let conv_storage = cuda_storage(&conv_guard, "convolution weight")?;
        let conv_weight = contiguous_bf16_view(conv_storage, conv_layout, "weight")?;
        let (dt_guard, dt_layout) = self.dt_bias.storage_and_layout();
        let dt_storage = cuda_storage(&dt_guard, "dt_bias")?;
        let dt_bias = contiguous_bf16_view(dt_storage, dt_layout, "dt_bias")?;
        let (negative_a_guard, negative_a_layout) =
            self.negative_a.storage_and_layout();
        let negative_a_storage = cuda_storage(&negative_a_guard, "negative A")?;
        let (negative_a_start, negative_a_end) =
            negative_a_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet negative A must be contiguous".to_owned(),
                )
            })?;
        let negative_a = negative_a_storage
            .as_cuda_slice::<f32>()?
            .slice(negative_a_start..negative_a_end);
        let (norm_guard, norm_layout) = self.norm_weight.storage_and_layout();
        let norm_storage = cuda_storage(&norm_guard, "normalization weight")?;
        let (norm_start, norm_end) =
            norm_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg(
                    "Gated DeltaNet normalization weight must be contiguous".to_owned(),
                )
            })?;
        let norm_weight = norm_storage
            .as_cuda_slice::<f32>()?
            .slice(norm_start..norm_end);

        let device = qkvz_storage.device.clone();
        let stream = device.cuda_stream();
        let mut workspace = self.lock_workspace()?;
        let GatedDeltaNetWorkspace {
            conv_state,
            recurrent_state,
            convolved_qkv,
            recurrence_output,
        } = &mut *workspace;

        let (qkvz_ptr, qkvz_pointer_guard) = qkvz.device_ptr(&stream);
        let (ba_ptr, ba_pointer_guard) = ba.device_ptr(&stream);
        let (conv_weight_ptr, conv_weight_pointer_guard) =
            conv_weight.device_ptr(&stream);
        let (dt_bias_ptr, dt_bias_pointer_guard) = dt_bias.device_ptr(&stream);
        let (negative_a_ptr, negative_a_pointer_guard) = negative_a.device_ptr(&stream);
        let (norm_weight_ptr, norm_weight_pointer_guard) =
            norm_weight.device_ptr(&stream);
        let (conv_state_ptr, conv_state_guard) = conv_state.device_ptr_mut(&stream);
        let (recurrent_state_ptr, recurrent_state_guard) =
            recurrent_state.device_ptr_mut(&stream);
        let (convolved_qkv_ptr, convolved_qkv_guard) =
            convolved_qkv.device_ptr_mut(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let recurrence_output_pointer = recurrence_output
            .as_mut()
            .map(|buffer| buffer.device_ptr_mut(&stream));
        let recurrence_output_ptr = recurrence_output_pointer
            .as_ref()
            .map_or(0, |(pointer, _guard)| *pointer);

        // Compute one token's Gated DeltaNet core from its already projected
        // BF16 Q/K/V/Z vector and beta/decay projections, the layer parameters,
        // and the state retained from previous tokens. CUDA advances the causal
        // convolution, normalizes Q/K, applies decay and the delta-rule memory
        // update, then applies gated RMSNorm using Z as the SiLU gate.
        // The expected result is a BF16 [1, 1, value_width] tensor in `output`,
        // ready for Candle's output projection. The BF16 convolution history and
        // FP32 recurrent state are updated in place for the next token; scratch
        // buffers hold intermediate convolution and recurrence results.
        // Work is queued on Candle's stream, so these results become available
        // in stream order rather than being synchronized back to the CPU here.
        //
        // SAFETY: Construction and dispatch validate all pointer shapes, dtypes,
        // devices, state lengths, dimensions, and the active Candle stream.
        unsafe {
            tau_cuda_kernels::launch_fused_bf16_gated_delta_net_decode(
                qkvz_ptr as *const c_void,
                ba_ptr as *const c_void,
                conv_weight_ptr as *const c_void,
                dt_bias_ptr as *const c_void,
                negative_a_ptr as *const c_void,
                norm_weight_ptr as *const f32,
                conv_state_ptr as *mut c_void,
                recurrent_state_ptr as *mut c_void,
                convolved_qkv_ptr as *mut c_void,
                recurrence_output_ptr as *mut c_void,
                output_ptr as *mut c_void,
                i32::try_from(dimensions.conv_width).expect("validated conv width"),
                i32::try_from(dimensions.conv_kernel_size)
                    .expect("validated convolution size"),
                i32::try_from(dimensions.num_key_heads).expect("validated key heads"),
                i32::try_from(dimensions.num_value_heads)
                    .expect("validated value heads"),
                i32::try_from(dimensions.key_head_dim)
                    .expect("validated key head dimension"),
                i32::try_from(dimensions.value_head_dim)
                    .expect("validated value head dimension"),
                self.epsilon,
                i32::from(self.reference_reduction),
                stream.cu_stream() as *mut c_void,
            )
        }
        .map_err(|error| {
            candle::Error::Cuda(Box::<dyn std::error::Error + Send + Sync>::from(
                error.to_string(),
            ))
        })?;

        drop(output_guard);
        drop(recurrence_output_pointer);
        drop(convolved_qkv_guard);
        drop(recurrent_state_guard);
        drop(conv_state_guard);
        drop(norm_weight_pointer_guard);
        drop(negative_a_pointer_guard);
        drop(dt_bias_pointer_guard);
        drop(conv_weight_pointer_guard);
        drop(ba_pointer_guard);
        drop(qkvz_pointer_guard);
        drop(workspace);
        drop(norm_guard);
        drop(negative_a_guard);
        drop(dt_guard);
        drop(conv_guard);

        Ok(())
    }

    /// Locks reusable state and turns poisoning into a Candle error.
    /// Runner clones share this workspace. The mutex protects host-side access
    /// while copies or kernels are enqueued; it does not wait for GPU completion.
    /// The caller must still serialize execution and graph replay on the stream.
    fn lock_workspace(
        &self,
    ) -> CandleResult<std::sync::MutexGuard<'_, GatedDeltaNetWorkspace>> {
        self.workspace.lock().map_err(|_| {
            candle::Error::Msg("Gated DeltaNet workspace lock is poisoned".to_owned())
        })
    }
}

/// Binary custom operator carrying one stateful native runner.
#[derive(Debug, Clone, Copy)]
struct GatedDeltaNetDecodeCuda<'a> {
    runner: &'a GatedDeltaNetRunner,
}

impl CustomOp2 for GatedDeltaNetDecodeCuda<'_> {
    /// Returns the operation name used by Candle diagnostics.
    fn name(&self) -> &'static str {
        "tau-fused-bf16-gated-delta-net-decode-cuda"
    }

    /// Rejects CPU execution because state storage is CUDA-only.
    fn cpu_fwd(
        &self,
        _qkvz_storage: &CpuStorage,
        _qkvz_layout: &Layout,
        _ba_storage: &CpuStorage,
        _ba_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        candle::bail!("{} is a CUDA-only operator", self.name())
    }

    /// Launches fused recurrent decode for validated CUDA tensors.
    /// This adapter delegates storage validation, allocation and kernel launches
    /// to the runner, which owns the weights and mutable state shared across calls.
    fn cuda_fwd(
        &self,
        qkvz_storage: &CudaStorage,
        qkvz_layout: &Layout,
        ba_storage: &CudaStorage,
        ba_layout: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        self.runner
            .launch(qkvz_storage, qkvz_layout, ba_storage, ba_layout)
    }
}

/// Returns CUDA storage or a descriptive parameter-location error.
/// Borrows from an existing tensor read lock; it performs no allocation or copy.
/// The label identifies which parameter had unexpected CPU/other backend storage.
fn cuda_storage<'a>(
    storage: &'a std::sync::RwLockReadGuard<'_, Storage>,
    label: &str,
) -> CandleResult<&'a CudaStorage> {
    match &**storage {
        Storage::Cuda(storage) => Ok(storage),
        _ => candle::bail!("Gated DeltaNet {label} must use CUDA storage"),
    }
}

/// Returns one contiguous BF16 device view for a dynamic or parameter tensor.
/// Honors the tensor's starting offset, so a packed slice need not begin at the
/// allocation's base address. Rejects strided layouts and incorrect storage dtype
/// instead of silently materializing a copy inside the native decode launch.
fn contiguous_bf16_view<'a>(
    storage: &'a CudaStorage,
    layout: &Layout,
    label: &str,
) -> CandleResult<candle::cuda_backend::cudarc::driver::CudaView<'a, bf16>> {
    let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
        candle::Error::Msg(format!("Gated DeltaNet {label} must be contiguous"))
    })?;
    Ok(storage.as_cuda_slice::<bf16>()?.slice(start..end))
}

struct DecodeInto<'a>(&'a GatedDeltaNetRunner);
impl candle::InplaceOp3 for DecodeInto<'_> {
    /// Identifies the retained-output GDN adapter in tensor error diagnostics.
    fn name(&self) -> &'static str {
        "tau-gdn-decode-into"
    }
    /// Rejects CPU storage because the retained native runner is CUDA-only.
    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<()> {
        candle::bail!("CUDA-only")
    }
    /// Checks that the output is a complete slot on the inputs' device, then
    /// delegates to the shared stateful launcher using that slot's BF16 storage.
    fn cuda_fwd(
        &self,
        out: &mut CudaStorage,
        ol: &Layout,
        q: &CudaStorage,
        ql: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> CandleResult<()> {
        if ol.start_offset() != 0
            || !ol.is_contiguous()
            || ol.shape().elem_count() != self.0.dimensions.value_width
            || out.device.id() != q.device.id()
        {
            candle::bail!("invalid GDN output slot")
        }
        self.0
            .launch_into(q, ql, b, bl, out.as_cuda_slice_mut::<bf16>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Rejects CPU parameters before allocating native state.
    #[test]
    fn unsupported_cpu_parameters_do_not_create_runner() {
        let device = Device::Cpu;
        let runner = GatedDeltaNetRunner::new(
            16,
            4,
            8,
            8,
            1,
            2,
            1e-6,
            Tensor::zeros((32, 1, 4), DType::BF16, &device).unwrap(),
            Tensor::zeros(2, DType::BF16, &device).unwrap(),
            Tensor::zeros(2, DType::F32, &device).unwrap(),
            Tensor::zeros(8, DType::BF16, &device).unwrap(),
        )
        .unwrap();

        assert!(runner.is_none());
    }

    /// Ensures the FFI cannot reinterpret a BF16 normalization buffer as FP32.
    #[test]
    fn rejects_non_fp32_normalization_before_native_dispatch() {
        let device = Device::new_cuda(0).expect("Gated DeltaNet test requires CUDA");
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let runner = GatedDeltaNetRunner::new(
                16,
                4,
                8,
                8,
                1,
                2,
                1e-6,
                Tensor::zeros((32, 1, 4), DType::BF16, &device).unwrap(),
                Tensor::zeros(2, DType::BF16, &device).unwrap(),
                Tensor::zeros(2, DType::F32, &device).unwrap(),
                Tensor::ones(8, dtype, &device).unwrap(),
            )
            .unwrap();
            assert_eq!(runner.is_some(), dtype == DType::F32);
        }
    }
}
