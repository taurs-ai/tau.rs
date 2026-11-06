//! Prepared, allocation-free single-token operations with explicit output slots.
//! Construction allocates tensors; run only queues native kernels/cuBLAS on the
//! private capture device's serialized stream. No tensor arithmetic is evaluated
//! while recording or replaying the operation list.
use candle::{
    CpuStorage, CudaStorage, DType, Device, InplaceOp1, Layout, Result, Tensor,
};
use std::rc::Rc;
#[cfg(test)]
mod tests;

mod blas;
mod moe;
mod operation;
pub(super) use blas::PreparedBlas;
use operation::{DecodeOp, Kind};

type Operation = Box<dyn Fn() -> Result<()>>;

pub(crate) struct DecodeProgram {
    device: Device,
    operations: Vec<Operation>,
    blas: Option<Rc<PreparedBlas>>,
}

impl DecodeProgram {
    /// Shares one dedicated cuBLAS handle/workspace across prepared projections
    /// and routed experts. All launches are serialized on this program's stream.
    pub(super) fn prepared_blas(&mut self) -> Result<Rc<PreparedBlas>> {
        if self.blas.is_none() {
            self.blas = Some(Rc::new(PreparedBlas::new(&self.device)?));
        }
        Ok(Rc::clone(self.blas.as_ref().unwrap()))
    }

    /// Starts an empty operation list on a private CUDA stream. Event tracking
    /// must be disabled because retained buffers are ordered by that one stream.
    pub fn new(device: &Device) -> Result<Self> {
        if device.as_cuda_device()?.is_event_tracking() {
            candle::bail!("prepared decode requires a private capture device")
        }
        Ok(Self {
            device: device.clone(),
            operations: Vec::new(),
            blas: None,
        })
    }

    /// Queues the prepared operations for warm-up, eager checks or graph capture.
    /// CUDA graph replay executes the recorded nodes without calling this loop.
    pub fn run(&self) -> Result<()> {
        for operation in &self.operations {
            operation()?;
        }
        Ok(())
    }

    /// Appends a launch closure that retains its inputs, output slots and state.
    pub fn push(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    /// Allocates a reusable BF16 output slot while building the program, before
    /// capture begins. Later operations can use the returned tensor as an input.
    pub fn buffer(&self, shape: impl Into<candle::Shape>) -> Result<Tensor> {
        Tensor::zeros(
            Into::<candle::Shape>::into(shape),
            DType::BF16,
            &self.device,
        )
    }

    /// Checks input storage and records a launch into the supplied output slot.
    /// Most inputs are BF16; metadata is U32 and attention scratch is FP32. The
    /// closure retains all tensors so their addresses survive capture and replay.
    fn operation(
        &mut self,
        output: Tensor,
        inputs: &[&Tensor],
        kind: Kind,
    ) -> Result<Tensor> {
        let metadata_index = match kind {
            Kind::Elementwise { kind: 0 | 4, .. } => Some(1),
            #[cfg(test)]
            Kind::Rope { .. } => Some(3),
            Kind::AttentionPrepare { .. } => Some(9),
            Kind::Attention { .. } => Some(4),
            _ => None,
        };
        for (index, input) in inputs.iter().enumerate() {
            let dtype = if Some(index) == metadata_index {
                DType::U32
            } else if (matches!(kind, Kind::Attention { .. }) && index == 3)
                || (matches!(kind, Kind::AttentionPrepare { fp32_rope: 1, .. })
                    && matches!(index, 5 | 6))
            {
                DType::F32
            } else {
                DType::BF16
            };
            if !input.is_contiguous()
                || !input.device().same_device(&self.device)
                || input.dtype() != dtype
                || (Some(index) == metadata_index && input.elem_count() != 2)
            {
                candle::bail!("invalid prepared decode input storage")
            }
        }
        let blas = if matches!(kind, Kind::Linear { .. }) {
            Some(self.prepared_blas()?)
        } else {
            None
        };
        let op = DecodeOp {
            inputs: inputs.iter().map(|t| (*t).clone()).collect(),
            kind,
            blas,
        };
        let target = output.clone();
        self.push(Box::new(move || target.inplace_op1(&op)));
        Ok(output)
    }

    /// Prepares one bias-free BF16 projection into a fixed output slot. Execution
    /// uses the eager path's cuBLAS entrypoint and geometry to preserve its math.
    pub fn linear(
        &mut self,
        input: &Tensor,
        layer: &candle_nn::Linear,
    ) -> Result<Tensor> {
        let weight = layer.weight();
        let (m, k) = weight.dims2()?;
        if layer.bias().is_some()
            || input.elem_count() != k
            || m == 0
            || k == 0
            || m > i32::MAX as usize
            || k > i32::MAX as usize
            || weight.dtype() != DType::BF16
        {
            candle::bail!("unsupported prepared linear contract")
        }
        let output = self.buffer((1, 1, m))?;
        self.operation(
            output,
            &[input, weight],
            Kind::Linear {
                m: m as i32,
                k: k as i32,
            },
        )
    }

    /// Prepares centered RMSNorm, retaining its scale and epsilon. Each row is
    /// normalized in FP32, scaled by `1 + weight`, and stored as BF16.
    pub fn norm(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f64,
    ) -> Result<Tensor> {
        let width = weight.elem_count();
        if width == 0
            || !input.elem_count().is_multiple_of(width)
            || weight.dtype() != DType::BF16
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || epsilon > f32::MAX as f64
        {
            candle::bail!("unsupported prepared norm contract")
        }
        let output = self.buffer(input.shape().clone())?;
        self.operation(
            output,
            &[input, weight],
            Kind::Norm {
                rows: i32::try_from(input.elem_count() / width).map_err(error)?,
                width: i32::try_from(width).map_err(error)?,
                epsilon: epsilon as f32,
            },
        )
    }

    /// Prepares SiLU(gate) * up from the two halves of a packed projection,
    /// producing one reusable output with half the input width.
    pub fn silu(&mut self, input: &Tensor) -> Result<Tensor> {
        if input.elem_count() == 0 || !input.elem_count().is_multiple_of(2) {
            candle::bail!("invalid prepared SiLU input width")
        }
        let width = input.elem_count() / 2;
        let output = self.buffer((1, 1, width))?;
        self.operation(
            output,
            &[input],
            Kind::Silu {
                width: i32::try_from(width).map_err(error)?,
            },
        )
    }

    /// Validates and records one small decode operation using the native kind:
    /// 0 embeds a token, 1 adds residuals, 2 extracts packed queries, 3 gates the
    /// attention result, 4 writes the new KV slot, and 5 applies a scalar sigmoid
    /// gate. Metadata supplies changing
    /// token/position values; dimensions and output addresses stay fixed.
    #[allow(clippy::too_many_arguments)]
    pub fn elementwise(
        &mut self,
        kind: i32,
        inputs: &[&Tensor],
        output: Tensor,
        elements: usize,
        width: usize,
        capacity: usize,
        transposed: bool,
    ) -> Result<Tensor> {
        let valid = elements > 0
            && width > 0
            && elements.is_multiple_of(width)
            && match (kind, inputs) {
                (0, [weight, _]) => {
                    weight.dims().len() == 2
                        && weight.dims()[1] == width
                        && elements == width
                        && output.elem_count() == elements
                }
                (1, [a, b]) => {
                    a.elem_count() == elements
                        && b.elem_count() == elements
                        && output.elem_count() == elements
                }
                (2, [packed]) => {
                    packed.elem_count() == 2 * elements
                        && output.elem_count() == elements
                }
                (3 | 6, [context, packed]) => {
                    context.elem_count() == elements
                        && packed.elem_count() == 2 * elements
                        && output.elem_count() == elements
                }
                (4, [input, _]) => {
                    capacity > 0
                        && input.elem_count() == elements
                        && elements.checked_mul(capacity) == Some(output.elem_count())
                }
                (5, [input, gate]) => {
                    input.elem_count() == elements
                        && gate.elem_count() == 1
                        && output.elem_count() == elements
                }
                (7, [input, gate]) => {
                    input.elem_count() == elements
                        && gate.elem_count() == elements
                        && output.elem_count() == elements
                }
                _ => false,
            };
        if !valid {
            candle::bail!("invalid prepared elementwise buffer contract")
        }
        self.operation(
            output,
            inputs,
            Kind::Elementwise {
                kind,
                n: i32::try_from(elements).map_err(error)?,
                width: i32::try_from(width).map_err(error)?,
                capacity: i32::try_from(capacity).map_err(error)?,
                transposed: i32::from(transposed),
            },
        )
    }

    /// Prepares a residual sum in a separate slot, preserving both input tensors.
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            candle::bail!("prepared residual shape mismatch")
        }
        let output = self.buffer(a.shape().clone())?;
        self.elementwise(1, &[a, b], output, a.elem_count(), a.elem_count(), 0, false)
    }

    /// Applies one learned sigmoid gate to a token's vector. Exponential, sum,
    /// reciprocal and product each preserve ordinary BF16 rounding boundaries.
    #[cfg(test)]
    pub fn scalar_gate(&mut self, input: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let output = self.buffer(input.shape().clone())?;
        self.elementwise(
            5,
            &[input, gate],
            output,
            input.elem_count(),
            input.elem_count(),
            0,
            false,
        )
    }

    /// Prepares partial RoPE using retained sine/cosine tables. Execution reads
    /// the current position from metadata, rotates the leading head dimensions,
    /// and copies the remaining dimensions without recapturing for each token.
    #[cfg(test)]
    pub fn rope(
        &mut self,
        input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        metadata: &Tensor,
        dim: usize,
        rotary: usize,
    ) -> Result<Tensor> {
        if dim == 0
            || rotary == 0
            || rotary > dim
            || !rotary.is_multiple_of(2)
            || !input.elem_count().is_multiple_of(dim)
            || cos.dims() != sin.dims()
            || cos.dims().len() != 2
            || cos.dims()[1] != rotary / 2
        {
            candle::bail!("invalid prepared RoPE buffer contract")
        }
        let output = self.buffer(input.shape().clone())?;
        self.operation(
            output,
            &[input, cos, sin, metadata],
            Kind::Rope {
                n: i32::try_from(input.elem_count()).map_err(error)?,
                dim: i32::try_from(dim).map_err(error)?,
                rotary: i32::try_from(rotary).map_err(error)?,
            },
        )
    }

    /// Fuses Q extraction, Q/K norm, partial RoPE and KV append into one launch.
    /// All outputs are retained before capture; metadata supplies the live position.
    /// Cache buffers remain exclusively owned and serialized by the model.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prepare(
        &mut self,
        query_gate: &Tensor,
        key: &Tensor,
        value: &Tensor,
        query_norm: (&Tensor, f64),
        key_norm: (&Tensor, f64),
        rotary: (&Tensor, &Tensor, usize),
        cache: (&Tensor, &Tensor),
        metadata: &Tensor,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        capacity: usize,
        transposed: bool,
    ) -> Result<Tensor> {
        let (query_weight, query_epsilon) = query_norm;
        let (key_weight, key_epsilon) = key_norm;
        let (cos, sin, rotary_dim) = rotary;
        let (cached_key, cached_value) = cache;
        let query_width = heads
            .checked_mul(dim)
            .ok_or_else(|| error("Q size overflow"))?;
        let kv_width = kv_heads
            .checked_mul(dim)
            .ok_or_else(|| error("KV size overflow"))?;
        if heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || dim == 0
            || dim > 256
            || capacity == 0
            || capacity > 8192
            || query_width.checked_mul(2) != Some(query_gate.elem_count())
            || kv_width != key.elem_count()
            || kv_width != value.elem_count()
            || query_weight.elem_count() != dim
            || key_weight.elem_count() != dim
            || kv_width.checked_mul(capacity) != Some(cached_key.elem_count())
            || cached_key.elem_count() != cached_value.elem_count()
            || rotary_dim == 0
            || rotary_dim > dim
            || !rotary_dim.is_multiple_of(2)
            || cos.dims() != sin.dims()
            || cos.dims().len() != 2
            || cos.dims()[0] == 0
            || cos.dims()[1] != rotary_dim / 2
            || [query_epsilon, key_epsilon].iter().any(|e| {
                !e.is_finite()
                    || *e <= 0.0
                    || *e > f64::from(f32::MAX)
                    || (*e as f32) == 0.0
            })
        {
            candle::bail!("invalid prepared attention preparation contract")
        }
        let output = self.buffer((1, 1, query_width))?;
        self.operation(
            output,
            &[
                query_gate,
                key,
                value,
                query_weight,
                key_weight,
                cos,
                sin,
                cached_key,
                cached_value,
                metadata,
            ],
            Kind::AttentionPrepare {
                heads: i32::try_from(heads).map_err(error)?,
                kv_heads: i32::try_from(kv_heads).map_err(error)?,
                dim: dim as i32,
                capacity: capacity as i32,
                rotary: rotary_dim as i32,
                transposed: i32::from(transposed),
                fp32_rope: i32::from(cos.dtype() == DType::F32),
                query_epsilon: query_epsilon as f32,
                key_epsilon: key_epsilon as f32,
            },
        )
    }

    /// Prepares grouped-query attention over retained KV without expanding heads.
    /// Fixed FP32 scratch holds each selected tile's maximum, softmax sum and
    /// context; a combine kernel merges tiles when capacity exceeds one tile.
    /// Metadata limits reads to the live prefix even when storage is larger.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        metadata: &Tensor,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        capacity: usize,
        transposed: bool,
        block_size: crate::AttentionBlockSize,
        bf16_products: bool,
    ) -> Result<Tensor> {
        if heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || dim == 0
            || dim > 256
            || capacity == 0
            || capacity > 8192
            || heads.checked_mul(dim) != Some(query.elem_count())
            || kv_heads
                .checked_mul(dim)
                .and_then(|n| n.checked_mul(capacity))
                != Some(key.elem_count())
            || key.elem_count() != value.elem_count()
        {
            candle::bail!("unsupported prepared attention dimensions")
        }
        let output = self.buffer((1, 1, heads * dim))?;
        let workspace = Tensor::zeros(
            (heads, capacity.div_ceil(block_size.tokens()), dim + 2),
            DType::F32,
            &self.device,
        )?;
        self.operation(
            output,
            &[query, key, value, &workspace, metadata],
            Kind::Attention {
                heads: heads as i32,
                kv_heads: kv_heads as i32,
                dim: dim as i32,
                capacity: capacity as i32,
                transposed: i32::from(transposed),
                block_size: block_size.tokens() as i32,
                bf16_products: i32::from(bf16_products),
            },
        )
    }
}

/// Converts dimension/launch errors into the tensor runtime's error type.
fn error(e: impl std::fmt::Display) -> candle::Error {
    candle::Error::Msg(e.to_string())
}

/// Updates `[token_id, position]` in the retained device allocation before replay.
/// The copy uses the same stream as the graph, preserving buffer addresses and
/// ordering the new values before kernels read them.
pub(crate) fn write_metadata(target: &Tensor, values: [u32; 2]) -> Result<()> {
    struct Write([u32; 2]);
    impl InplaceOp1 for Write {
        /// Identifies the in-place metadata update in tensor error diagnostics.
        fn name(&self) -> &'static str {
            "tau-decode-metadata"
        }
        /// Rejects CPU storage because these metadata belong to a CUDA graph.
        fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
            candle::bail!("CUDA-only")
        }
        /// Checks the two-value slot and copies the host values into its existing
        /// U32 storage, rather than allocating a replacement token tensor.
        fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
            if layout.shape().dims() != [2] || layout.start_offset() != 0 {
                candle::bail!("metadata shape mismatch")
            }
            storage
                .device
                .clone()
                .memcpy_htod(&self.0, storage.as_cuda_slice_mut::<u32>()?)
        }
    }
    target.inplace_op1(&Write(values))
}

#[cfg(test)]
mod precision_tests;
