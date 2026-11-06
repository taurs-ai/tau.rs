//! Retained key/value storage for gated attention, including request limits and reuse.

use super::GROUPED_DECODE_MIN_CONTEXT;
use candle::{DType, Device, Result as CandleResult, Tensor};

/// Capacity is physical storage; `len` is the initialized prefix visible to attention.
/// V is contiguous [batch, KV heads, capacity, head dimension]. Optimized BF16
/// layer/eager decode may store K as [batch, KV heads, head dimension, capacity]
/// to consume its transpose without rebuilding history. Prepared full decode
/// prefers K in V's layout for coalesced reads. `prefix` always exposes logical
/// [batch, KV heads, length, dim] views regardless of the selected physical layout.
#[derive(Debug)]
pub(super) struct KvCache {
    pub(super) buffers: Option<(Tensor, Tensor)>,
    pub(super) transposed_keys: bool,
    pub(super) prefer_transposed_keys: bool,
    pub(super) len: usize,
    pub(super) request_limit: Option<usize>,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) max_len: usize,
    pub(super) dtype: DType,
    pub(super) device: Device,
}

impl KvCache {
    /// Returns the number of token slots in the allocated buffers, or zero if absent.
    /// This includes unused slots; `len` counts only tokens available to attention.
    pub(super) fn capacity(&self) -> usize {
        self.buffers
            .as_ref()
            .map_or(0, |(_, value)| value.dims()[2])
    }

    /// Prepares room for one request's prompt and output tokens while the cache is
    /// empty.
    /// Reuses large enough buffers, chooses the key layout, and sets the request's
    /// token limit even if the buffers have more space. Serving uses batch size one.
    /// This runs outside CUDA graphs; never resize storage inside future capture.
    pub(super) fn reserve(&mut self, capacity: usize) -> CandleResult<()> {
        self.reserve_decode(capacity, false)
    }

    /// Chooses physical storage for a request. Prepared graphs round to a power
    /// of two for reuse, capped by the configured budget; ordinary reservations
    /// use the requested size. This calculation does not change logical limits.
    pub(super) fn reservation_capacity(&self, capacity: usize, graph: bool) -> usize {
        if graph {
            capacity
                .checked_next_power_of_two()
                .unwrap_or(self.max_len)
                .min(self.max_len)
        } else {
            capacity
        }
    }

    /// Graph workspace buckets stabilize addresses across small request-length
    /// differences. The visible request limit stays exact and the allocation
    /// remains bounded by the already configured model/VRAM capacity limit.
    /// Choose the K layout only while the cache is empty, reusing large enough
    /// buffers; the graph owner must invalidate captured views before this call.
    pub(super) fn reserve_decode(
        &mut self,
        capacity: usize,
        graph: bool,
    ) -> CandleResult<()> {
        if self.len != 0 || capacity == 0 || capacity > self.max_len {
            candle::bail!("KV reservation requires an empty cache and a valid capacity")
        }
        // Empty storage can be reinterpreted without reallocating or copying it.
        let allocation = self.reservation_capacity(capacity, graph);
        self.transposed_keys = self.prefer_transposed_keys
            && !graph
            && allocation >= GROUPED_DECODE_MIN_CONTEXT;
        self.ensure_capacity(1, allocation, true)?;
        // Limit visibility to this request's budget even if retained storage is larger.
        self.request_limit = Some(capacity);
        Ok(())
    }

    /// Checks that new tokens start at the current end and fit within the token limit.
    /// Rejects empty input and batch changes within a request. Returns the new length
    /// without changing the cache or allocating storage.
    pub(super) fn validate_append(
        &self,
        batch: usize,
        tokens: usize,
        offset: usize,
    ) -> CandleResult<usize> {
        if offset != self.len || batch == 0 || tokens == 0 {
            candle::bail!(
                "KV append requires offset == cache length and nonempty input"
            )
        }
        let end = offset
            .checked_add(tokens)
            .ok_or_else(|| candle::Error::Msg("KV cache length overflow".to_owned()))?;
        if end > self.request_limit.unwrap_or(self.max_len) {
            candle::bail!(
                "KV append exceeds reserved request or model context capacity"
            )
        }
        if let Some((key, _)) = &self.buffers
            && key.dims()[0] != batch
            && (self.len != 0 || self.request_limit.is_some())
        {
            candle::bail!("KV cache batch size changed within a sequence")
        }
        Ok(end)
    }

    /// Reuses compatible buffers or allocates enough space for `needed` tokens.
    /// New allocations use exactly `needed` slots when `exact` is true; otherwise
    /// they round up to a power of two, capped by the token limit. Existing tokens
    /// are copied into replacement buffers so an active sequence can continue.
    fn ensure_capacity(
        &mut self,
        batch: usize,
        needed: usize,
        exact: bool,
    ) -> CandleResult<()> {
        let compatible = self
            .buffers
            .as_ref()
            .is_some_and(|(key, _)| key.dims()[0] == batch);
        if compatible && needed <= self.capacity() {
            return Ok(());
        }
        // Serving reserves the known request budget exactly. Direct model calls
        // have no output budget, so double capacity at boundaries (capped by RoPE's
        // context limit). Old tokens are copied only on growth, not on every token.
        let capacity = if exact {
            needed
        } else {
            needed
                .checked_next_power_of_two()
                .unwrap_or(self.max_len)
                .min(self.max_len)
        };
        let elements = batch
            .checked_mul(self.num_heads)
            .and_then(|n| n.checked_mul(capacity))
            .and_then(|n| n.checked_mul(self.head_dim));
        // Candle's CUDA copy2d uses u32 dimensions/strides and an element-count grid.
        if capacity < needed || elements.is_none_or(|n| n > u32::MAX as usize) {
            candle::bail!("KV allocation exceeds supported copy dimensions")
        }
        // At a request boundary the old contents are invalid. Release the smaller
        // allocation before replacing it so growth does not temporarily retain two
        // complete caches. Active, unreserved sequences still copy their live prefix.
        if self.len == 0 {
            self.buffers = None;
        }
        let shape = (batch, self.num_heads, capacity, self.head_dim);
        // Zero only on allocation. Resetting a request changes `len`, not these bytes.
        let key = Tensor::zeros(shape, self.dtype, &self.device)?;
        let value = Tensor::zeros(shape, self.dtype, &self.device)?;
        if self.len != 0 {
            let (old_key, old_value) = self.prefix()?;
            if self.transposed_keys {
                key.reshape((batch, self.num_heads, self.head_dim, capacity))?
                    .slice_set(&old_key.transpose(2, 3)?.contiguous()?, 3, 0)?;
            } else {
                key.slice_set(&old_key.contiguous()?, 2, 0)?;
            }
            value.slice_set(&old_value.contiguous()?, 2, 0)?;
        }
        self.buffers = Some((key, value));
        Ok(())
    }

    /// Adds new keys and values immediately after the tokens already in the cache.
    /// Checks their shape, dtype, device and position, then copies them into retained
    /// storage, growing it if needed. The new length is published only after both
    /// copies succeed; callers must reset the request if an append fails.
    pub(super) fn append(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        offset: usize,
    ) -> CandleResult<()> {
        let (batch, heads, tokens, dim) = key.dims4()?;
        let end = self.validate_append(batch, tokens, offset)?;
        if heads != self.num_heads
            || dim != self.head_dim
            || value.shape() != key.shape()
            || key.dtype() != self.dtype
            || value.dtype() != self.dtype
            || !key.device().same_device(&self.device)
            || !value.device().same_device(&self.device)
        {
            candle::bail!("KV append shape, dtype or device mismatch")
        }
        // Direct callers without a reservation choose once at sequence start.
        // Growing an active sequence retains its chosen layout.
        if self.len == 0 && self.request_limit.is_none() {
            self.transposed_keys =
                self.prefer_transposed_keys && end >= GROUPED_DECODE_MIN_CONTEXT;
        }
        // Pack only incoming tokens in the retained layout. In transposed K,
        // append writes their columns; it never transposes the complete history.
        let key = if self.transposed_keys {
            key.transpose(2, 3)?.contiguous()?
        } else {
            key.contiguous()?
        };
        let value = value.contiguous()?;
        self.ensure_capacity(batch, end, self.request_limit.is_some())?;
        let (cached_key, cached_value) = self.buffers.as_ref().unwrap();
        if self.transposed_keys {
            cached_key
                .reshape((batch, self.num_heads, self.head_dim, self.capacity()))?
                .slice_set(&key, 3, offset)?;
        } else {
            cached_key.slice_set(&key, 2, offset)?;
        }
        cached_value.slice_set(&value, 2, offset)?;
        // Publish both copies together. A failed append must abort/reset the request.
        self.len = end;
        Ok(())
    }

    /// Returns views of the stored tokens, excluding all unused buffer capacity.
    /// Both views have shape [batch, KV heads, length, head dimension], regardless
    /// of the stored key layout. No token data is copied; missing storage is an error.
    pub(super) fn prefix(&self) -> CandleResult<(Tensor, Tensor)> {
        let (key, value) = self
            .buffers
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("KV cache has no storage".to_owned()))?;
        let key = if self.transposed_keys {
            key.reshape((
                key.dims()[0],
                self.num_heads,
                self.head_dim,
                self.capacity(),
            ))?
            .narrow(3, 0, self.len)?
            .transpose(2, 3)?
        } else {
            key.narrow(2, 0, self.len)?
        };
        Ok((key, value.narrow(2, 0, self.len)?))
    }

    /// Starts an empty request while keeping allocated buffers for reuse.
    /// Clears the visible length and request limit without erasing old token data.
    pub(super) fn reset(&mut self) {
        // Stale tail values, even NaNs, are harmless: only [0, len) is ever exposed.
        // Retain one high-water allocation per layer, bounded by the model context;
        // it is released when the owning model is dropped.
        self.len = 0;
        self.request_limit = None;
    }

    /// Clears request state and drops the retained key and value buffers.
    /// The next reservation or append must allocate storage again.
    pub(super) fn release(&mut self) {
        self.reset();
        self.buffers = None;
    }

    /// Lowers the maximum token capacity before any cache buffers are allocated.
    /// The positive limit bounds both request reservations and later buffer growth;
    /// it can never raise the existing model or memory limit.
    pub(super) fn set_capacity_limit(&mut self, capacity: usize) -> CandleResult<()> {
        if capacity == 0 || self.buffers.is_some() {
            candle::bail!(
                "KV capacity limit must be set before allocating cache storage"
            )
        }
        self.max_len = self.max_len.min(capacity);
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    fn cache(device: &Device, dtype: DType, heads: usize, dim: usize) -> KvCache {
        KvCache {
            buffers: None,
            transposed_keys: false,
            prefer_transposed_keys: device.is_cuda() && dtype == DType::BF16,
            len: 0,
            request_limit: None,
            num_heads: heads,
            head_dim: dim,
            max_len: 17,
            dtype,
            device: device.clone(),
        }
    }

    #[test]
    fn startup_limit_bounds_reservation_growth_and_survives_release() {
        let mut c = cache(&Device::Cpu, DType::F32, 2, 8);
        assert!(c.set_capacity_limit(0).is_err());
        c.set_capacity_limit(6).unwrap();
        assert!(c.reserve(7).is_err());
        c.reserve(6).unwrap();
        assert!(c.set_capacity_limit(4).is_err());
        c.release();
        assert!(c.buffers.is_none());
        assert_eq!(c.max_len, 6);
        let x = chunk(&c, 1, 3, 0);
        c.append(&x, &x, 0).unwrap();
        assert_eq!(c.capacity(), 4);
        c.append(&x, &x, 3).unwrap();
        assert_eq!(c.capacity(), 6);
        assert!(c.append(&x, &x, 6).is_err());
        c.reset();
        assert!(c.reserve(7).is_err());
    }

    pub(in super::super) fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    pub(in super::super) fn exact(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        for (a, b) in values(a).into_iter().zip(values(b)) {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    pub(in super::super) fn patterned(
        shape: &[usize],
        phase: usize,
        dtype: DType,
        device: &Device,
    ) -> Tensor {
        let data = (0..shape.iter().product())
            .map(|i| (((i * 13 + phase) % 67) as f32 - 33.0) / 64.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    /// Projection-like chunks are strided, as in real multi-token prefill.
    fn chunk(c: &KvCache, batch: usize, tokens: usize, phase: usize) -> Tensor {
        patterned(
            &[batch, tokens, c.num_heads, c.head_dim],
            phase,
            c.dtype,
            &c.device,
        )
        .transpose(1, 2)
        .unwrap()
    }

    fn append_reference(cache: &mut Option<Tensor>, input: &Tensor) {
        *cache = Some(match cache {
            Some(old) => Tensor::cat(&[&*old, input], 2).unwrap(),
            None => input.copy().unwrap(),
        });
    }

    fn check_reserved_chunks(device: &Device, dtype: DType) {
        // Includes Qwen3.5-4B's real KV shape, plus a small shape for readable
        // failures.
        for (heads, dim) in [(2, 8), (4, 256)] {
            let mut c = cache(device, dtype, heads, dim);
            c.reserve(17).unwrap();
            let (key, value) = c.buffers.as_ref().unwrap();
            let ids = (key.id(), value.id());
            assert!(values(key).iter().all(|&x| x == 0.0));
            assert!(values(value).iter().all(|&x| x == 0.0));
            let (mut rk, mut rv) = (None, None);
            for tokens in [3, 1, 5, 8] {
                let k = chunk(&c, 1, tokens, c.len);
                let v = chunk(&c, 1, tokens, 23 + c.len);
                append_reference(&mut rk, &k);
                append_reference(&mut rv, &v);
                c.append(&k, &v, c.len).unwrap();
                let (k, v) = c.prefix().unwrap();
                exact(&k, rk.as_ref().unwrap());
                exact(&v, rv.as_ref().unwrap());
                // Exercise both the no-expansion fallback and the production 4B/27B
                // repetition factors. Capacity strides must never expose padding.
                for groups in [1, 4, 6] {
                    let repeated =
                        candle_transformers::utils::repeat_kv(k.clone(), groups)
                            .unwrap();
                    let expected = candle_transformers::utils::repeat_kv(
                        rk.as_ref().unwrap().clone(),
                        groups,
                    )
                    .unwrap();
                    exact(
                        &repeated.contiguous().unwrap(),
                        &expected.contiguous().unwrap(),
                    );
                }
                let (key, value) = c.buffers.as_ref().unwrap();
                assert_eq!((key.id(), value.id()), ids);
                assert_eq!(c.capacity(), 17);
            }
        }
    }

    #[test]
    fn reserved_chunks_match_concatenation_cpu() {
        check_reserved_chunks(&Device::Cpu, DType::F32);
    }

    #[test]
    fn reserved_chunks_match_concatenation_cuda_bf16() {
        check_reserved_chunks(&Device::new_cuda(0).unwrap(), DType::BF16);
    }

    fn check_growth(device: &Device, dtype: DType) {
        let mut c = cache(device, dtype, 2, 8);
        let (mut rk, mut rv) = (None, None);
        let mut previous = None;
        for pos in 0..17 {
            let k = chunk(&c, 2, 1, pos);
            let v = chunk(&c, 2, 1, pos + 19);
            append_reference(&mut rk, &k);
            append_reference(&mut rv, &v);
            c.append(&k, &v, pos).unwrap();
            assert_eq!(c.capacity(), (pos + 1).next_power_of_two().min(17));
            let (key, value) = c.buffers.as_ref().unwrap();
            let current = (c.capacity(), key.id(), value.id());
            if let Some((cap, k_id, v_id)) = previous {
                if cap == current.0 {
                    assert_eq!((current.1, current.2), (k_id, v_id));
                } else {
                    assert_ne!((current.1, current.2), (k_id, v_id));
                }
            }
            previous = Some(current);
            let (k, v) = c.prefix().unwrap();
            exact(&k, rk.as_ref().unwrap());
            exact(&v, rv.as_ref().unwrap());
        }
        let k = chunk(&c, 2, 1, 9);
        assert!(c.append(&k, &k, 17).is_err());
        c.reset();
        c.reserve(4).unwrap(); // Idle batch change releases the old batch-two storage.
        let k = chunk(&c, 1, 4, 1);
        c.append(&k, &k, 0).unwrap();
        exact(&c.prefix().unwrap().0, &k);
    }

    #[test]
    fn geometric_growth_preserves_prefix_cpu() {
        check_growth(&Device::Cpu, DType::F32);
    }

    #[test]
    fn geometric_growth_preserves_prefix_cuda_bf16() {
        check_growth(&Device::new_cuda(0).unwrap(), DType::BF16);
    }

    #[test]
    fn rejected_appends_preserve_visible_state() {
        let mut c = cache(&Device::Cpu, DType::F32, 2, 8);
        assert!(c.reserve(0).is_err());
        assert!(c.reserve(18).is_err());
        c.reserve(8).unwrap();
        let k = chunk(&c, 1, 2, 3);
        c.append(&k, &k, 0).unwrap();
        assert!(c.reserve(9).is_err());
        for offset in [0, 1, 3, usize::MAX] {
            assert!(c.append(&k, &k, offset).is_err());
        }
        let bad_shape = Tensor::zeros((1, 1, 2, 8), DType::F32, &Device::Cpu).unwrap();
        assert!(c.append(&k, &bad_shape, 2).is_err());
        assert!(c.append(&k, &k.to_dtype(DType::F64).unwrap(), 2).is_err());
        assert!(
            c.append(&chunk(&c, 2, 1, 0), &chunk(&c, 2, 1, 0), 2)
                .is_err()
        );
        assert!(
            c.append(&chunk(&c, 1, 7, 0), &chunk(&c, 1, 7, 0), 2)
                .is_err()
        );
        assert_eq!(c.len, 2);
        exact(&c.prefix().unwrap().0, &k);
        exact(&c.prefix().unwrap().1, &k);
        c.reset();
        let id = c.buffers.as_ref().unwrap().0.id();
        c.reserve(3).unwrap();
        assert_eq!(c.capacity(), 8);
        assert_eq!(c.buffers.as_ref().unwrap().0.id(), id);
        assert!(
            c.append(&chunk(&c, 1, 4, 0), &chunk(&c, 1, 4, 0), 0)
                .is_err()
        );
        // The old bytes remain allocated, but an empty request cannot see them.
        assert!(values(&c.prefix().unwrap().0).is_empty());
    }

    #[test]
    fn transposed_cache_grows_and_reuses_storage_across_layout_changes() {
        let device = Device::new_cuda(0).unwrap();
        let mut c = cache(&device, DType::BF16, 4, 256);
        c.max_len = 1024;
        let (mut rk, mut rv) = (None, None);
        for tokens in [257, 256] {
            let k = chunk(&c, 1, tokens, c.len);
            let v = chunk(&c, 1, tokens, c.len + 31);
            append_reference(&mut rk, &k);
            append_reference(&mut rv, &v);
            c.append(&k, &v, c.len).unwrap();
            assert!(c.transposed_keys);
            exact(&c.prefix().unwrap().0, rk.as_ref().unwrap());
            exact(&c.prefix().unwrap().1, rv.as_ref().unwrap());
        }
        assert_eq!(c.capacity(), 1024);
        let ids = {
            let (k, v) = c.buffers.as_ref().unwrap();
            (k.id(), v.id())
        };
        for budget in [32, 768, 32] {
            c.reset();
            c.reserve(budget).unwrap();
            assert_eq!(c.transposed_keys, budget >= GROUPED_DECODE_MIN_CONTEXT);
            let k = chunk(&c, 1, 3, budget);
            c.append(&k, &k, 0).unwrap();
            exact(&c.prefix().unwrap().0, &k);
            exact(&c.prefix().unwrap().1, &k);
            let (key, value) = c.buffers.as_ref().unwrap();
            assert_eq!((key.id(), value.id()), ids);
        }
    }
}
