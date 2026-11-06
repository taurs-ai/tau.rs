//! General-purpose partial rotary position embedding.
//!
//! Partial RoPE rotates a configured prefix of every query and key head while
//! preserving the remaining dimensions. This module validates concrete runtime
//! dimensions, precomputes sine/cosine tables, and applies them at the cache offset
//! supplied by incremental decoding. Model-family configuration parsing belongs to
//! the caller.

use candle::{D, DType, Device, Result as CandleResult, Tensor};

/// Concrete dimensions and frequency base for a partial RoPE component.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PartialRotaryConfig {
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) theta: f64,
}

impl PartialRotaryConfig {
    /// Checks dimensions and numerical values required to build rotary tables.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0 || self.max_position_embeddings == 0 {
            return Err(
                "partial RoPE head and context dimensions must be non-zero".to_owned()
            );
        }
        if self.rotary_dim == 0
            || !self.rotary_dim.is_multiple_of(2)
            || self.rotary_dim > self.head_dim
        {
            return Err(format!(
                "partial RoPE produces invalid rotary dimension {} for head_dim {}",
                self.rotary_dim, self.head_dim
            ));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err("partial RoPE theta must be positive and finite".to_owned());
        }
        if self.max_position_embeddings > u32::MAX as usize {
            return Err(
                "partial RoPE context length exceeds u32 position range".to_owned()
            );
        }
        Ok(())
    }
}

/// Precomputed partial-RoPE tables for one model instance.
#[derive(Debug, Clone)]
pub(crate) struct PartialRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    /// Borrows the retained coefficient tables for a fused prepared operation.
    pub(super) fn prepared_parameters(&self) -> (&Tensor, &Tensor, usize) {
        (&self.cos, &self.sin, self.rotary_dim)
    }

    /// Adds partial RoPE using this layer's cached sine/cosine tables. The kernel
    /// reads the position from device metadata, so advancing decode does not
    /// replace the tables or require a new graph.
    #[cfg(test)]
    pub(crate) fn prepare_decode_program(
        &self,
        program: &mut super::DecodeProgram,
        input: &Tensor,
        metadata: &Tensor,
        dim: usize,
    ) -> CandleResult<Tensor> {
        program.rope(input, &self.cos, &self.sin, metadata, dim, self.rotary_dim)
    }

    /// Cache reservations and appends must stay within the precomputed positions.
    pub(crate) fn max_sequence_length(&self) -> usize {
        self.cos.dims()[0]
    }

    /// Precomputes sine and cosine tables for the rotary portion of each head.
    pub(crate) fn new(
        cfg: &PartialRotaryConfig,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        if let Err(message) = cfg.validate() {
            candle::bail!("{message}")
        }
        let inverse_frequencies = (0..cfg.rotary_dim)
            .step_by(2)
            .map(|index| {
                1f32 / cfg.theta.powf(index as f64 / cfg.rotary_dim as f64) as f32
            })
            .collect::<Vec<_>>();
        let inverse_frequencies = Tensor::new(inverse_frequencies, device)?
            .reshape((1, cfg.rotary_dim / 2))?;
        let positions =
            Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
                .to_dtype(DType::F32)?
                .reshape((cfg.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inverse_frequencies)?;
        Ok(Self {
            cos: frequencies.cos()?.to_dtype(dtype)?,
            sin: frequencies.sin()?.to_dtype(dtype)?,
            rotary_dim: cfg.rotary_dim,
        })
    }

    /// Rotates the configured leading dimensions of Q and K at `offset`.
    ///
    /// Dimensions outside `rotary_dim` pass through unchanged, implementing
    /// partial-RoPE attention.
    pub(crate) fn apply(
        &self,
        query: &Tensor,
        key: &Tensor,
        offset: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (_, _, sequence_length, head_dim) = query.dims4()?;
        let cos = self.cos.narrow(0, offset, sequence_length)?;
        let sin = self.sin.narrow(0, offset, sequence_length)?;
        let query_rotary = query.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
        let key_rotary = key.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
        // FP32 coefficient tables opt into one final storage round after the
        // rotation. Existing same-dtype tables retain their original arithmetic.
        let query_rotary = candle_nn::rotary_emb::rope(
            &query_rotary.to_dtype(cos.dtype())?,
            &cos,
            &sin,
        )?
        .to_dtype(query.dtype())?;
        let key_rotary = candle_nn::rotary_emb::rope(
            &key_rotary.to_dtype(cos.dtype())?,
            &cos,
            &sin,
        )?
        .to_dtype(key.dtype())?;

        if self.rotary_dim == head_dim {
            return Ok((query_rotary, key_rotary));
        }

        let query_pass =
            query.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let key_pass =
            key.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        Ok((
            Tensor::cat(&[&query_rotary, &query_pass], D::Minus1)?,
            Tensor::cat(&[&key_rotary, &key_pass], D::Minus1)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a conventional partial-RoPE configuration for focused unit tests.
    fn config(rotary_dim: usize) -> PartialRotaryConfig {
        PartialRotaryConfig {
            head_dim: 64,
            rotary_dim,
            max_position_embeddings: 128,
            theta: 10_000_000.0,
        }
    }

    /// Confirms validation accepts even partial widths and rejects malformed ones.
    #[test]
    fn validates_partial_rotary_dimensions() {
        config(16).validate().unwrap();

        let error = config(3).validate().unwrap_err();
        assert!(error.contains("invalid rotary dimension 3"));

        let error = config(0).validate().unwrap_err();
        assert!(error.contains("invalid rotary dimension 0"));
    }

    /// Confirms invalid frequency bases are rejected before table construction.
    #[test]
    fn rejects_invalid_theta() {
        let mut config = config(16);
        config.theta = f64::INFINITY;

        let error = config.validate().unwrap_err();
        assert!(error.contains("theta must be positive and finite"));
    }

    /// Confirms partial RoPE rotates its prefix while preserving trailing features.
    #[test]
    fn leaves_non_rotary_dimensions_unchanged() {
        let device = Device::Cpu;
        let config = PartialRotaryConfig {
            head_dim: 4,
            rotary_dim: 2,
            max_position_embeddings: 8,
            theta: 10_000_000.0,
        };
        let rope = PartialRotaryEmbedding::new(&config, DType::F32, &device).unwrap();
        let query = Tensor::new(
            (1..=8).map(|value| value as f32).collect::<Vec<_>>(),
            &device,
        )
        .unwrap()
        .reshape((1, 1, 2, 4))
        .unwrap();
        let key = (&query * 2.0).unwrap();

        let (rotated_query, rotated_key) = rope.apply(&query, &key, 0).unwrap();
        let query_tail = query
            .narrow(D::Minus1, 2, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rotated_query_tail = rotated_query
            .narrow(D::Minus1, 2, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let key_tail = key
            .narrow(D::Minus1, 2, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rotated_key_tail = rotated_key
            .narrow(D::Minus1, 2, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(rotated_query_tail, query_tail);
        assert_eq!(rotated_key_tail, key_tail);
        assert_ne!(
            rotated_query
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            query.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}
