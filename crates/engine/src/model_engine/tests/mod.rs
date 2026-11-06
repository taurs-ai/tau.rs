//! Shared deterministic model/tokenizer fixtures for engine contract tests.

mod factory;
mod generation;
mod gpu_generation;
mod kv_budget;
mod options;

use super::generation as token_generation;
use super::kv_budget::KvCacheBudget;
use super::*;
use crate::model::{ModelLoadError, ModelLoadOptions};
use candle::Tensor as CandleTensor;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// A deterministic CPU model exercises real tokenizer, generation, and cache
/// boundaries.
struct SequenceModel {
    calls: Vec<usize>,
    resets: usize,
    fail_decode: bool,
    eos_offset: usize,
    decode_calls: usize,
    prefill_calls: Vec<(usize, usize, bool, bool)>,
    reservations: Vec<(usize, usize)>,
    fail_reservation: bool,
    graph_activity: crate::model::CudaGraphStats,
}

impl CausalLanguageModel for SequenceModel {
    fn reserve_kv_cache(&mut self, capacity: usize) -> candle::Result<()> {
        self.reservations.push((capacity, self.resets));
        if self.fail_reservation {
            candle::bail!("intentional reservation failure")
        }
        Ok(())
    }

    fn forward(
        &mut self,
        _: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        self.calls.push(offset);
        if self.fail_decode && offset != 0 {
            candle::bail!("intentional decode failure");
        }
        let values = if offset < self.eos_offset {
            [0., 0., 3., 0.]
        } else {
            [0., 0., 0., 3.]
        };
        CandleTensor::from_vec(values.to_vec(), (1, 1, 4), &Device::Cpu)
    }

    fn forward_prefill_chunk(
        &mut self,
        input: &CandleTensor,
        offset: usize,
        full_chunk: bool,
        last_chunk: bool,
    ) -> candle::Result<Option<CandleTensor>> {
        self.prefill_calls
            .push((offset, input.dim(1)?, full_chunk, last_chunk));
        let logits = self.forward(input, offset)?;
        Ok(last_chunk.then_some(logits))
    }

    fn forward_decode(
        &mut self,
        input: &CandleTensor,
        offset: usize,
    ) -> candle::Result<CandleTensor> {
        self.decode_calls += 1;
        self.forward(input, offset)
    }

    fn cuda_graph_counts(&self) -> crate::model::CudaGraphStats {
        self.calls.iter().fold(
            Default::default(),
            |counts: crate::model::CudaGraphStats, _| counts.merge(self.graph_activity),
        )
    }

    fn clear_kv_cache(&mut self) {
        self.resets += 1;
    }
}

fn sequence_engine(options: EngineOptions) -> CausalModelEngine<SequenceModel> {
    let vocabulary = [("<unk>", 0), ("hello", 1), ("answer", 2), ("eos", 3)]
        .into_iter()
        .map(|(token, id)| (token.to_owned(), id))
        .collect();
    let tokenizer = Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap(),
    );
    CausalModelEngine::from_loaded(
        PathBuf::from("fixture"),
        Device::Cpu,
        LoadedCausalModel {
            model: SequenceModel {
                calls: Vec::new(),
                resets: 0,
                fail_decode: false,
                eos_offset: 1,
                decode_calls: 0,
                prefill_calls: Vec::new(),
                reservations: Vec::new(),
                fail_reservation: false,
                graph_activity: Default::default(),
            },
            tokenizer,
            generation_config: GenerationConfig {
                eos_token_ids: vec![3],
                max_new_tokens: 4,
            },
            context_window: 32,
            dtype: ModelDType::F32,
        },
        options,
    )
}
