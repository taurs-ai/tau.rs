//! Test-only views of actual grouped execution; production contains no readbacks.
#[cfg(test)]
use super::*;

#[cfg(test)]
impl MoePrefillRunner {
    /// Canonicalize actual scratch by its device permutation for the existing
    /// SGLang stage checker. This copies computed values and never reruns an MLP.
    pub(super) fn observe(&self, tokens: usize) -> Result<()> {
        use crate::test_support::tensor_capture as capture;
        let owner = self.workspace.lock().unwrap();
        let w = owner.as_ref().unwrap();
        let stream = self.gate_up.device().as_cuda_device()?.cuda_stream();
        let read = |buffer: &CudaSlice<bf16>| -> Result<Vec<f32>> {
            Ok(stream
                .clone_dtoh(buffer)
                .map_err(candle::Error::wrap)?
                .iter()
                .map(|v| v.to_f32())
                .collect())
        };
        let routes = stream.clone_dtoh(&w.routes).map_err(candle::Error::wrap)?;
        let meta = stream.clone_dtoh(&w.meta).map_err(candle::Error::wrap)?;
        let slots = stream.clone_dtoh(&w.slots).map_err(candle::Error::wrap)?;
        let tiles = stream.clone_dtoh(&w.tiles).map_err(candle::Error::wrap)?;
        assert_eq!(meta[512], 0, "invalid native dispatch");
        let padded = meta[513] as usize;
        let assignments = tokens * 8;
        let mut ids = Vec::new();
        let mut weights = Vec::new();
        for token in 0..tokens {
            ids.extend_from_slice(&routes[token * 16..token * 16 + 8]);
            weights.extend_from_slice(&routes[token * 16 + 8..token * 16 + 16]);
        }
        capture::host("ids", vec![tokens, 8], ids);
        capture::host("weights", vec![tokens, 8], weights);
        capture::host(
            "grouped-dispatch-slots",
            vec![padded],
            slots[..padded].iter().map(|v| *v as f32).collect(),
        );
        capture::host(
            "grouped-dispatch-experts",
            vec![padded / self.config.gate_up().rows()],
            tiles[..padded / self.config.gate_up().rows()]
                .iter()
                .map(|v| *v as f32)
                .collect(),
        );
        for (name, width, data) in [
            (
                "gate_up",
                self.dimensions.intermediate_size * 2,
                read(&w.projected)?,
            ),
            (
                "activation",
                self.dimensions.intermediate_size,
                read(&w.activated)?,
            ),
        ] {
            let mut canonical = vec![0f32; assignments * width];
            let mut visited = vec![false; assignments];
            for (row, slot) in slots[..padded].iter().enumerate() {
                if (*slot as usize) < assignments {
                    let slot = *slot as usize;
                    assert!(!std::mem::replace(&mut visited[slot], true));
                    canonical[slot * width..(slot + 1) * width]
                        .copy_from_slice(&data[row * width..(row + 1) * width]);
                }
            }
            assert!(visited.iter().all(|v| *v));
            capture::host(
                &format!("grouped-{name}"),
                vec![tokens, 8, width],
                canonical,
            );
        }
        let width = self.dimensions.hidden_size;
        capture::host(
            "grouped-weighted_down",
            vec![tokens, 8, width],
            read(&w.contributions)?[..assignments * width].to_vec(),
        );
        let receipts = stream
            .clone_dtoh(&w.receipts)
            .map_err(candle::Error::wrap)?;
        capture::host(
            "grouped-kernel-config",
            vec![2, 5],
            receipts.iter().map(|v| *v as f32).collect(),
        );
        capture::host("grouped-executed", vec![1], vec![1.]);
        Ok(())
    }
}
