//! Opt-in sampled process VRAM, shared across reporting modes and GPU requests.

mod nvml;
mod sampler;

use sampler::Sampler;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

// Internal configuration: a future CLI option can supply this interval instead.
const VRAM_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const QUERY_TIMEOUT: Duration = Duration::from_secs(1);

/// A sampled process footprint on one device, including retained allocations.
/// This is not an allocator high-water mark; brief spikes can escape observation.
#[derive(Debug, Clone, Serialize)]
pub(super) struct VramSnapshot {
    vram_start_bytes: Option<u64>,
    vram_peak_sampled_bytes: Option<u64>,
    vram_end_bytes: Option<u64>,
    vram_peak_delta_bytes: Option<u64>,
    vram_sample_interval_ms: u64,
    vram_samples: usize,
    vram_scope: &'static str,
    vram_device_uuid: Option<String>,
    // Preserve evidence of missing samples even if later queries succeed.
    vram_error: Option<String>,
}

impl VramSnapshot {
    fn new(uuid: Option<String>, interval: Duration) -> Self {
        Self {
            vram_start_bytes: None,
            vram_peak_sampled_bytes: None,
            vram_end_bytes: None,
            vram_peak_delta_bytes: None,
            vram_sample_interval_ms: interval.as_millis() as u64,
            vram_samples: 0,
            vram_scope: "process_on_device",
            vram_device_uuid: uuid,
            vram_error: None,
        }
    }

    fn observe(&mut self, sample: &Result<u64, String>) {
        match sample {
            Ok(bytes) => {
                self.vram_peak_sampled_bytes =
                    Some(self.vram_peak_sampled_bytes.unwrap_or(0).max(*bytes));
                self.vram_samples += 1;
            }
            Err(error) => self.vram_error = Some(error.clone()),
        }
    }

    fn complete(&mut self, sample: &Result<u64, String>) {
        self.vram_end_bytes = sample.as_ref().ok().copied();
        self.vram_peak_delta_bytes = self
            .vram_peak_sampled_bytes
            .zip(self.vram_start_bytes)
            .map(|(peak, start)| peak.saturating_sub(start));
    }
}

/// Request ownership only; the shared worker owns samples and the NVML handle.
#[derive(Debug)]
pub(super) struct VramMetrics {
    request_id: u64,
    model: String,
    sampler: Option<Arc<Sampler>>,
    unavailable: VramSnapshot,
}

impl VramMetrics {
    pub(super) fn start(model: &str, request_id: u64, device: &candle::Device) -> Self {
        let mut metrics = Self {
            request_id,
            model: model.to_owned(),
            sampler: None,
            unavailable: VramSnapshot::new(None, VRAM_SAMPLE_INTERVAL),
        };
        let result = (|| {
            if !device.is_cuda() {
                return Err("VRAM metrics require a CUDA device".to_owned());
            }
            let cuda = device.as_cuda_device().map_err(|e| e.to_string())?;
            let uuid = cuda
                .cuda_stream()
                .context()
                .uuid()
                .map_err(|e| e.to_string())?;
            // Match the actual CUDA UUID, rather than CUDA/NVML ordinals, which
            // can differ when CUDA_VISIBLE_DEVICES remaps or hides devices.
            let uuid = format_uuid(uuid.bytes.map(|byte| byte as u8));
            metrics.unavailable.vram_device_uuid = Some(uuid.clone());
            let sampler = Sampler::shared(&uuid)?;
            sampler.begin(request_id)?;
            Ok(sampler)
        })();
        match result {
            Ok(sampler) => metrics.sampler = Some(sampler),
            Err(error) => metrics.unavailable.vram_error = Some(error),
        }
        metrics
    }

    pub(super) fn finish(mut self, error: Option<&str>) -> VramSnapshot {
        let snapshot = match self.sampler.take() {
            Some(sampler) => match sampler.finish(self.request_id) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.unavailable.vram_error = Some(error);
                    self.unavailable.clone()
                }
            },
            None => self.unavailable.clone(),
        };
        #[derive(Serialize)]
        struct Report<'a> {
            request_id: u64,
            model: &'a str,
            status: &'static str,
            error: Option<&'a str>,
            #[serde(flatten)]
            memory: &'a VramSnapshot,
        }
        let report = Report {
            request_id: self.request_id,
            model: &self.model,
            status: if error.is_some() { "error" } else { "ok" },
            error,
            memory: &snapshot,
        };
        eprintln!(
            "inference_vram_metrics {}",
            serde_json::to_string(&report).expect("VRAM metrics are serializable")
        );
        snapshot
    }
}

impl Drop for VramMetrics {
    fn drop(&mut self) {
        if let Some(sampler) = self.sampler.take() {
            // Cancel dropped/panicked requests without blocking the serving thread.
            sampler.cancel(self.request_id);
        }
    }
}

fn format_uuid(bytes: [u8; 16]) -> String {
    let mut uuid = String::from("GPU-");
    use std::fmt::Write;
    for (i, byte) in bytes.iter().enumerate() {
        if [4, 6, 8, 10].contains(&i) {
            uuid.push('-');
        }
        write!(uuid, "{byte:02x}").expect("writing to a String cannot fail");
    }
    uuid
}

#[cfg(test)]
mod tests;
