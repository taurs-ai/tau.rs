//! Trace-only decode phases. Host spans overlap asynchronous GPU execution;
//! CUDA events measure ordered stream intervals, including gaps between kernels.

mod cuda;
use candle::{Device, Result};
use cuda::DeviceClock;
use serde::Serialize;
use std::time::Instant;

/// Wall-clock host spans and optional CUDA stream spans, all in microseconds.
/// Readback host time includes waiting for queued forward/conversion work, so
/// these fields must not be added to GPU durations as independent costs.
#[derive(Debug, Default, Clone, Serialize)]
pub struct DecodePhaseTotals {
    pub input_setup_host_us: f64,
    pub forward_submit_host_us: f64,
    pub pre_readback_host_us: f64,
    pub logits_to_f32_host_us: f64,
    pub logits_readback_host_us: f64,
    pub cpu_selection_host_us: f64,
    pub gpu_forward_us: Option<f64>,
    pub gpu_logits_to_f32_us: Option<f64>,
    pub logits_readback_bytes: usize,
    pub timing_collection_host_us: f64,
}

impl DecodePhaseTotals {
    /// Adds one completed step without inventing GPU times on CPU/non-CUDA runs.
    fn add(&mut self, sample: &Self) {
        self.input_setup_host_us += sample.input_setup_host_us;
        self.forward_submit_host_us += sample.forward_submit_host_us;
        self.pre_readback_host_us += sample.pre_readback_host_us;
        self.logits_to_f32_host_us += sample.logits_to_f32_host_us;
        self.logits_readback_host_us += sample.logits_readback_host_us;
        self.cpu_selection_host_us += sample.cpu_selection_host_us;
        self.logits_readback_bytes += sample.logits_readback_bytes;
        self.timing_collection_host_us += sample.timing_collection_host_us;
        for (total, value) in [
            (&mut self.gpu_forward_us, sample.gpu_forward_us),
            (&mut self.gpu_logits_to_f32_us, sample.gpu_logits_to_f32_us),
        ] {
            if let Some(value) = value {
                *total = Some(total.unwrap_or(0.0) + value);
            }
        }
    }
}

/// One decode sample, retaining the actual forward boundary and cold captures.
#[derive(Debug, Default, Clone, Serialize)]
pub struct DecodePhases {
    pub forward_scope: &'static str,
    pub cold_graph_capture: bool,
    #[serde(flatten)]
    pub totals: DecodePhaseTotals,
}

/// Request totals retain cold steps while providing separate warmed aggregates.
#[derive(Debug, Default, Serialize)]
pub(crate) struct DecodePhaseSummary {
    pub steps: usize,
    pub cold_graph_steps: usize,
    pub total: DecodePhaseTotals,
    pub steady: DecodePhaseTotals,
}

impl DecodePhaseSummary {
    /// Records one complete sample; a capture step never enters warmed totals.
    pub(super) fn record(&mut self, sample: &DecodePhases) {
        self.steps += 1;
        self.cold_graph_steps += usize::from(sample.cold_graph_capture);
        self.total.add(&sample.totals);
        if !sample.cold_graph_capture {
            self.steady.add(&sample.totals);
        }
    }
}

/// Created only with tracing enabled. CPU runs retain host spans and null GPU
/// fields. CUDA elapsed values are collected after the existing blocking logit
/// readback, without adding a synchronization before forward or conversion.
#[derive(Debug)]
pub struct DecodeTimer {
    gpu: Option<DeviceClock>,
    boundary: Option<Instant>,
    sample: DecodePhases,
}

impl DecodeTimer {
    /// Creates the optional device clock; untraced callers never call this method.
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            gpu: device
                .is_cuda()
                .then(|| DeviceClock::new(device))
                .transpose()?,
            boundary: None,
            sample: DecodePhases::default(),
        })
    }

    /// Starts a fresh host phase record while reusing the request's CUDA events.
    pub fn start_step(&mut self) {
        self.sample = DecodePhases::default();
        self.boundary = Some(Instant::now());
    }

    /// Ends input/metadata/setup timing and brackets forward on the device stream.
    /// Full replay calls this after warm-up/capture and metadata/state preparation.
    pub fn begin_forward(&mut self, scope: &'static str) -> Result<()> {
        self.sample.forward_scope = scope;
        self.sample.totals.input_setup_host_us = self.mark();
        if let Some(gpu) = &self.gpu {
            gpu.record_forward_start()?;
        }
        Ok(())
    }

    /// Ends forward submission without waiting for GPU execution to complete.
    pub fn end_forward(&mut self) -> Result<()> {
        if let Some(gpu) = &self.gpu {
            gpu.record_forward_end()?;
        }
        self.sample.totals.forward_submit_host_us = self.mark();
        Ok(())
    }

    /// Separates post-forward bookkeeping from the logit view/cast submission.
    pub fn begin_conversion(&mut self) -> Result<()> {
        self.sample.totals.pre_readback_host_us = self.mark();
        if let Some(gpu) = &self.gpu {
            gpu.record_conversion_start()?;
        }
        Ok(())
    }

    /// Ends the FP32 conversion event span before the existing blocking readback.
    pub fn begin_readback(&mut self) -> Result<()> {
        if let Some(gpu) = &self.gpu {
            gpu.record_conversion_end()?;
        }
        self.sample.totals.logits_to_f32_host_us = self.mark();
        Ok(())
    }

    /// Records transfer/allocation/wait wall time and the FP32 vector's byte count.
    /// The following host interval is the CPU greedy selection and EOS check.
    pub fn end_readback(&mut self, elements: usize) {
        self.sample.totals.logits_readback_host_us = self.mark();
        self.sample.totals.logits_readback_bytes = elements * size_of::<f32>();
    }

    /// Stops CPU selection timing before trace hashing and reporting begin.
    pub fn end_selection(&mut self) {
        self.sample.totals.cpu_selection_host_us = self.mark();
    }

    /// Reads completed event intervals after sampling and labels capture steps.
    /// Driver query overhead is reported separately from model-step duration.
    pub fn finish(&mut self, cold_graph_capture: bool) -> Result<DecodePhases> {
        let started = Instant::now();
        self.sample.cold_graph_capture = cold_graph_capture;
        if let Some(gpu) = &self.gpu {
            self.sample.totals.gpu_forward_us = Some(gpu.forward_us()?);
            self.sample.totals.gpu_logits_to_f32_us = Some(gpu.conversion_us()?);
        }
        self.sample.totals.timing_collection_host_us =
            started.elapsed().as_secs_f64() * 1e6;
        Ok(self.sample.clone())
    }

    /// Advances the host wall-clock boundary; these spans can overlap GPU work.
    fn mark(&mut self) -> f64 {
        let now = Instant::now();
        now.duration_since(
            self.boundary
                .replace(now)
                .expect("decode timing step started"),
        )
        .as_secs_f64()
            * 1e6
    }
}

#[cfg(test)]
mod tests;
