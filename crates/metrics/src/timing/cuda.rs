//! CUDA event ownership; no model, cache or execution-policy decisions.

use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaStream, sys};
use candle::{Device, Result};
use std::sync::Arc;

/// Owns four timing-enabled events on the inference stream, reused each step.
/// They are recorded outside capture and never become nodes in a model graph.
#[derive(Debug)]
pub(super) struct DeviceClock {
    stream: Arc<CudaStream>,
    forward_start: CudaEvent,
    forward_end: CudaEvent,
    conversion_start: CudaEvent,
    conversion_end: CudaEvent,
}

impl DeviceClock {
    /// Allocates events once per traced decode request, before timing any step.
    pub(super) fn new(device: &Device) -> Result<Self> {
        let stream = device.as_cuda_device()?.cuda_stream();
        let event = || {
            stream
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(error)
        };
        Ok(Self {
            forward_start: event()?,
            forward_end: event()?,
            conversion_start: event()?,
            conversion_end: event()?,
            stream,
        })
    }

    /// Refuses to record during capture, guarding graph ownership and node counts.
    fn record(&self, event: &CudaEvent) -> Result<()> {
        if self.stream.capture_status().map_err(error)?
            != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            candle::bail!("decode timing events must remain outside CUDA capture")
        }
        event.record(&self.stream).map_err(error)
    }

    /// Marks the beginning of submitted forward work on the existing stream.
    pub(super) fn record_forward_start(&self) -> Result<()> {
        self.record(&self.forward_start)
    }

    /// Marks the end of forward submission without synchronizing the stream.
    pub(super) fn record_forward_end(&self) -> Result<()> {
        self.record(&self.forward_end)
    }

    /// Marks the beginning of the existing logits-to-FP32 conversion interval.
    pub(super) fn record_conversion_start(&self) -> Result<()> {
        self.record(&self.conversion_start)
    }

    /// Ends conversion timing before the caller's existing blocking readback.
    pub(super) fn record_conversion_end(&self) -> Result<()> {
        self.record(&self.conversion_end)
    }

    /// Reads the completed forward interval without adding another device wait.
    pub(super) fn forward_us(&self) -> Result<f64> {
        Ok(f64::from(
            self.forward_start
                .elapsed_ms(&self.forward_end)
                .map_err(error)?,
        ) * 1000.0)
    }

    /// Reads the completed conversion interval after the existing host readback.
    pub(super) fn conversion_us(&self) -> Result<f64> {
        Ok(f64::from(
            self.conversion_start
                .elapsed_ms(&self.conversion_end)
                .map_err(error)?,
        ) * 1000.0)
    }
}

/// Preserves CUDA timing failures as request errors instead of plausible timings.
fn error(source: impl std::fmt::Display) -> candle::Error {
    candle::Error::Msg(format!("decode timing failed: {source}"))
}
