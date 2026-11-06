//! Scoped CUDA stream capture with explicit resource ownership and no CPU fallback.
//!
//! This primitive captures GPU work only. Model eligibility, input slots, state
//! preparation and warm-up remain the caller's responsibility.

use candle::cuda_backend::cudarc::driver::{CudaStream, result, sys};
use candle::{CudaDevice, Device, Result};
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

/// Creates a CUDA device for single-stream capture, or returns an explicit error.
///
/// Candle/cudarc's per-allocation events are unsuitable for replay: captured
/// events can become invalid when the graph is destroyed. Disable tracking before
/// allocating any tensors. All use must stay on its active stream and thread.
///
/// # Safety
/// The caller must serialize all operations on this device's active stream/thread and
/// must not share its tensors with other streams without explicit synchronization.
pub unsafe fn new_capture_device(ordinal: usize) -> Result<Device> {
    let device = Device::new_cuda(ordinal).map_err(|error| {
        candle::Error::Msg(format!(
            "CUDA graph capture requires a usable NVIDIA CUDA GPU {ordinal}: {error}"
        ))
    })?;
    // SAFETY: No user tensors exist yet, and the caller accepts stream ordering.
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    Ok(device)
}

/// A captured operation whose closure retains the resources referenced by its graph.
///
/// Replay is asynchronous and never calls the Rust closure again. The object is
/// deliberately neither Send nor Sync: Candle may use a per-thread CUDA stream.
pub(crate) struct CapturedGraph<F>
where
    F: FnMut() -> Result<()>,
{
    // Drop the graph (and wait for outstanding work) before releasing the closure.
    graph: GraphHandles,
    _operation: F,
}

impl<F: FnMut() -> Result<()>> CapturedGraph<F> {
    /// Captures an already-warmed operation on the supplied Candle device's stream.
    ///
    /// Enable Candle's CUDA parameter cache during warm-up on this same thread.
    /// Initialization/reset and host readback must happen outside capture.
    ///
    /// # Safety
    /// The closure must own or borrow every resource needed through replay (including
    /// weights and persistent input/output/state buffers). All work must use this
    /// device's active stream. All tensors must have been allocated with event tracking
    /// disabled (use new_capture_device before allocating them). Temporary graph
    /// allocations must be freed within the captured operation; persistent
    /// allocations must predate capture. The caller
    /// must serialize buffer updates/readback with replay on that stream, and must
    /// not replace retained storage or concurrently use captured mutable state.
    pub(crate) unsafe fn capture(device: &Device, mut operation: F) -> Result<Self> {
        let device = match device {
            Device::Cuda(device) => device.clone(),
            _ => candle::bail!(
                "CUDA graph capture requires an NVIDIA CUDA device; no CPU or non-CUDA \
                 GPU fallback is supported"
            ),
        };
        let stream = device.cuda_stream();
        require_idle(&stream)?;
        if device.is_event_tracking() {
            candle::bail!(
                "CUDA graph capture requires event tracking disabled before tensor \
                 allocation; use new_capture_device"
            )
        }
        stream.context().check_err().map_err(cuda_error)?;
        let _parameters = device.enable_cuda_graph_htod_cache();
        stream
            .begin_capture(
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(cuda_error)?;
        let capture = StreamCapture {
            stream,
            active: true,
        };
        let operation_result = operation();
        // Always end capture, including when the closure returns an error.
        let graph_result = capture.finish(device);
        let mut graph = match (operation_result, graph_result) {
            (Ok(()), Ok(graph)) => graph,
            (Err(error), Ok(_)) => return Err(error),
            (Ok(()), Err(error)) => return Err(error),
            (Err(error), Err(cleanup)) => candle::bail!(
                "CUDA graph operation failed: {error}; ending capture also failed: \
                 {cleanup}"
            ),
        };
        graph
            .device
            .cuda_stream()
            .context()
            .check_err()
            .map_err(cuda_error)?;
        if graph.node_count()? == 0 {
            candle::bail!("CUDA graph capture recorded no GPU work")
        }
        // SAFETY: This owner retains the graph on failure as well as success. Use
        // zero flags so unfreed graph allocations cause replay errors, not silent
        // resets.
        unsafe {
            sys::cuGraphInstantiateWithFlags(&mut graph.executable, graph.graph, 0)
                .result()
                .map_err(cuda_error)?;
        }
        Ok(Self {
            graph,
            _operation: operation,
        })
    }

    /// Enqueues one replay on Candle's original stream; execution errors surface at
    /// sync.
    pub(crate) fn replay(&mut self) -> Result<()> {
        let stream = self.graph.device.cuda_stream();
        require_idle(&stream)?;
        stream.context().check_err().map_err(cuda_error)?;
        // SAFETY: Handles and resources are owned by self; &mut self serializes
        // launches.
        unsafe { result::graph::launch(self.graph.executable, stream.cu_stream()) }
            .map_err(cuda_error)
    }

    /// Completes queued work and reports asynchronous execution or recorded driver
    /// errors.
    pub(crate) fn synchronize(&mut self) -> Result<()> {
        let stream = self.graph.device.cuda_stream();
        require_idle(&stream)?;
        stream.synchronize().map_err(cuda_error)?;
        stream.context().check_err().map_err(cuda_error)
    }

    /// Audit once after instantiation: prepared forward must own all workspace
    /// before capture, including library scratch. Allocation nodes are forbidden.
    pub(crate) fn allocation_nodes(&self) -> Result<usize> {
        let mut count = self.graph.node_count()?;
        let mut nodes = vec![ptr::null_mut(); count];
        // SAFETY: Graph handles are live and accessed on the owning thread.
        unsafe {
            sys::cuGraphGetNodes(self.graph.graph, nodes.as_mut_ptr(), &mut count)
                .result()
                .map_err(cuda_error)?;
        }
        let mut allocations = 0;
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            // SAFETY: Every node belongs to this live graph.
            unsafe {
                sys::cuGraphNodeGetType(node, &mut kind)
                    .result()
                    .map_err(cuda_error)?;
            }
            if matches!(
                kind,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC
                    | sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE
            ) {
                allocations += 1;
            }
        }
        Ok(allocations)
    }
}

fn cuda_error(error: result::DriverError) -> candle::Error {
    candle::Error::Cuda(Box::<dyn std::error::Error + Send + Sync>::from(format!(
        "CUDA graph operation failed: {error}"
    )))
}

fn require_idle(stream: &CudaStream) -> Result<()> {
    if stream.capture_status().map_err(cuda_error)?
        != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        candle::bail!("CUDA graph operation requires a stream outside capture")
    }
    Ok(())
}

/// Ends stream capture on early return or panic, without instantiating discarded work.
struct StreamCapture {
    stream: Arc<CudaStream>,
    active: bool,
}

impl StreamCapture {
    fn finish(mut self, device: CudaDevice) -> Result<GraphHandles> {
        self.active = false;
        // SAFETY: This guard owns the capture begun on this stream/thread.
        let graph = unsafe { result::stream::end_capture(self.stream.cu_stream()) }
            .map_err(cuda_error)?;
        if graph.is_null() {
            candle::bail!("CUDA stream capture returned no graph")
        }
        Ok(GraphHandles {
            device,
            graph,
            executable: ptr::null_mut(),
            _thread_bound: PhantomData,
        })
    }
}

impl Drop for StreamCapture {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: Panic unwinding remains on the capturing thread. Discard the
            // unexecuted graph and preserve any cleanup error in cudarc's context.
            let cleanup = unsafe {
                result::stream::end_capture(self.stream.cu_stream()).and_then(|graph| {
                    if graph.is_null() {
                        Ok(())
                    } else {
                        result::graph::destroy(graph)
                    }
                })
            };
            self.stream.context().record_err(cleanup);
        }
    }
}

/// Owns both handles immediately after capture, including on instantiation failure.
struct GraphHandles {
    device: CudaDevice,
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GraphHandles {
    fn node_count(&self) -> Result<usize> {
        let mut count = 0;
        // SAFETY: The graph is live and only accessed on its owning thread.
        unsafe { sys::cuGraphGetNodes(self.graph, ptr::null_mut(), &mut count) }
            .result()
            .map_err(cuda_error)?;
        Ok(count)
    }
}

impl Drop for GraphHandles {
    fn drop(&mut self) {
        let stream = self.device.cuda_stream();
        let context = stream.context();
        context.record_err(context.bind_to_thread());
        if !self.executable.is_null() {
            // External tensor lifetimes must outlive the last asynchronous replay.
            context.record_err(stream.synchronize());
            // SAFETY: These handles belong exclusively to this owner.
            context.record_err(unsafe { result::graph::exec_destroy(self.executable) });
        }
        context.record_err(unsafe { result::graph::destroy(self.graph) });
    }
}

#[cfg(test)]
pub(crate) mod tests;
