//! Minimal dynamically loaded NVML process-memory API. No driver calls at startup.
//!
//! ABI: NVIDIA nvml.h, nvmlProcessInfo_t and the _v3 process enumeration API.
//! https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html

use libloading::Library;
use std::ffi::{CString, c_char, c_uint, c_void};

const NVML_ERROR_NOT_FOUND: c_uint = 6;
const NVML_ERROR_INSUFFICIENT_SIZE: c_uint = 7;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

type Shutdown = unsafe extern "C" fn() -> c_uint;
type Processes =
    unsafe extern "C" fn(*mut c_void, *mut c_uint, *mut ProcessInfo) -> c_uint;
type HandleByUuid = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_uint;

pub(super) struct Nvml {
    // Function pointers remain valid until after shutdown and all handle use.
    _library: Library,
    shutdown: Shutdown,
    processes: Processes,
    device: *mut c_void,
    buffer: Vec<ProcessInfo>,
}

impl Nvml {
    pub(super) fn open(uuid: &str) -> Result<Self, String> {
        let name = if cfg!(target_os = "windows") {
            "nvml.dll"
        } else {
            "libnvidia-ml.so.1"
        };
        // SAFETY: Load NVIDIA's standard NVML library and resolve exact C ABIs.
        // The owning Library is retained for every copied function pointer's use.
        let (library, init, shutdown, handle, processes) = unsafe {
            let library = Library::new(name).map_err(|e| e.to_string())?;
            let init = *library
                .get::<Shutdown>(b"nvmlInit_v2\0")
                .map_err(|e| e.to_string())?;
            let shutdown = *library
                .get::<Shutdown>(b"nvmlShutdown\0")
                .map_err(|e| e.to_string())?;
            let handle = *library
                .get::<HandleByUuid>(b"nvmlDeviceGetHandleByUUID\0")
                .map_err(|e| e.to_string())?;
            let processes = *library
                .get::<Processes>(b"nvmlDeviceGetComputeRunningProcesses_v3\0")
                .map_err(|e| e.to_string())?;
            (library, init, shutdown, handle, processes)
        };
        // SAFETY: Resolved zero-argument NVML initializer with no pointer inputs.
        check("nvmlInit_v2", unsafe { init() })?;
        let mut nvml = Self {
            _library: library,
            shutdown,
            processes,
            device: std::ptr::null_mut(),
            buffer: vec![ProcessInfo::default(); 16],
        };
        let uuid = CString::new(uuid).map_err(|e| e.to_string())?;
        // SAFETY: UUID is NUL-terminated; device points to writable handle storage.
        let status = unsafe { handle(uuid.as_ptr(), &mut nvml.device) };
        if status == NVML_ERROR_NOT_FOUND {
            // CUDA UUIDs identify MIG instances too; NVML uses a MIG- prefix.
            let mig = CString::new(uuid.to_str().unwrap().replacen("GPU-", "MIG-", 1))
                .map_err(|e| e.to_string())?;
            // SAFETY: Same handle ABI and owned writable output as above.
            check("nvmlDeviceGetHandleByUUID", unsafe {
                handle(mig.as_ptr(), &mut nvml.device)
            })?;
        } else {
            check("nvmlDeviceGetHandleByUUID", status)?;
        }
        Ok(nvml)
    }

    pub(super) fn sample(&mut self) -> Result<u64, String> {
        process_memory(
            self.processes,
            self.device,
            &mut self.buffer,
            std::process::id(),
        )
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: Balances this instance's successful init before unloading NVML.
        let _ = unsafe { (self.shutdown)() };
    }
}

fn check(operation: &str, status: c_uint) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed (NVML status {status})"))
    }
}

fn process_memory(
    processes: Processes,
    device: *mut c_void,
    buffer: &mut Vec<ProcessInfo>,
    pid: u32,
) -> Result<u64, String> {
    // Process lists can grow between queries. Bound retries under process churn.
    for _ in 0..3 {
        let mut count = buffer.len() as c_uint;
        // SAFETY: The function and handle come from live NVML on this worker.
        // The buffer has count initialized entries of the documented _v3 ABI.
        let status = unsafe { processes(device, &mut count, buffer.as_mut_ptr()) };
        if status == NVML_ERROR_INSUFFICIENT_SIZE {
            buffer.resize(
                (count as usize).max(buffer.len() * 2),
                ProcessInfo::default(),
            );
            continue;
        }
        check("nvmlDeviceGetComputeRunningProcesses_v3", status)?;
        let entries = buffer
            .get(..count as usize)
            .ok_or_else(|| "NVML returned an invalid process count".to_owned())?;
        let info = entries
            .iter()
            .find(|info| info.pid == pid)
            .ok_or_else(|| "server PID is absent from NVML process list".to_owned())?;
        if info.used_gpu_memory == u64::MAX {
            return Err("NVML process VRAM measurement is unavailable".to_owned());
        }
        return Ok(info.used_gpu_memory);
    }
    Err("NVML process list kept growing during sampling".to_owned())
}

#[cfg(test)]
mod tests;
