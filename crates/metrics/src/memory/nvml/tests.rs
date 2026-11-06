use super::*;

struct Query {
    entries: Vec<ProcessInfo>,
    status: u32,
    calls: usize,
}

unsafe extern "C" fn fake_processes(
    device: *mut c_void,
    count: *mut c_uint,
    entries: *mut ProcessInfo,
) -> c_uint {
    // SAFETY: Tests pass a live exclusive Query and the initialized output buffer.
    let query = unsafe { &mut *device.cast::<Query>() };
    query.calls += 1;
    if query.status != 0 {
        return query.status;
    }
    // SAFETY: count is writable storage supplied by process_memory.
    let capacity = unsafe { *count } as usize;
    unsafe {
        *count = query.entries.len() as c_uint;
    }
    if capacity < query.entries.len() {
        return 7;
    }
    // SAFETY: The capacity check above guarantees enough writable entries.
    unsafe {
        std::ptr::copy_nonoverlapping(
            query.entries.as_ptr(),
            entries,
            query.entries.len(),
        );
    }
    0
}

#[test]
fn filters_server_pid_and_retries_growing_process_lists() {
    let mut query = Query {
        entries: vec![
            ProcessInfo {
                pid: 123,
                used_gpu_memory: 999,
                ..Default::default()
            },
            ProcessInfo {
                pid: 456,
                used_gpu_memory: 200,
                ..Default::default()
            },
        ],
        status: 0,
        calls: 0,
    };
    let mut buffer = vec![ProcessInfo::default(); 1];
    let sample = process_memory(
        fake_processes,
        (&mut query as *mut Query).cast(),
        &mut buffer,
        456,
    );
    assert_eq!(sample.unwrap(), 200);
    assert_eq!(query.calls, 2);
}

#[test]
fn unavailable_process_and_driver_failures_are_not_zero_usage() {
    let mut query = Query {
        entries: vec![ProcessInfo {
            pid: 123,
            used_gpu_memory: u64::MAX,
            ..Default::default()
        }],
        status: 0,
        calls: 0,
    };
    let mut buffer = vec![ProcessInfo::default(); 2];
    assert!(
        process_memory(
            fake_processes,
            (&mut query as *mut Query).cast(),
            &mut buffer,
            123
        )
        .unwrap_err()
        .contains("unavailable")
    );
    assert!(
        process_memory(
            fake_processes,
            (&mut query as *mut Query).cast(),
            &mut buffer,
            456
        )
        .unwrap_err()
        .contains("absent")
    );
    query.status = 3;
    assert!(
        process_memory(
            fake_processes,
            (&mut query as *mut Query).cast(),
            &mut buffer,
            123
        )
        .unwrap_err()
        .contains("NVML status 3")
    );
}
