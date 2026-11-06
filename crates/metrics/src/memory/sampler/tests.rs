use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn worker(
    query: impl FnMut() -> Result<u64, String> + Send + 'static,
    interval: Duration,
) -> (Arc<Sampler>, thread::JoinHandle<()>) {
    let (commands, receiver) = mpsc::channel();
    let handle = thread::spawn(move || run(receiver, query, "GPU-test", interval));
    (Arc::new(Sampler { commands }), handle)
}

#[test]
fn samples_between_boundaries_and_stops_promptly_without_polling_idle() {
    let (observed, samples) = mpsc::channel();
    let mut calls = 0;
    let (sampler, handle) = worker(
        move || {
            calls += 1;
            observed.send(calls).unwrap();
            Ok(if calls == 2 { 400 } else { 100 })
        },
        Duration::from_millis(1),
    );
    assert!(samples.try_recv().is_err());
    sampler.begin(1).unwrap();
    assert_eq!(samples.recv_timeout(QUERY_TIMEOUT).unwrap(), 1);
    assert_eq!(samples.recv_timeout(QUERY_TIMEOUT).unwrap(), 2);
    let snapshot = sampler.finish(1).unwrap();
    assert_eq!(snapshot.vram_start_bytes, Some(100));
    assert_eq!(snapshot.vram_peak_sampled_bytes, Some(400));
    assert_eq!(snapshot.vram_end_bytes, Some(100));
    assert_eq!(snapshot.vram_peak_delta_bytes, Some(300));
    drop(sampler);
    handle.join().unwrap();
}

#[test]
fn shared_worker_keeps_overlapping_requests_and_resets_each_peak() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let (sampler, handle) = worker(
        move || {
            Ok([100, 400, 200, 150, 120, 110][counter.fetch_add(1, Ordering::Relaxed)])
        },
        Duration::from_secs(60),
    );
    sampler.begin(1).unwrap();
    sampler.begin(2).unwrap();
    let first = sampler.finish(1).unwrap();
    let second = sampler.finish(2).unwrap();
    assert_eq!(first.vram_peak_sampled_bytes, Some(400));
    assert_eq!(first.vram_peak_delta_bytes, Some(300));
    assert_eq!(second.vram_start_bytes, Some(400));
    assert_eq!(second.vram_peak_delta_bytes, Some(0));
    sampler.begin(3).unwrap();
    let third = sampler.finish(3).unwrap();
    assert_eq!(third.vram_peak_sampled_bytes, Some(120));
    assert_eq!(third.vram_end_bytes, Some(110));
    assert_eq!(calls.load(Ordering::Relaxed), 6);
    drop(sampler);
    handle.join().unwrap();
}

#[test]
fn dropped_requests_cancel_without_an_end_query() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let (sampler, handle) = worker(
        move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(100)
        },
        Duration::from_secs(60),
    );
    sampler.begin(1).unwrap();
    let request = super::super::VramMetrics {
        request_id: 1,
        model: "fixture".to_owned(),
        sampler: Some(Arc::clone(&sampler)),
        unavailable: VramSnapshot::new(None, VRAM_SAMPLE_INTERVAL),
    };
    drop(request);
    // Channel closure joins promptly even with a long configured interval.
    drop(sampler);
    handle.join().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn end_query_failure_keeps_samples_collected_during_generation() {
    let (observed, samples) = mpsc::channel();
    let mut calls = 0;
    let (sampler, handle) = worker(
        move || {
            calls += 1;
            observed.send(calls).unwrap();
            match calls {
                1 => Ok(100),
                2 => Ok(400),
                _ => Err("query failed".to_owned()),
            }
        },
        Duration::from_millis(1),
    );
    sampler.begin(1).unwrap();
    assert_eq!(samples.recv_timeout(QUERY_TIMEOUT).unwrap(), 1);
    assert_eq!(samples.recv_timeout(QUERY_TIMEOUT).unwrap(), 2);
    let snapshot = sampler.finish(1).unwrap();
    assert_eq!(snapshot.vram_peak_sampled_bytes, Some(400));
    assert_eq!(snapshot.vram_peak_delta_bytes, Some(300));
    assert_eq!(snapshot.vram_end_bytes, None);
    assert_eq!(snapshot.vram_error.as_deref(), Some("query failed"));
    drop(sampler);
    handle.join().unwrap();
}
