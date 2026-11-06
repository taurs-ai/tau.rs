//! One lazy worker per process/device; idle workers perform no queries.

use super::{QUERY_TIMEOUT, VRAM_SAMPLE_INTERVAL, VramSnapshot, nvml::Nvml};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};

type SamplerCache = HashMap<String, Result<Arc<Sampler>, String>>;
static SAMPLERS: OnceLock<Mutex<SamplerCache>> = OnceLock::new();

#[derive(Debug)]
pub(super) struct Sampler {
    commands: mpsc::Sender<Command>,
}

enum Command {
    Begin(u64, mpsc::Sender<()>),
    Finish(u64, mpsc::Sender<VramSnapshot>),
    Cancel(u64),
}

impl Sampler {
    pub(super) fn shared(uuid: &str) -> Result<Arc<Self>, String> {
        // The cache is initialized only by explicit VRAM collection. Workers live
        // for the process lifetime and sleep without polling between requests.
        let mut cache = SAMPLERS
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| "VRAM sampler registry is unavailable".to_owned())?;
        cache
            .entry(uuid.to_owned())
            .or_insert_with(|| Self::spawn(uuid))
            .clone()
    }

    fn spawn(uuid: &str) -> Result<Arc<Self>, String> {
        let (commands, receiver) = mpsc::channel();
        let (ready, initialized) = mpsc::channel();
        let uuid = uuid.to_owned();
        thread::Builder::new()
            .name("tau-vram-metrics".to_owned())
            .spawn(move || {
                // Load NVML and own its raw handle on this worker; no CUDA device,
                // tensor, stream or graph handle crosses into the sampling thread.
                match Nvml::open(&uuid) {
                    Ok(mut nvml) => {
                        if ready.send(Ok(())).is_ok() {
                            run(
                                receiver,
                                || nvml.sample(),
                                &uuid,
                                VRAM_SAMPLE_INTERVAL,
                            );
                        }
                    }
                    Err(error) => {
                        let _ = ready.send(Err(error));
                    }
                }
            })
            .map_err(|e| format!("cannot start VRAM sampler: {e}"))?;
        initialized
            .recv_timeout(QUERY_TIMEOUT)
            .map_err(|e| format!("VRAM sampler initialization failed: {e}"))??;
        Ok(Arc::new(Self { commands }))
    }

    pub(super) fn begin(&self, request_id: u64) -> Result<(), String> {
        let (reply, received) = mpsc::channel();
        self.commands
            .send(Command::Begin(request_id, reply))
            .map_err(|e| e.to_string())?;
        received.recv_timeout(QUERY_TIMEOUT).map_err(|e| {
            self.cancel(request_id);
            format!("VRAM start sample failed: {e}")
        })
    }

    pub(super) fn finish(&self, request_id: u64) -> Result<VramSnapshot, String> {
        let (reply, received) = mpsc::channel();
        self.commands
            .send(Command::Finish(request_id, reply))
            .map_err(|e| e.to_string())?;
        received.recv_timeout(QUERY_TIMEOUT).map_err(|e| {
            self.cancel(request_id);
            format!("VRAM end sample failed: {e}")
        })
    }

    pub(super) fn cancel(&self, request_id: u64) {
        let _ = self.commands.send(Command::Cancel(request_id));
    }
}

fn observe_all(active: &mut HashMap<u64, VramSnapshot>, sample: &Result<u64, String>) {
    for snapshot in active.values_mut() {
        snapshot.observe(sample);
    }
}

fn run(
    receiver: mpsc::Receiver<Command>,
    mut sample: impl FnMut() -> Result<u64, String>,
    uuid: &str,
    interval: Duration,
) {
    let mut active = HashMap::<u64, VramSnapshot>::new();
    let mut next_sample = Instant::now() + interval;
    loop {
        // Commands from other requests must not postpone periodic observation.
        if !active.is_empty() && Instant::now() >= next_sample {
            observe_all(&mut active, &sample());
            next_sample = Instant::now() + interval;
        }
        let command = if active.is_empty() {
            receiver
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
        } else {
            receiver.recv_timeout(next_sample.saturating_duration_since(Instant::now()))
        };
        match command {
            Ok(Command::Begin(id, reply)) => {
                let was_idle = active.is_empty();
                let sample = sample();
                observe_all(&mut active, &sample);
                let mut snapshot = VramSnapshot::new(Some(uuid.to_owned()), interval);
                snapshot.vram_start_bytes = sample.as_ref().ok().copied();
                snapshot.observe(&sample);
                active.insert(id, snapshot);
                if was_idle {
                    next_sample = Instant::now() + interval;
                }
                let _ = reply.send(());
            }
            Ok(Command::Finish(id, reply)) => {
                let sample = sample();
                observe_all(&mut active, &sample);
                if let Some(mut snapshot) = active.remove(&id) {
                    snapshot.complete(&sample);
                    let _ = reply.send(snapshot);
                }
            }
            Ok(Command::Cancel(id)) => {
                active.remove(&id);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                observe_all(&mut active, &sample());
                next_sample = Instant::now() + interval;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[cfg(test)]
mod tests;
