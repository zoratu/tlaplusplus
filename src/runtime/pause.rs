//! Checkpoint pause coordination.
//!
//! `PauseController` parks worker threads at quiescence so the checkpoint
//! thread can serialize a consistent snapshot. The protocol is lock-free
//! (workers `thread::park` independently) and the visibility ordering between
//! `pause.requested` and `queue.set_checkpoint_pause_requested` is load-bearing
//! — see `request_checkpoint_pause`.

use crate::storage::numa::NumaDiagnostics;
use crate::storage::spillable_work_stealing::SpillableWorkStealingQueues;
use crate::storage::work_stealing_queues::WorkStealingQueues;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub(super) struct PauseController {
    pub(super) requested: AtomicBool,
    pub(super) paused_workers: AtomicUsize,
    /// Per-worker thread handles for lock-free unparking.
    /// Workers call thread::park() instead of blocking on a shared mutex.
    /// The checkpoint thread calls unpark() on each handle to resume.
    worker_threads: parking_lot::RwLock<Vec<std::thread::Thread>>,
    /// Per-worker pause status for debugging (worker_id -> is_paused)
    worker_pause_status: parking_lot::RwLock<Vec<AtomicBool>>,
    /// Per-worker NUMA node assignment (for diagnostics)
    worker_numa_nodes: parking_lot::RwLock<Vec<usize>>,
    /// NUMA diagnostics for stuck worker analysis
    numa_diagnostics: parking_lot::RwLock<Option<NumaDiagnostics>>,
}

impl Default for PauseController {
    fn default() -> Self {
        Self {
            requested: AtomicBool::new(false),
            paused_workers: AtomicUsize::new(0),
            worker_threads: parking_lot::RwLock::new(Vec::new()),
            worker_pause_status: parking_lot::RwLock::new(Vec::new()),
            worker_numa_nodes: parking_lot::RwLock::new(Vec::new()),
            numa_diagnostics: parking_lot::RwLock::new(None),
        }
    }
}

pub(super) const QUIESCENCE_INITIAL_TIMEOUT_SECS: u64 = 60;
pub(super) const QUIESCENCE_MAX_TOTAL_TIMEOUT_SECS: u64 = 300;
pub(super) const QUIESCENCE_MAX_ATTEMPTS: u32 = 3;
pub(super) const QUIESCENCE_RETRY_SETTLE_DELAY_MS: u64 = 100;

pub(super) fn next_quiescence_timeout_secs(attempt_index: u32, elapsed: Duration) -> Option<u64> {
    if attempt_index >= QUIESCENCE_MAX_ATTEMPTS {
        return None;
    }

    let elapsed_secs = elapsed.as_secs();
    if elapsed_secs >= QUIESCENCE_MAX_TOTAL_TIMEOUT_SECS {
        return None;
    }

    let remaining_secs = QUIESCENCE_MAX_TOTAL_TIMEOUT_SECS - elapsed_secs;
    let planned_timeout =
        QUIESCENCE_INITIAL_TIMEOUT_SECS.saturating_mul(1u64 << attempt_index.min(62));
    Some(planned_timeout.min(remaining_secs.max(1)))
}

impl PauseController {
    /// Initialize per-worker tracking for debugging
    pub(super) fn init_worker_tracking(&self, num_workers: usize, worker_numa_nodes: &[usize]) {
        let mut status = self.worker_pause_status.write();
        status.clear();
        for _ in 0..num_workers {
            status.push(AtomicBool::new(false));
        }
        drop(status);

        // Store NUMA node assignments for each worker
        let mut numa_nodes = self.worker_numa_nodes.write();
        numa_nodes.clear();
        numa_nodes.extend_from_slice(worker_numa_nodes);
    }

    /// Set NUMA diagnostics for stuck worker analysis
    pub(super) fn set_numa_diagnostics(&self, diagnostics: NumaDiagnostics) {
        let mut diag = self.numa_diagnostics.write();
        *diag = Some(diagnostics);
    }

    /// Get list of worker IDs that are NOT paused (for debugging)
    pub(super) fn get_unpaused_workers(&self) -> Vec<usize> {
        let status = self.worker_pause_status.read();
        status
            .iter()
            .enumerate()
            .filter(|(_, paused)| !paused.load(Ordering::Acquire))
            .map(|(id, _)| id)
            .collect()
    }

    /// Get the NUMA node assignments for workers
    pub(super) fn get_worker_numa_nodes(&self) -> Vec<usize> {
        self.worker_numa_nodes.read().clone()
    }
}

impl PauseController {
    /// Register a worker thread for lock-free unparking during checkpoint resume.
    pub(super) fn register_worker_thread(&self, thread: std::thread::Thread) {
        self.worker_threads.write().push(thread);
    }

    pub(super) fn worker_pause_point(&self, stop: &AtomicBool, worker_id: usize) {
        if !self.requested.load(Ordering::Acquire) {
            return;
        }

        // Mark this worker as paused (for debugging)
        {
            let status = self.worker_pause_status.read();
            if let Some(paused) = status.get(worker_id) {
                paused.store(true, Ordering::Release);
            }
        }

        let paused_count = self.paused_workers.fetch_add(1, Ordering::AcqRel) + 1;
        if paused_count <= 5 || paused_count % 10 == 0 {
            eprintln!(
                "Worker {} pausing: {} workers now paused",
                worker_id, paused_count
            );
        }

        // Lock-free pause: park this thread instead of blocking on a shared mutex.
        // Each worker parks independently — zero contention between workers.
        while self.requested.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
            std::thread::park_timeout(Duration::from_millis(50));
        }

        self.paused_workers.fetch_sub(1, Ordering::AcqRel);

        // Mark this worker as unpaused
        {
            let status = self.worker_pause_status.read();
            if let Some(paused) = status.get(worker_id) {
                paused.store(false, Ordering::Release);
            }
        }
    }

    pub(super) fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Wait for all workers to pause (quiescence).
    /// Returns true if quiescence was achieved, false if timeout occurred.
    ///
    /// Uses exponential backoff with a hard overall budget:
    /// starts at 60s, doubles on each retry, and never exceeds 5 minutes total.
    pub(super) fn wait_for_quiescence(
        &self,
        stop: &AtomicBool,
        active_workers: &AtomicUsize,
        live_workers: &AtomicUsize,
    ) -> bool {
        let mut attempt_index = 0u32;
        let overall_start = Instant::now();

        loop {
            let Some(current_timeout_secs) =
                next_quiescence_timeout_secs(attempt_index, overall_start.elapsed())
            else {
                eprintln!(
                    "Checkpoint: giving up after {:.1}s total, skipping checkpoint",
                    overall_start.elapsed().as_secs_f64()
                );
                return false;
            };

            let timeout = Duration::from_secs(current_timeout_secs);
            eprintln!(
                "Checkpoint: entering wait_for_quiescence (attempt {}/{}, timeout: {}s)",
                attempt_index + 1,
                QUIESCENCE_MAX_ATTEMPTS,
                current_timeout_secs
            );

            if let Some(success) =
                self.wait_for_quiescence_attempt(stop, active_workers, live_workers, timeout)
            {
                if success {
                    eprintln!(
                        "Checkpoint: quiescence achieved after {:.1}s total ({} retries)",
                        overall_start.elapsed().as_secs_f64(),
                        attempt_index
                    );
                    return true;
                }
            } else {
                // Stopped
                return false;
            }

            // Quiescence failed for this attempt
            attempt_index += 1;
            let Some(next_timeout_secs) =
                next_quiescence_timeout_secs(attempt_index, overall_start.elapsed())
            else {
                eprintln!(
                    "Checkpoint: giving up after {:.1}s total, skipping checkpoint",
                    overall_start.elapsed().as_secs_f64()
                );
                return false;
            };

            // Exponential backoff: double the timeout
            eprintln!(
                "Checkpoint: quiescence timeout, backing off (next timeout: {}s, {:.1}s budget remaining)",
                next_timeout_secs,
                (QUIESCENCE_MAX_TOTAL_TIMEOUT_SECS as f64) - overall_start.elapsed().as_secs_f64()
            );

            // Brief pause before retry to let workers settle
            std::thread::sleep(Duration::from_millis(QUIESCENCE_RETRY_SETTLE_DELAY_MS));
        }
    }

    /// Single attempt to wait for quiescence with a specific timeout.
    /// Returns Some(true) if achieved, Some(false) if timeout, None if stopped.
    fn wait_for_quiescence_attempt(
        &self,
        stop: &AtomicBool,
        active_workers: &AtomicUsize,
        live_workers: &AtomicUsize,
        timeout: Duration,
    ) -> Option<bool> {
        let mut iterations = 0u64;
        let start = Instant::now();

        loop {
            if stop.load(Ordering::Acquire) {
                eprintln!("Checkpoint: wait_for_quiescence breaking due to stop");
                return None;
            }

            // Check timeout
            if start.elapsed() > timeout {
                let paused = self.paused_workers.load(Ordering::Acquire);
                let live = live_workers.load(Ordering::Acquire);
                let active = active_workers.load(Ordering::Acquire);
                let unpaused_workers = self.get_unpaused_workers();
                eprintln!(
                    "Checkpoint: TIMEOUT waiting for quiescence after {:.1}s! paused={}/{}, active={}",
                    start.elapsed().as_secs_f64(),
                    paused,
                    live,
                    active
                );
                if !unpaused_workers.is_empty() {
                    eprintln!("Checkpoint: STUCK WORKERS: {:?}", unpaused_workers);
                    for worker_id in &unpaused_workers {
                        eprintln!(
                            "  Worker {}: not at pause point (may be blocked on lock contention)",
                            worker_id
                        );
                    }

                    let worker_numa_nodes = self.get_worker_numa_nodes();
                    if let Some(ref diagnostics) = *self.numa_diagnostics.read() {
                        diagnostics
                            .print_stuck_worker_diagnostics(&unpaused_workers, &worker_numa_nodes);
                    } else {
                        eprintln!("NUMA node assignments for stuck workers:");
                        for &worker_id in &unpaused_workers {
                            if let Some(&node) = worker_numa_nodes.get(worker_id) {
                                eprintln!("  Worker {} -> NUMA node {}", worker_id, node);
                            }
                        }
                    }
                }
                return Some(false);
            }

            let paused = self.paused_workers.load(Ordering::Acquire);
            let live = live_workers.load(Ordering::Acquire);
            let active = active_workers.load(Ordering::Acquire);

            // Debug: log progress every 5 seconds
            iterations += 1;
            if iterations % 5000 == 0 {
                let unpaused_workers = self.get_unpaused_workers();
                eprintln!(
                    "Checkpoint: waiting for quiescence: paused={}/{}, active={}, elapsed={:.1}s, stuck_workers={:?}",
                    paused,
                    live,
                    active,
                    start.elapsed().as_secs_f64(),
                    unpaused_workers
                );
            }

            // All workers have terminated — quiescence is trivially
            // achieved (no workers to pause).
            if live == 0 {
                eprintln!(
                    "Checkpoint: all workers terminated (live=0), quiescence trivially achieved, elapsed={:.1}s",
                    start.elapsed().as_secs_f64()
                );
                return Some(true);
            }

            if paused >= live && active == 0 {
                eprintln!(
                    "Checkpoint: quiescence achieved: paused={}/{}, active={}, elapsed={:.1}s",
                    paused,
                    live,
                    active,
                    start.elapsed().as_secs_f64()
                );
                return Some(true);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    pub(super) fn resume(&self) {
        self.requested.store(false, Ordering::Release);
        // Unpark all registered worker threads — lock-free, no contention.
        let threads = self.worker_threads.read();
        for t in threads.iter() {
            t.unpark();
        }
    }
}

pub(super) trait CheckpointPauseQueue {
    fn set_checkpoint_pause_requested(&self, requested: bool);
    fn checkpoint_is_in_progress(&self) -> bool;
}

impl<T: 'static> CheckpointPauseQueue for WorkStealingQueues<T> {
    fn set_checkpoint_pause_requested(&self, requested: bool) {
        self.set_pause_requested(requested);
    }

    fn checkpoint_is_in_progress(&self) -> bool {
        self.is_checkpoint_in_progress()
    }
}

impl<T> CheckpointPauseQueue for SpillableWorkStealingQueues<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
{
    fn set_checkpoint_pause_requested(&self, requested: bool) {
        self.set_pause_requested(requested);
    }

    fn checkpoint_is_in_progress(&self) -> bool {
        self.is_checkpoint_in_progress()
    }
}

pub(super) fn request_checkpoint_pause<Q: CheckpointPauseQueue>(
    queue: &Q,
    pause: &PauseController,
) {
    // Request the pause before flipping the queue flag so workers that
    // observe pause_requested on the queue can park immediately.
    pause.request_pause();
    std::sync::atomic::fence(Ordering::SeqCst);
    queue.set_checkpoint_pause_requested(true);
}

pub(super) fn pause_worker_after_empty_pop_during_checkpoint<Q: CheckpointPauseQueue>(
    worker_queue: &Q,
    worker_pause: &PauseController,
    worker_stop: &AtomicBool,
    worker_id: usize,
) -> bool {
    if !worker_queue.checkpoint_is_in_progress() {
        return false;
    }

    // Give the checkpoint thread a chance to publish the pause request,
    // then re-enter the production pause point directly from the empty-pop path.
    std::thread::sleep(Duration::from_millis(1));
    worker_pause.worker_pause_point(worker_stop, worker_id);
    true
}

#[cfg(test)]
mod tests {
    use super::{
        PauseController, next_quiescence_timeout_secs,
        pause_worker_after_empty_pop_during_checkpoint, request_checkpoint_pause,
    };
    use crate::storage::work_stealing_queues::WorkStealingQueues;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    #[test]
    fn quiescence_schedule_respects_total_budget() {
        assert_eq!(next_quiescence_timeout_secs(0, Duration::ZERO), Some(60));
        assert_eq!(
            next_quiescence_timeout_secs(1, Duration::from_secs(60)),
            Some(120)
        );
        assert_eq!(
            next_quiescence_timeout_secs(2, Duration::from_secs(180)),
            Some(120)
        );
        assert_eq!(
            next_quiescence_timeout_secs(3, Duration::from_secs(180)),
            None
        );
    }

    #[test]
    fn quiescence_schedule_clamps_to_remaining_budget() {
        assert_eq!(
            next_quiescence_timeout_secs(2, Duration::from_secs(250)),
            Some(50)
        );
        assert_eq!(
            next_quiescence_timeout_secs(0, Duration::from_secs(300)),
            None
        );
    }

    #[test]
    fn worker_pauses_when_checkpoint_is_requested_between_pause_point_and_pop() {
        let pause = Arc::new(PauseController::default());
        pause.init_worker_tracking(1, &[0]);
        let stop = Arc::new(AtomicBool::new(false));
        let (queue, mut worker_states) = WorkStealingQueues::<u64>::new(1, vec![0]);
        queue.set_checkpoint_in_progress(true);
        let worker_state = worker_states.pop().expect("missing worker state");

        let (after_top_pause_tx, after_top_pause_rx) = mpsc::channel();
        let (checkpoint_requested_tx, checkpoint_requested_rx) = mpsc::channel();

        let worker = {
            let pause = Arc::clone(&pause);
            let stop = Arc::clone(&stop);
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || {
                let mut worker_state = worker_state;

                // Simulate the worker passing the normal top-of-loop pause point
                // just before the checkpoint thread requests quiescence.
                pause.worker_pause_point(&stop, 0);
                after_top_pause_tx.send(()).unwrap();
                checkpoint_requested_rx.recv().unwrap();

                assert!(queue.pop_for_worker(&mut worker_state).is_none());
                assert!(pause_worker_after_empty_pop_during_checkpoint(
                    queue.as_ref(),
                    pause.as_ref(),
                    &stop,
                    0,
                ));
            })
        };

        after_top_pause_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker never reached the pre-checkpoint pause point");

        request_checkpoint_pause(queue.as_ref(), pause.as_ref());
        assert!(pause.requested.load(Ordering::Acquire));
        assert!(queue.is_pause_requested());
        checkpoint_requested_tx.send(()).unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        let mut paused = false;
        while Instant::now() < deadline {
            if pause.paused_workers.load(Ordering::Acquire) == 1 {
                paused = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        if !paused {
            stop.store(true, Ordering::Release);
            pause.resume();
            let _ = worker.join();
            panic!("worker did not enter the checkpoint pause path");
        }

        pause.resume();
        worker.join().unwrap();
        assert_eq!(pause.paused_workers.load(Ordering::Acquire), 0);
    }

    #[test]
    fn request_checkpoint_pause_sets_controller_and_queue_flags() {
        let pause = PauseController::default();
        let (queue, _) = WorkStealingQueues::<u64>::new(1, vec![0]);

        request_checkpoint_pause(queue.as_ref(), &pause);

        assert!(
            pause.requested.load(Ordering::Acquire),
            "pause controller should become visible before workers observe the queue pause flag"
        );
        assert!(queue.is_pause_requested());
    }

    #[test]
    fn pause_worker_after_empty_pop_is_noop_without_checkpoint() {
        let pause = PauseController::default();
        pause.init_worker_tracking(1, &[0]);
        let stop = AtomicBool::new(false);
        let (queue, _) = WorkStealingQueues::<u64>::new(1, vec![0]);

        assert!(!pause_worker_after_empty_pop_during_checkpoint(
            queue.as_ref(),
            &pause,
            &stop,
            0,
        ));
        assert_eq!(pause.paused_workers.load(Ordering::Acquire), 0);
    }
}
