//! Shutdown orchestration — the 13-step end-of-`run_model` phase.
//!
//! This module bundles the textually fixed sequence of teardown actions
//! that runs after the worker spawn loop completes, plus the final
//! `RunOutcome` build. Every step is a verbatim move from
//! `runtime.rs::run_model`; the constraints documented in
//! `docs/runtime-refactor-plan.md` "Concurrency-coupling analysis" §7
//! pin the ordering of these steps. Reordering any of them risks
//! crash-recovery / quiescence / liveness regressions.
//!
//! ### Step ordering (must not change)
//!
//! 1. Worker join (with crash recovery — N-1 workers tolerated).
//! 2. Init producer join (T5.4) — happens **after** worker join because
//!    workers' empty-pop branch reads `init_producing` whose Drop guard
//!    is owned by the producer thread.
//! 3. `stop.store(true)` + progress thread join.
//! 4. Auto-tuner join.
//! 5. `queue.finish()` — signals queue completion to any remaining
//!    drainers / loaders.
//! 6. `mem_monitor_stop.store(true)` + unpark + monitor join.
//! 7. `checkpoint_thread_stop.store(true)` + `pause.resume()` +
//!    checkpoint thread join.
//! 8. Exit checkpoint write (when `config.checkpoint_on_exit`).
//! 9. `fp_store.flush()` (best-effort).
//! 10. Compression-stats summary print (when present).
//! 11. Worker error drain (first error becomes the run-level error).
//! 12. Violation collection from the bounded channel.
//! 13. Liveness post-processing (T10) — reads `labeled_transitions`
//!     populated by workers, must run after worker join (step 1).
//!
//! Steps 1-12 are fixed; reordering breaks crash-recovery semantics
//! or risks deadlock. Step 13 must come after step 11 (errors) but
//! before the final `RunOutcome` build because `violation` may be
//! reassigned by liveness.

use crate::autotune::AutoTuner;
use crate::model::{LabeledTransition, Model};
use crate::storage::fingerprint_store::FingerprintStats as OldFingerprintStats;
use crate::storage::queue::QueueStats;
use crate::storage::spillable_work_stealing::SpillableWorkStealingQueues;
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use crate::system::WorkerPlan;
use anyhow::{Result, anyhow};
use crossbeam_channel::Receiver;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::pause::PauseController;
use super::stats::AtomicRunStats;
use super::{EngineConfig, RunOutcome, RunStats, Violation, liveness};

/// Bundle of every handle, Arc, and config slice the shutdown phase
/// needs. Built by `run_model` immediately after the worker spawn loop;
/// consumed by `orchestrate`.
///
/// Fields are grouped by step (see module-level docs) to make the
/// per-step provenance obvious to a future reader.
pub(super) struct ShutdownContext<'a, M: Model> {
    // Step 1: workers
    pub workers: Vec<JoinHandle<()>>,
    pub stop: Arc<AtomicBool>,

    // Step 2: init producer (T5.4)
    pub init_producer: Option<JoinHandle<Result<u64>>>,

    // Step 3: progress reporting
    pub progress_thread: JoinHandle<()>,

    // Step 4: auto-tuner
    pub auto_tuner: Option<AutoTuner>,

    // Step 5: queue finish
    pub queue: Arc<SpillableWorkStealingQueues<M::State>>,

    // Step 6: memory monitor (file-backed FP store path)
    pub mem_monitor_stop: Arc<AtomicBool>,
    pub mem_monitor_thread_handle: Arc<parking_lot::RwLock<Option<std::thread::Thread>>>,
    pub mem_monitor_thread: Option<JoinHandle<()>>,

    // Step 7: checkpoint thread
    pub checkpoint_thread_stop: Arc<AtomicBool>,
    pub pause: Arc<PauseController>,
    pub checkpoint_thread: Option<JoinHandle<()>>,

    // Step 8: exit-checkpoint manifest fields
    pub fp_store: Arc<UnifiedFingerprintStore>,
    pub run_stats: Arc<AtomicRunStats>,
    pub config: &'a EngineConfig,
    pub worker_plan: &'a WorkerPlan,
    pub effective_memory_max: Option<u64>,
    pub resumed_from_checkpoint: bool,
    pub started_at: Instant,

    // Steps 11 + 12: error / violation drain
    pub error_rx: Receiver<String>,
    pub violation_rx: Receiver<Violation<M::State>>,

    // Step 13: liveness post-processing
    pub model: Arc<M>,
    pub labeled_transitions: Option<Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>>,
    /// T10.2 stage 4 — when set to `true` by the DFS worker
    /// (`runtime/dfs_worker.rs`), shutdown skips the
    /// `liveness::run_post_processing` Tarjan pass because the DFS worker
    /// has already produced the in-band fairness verdict and emitted any
    /// resulting violation through `violation_tx`. When `false` (BFS
    /// path), post-processing runs as before.
    pub dfs_inband_verdict_done: Arc<AtomicBool>,
}

impl<'a, M: Model> ShutdownContext<'a, M> {
    /// Run the full 13-step shutdown sequence and build the final
    /// `RunOutcome`.
    ///
    /// `auto_tuner`, `mem_monitor_thread`, `checkpoint_thread`, and
    /// `init_producer` are all `Option<_>` because each may be absent in
    /// some configurations (auto-tune off, file-backed FP off, etc.).
    pub(super) fn orchestrate(mut self) -> Result<RunOutcome<M::State>> {
        // ---- Step 1: worker join with crash recovery ----------------
        //
        // Continue with N-1 workers if any panic, propagate "all crashed"
        // as a hard error. The work-stealing queue redistributes work
        // from a crashed worker so siblings can finish.
        let mut crashed_workers = 0usize;
        let total_workers = self.workers.len();
        for (worker_id, worker) in self.workers.into_iter().enumerate() {
            if worker.join().is_err() {
                crashed_workers += 1;
                eprintln!(
                    "Warning: worker {} crashed ({}/{} workers still running)",
                    worker_id,
                    total_workers - crashed_workers,
                    total_workers
                );
                if crashed_workers == total_workers {
                    self.stop.store(true, Ordering::Release);
                    return Err(anyhow!("all {} worker threads panicked", total_workers));
                }
            }
        }
        if crashed_workers > 0 {
            eprintln!(
                "Run completed with {}/{} workers crashed (recovered gracefully)",
                crashed_workers, total_workers
            );
        }

        // ---- Step 2: init producer join (T5.4) ----------------------
        //
        // The Drop guard inside the producer thread already cleared
        // `init_producing`, so workers exited above. Producer errors are
        // logged but do not fail the run (best-effort post-exploration).
        if let Some(handle) = self.init_producer {
            match handle.join() {
                Ok(Ok(_distinct)) => {}
                Ok(Err(e)) => {
                    eprintln!("Warning: Init producer reported error: {e}");
                }
                Err(panic) => {
                    eprintln!("Warning: Init producer thread panicked: {panic:?}");
                }
            }
        }

        // ---- Step 3: stop progress reporting ------------------------
        self.stop.store(true, Ordering::Release);
        if let Err(panic) = self.progress_thread.join() {
            eprintln!("warning: progress thread panicked during shutdown: {panic:?}");
        }

        // ---- Step 4: auto-tuner ------------------------------------
        if let Some(tuner) = self.auto_tuner.take() {
            tuner.join();
        }

        // ---- Step 5: signal queue completion ------------------------
        self.queue.finish();

        // ---- Step 6: memory monitor ---------------------------------
        //
        // Unpark first to skip the up-to-5s park_timeout that the monitor
        // thread sleeps in (avoids a 5s shutdown stall on every run).
        self.mem_monitor_stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.mem_monitor_thread_handle.read().as_ref() {
            thread.unpark();
        }
        if let Some(handle) = self.mem_monitor_thread
            && let Err(panic) = handle.join()
        {
            eprintln!("warning: memory monitor thread panicked during shutdown: {panic:?}");
        }

        // ---- Step 7: checkpoint thread ------------------------------
        //
        // resume() is required: if a checkpoint was mid-pause when stop
        // fired, workers are parked on the pause controller. Resuming
        // unparks them so they can observe stop and exit.
        self.checkpoint_thread_stop.store(true, Ordering::Release);
        self.pause.resume();
        if let Some(handle) = self.checkpoint_thread
            && handle.join().is_err()
        {
            return Err(anyhow!("checkpoint thread panicked"));
        }

        // ---- Step 8: exit-checkpoint write --------------------------
        if self.config.checkpoint_on_exit {
            let checkpoint_path = self
                .config
                .work_dir
                .join("checkpoints")
                .join("latest.json");
            let (
                states_generated,
                states_processed,
                states_distinct,
                duplicates,
                enqueued,
                checkpoints,
            ) = self.run_stats.snapshot();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let manifest = super::checkpoint::CheckpointManifest {
                version: 1,
                model: std::any::type_name::<M>().to_string(),
                created_unix_secs: now,
                duration_millis: self.started_at.elapsed().as_millis() as u64,
                states_generated,
                states_processed,
                states_distinct,
                duplicates,
                enqueued,
                checkpoints,
                configured_workers: self.config.workers,
                actual_workers: self.worker_plan.worker_count,
                allowed_cpu_count: self.worker_plan.allowed_cpus.len(),
                cgroup_cpuset_cores: self.worker_plan.cgroup_cpuset_cores,
                cgroup_quota_cores: self.worker_plan.cgroup_quota_cores,
                numa_nodes_used: self.worker_plan.numa_nodes_used,
                effective_memory_max_bytes: self.effective_memory_max,
                resumed_from_checkpoint: self.resumed_from_checkpoint,
                queue: self.queue.stats(),
                fingerprints: {
                    let stats = self.fp_store.stats();
                    OldFingerprintStats {
                        checks: stats.checks,
                        hits: stats.hits,
                        inserts: stats.inserts,
                        batch_calls: stats.batch_calls,
                        batch_items: stats.batch_items,
                    }
                },
            };
            if let Err(e) =
                super::checkpoint::write_validated_rolling_checkpoint(&checkpoint_path, &manifest)
            {
                eprintln!(
                    "Warning: failed to write/validate checkpoint manifest: {}",
                    e
                );
            } else {
                eprintln!(
                    "Exit checkpoint written and validated [keeping last {}]",
                    super::checkpoint::MAX_CHECKPOINT_FILES
                );
            }
        }

        // ---- Step 9: best-effort FP-store flush ---------------------
        //
        // The exit-checkpoint manifest above already wrote the canonical
        // verdict; this flush is purely about persistence so operators
        // notice store-level failures, not about the run's verdict.
        if let Err(e) = self.fp_store.flush() {
            eprintln!("warning: fingerprint store flush at shutdown failed: {e}");
        }

        // ---- Step 10: compression stats summary ---------------------
        if let Some(snap) = self.queue.compression_stats() {
            if snap.segments_compressed > 0 || snap.segments_decompressed > 0 {
                eprintln!(
                    "Queue compression: ratio={:.2}x compressed_bytes={} uncompressed_bytes={} segments_in/out={}/{} compress_ms={} decompress_ms={} ring_full_rejects={}",
                    snap.ratio(),
                    snap.bytes_compressed,
                    snap.bytes_uncompressed,
                    snap.segments_compressed,
                    snap.segments_decompressed,
                    snap.compress_time_us / 1000,
                    snap.decompress_time_us / 1000,
                    snap.push_rejected_full,
                );
            }
        }

        // ---- Step 11: worker error drain ----------------------------
        //
        // Surface only the first error as the run-level Err; log
        // additional errors so they show up in the run log.
        if let Ok(first_err) = self.error_rx.try_recv() {
            while let Ok(more) = self.error_rx.try_recv() {
                eprintln!("Additional worker error: {more}");
            }
            return Err(anyhow!(first_err));
        }

        // ---- Step 12: violation collection --------------------------
        let mut all_violations: Vec<Violation<M::State>> = Vec::new();
        while let Ok(v) = self.violation_rx.try_recv() {
            all_violations.push(v);
        }
        let mut violation = if all_violations.is_empty() {
            None
        } else {
            Some(all_violations.remove(0))
        };

        // ---- Step 13: liveness post-processing (T10) ----------------
        //
        // T10.2 stage 4 gate: when the DFS exploration path was used
        // (`--liveness-streaming-exploration` + fairness constraints +
        // single-node), the DFS worker has already produced the in-band
        // fairness verdict and emitted any violation through
        // `violation_tx` (collected at step 12 above). Skip the Tarjan
        // post-processing pass entirely — re-running it on the (None)
        // labeled-transitions map would emit "no transitions recorded"
        // and on a populated map would double-count.
        let dfs_inband_done = self.dfs_inband_verdict_done.load(Ordering::Acquire);
        if dfs_inband_done {
            eprintln!(
                "Liveness post-processing: skipped — DFS worker already produced \
                 in-band fairness verdict (T10.2 stage 4)"
            );
        } else if let Some(liveness_violation) = liveness::run_post_processing(
            violation.is_some(),
            &self.model,
            self.labeled_transitions.as_ref(),
            self.config,
        ) {
            violation = Some(liveness_violation);
        }

        let (
            states_generated,
            states_processed,
            states_distinct,
            duplicates,
            enqueued,
            checkpoints,
        ) = self.run_stats.snapshot();

        Ok(RunOutcome {
            stats: RunStats {
                duration: self.started_at.elapsed(),
                states_generated,
                states_processed,
                states_distinct,
                duplicates,
                enqueued,
                checkpoints,
                configured_workers: self.config.workers,
                actual_workers: self.worker_plan.worker_count,
                allowed_cpu_count: self.worker_plan.allowed_cpus.len(),
                cgroup_cpuset_cores: self.worker_plan.cgroup_cpuset_cores,
                cgroup_quota_cores: self.worker_plan.cgroup_quota_cores,
                numa_nodes_used: self.worker_plan.numa_nodes_used,
                effective_memory_max_bytes: self.effective_memory_max,
                resumed_from_checkpoint: self.resumed_from_checkpoint,
                queue: {
                    let ws_stats = self.queue.stats();
                    QueueStats {
                        pushed: ws_stats.pushed,
                        popped: ws_stats.popped,
                        spilled_items: 0,
                        spill_batches: 0,
                        loaded_segments: 0,
                        loaded_items: 0,
                        max_inmem_len: 0,
                        spill_lost_permanently: 0,
                    }
                },
                fingerprints: self.fp_store.stats(),
            },
            violation,
            violations: all_violations,
        })
    }
}
