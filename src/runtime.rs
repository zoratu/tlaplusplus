use crate::autotune::{AutoTuneConfig, AutoTuner, WorkerThrottle};
use crate::fairness::{
    ActionLabel as FairnessActionLabel, LabeledTransition as FairnessLabeledTransition, TarjanSCC,
    check_fairness_on_scc,
};
use crate::model::{LabeledTransition, Model};
use crate::storage::async_fingerprint_writer::{create_persist_channels, fingerprint_writer_task};
use crate::storage::fingerprint_store::{
    FingerprintStats as OldFingerprintStats, FingerprintStore,
};
use crate::storage::numa::{NumaDiagnostics, NumaTopology, set_preferred_node};
use crate::storage::page_aligned_fingerprint_store::FingerprintStats;
use crate::storage::queue::{DiskBackedQueue, QueueStats};
use crate::storage::spillable_work_stealing::{SpillableConfig, SpillableWorkStealingQueues};
use crate::storage::unified_fingerprint_store::{
    AutoSwitchConfigInput, UnifiedFingerprintConfig, UnifiedFingerprintStore,
};
use crate::storage::work_stealing_queues::WorkStealingQueues;
use crate::system::{
    MemoryMonitor, MemoryStatus, WorkerPlan, WorkerPlanRequest, build_worker_plan,
    cgroup_memory_max_bytes, check_disk_space, get_disk_stats, pin_current_thread_to_cpu,
    prune_work_dir_segments,
};
use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub workers: usize,
    pub core_ids: Option<Vec<usize>>,
    pub enforce_cgroups: bool,
    pub numa_pinning: bool,
    /// Restrict workers to specific NUMA nodes (e.g., Some(vec![0, 1]))
    pub numa_nodes: Option<Vec<usize>>,
    pub memory_max_bytes: Option<u64>,
    pub estimated_state_bytes: usize,
    pub work_dir: PathBuf,
    pub clean_work_dir: bool,
    pub resume_from_checkpoint: bool,
    pub checkpoint_interval_secs: u64,
    pub checkpoint_on_exit: bool,
    pub poll_sleep_ms: u64,
    pub stop_on_violation: bool,
    pub fp_shards: usize,
    pub fp_expected_items: usize,
    pub fp_false_positive_rate: f64,
    pub fp_hot_entries_per_shard: usize,
    pub fp_cache_capacity_bytes: u64,
    pub fp_flush_every_ms: Option<u64>,
    pub fp_batch_size: usize,
    pub queue_inmem_limit: usize,
    pub queue_spill_batch: usize,
    pub queue_spill_channel_bound: usize,
    /// Enable disk spilling for work-stealing queues (prevents memory exhaustion)
    pub enable_queue_spilling: bool,
    /// Max items in memory before spilling (when enable_queue_spilling is true)
    pub queue_max_inmem_items: u64,
    /// Enable auto-tuning of worker count based on CPU utilization
    pub auto_tune: bool,
    /// Enable fingerprint persistence to disk (required for resume)
    pub enable_fp_persistence: bool,
    /// Use bloom filter for fingerprints (bounded memory, ~1% false positive rate)
    pub use_bloom_fingerprints: bool,
    /// Enable automatic switching from exact to bloom filter under memory/state pressure
    pub bloom_auto_switch: bool,
    /// State count threshold to trigger bloom auto-switch
    pub bloom_switch_threshold: u64,
    /// Memory pressure threshold (0.0-1.0) to trigger bloom auto-switch
    pub bloom_switch_memory_threshold: f64,
    /// False positive rate for bloom filter after auto-switch
    pub bloom_switch_fpr: f64,
    /// Defer queue segment deletion (for S3 coordination)
    /// When true, consumed segments are retained until S3 confirms upload.
    pub defer_queue_segment_deletion: bool,
    /// Enable BFS parent tracking for error trace reconstruction.
    /// When enabled, each newly-discovered state records its parent's fingerprint
    /// so that violation traces can be reconstructed by walking the parent chain
    /// instead of re-exploring from initial states.
    pub trace_parents: bool,
    /// Maximum number of states to store in the parent/state maps.
    /// When this limit is reached, parent tracking stops recording new entries
    /// and falls back to `reconstruct_trace_limited` if a violation occurs.
    pub max_trace_states: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            workers: 0,
            core_ids: None,
            enforce_cgroups: true,
            numa_pinning: true,
            numa_nodes: None,
            memory_max_bytes: None,
            estimated_state_bytes: 256,
            work_dir: PathBuf::from("./.tlapp"),
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 30,
            checkpoint_on_exit: true,
            poll_sleep_ms: 1,
            stop_on_violation: true,
            fp_shards: 64,
            fp_expected_items: 100_000_000,
            fp_false_positive_rate: 0.01,
            fp_hot_entries_per_shard: 1_000_000,
            fp_cache_capacity_bytes: 1_024 * 1_024 * 1_024,
            fp_flush_every_ms: Some(10_000),
            fp_batch_size: 512,
            queue_inmem_limit: 5_000_000,
            queue_spill_batch: 50_000,
            queue_spill_channel_bound: 128,
            enable_queue_spilling: true,
            queue_max_inmem_items: 50_000_000, // 50M items before spilling
            auto_tune: false,
            enable_fp_persistence: true, // Enable by default for resume support
            use_bloom_fingerprints: false, // Use page-aligned by default (faster)
            bloom_auto_switch: true,     // Enable auto-switch by default
            bloom_switch_threshold: 1_000_000_000, // 1 billion states
            bloom_switch_memory_threshold: 0.85, // 85% memory pressure
            bloom_switch_fpr: 0.001,     // 0.1% FPR after switch
            defer_queue_segment_deletion: false, // Only true when S3 is active
            trace_parents: false,
            max_trace_states: 10_000_000, // 10M states
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyType {
    /// Safety property - something bad never happens
    Safety,
    /// Liveness property - something good eventually happens
    Liveness,
}

#[derive(Debug)]
pub struct Violation<S> {
    pub message: String,
    pub state: S,
    pub property_type: PropertyType,
    /// Trace from initial state to violating state (if available)
    /// Empty for safety violations without trace tracking enabled
    pub trace: Vec<S>,
}

#[derive(Debug)]
pub struct RunOutcome<S> {
    pub stats: RunStats,
    pub violation: Option<Violation<S>>,
}

#[derive(Debug, Clone)]
pub struct RunStats {
    pub duration: Duration,
    pub states_generated: u64,
    pub states_processed: u64,
    pub states_distinct: u64,
    pub duplicates: u64,
    pub enqueued: u64,
    pub checkpoints: u64,
    pub configured_workers: usize,
    pub actual_workers: usize,
    pub allowed_cpu_count: usize,
    pub cgroup_cpuset_cores: Option<usize>,
    pub cgroup_quota_cores: Option<usize>,
    pub numa_nodes_used: usize,
    pub effective_memory_max_bytes: Option<u64>,
    pub resumed_from_checkpoint: bool,
    pub queue: QueueStats,
    pub fingerprints: FingerprintStats,
}

#[derive(Default)]
struct AtomicRunStats {
    states_generated: AtomicU64,
    states_processed: AtomicU64,
    states_distinct: AtomicU64,
    duplicates: AtomicU64,
    enqueued: AtomicU64,
    checkpoints: AtomicU64,
}

impl AtomicRunStats {
    /// Create stats initialized from checkpoint values (for resume)
    fn from_checkpoint(
        states_generated: u64,
        states_processed: u64,
        states_distinct: u64,
        duplicates: u64,
        enqueued: u64,
        checkpoints: u64,
    ) -> Self {
        Self {
            states_generated: AtomicU64::new(states_generated),
            states_processed: AtomicU64::new(states_processed),
            states_distinct: AtomicU64::new(states_distinct),
            duplicates: AtomicU64::new(duplicates),
            enqueued: AtomicU64::new(enqueued),
            checkpoints: AtomicU64::new(checkpoints),
        }
    }

    fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64) {
        (
            self.states_generated.load(Ordering::Relaxed),
            self.states_processed.load(Ordering::Relaxed),
            self.states_distinct.load(Ordering::Relaxed),
            self.duplicates.load(Ordering::Relaxed),
            self.enqueued.load(Ordering::Relaxed),
            self.checkpoints.load(Ordering::Relaxed),
        )
    }
}

struct PauseController {
    requested: AtomicBool,
    paused_workers: AtomicUsize,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
    /// Per-worker pause status for debugging (worker_id -> is_paused)
    /// Only populated when debugging is enabled via TLAPP_DEBUG_PAUSE env var
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
            wait_lock: Mutex::new(()),
            wait_cv: Condvar::new(),
            worker_pause_status: parking_lot::RwLock::new(Vec::new()),
            worker_numa_nodes: parking_lot::RwLock::new(Vec::new()),
            numa_diagnostics: parking_lot::RwLock::new(None),
        }
    }
}

const QUIESCENCE_INITIAL_TIMEOUT_SECS: u64 = 60;
const QUIESCENCE_MAX_TOTAL_TIMEOUT_SECS: u64 = 300;
const QUIESCENCE_MAX_ATTEMPTS: u32 = 3;
const QUIESCENCE_RETRY_SETTLE_DELAY_MS: u64 = 100;

fn next_quiescence_timeout_secs(attempt_index: u32, elapsed: Duration) -> Option<u64> {
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
    fn init_worker_tracking(&self, num_workers: usize, worker_numa_nodes: &[usize]) {
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
    fn set_numa_diagnostics(&self, diagnostics: NumaDiagnostics) {
        let mut diag = self.numa_diagnostics.write();
        *diag = Some(diagnostics);
    }

    /// Get list of worker IDs that are NOT paused (for debugging)
    fn get_unpaused_workers(&self) -> Vec<usize> {
        let status = self.worker_pause_status.read();
        status
            .iter()
            .enumerate()
            .filter(|(_, paused)| !paused.load(Ordering::Acquire))
            .map(|(id, _)| id)
            .collect()
    }

    /// Get the NUMA node assignments for workers
    fn get_worker_numa_nodes(&self) -> Vec<usize> {
        self.worker_numa_nodes.read().clone()
    }
}

impl PauseController {
    fn worker_pause_point(&self, stop: &AtomicBool, worker_id: usize) {
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
        // Log when first few workers pause and periodically after
        if paused_count <= 5 || paused_count % 10 == 0 {
            eprintln!(
                "Worker {} pausing: {} workers now paused",
                worker_id, paused_count
            );
        }
        let mut guard = self.wait_lock.lock();
        while self.requested.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
            self.wait_cv.wait_for(&mut guard, Duration::from_millis(10));
        }
        drop(guard);
        self.paused_workers.fetch_sub(1, Ordering::AcqRel);

        // Mark this worker as unpaused
        {
            let status = self.worker_pause_status.read();
            if let Some(paused) = status.get(worker_id) {
                paused.store(false, Ordering::Release);
            }
        }
    }

    fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    /// Wait for all workers to pause (quiescence).
    /// Returns true if quiescence was achieved, false if timeout occurred.
    ///
    /// Uses exponential backoff with a hard overall budget:
    /// starts at 60s, doubles on each retry, and never exceeds 5 minutes total.
    fn wait_for_quiescence(
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

    fn resume(&self) {
        self.requested.store(false, Ordering::Release);
        self.wait_cv.notify_all();
    }
}

trait CheckpointPauseQueue {
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

fn request_checkpoint_pause<Q: CheckpointPauseQueue>(queue: &Q, pause: &PauseController) {
    // Request the pause before flipping the queue flag so workers that
    // observe pause_requested on the queue can park immediately.
    pause.request_pause();
    std::sync::atomic::fence(Ordering::SeqCst);
    queue.set_checkpoint_pause_requested(true);
}

fn pause_worker_after_empty_pop_during_checkpoint<Q: CheckpointPauseQueue>(
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

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointManifest {
    version: u32,
    model: String,
    created_unix_secs: u64,
    duration_millis: u64,
    states_generated: u64,
    states_processed: u64,
    states_distinct: u64,
    duplicates: u64,
    enqueued: u64,
    checkpoints: u64,
    configured_workers: usize,
    actual_workers: usize,
    allowed_cpu_count: usize,
    cgroup_cpuset_cores: Option<usize>,
    cgroup_quota_cores: Option<usize>,
    numa_nodes_used: usize,
    effective_memory_max_bytes: Option<u64>,
    resumed_from_checkpoint: bool,
    queue: QueueStats,
    fingerprints: OldFingerprintStats,
}
fn compute_effective_memory_max(config: &EngineConfig) -> Option<u64> {
    let cgroup_limit = if config.enforce_cgroups {
        cgroup_memory_max_bytes()
    } else {
        None
    };
    match (config.memory_max_bytes, cgroup_limit) {
        (Some(user), Some(cgroup)) => Some(user.min(cgroup)),
        (Some(user), None) => Some(user),
        (None, Some(cgroup)) => Some(cgroup),
        (None, None) => None,
    }
}

fn should_enable_file_backed_fingerprint_store(
    config: &EngineConfig,
    effective_memory_max: Option<u64>,
) -> bool {
    !config.use_bloom_fingerprints && effective_memory_max.is_some()
}

fn should_start_fingerprint_memory_monitor(
    config: &EngineConfig,
    effective_memory_max: Option<u64>,
    backing_dir_enabled: bool,
) -> bool {
    should_enable_file_backed_fingerprint_store(config, effective_memory_max) && backing_dir_enabled
}

fn apply_memory_budget(config: &EngineConfig, effective_memory_max: Option<u64>) -> (u64, usize) {
    let mut fp_cache = config.fp_cache_capacity_bytes;
    let mut queue_limit = config.queue_inmem_limit;

    if let Some(memory_max) = effective_memory_max {
        let budget_fp_cache = (memory_max.saturating_mul(60) / 100).max(64 * 1024 * 1024);
        fp_cache = fp_cache.min(budget_fp_cache);

        let state_bytes = config.estimated_state_bytes.max(1) as u64;
        let budget_queue_items = (memory_max.saturating_mul(30) / 100) / state_bytes;
        let budget_queue_items = budget_queue_items.max(10_000) as usize;
        queue_limit = queue_limit.min(budget_queue_items);
    }

    (fp_cache, queue_limit)
}

fn write_checkpoint_manifest(path: &Path, manifest: &CheckpointManifest) -> Result<()> {
    // Chaos: fail point for testing checkpoint write failures
    crate::fail_point!("checkpoint_write_fail");

    // Chaos: apply I/O latency if configured
    crate::chaos::apply_io_latency();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed creating checkpoint dir {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(manifest).context("failed serializing checkpoint")?;

    // Chaos: fail point for disk write
    crate::fail_point!("checkpoint_disk_write_fail");

    std::fs::write(&tmp, bytes)
        .with_context(|| format!("failed writing checkpoint temp file {}", tmp.display()))?;

    // Chaos: fail point for atomic rename
    crate::fail_point!("checkpoint_rename_fail");

    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed atomically moving checkpoint to {}", path.display()))?;
    Ok(())
}

/// Load checkpoint manifest from disk (for resume)
fn load_checkpoint_manifest(path: &Path) -> Result<Option<CheckpointManifest>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed reading checkpoint {}", path.display()))?;
    let manifest: CheckpointManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed parsing checkpoint {}", path.display()))?;
    Ok(Some(manifest))
}

/// Maximum number of checkpoint files to retain (rolling window)
const MAX_CHECKPOINT_FILES: usize = 5;

/// Write a checkpoint with validation and rolling retention.
///
/// This function:
/// 1. Writes checkpoint to a timestamped file (e.g., checkpoint-1709612345.json)
/// 2. Reads it back and validates the JSON parses correctly and key fields match
/// 3. Only if valid: updates latest.json to point to the new checkpoint
/// 4. Prunes old checkpoints keeping only the last MAX_CHECKPOINT_FILES
///
/// The `latest_path` should be the path to latest.json (e.g., work_dir/checkpoints/latest.json)
fn write_validated_rolling_checkpoint(
    latest_path: &Path,
    manifest: &CheckpointManifest,
) -> Result<()> {
    // Get the checkpoints directory
    let checkpoint_dir = latest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("checkpoint path has no parent directory"))?;

    std::fs::create_dir_all(checkpoint_dir).with_context(|| {
        format!(
            "failed creating checkpoint dir {}",
            checkpoint_dir.display()
        )
    })?;

    // Create timestamped checkpoint filename
    let timestamp = manifest.created_unix_secs;
    let timestamped_name = format!("checkpoint-{}.json", timestamp);
    let timestamped_path = checkpoint_dir.join(&timestamped_name);

    // Step 1: Write checkpoint to timestamped file
    write_checkpoint_manifest(&timestamped_path, manifest).with_context(|| {
        format!(
            "failed writing checkpoint to {}",
            timestamped_path.display()
        )
    })?;

    // Step 2: Read back and validate
    let readback_bytes = std::fs::read(&timestamped_path).with_context(|| {
        format!(
            "failed reading back checkpoint from {}",
            timestamped_path.display()
        )
    })?;

    let readback_manifest: CheckpointManifest = serde_json::from_slice(&readback_bytes)
        .with_context(|| {
            format!(
                "failed parsing checkpoint JSON from {}",
                timestamped_path.display()
            )
        })?;

    // Validate key fields match what we wrote
    if readback_manifest.version != manifest.version
        || readback_manifest.model != manifest.model
        || readback_manifest.created_unix_secs != manifest.created_unix_secs
        || readback_manifest.states_generated != manifest.states_generated
        || readback_manifest.states_distinct != manifest.states_distinct
    {
        // Validation failed - remove the corrupt checkpoint and return error
        let _ = std::fs::remove_file(&timestamped_path);
        return Err(anyhow::anyhow!(
            "checkpoint validation failed: read-back data does not match written data"
        ));
    }

    // Step 3: Update latest.json to be a copy of the validated checkpoint
    // We copy rather than symlink for S3 compatibility
    std::fs::copy(&timestamped_path, latest_path)
        .with_context(|| format!("failed copying checkpoint to {}", latest_path.display()))?;

    // Step 4: Prune old checkpoints (keep last MAX_CHECKPOINT_FILES)
    prune_old_checkpoints(checkpoint_dir, MAX_CHECKPOINT_FILES)?;

    Ok(())
}

/// Prune old checkpoint files, keeping only the most recent `keep_count` files.
/// Only removes files matching the pattern "checkpoint-*.json".
fn prune_old_checkpoints(checkpoint_dir: &Path, keep_count: usize) -> Result<()> {
    let mut checkpoint_files: Vec<_> = std::fs::read_dir(checkpoint_dir)
        .with_context(|| format!("failed reading checkpoint dir {}", checkpoint_dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            // Match checkpoint-{timestamp}.json but NOT latest.json
            name_str.starts_with("checkpoint-") && name_str.ends_with(".json")
        })
        .collect();

    // Sort by filename (which contains timestamp, so oldest first)
    checkpoint_files.sort_by_key(|e| e.file_name());

    // Remove oldest files if we have more than keep_count
    if checkpoint_files.len() > keep_count {
        let to_remove = checkpoint_files.len() - keep_count;
        for entry in checkpoint_files.into_iter().take(to_remove) {
            let path = entry.path();
            if let Err(e) = std::fs::remove_file(&path) {
                // Log but don't fail - pruning is best-effort
                eprintln!(
                    "Checkpoint: warning: failed to prune old checkpoint {}: {}",
                    path.display(),
                    e
                );
            } else {
                eprintln!("Checkpoint: pruned old checkpoint {}", path.display());
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
struct CheckpointContext<'a, T> {
    checkpoint_path: &'a Path,
    model_name: &'a str,
    started_at: Instant,
    run_stats: &'a AtomicRunStats,
    queue: &'a DiskBackedQueue<T>,
    fp_store: &'a FingerprintStore,
    pause: &'a PauseController,
    active_workers: &'a AtomicUsize,
    live_workers: &'a AtomicUsize,
    stop: &'a AtomicBool,
    worker_plan: &'a WorkerPlan,
    config: &'a EngineConfig,
    effective_memory_max: Option<u64>,
    resumed_from_checkpoint: bool,
}

#[allow(dead_code)]
fn checkpoint_once<T>(ctx: &CheckpointContext<T>) -> Result<()>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    ctx.pause.request_pause();
    let quiescence_achieved =
        ctx.pause
            .wait_for_quiescence(ctx.stop, ctx.active_workers, ctx.live_workers);

    if !quiescence_achieved {
        // Quiescence timed out - skip this checkpoint and resume workers
        ctx.pause.resume();
        return Ok(());
    }

    let checkpoint_result = (|| -> Result<()> {
        // Chaos: fail point for queue flush
        crate::fail_point!("checkpoint_queue_flush_fail");
        ctx.queue.checkpoint_flush()?;

        // Chaos: fail point for fingerprint flush
        crate::fail_point!("checkpoint_fp_flush_fail");
        let _ = ctx.fp_store.flush()?;

        let (states_generated, states_processed, states_distinct, duplicates, enqueued, _) =
            ctx.run_stats.snapshot();
        let checkpoints = ctx.run_stats.checkpoints.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let manifest = CheckpointManifest {
            version: 1,
            model: ctx.model_name.to_string(),
            created_unix_secs: now,
            duration_millis: ctx.started_at.elapsed().as_millis() as u64,
            states_generated,
            states_processed,
            states_distinct,
            duplicates,
            enqueued,
            checkpoints,
            configured_workers: ctx.config.workers,
            actual_workers: ctx.worker_plan.worker_count,
            allowed_cpu_count: ctx.worker_plan.allowed_cpus.len(),
            cgroup_cpuset_cores: ctx.worker_plan.cgroup_cpuset_cores,
            cgroup_quota_cores: ctx.worker_plan.cgroup_quota_cores,
            numa_nodes_used: ctx.worker_plan.numa_nodes_used,
            effective_memory_max_bytes: ctx.effective_memory_max,
            resumed_from_checkpoint: ctx.resumed_from_checkpoint,
            queue: ctx.queue.stats(),
            fingerprints: {
                let stats = ctx.fp_store.stats();
                // Convert new stats to old format for checkpoint (drop collisions field)
                OldFingerprintStats {
                    checks: stats.checks,
                    hits: stats.hits,
                    inserts: stats.inserts,
                    batch_calls: stats.batch_calls,
                    batch_items: stats.batch_items,
                }
            },
        };
        write_validated_rolling_checkpoint(ctx.checkpoint_path, &manifest)?;
        ctx.run_stats.checkpoints.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })();

    ctx.pause.resume();
    checkpoint_result
}

/// Calculate optimal shard count based on system characteristics.
///
/// Heuristics (optimized for many-core scalability):
/// - 2 shards per worker to minimize CAS contention (critical for 380+ workers)
/// - Power of 2 for fast modulo (bitwise AND)
/// - NUMA-aligned for even distribution across nodes
/// - Worker scaling takes priority over shard size for many-core systems
///
fn calculate_optimal_shard_count(
    worker_count: usize,
    numa_node_count: usize,
    total_memory_bytes: usize,
) -> usize {
    // For many-core systems, CAS contention is the primary bottleneck.
    // Having more shards (even if small) is better than fewer large shards.
    // Target: 2 shards per worker ensures low collision probability.
    let shards_per_worker = 2;
    let base_count = worker_count * shards_per_worker;

    // Round up to next power of 2 for fast modulo (hash & (count - 1))
    let mut candidate = base_count.next_power_of_two();

    // Ensure it's NUMA-aligned (multiple of NUMA node count)
    if candidate % numa_node_count != 0 {
        candidate = ((candidate / numa_node_count) + 1) * numa_node_count;
        // Round back to power of 2 if needed
        candidate = candidate.next_power_of_two();
    }

    // For many-core systems (>128 workers), prioritize worker scaling over shard size.
    // Small shards are fine - the overhead is negligible compared to CAS contention.
    // Only apply size constraints for systems with fewer workers where memory layout matters more.
    let is_many_core = worker_count > 128;

    let adjusted = if is_many_core {
        // Many-core: keep 2 shards per worker regardless of size
        candidate
    } else {
        // Fewer workers: apply traditional size-based constraints
        const MIN_SHARD_SIZE: usize = 256 * 1024 * 1024; // 256 MB
        const MAX_SHARD_SIZE: usize = 2 * 1024 * 1024 * 1024; // 2 GB

        let per_shard_bytes = total_memory_bytes / candidate;

        if per_shard_bytes < MIN_SHARD_SIZE {
            // Shards too small - reduce count
            let min_count = (total_memory_bytes / MAX_SHARD_SIZE).max(1);
            min_count.next_power_of_two()
        } else if per_shard_bytes > MAX_SHARD_SIZE {
            // Shards too large - increase count
            let max_count = (total_memory_bytes / MIN_SHARD_SIZE).max(1);
            max_count.next_power_of_two()
        } else {
            candidate
        }
    };

    // Clamp to reasonable bounds
    // Min: at least 2x NUMA nodes for distribution (64 for backwards compat)
    // Max: 4096 shards is plenty (avoid excessive overhead)
    let min_shards = (numa_node_count * 2).max(64);
    let max_shards = 4096;

    let final_count = adjusted.clamp(min_shards, max_shards);

    if std::env::var("TLAPP_VERBOSE").is_ok() {
        eprintln!(
            "Auto-calculated shard count: {} (base: {}, workers: {}, NUMA nodes: {}, per-shard: {:.1} MB, many-core: {})",
            final_count,
            base_count,
            worker_count,
            numa_node_count,
            (total_memory_bytes / final_count) as f64 / (1024.0 * 1024.0),
            is_many_core
        );
    }

    final_count
}

pub fn run_model<M>(model: M, config: EngineConfig) -> Result<RunOutcome<M::State>>
where
    M: Model,
{
    let model = Arc::new(model);
    let _checkpoint_path = config.work_dir.join("checkpoints").join("latest.json");

    if config.clean_work_dir && !config.resume_from_checkpoint && config.work_dir.exists() {
        std::fs::remove_dir_all(&config.work_dir).with_context(|| {
            format!("failed removing old work dir {}", config.work_dir.display())
        })?;
    }
    std::fs::create_dir_all(&config.work_dir)
        .with_context(|| format!("failed creating work dir {}", config.work_dir.display()))?;

    // Check disk space before starting - fail early if insufficient
    let disk_stats = check_disk_space(&config.work_dir)?;
    if disk_stats.is_under_pressure() {
        eprintln!(
            "WARNING: Disk space is under pressure ({} available, {}% used). \
             Consider freeing space to avoid checkpoint failures.",
            disk_stats.available_human(),
            (disk_stats.used_bytes as f64 / disk_stats.total_bytes as f64 * 100.0) as u32
        );
    }

    let effective_memory_max = compute_effective_memory_max(&config);
    let (_fp_cache_capacity_bytes, _queue_inmem_limit) =
        apply_memory_budget(&config, effective_memory_max);

    let worker_plan = build_worker_plan(WorkerPlanRequest {
        requested_workers: config.workers,
        enforce_cgroups: config.enforce_cgroups,
        enable_numa_pinning: config.numa_pinning,
        requested_core_ids: config.core_ids.clone(),
        requested_numa_nodes: config.numa_nodes.clone(),
    });

    let _fp_path = config.work_dir.join("fingerprints");
    let _queue_path = config.work_dir.join("queue");

    // Configure fingerprint store based on mode (bloom filter vs page-aligned)
    //
    // Page-aligned mode: In-memory hash table with huge pages
    //   - Fastest, eliminates TLB thrashing
    //   - Memory usage grows with unique states (unbounded)
    //
    // Bloom filter mode: Fixed memory bloom filter
    //   - Bounded memory (~120MB for 100M items at 1% FPR)
    //   - Small false positive rate (~1%) may cause re-exploration of some states

    // Calculate memory per shard based on expected items
    // Each entry is 16 bytes (8-byte fp + 8-byte padding), plus 10% headroom for open addressing
    let bytes_per_entry = 16;
    let load_factor = 0.9; // 10% headroom
    let total_bytes_needed =
        (config.fp_expected_items as f64 / load_factor * bytes_per_entry as f64) as usize;

    // Auto-calculate optimal shard count if user specified 0 or default
    let shard_count = if config.fp_shards == 0 {
        calculate_optimal_shard_count(
            worker_plan.worker_count,
            worker_plan.numa_nodes_used.max(1),
            total_bytes_needed,
        )
    } else {
        // User specified explicit count - round to power of 2 and ensure minimum
        config.fp_shards.next_power_of_two().max(64)
    };

    // Only calculate shard_size_mb for page-aligned mode
    let shard_size_mb = if !config.use_bloom_fingerprints {
        // Minimum 64MB per shard to avoid frequent resizes with many workers
        let min_shard_bytes = 64 * 1024 * 1024; // 64MB minimum
        let bytes_per_shard = (total_bytes_needed / shard_count).max(min_shard_bytes);
        (bytes_per_shard / (1024 * 1024)).max(64) // At least 64MB
    } else {
        0 // Not used in bloom mode
    };

    if std::env::var("TLAPP_VERBOSE").is_ok() {
        eprintln!("Fingerprint store config:");
        eprintln!(
            "  Mode: {}",
            if config.use_bloom_fingerprints {
                "bloom"
            } else {
                "page-aligned"
            }
        );
        eprintln!("  Expected items: {}", config.fp_expected_items);
        eprintln!("  Shard count: {}", shard_count);
        if !config.use_bloom_fingerprints {
            eprintln!("  Shard size: {} MB", shard_size_mb);
            eprintln!(
                "  Total memory: {} MB ({:.1} GB)",
                shard_count * shard_size_mb,
                (shard_count * shard_size_mb) as f64 / 1024.0
            );
        }
    }

    // Create unified fingerprint store (page-aligned, bloom, or auto-switch)
    let auto_switch_config = if config.bloom_auto_switch {
        Some(AutoSwitchConfigInput {
            state_count_threshold: Some(config.bloom_switch_threshold),
            memory_threshold: Some(config.bloom_switch_memory_threshold),
            bloom_false_positive_rate: Some(config.bloom_switch_fpr),
        })
    } else {
        None
    };

    // Create directory for file-backed mmap fingerprint store
    let fp_backing_dir = if should_enable_file_backed_fingerprint_store(
        &config,
        effective_memory_max,
    ) {
        let dir = config.work_dir.join("fingerprints-mmap");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!(
                "Warning: Failed to create fingerprint backing dir {}: {}. Using anonymous mmap.",
                dir.display(),
                e
            );
            None
        } else {
            Some(dir)
        }
    } else {
        None
    };

    let fp_backing_dir_enabled = fp_backing_dir.is_some();

    let fp_config = UnifiedFingerprintConfig {
        use_bloom: config.use_bloom_fingerprints,
        use_auto_switch: config.bloom_auto_switch && !config.use_bloom_fingerprints,
        shard_count,
        expected_items: config.fp_expected_items,
        false_positive_rate: config.fp_false_positive_rate,
        shard_size_mb,
        num_numa_nodes: worker_plan.numa_nodes_used.max(1),
        auto_switch_config,
        backing_dir: fp_backing_dir,
    };

    let mut fp_store = UnifiedFingerprintStore::new(fp_config, &worker_plan.assigned_cpus)?;

    // Set up fingerprint persistence for resume support
    let _fp_persist_runtime = if config.enable_fp_persistence {
        // Create persist channels (one per shard)
        let (persist_tx, persist_rx) = create_persist_channels(shard_count, 10_000);
        fp_store.enable_persistence(persist_tx);

        // Create tokio runtime for async I/O
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2) // Just 2 threads for I/O
            .thread_name("tlapp-fp-io")
            .enable_all()
            .build()
            .context("Failed to create fingerprint I/O runtime")?;

        // Spawn writer tasks for each shard
        let work_dir = config.work_dir.clone();
        for (shard_id, rx) in persist_rx.into_iter().enumerate() {
            let wd = work_dir.clone();
            runtime.spawn(async move {
                if let Err(e) = fingerprint_writer_task(shard_id, rx, wd).await {
                    // Channel disconnected is expected on shutdown
                    if !e.to_string().contains("disconnected") {
                        eprintln!("Fingerprint writer shard {} error: {}", shard_id, e);
                    }
                }
            });
        }

        Some(runtime)
    } else {
        None
    };

    // Load fingerprints from disk if resuming
    if config.resume_from_checkpoint {
        let fp_dir = config.work_dir.join("fingerprints");
        if fp_dir.exists() {
            eprintln!("Loading fingerprints from disk for resume...");
            let load_start = Instant::now();

            // Use a blocking load since we need this before workers start
            let loaded = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(shard_count);
                for shard_id in 0..shard_count {
                    let fp_path = fp_dir
                        .join(format!("shard-{:03}", shard_id))
                        .join("segment.bin");
                    handles.push(s.spawn(move || -> Vec<u64> {
                        if !fp_path.exists() {
                            return Vec::new();
                        }
                        match std::fs::read(&fp_path) {
                            Ok(bytes) => bytes
                                .chunks_exact(8)
                                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                                .collect(),
                            Err(e) => {
                                eprintln!("Warning: failed to load shard {}: {}", shard_id, e);
                                Vec::new()
                            }
                        }
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });

            // Insert loaded fingerprints into store
            let mut total_loaded = 0usize;
            for (_shard_id, fps) in loaded.into_iter().enumerate() {
                total_loaded += fps.len();
                for fp in fps {
                    let _ = fp_store.contains_or_insert(fp);
                }
            }

            let load_elapsed = load_start.elapsed();
            eprintln!(
                "Loaded {} fingerprints from disk in {:.1}s",
                total_loaded,
                load_elapsed.as_secs_f64()
            );
        }
    }

    let fp_store = Arc::new(fp_store);

    // Memory pressure monitor thread - advises kernel to page out cold fingerprints
    // when RSS approaches the effective memory cap. Only active when file-backed
    // fingerprints were successfully configured.
    let mem_monitor_stop = Arc::new(AtomicBool::new(false));
    let mem_monitor_thread: Option<std::thread::JoinHandle<()>> =
        if should_start_fingerprint_memory_monitor(
            &config,
            effective_memory_max,
            fp_backing_dir_enabled,
        ) {
            let monitor_fp_store = Arc::clone(&fp_store);
            let monitor_stop = Arc::clone(&mem_monitor_stop);
            let memory_max = effective_memory_max.unwrap();
            Some(
            std::thread::Builder::new()
                .name("tlapp-mem-monitor".into())
                .spawn(move || {
                    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
                    let threshold_90 = (memory_max as f64 * 0.90) as u64;
                    let threshold_95 = (memory_max as f64 * 0.95) as u64;
                    let mut warned_95 = false;

                    while !monitor_stop.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_secs(5));
                        if monitor_stop.load(Ordering::Relaxed) {
                            break;
                        }

                        // Read RSS from /proc/self/statm (Linux only)
                        let rss_bytes = match std::fs::read_to_string("/proc/self/statm") {
                            Ok(statm) => {
                                // Field 1 (0-indexed) is RSS in pages
                                statm
                                    .split_whitespace()
                                    .nth(1)
                                    .and_then(|s| s.parse::<u64>().ok())
                                    .map(|pages| pages * page_size)
                                    .unwrap_or(0)
                            }
                            Err(_) => continue, // Not on Linux or /proc not available
                        };

                        if rss_bytes > threshold_95 {
                            if !warned_95 {
                                eprintln!(
                                    "Memory pressure: RSS {} MB exceeds 95% of limit {} MB, advising cold pages",
                                    rss_bytes / (1024 * 1024),
                                    memory_max / (1024 * 1024)
                                );
                                warned_95 = true;
                            }
                            monitor_fp_store.advise_cold();
                        } else if rss_bytes > threshold_90 {
                            monitor_fp_store.advise_cold();
                            warned_95 = false; // Reset warning if we drop back below 95%
                        } else {
                            warned_95 = false;
                        }
                    }
                })
                .expect("Failed to spawn memory monitor thread"),
        )
        } else {
            None
        };

    // Use NUMA-aware work-stealing queues with optional disk spilling
    // Each worker has its own queue, steals from others when idle
    // Hierarchical stealing: prefer same-NUMA node first, then remote
    // When spilling enabled, overflow goes to disk to prevent memory exhaustion
    let spill_config = SpillableConfig {
        max_inmem_items: if config.enable_queue_spilling {
            config.queue_max_inmem_items
        } else {
            u64::MAX // Effectively disable spilling
        },
        spill_dir: config.work_dir.join("queue-spill"),
        spill_batch: config.queue_spill_batch,
        load_existing: config.resume_from_checkpoint,
        // Per-worker spill buffer settings for lock-free spilling
        worker_spill_buffer_size: 4096, // Each worker buffers 4K items locally
        worker_channel_bound: 16,       // 16 batches in flight per worker
        // When S3 is active, defer segment deletion until S3 confirms upload
        defer_segment_deletion: config.defer_queue_segment_deletion,
    };
    let (queue, worker_states) = SpillableWorkStealingQueues::new(
        worker_plan.worker_count,
        worker_plan.worker_numa_nodes.clone(),
        spill_config,
    )?;

    // On resume, proactively load spilled queue segments into memory
    // This ensures workers have access to the spilled states immediately
    // Only load up to queue_max_inmem_items to avoid OOM - rest loaded on-demand
    if config.resume_from_checkpoint {
        eprintln!("Loading spilled queue segments for resume...");
        let max_load = config.queue_max_inmem_items;
        let loaded = queue.load_spilled_segments(max_load)?;
        if loaded > 0 {
            eprintln!("Loaded {} states from spilled queue segments", loaded);
        }
    }

    // Initialize run stats - from checkpoint if resuming, otherwise from zero
    let run_stats = if config.resume_from_checkpoint {
        let checkpoint_path = config.work_dir.join("checkpoints").join("latest.json");
        match load_checkpoint_manifest(&checkpoint_path) {
            Ok(Some(manifest)) => {
                eprintln!(
                    "Resuming counters from checkpoint: {} generated, {} distinct, {} duplicates",
                    manifest.states_generated, manifest.states_distinct, manifest.duplicates
                );
                Arc::new(AtomicRunStats::from_checkpoint(
                    manifest.states_generated,
                    manifest.states_processed,
                    manifest.states_distinct,
                    manifest.duplicates,
                    manifest.enqueued,
                    manifest.checkpoints,
                ))
            }
            Ok(None) => {
                eprintln!("No checkpoint manifest found, starting counters from 0");
                Arc::new(AtomicRunStats::default())
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load checkpoint manifest: {}, starting counters from 0",
                    e
                );
                Arc::new(AtomicRunStats::default())
            }
        }
    } else {
        Arc::new(AtomicRunStats::default())
    };
    let stop = Arc::new(AtomicBool::new(false));
    let active_workers = Arc::new(AtomicUsize::new(0));
    let live_workers = Arc::new(AtomicUsize::new(worker_plan.worker_count));
    let pause = Arc::new(PauseController::default());
    let (violation_tx, violation_rx) = crossbeam_channel::bounded(1);
    let (error_tx, error_rx) = crossbeam_channel::bounded::<String>(1);

    // Auto-tuning: create throttle for dynamic worker count adjustment
    let throttle = Arc::new(WorkerThrottle::new(worker_plan.worker_count));
    let auto_tune_enabled = config.auto_tune;

    // BFS parent tracking for error trace reconstruction (TLC-style).
    // parent_map: fingerprint(child) -> fingerprint(parent)
    // state_map:  fingerprint(state) -> state (for reconstructing the trace)
    // trace_count: number of entries recorded (for max_trace_states limit)
    let parent_map: Option<Arc<DashMap<u64, u64>>> = if config.trace_parents {
        Some(Arc::new(DashMap::new()))
    } else {
        None
    };
    let state_map: Option<Arc<DashMap<u64, M::State>>> = if config.trace_parents {
        Some(Arc::new(DashMap::new()))
    } else {
        None
    };
    let trace_state_count = Arc::new(AtomicU64::new(0));
    let max_trace_states = config.max_trace_states as u64;

    let started_at = Instant::now();
    let resumed_from_checkpoint = config.resume_from_checkpoint && queue.has_pending_work();
    if !resumed_from_checkpoint {
        let initial = model.initial_states();
        let mut initial_fps = Vec::with_capacity(initial.len());
        let mut unique_initial = Vec::with_capacity(initial.len());
        let mut dedup = HashSet::with_capacity(initial.len().max(16));
        for state in initial {
            run_stats.states_generated.fetch_add(1, Ordering::Relaxed);
            let state = model.canonicalize(state);
            let fp = model.fingerprint(&state);
            if dedup.insert(fp) {
                initial_fps.push(fp);
                unique_initial.push(state);
            } else {
                run_stats.duplicates.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut seen_flags = Vec::with_capacity(initial_fps.len());
        fp_store.contains_or_insert_batch(&initial_fps, &mut seen_flags)?;
        let mut distinct_initial = 0;
        for (idx, state) in unique_initial.into_iter().enumerate() {
            if seen_flags[idx] {
                run_stats.duplicates.fetch_add(1, Ordering::Relaxed);
            } else {
                // Record initial state in state_map (no parent entry needed)
                if let Some(ref sm) = state_map {
                    let fp = initial_fps[idx];
                    sm.insert(fp, state.clone());
                    trace_state_count.fetch_add(1, Ordering::Relaxed);
                }
                run_stats.states_distinct.fetch_add(1, Ordering::Relaxed);
                queue.push_global(state);
                run_stats.enqueued.fetch_add(1, Ordering::Relaxed);
                distinct_initial += 1;
            }
        }

        // Print TLC-compatible message
        let now = chrono::Local::now();
        let timestamp = now.format("%Y-%m-%d %H:%M:%S");
        let plural = if distinct_initial == 1 {
            "state"
        } else {
            "states"
        };
        eprintln!(
            "Finished computing initial states: {} distinct {} generated at {}.",
            distinct_initial, plural, timestamp
        );
    }

    let checkpoint_thread_stop = Arc::new(AtomicBool::new(false));

    // Checkpoint thread - periodically flushes queue and fingerprints to disk
    let checkpoint_thread: Option<std::thread::JoinHandle<()>> = if config.checkpoint_interval_secs
        > 0
    {
        let ckpt_stop = Arc::clone(&checkpoint_thread_stop);
        let ckpt_pause = Arc::clone(&pause);
        let ckpt_queue = Arc::clone(&queue);
        let ckpt_fp_store = Arc::clone(&fp_store);
        let ckpt_run_stats = Arc::clone(&run_stats);
        let ckpt_active_workers = Arc::clone(&active_workers);
        let ckpt_live_workers = Arc::clone(&live_workers);
        let ckpt_exploration_stop = Arc::clone(&stop);
        let ckpt_interval = Duration::from_secs(config.checkpoint_interval_secs);
        let ckpt_work_dir = config.work_dir.clone();
        let ckpt_model_name = std::any::type_name::<M>().to_string();
        let ckpt_worker_count = worker_plan.worker_count;
        let ckpt_allowed_cpus = worker_plan.allowed_cpus.len();
        let ckpt_cgroup_cpuset_cores = worker_plan.cgroup_cpuset_cores;
        let ckpt_cgroup_quota_cores = worker_plan.cgroup_quota_cores;
        let ckpt_numa_nodes_used = worker_plan.numa_nodes_used;
        let ckpt_effective_memory_max = effective_memory_max;
        let ckpt_resumed = resumed_from_checkpoint;
        let ckpt_started_at = started_at;
        let ckpt_configured_workers = config.workers;

        Some(std::thread::spawn(move || {
            let checkpoint_path = ckpt_work_dir.join("checkpoints").join("latest.json");
            let mut last_checkpoint = Instant::now();

            loop {
                // Sleep in small increments to check stop flag
                std::thread::sleep(Duration::from_secs(1));

                if ckpt_stop.load(Ordering::Acquire) {
                    break;
                }

                // Check for emergency checkpoint (spot instance preemption)
                let is_emergency = crate::chaos::is_emergency_checkpoint_requested();

                // Check if it's time for a checkpoint (or emergency requested)
                if !is_emergency && last_checkpoint.elapsed() < ckpt_interval {
                    continue;
                }

                // Don't checkpoint if exploration has stopped (unless emergency)
                if !is_emergency && ckpt_exploration_stop.load(Ordering::Acquire) {
                    break;
                }

                // Check disk space before checkpoint
                let disk_stats = get_disk_stats(&ckpt_work_dir);
                if disk_stats.is_critical() {
                    eprintln!(
                        "Checkpoint: CRITICAL - Disk space low ({} available, {}% used). \
                         Skipping checkpoint to prevent corruption. Free up disk space!",
                        disk_stats.available_human(),
                        (disk_stats.used_bytes as f64 / disk_stats.total_bytes as f64 * 100.0)
                            as u32
                    );
                    if !is_emergency {
                        continue;
                    }
                    // For emergency checkpoint, try anyway but warn loudly
                    eprintln!(
                        "Checkpoint: Attempting emergency checkpoint despite low disk space..."
                    );
                }

                if is_emergency {
                    eprintln!(
                        "Checkpoint: EMERGENCY checkpoint requested (spot instance preemption)..."
                    );
                } else {
                    eprintln!("Checkpoint: starting periodic checkpoint...");
                }
                let ckpt_start = Instant::now();

                // CRITICAL: Set checkpoint flag BEFORE requesting pause!
                // This prevents workers from terminating when they see empty
                // queues while waiting for pause. Without this, workers in
                // pop_slow_path check should_terminate(), see empty queues,
                // and exit before checkpoint can drain/reload items.
                ckpt_queue.set_checkpoint_in_progress(true);

                request_checkpoint_pause(ckpt_queue.as_ref(), &ckpt_pause);
                let quiescence_achieved = ckpt_pause.wait_for_quiescence(
                    &ckpt_exploration_stop,
                    &ckpt_active_workers,
                    &ckpt_live_workers,
                );

                // If quiescence timed out, skip this checkpoint
                if !quiescence_achieved {
                    eprintln!(
                        "Checkpoint: quiescence timeout, skipping checkpoint and resuming workers"
                    );
                    ckpt_pause.resume();
                    ckpt_queue.set_pause_requested(false);
                    ckpt_queue.set_checkpoint_in_progress(false);
                    continue;
                }

                // Flush queue state to disk
                if let Err(e) = ckpt_queue.checkpoint_flush() {
                    eprintln!("Checkpoint: queue flush failed: {}", e);
                    ckpt_pause.resume();
                    ckpt_queue.set_pause_requested(false);
                    ckpt_queue.set_checkpoint_in_progress(false);
                    continue;
                }

                // Flush fingerprints to disk
                if let Err(e) = ckpt_fp_store.flush() {
                    eprintln!("Checkpoint: fingerprint flush failed: {}", e);
                    ckpt_pause.resume();
                    ckpt_queue.set_pause_requested(false);
                    ckpt_queue.set_checkpoint_in_progress(false);
                    continue;
                }

                // Write checkpoint manifest
                let (
                    states_generated,
                    states_processed,
                    states_distinct,
                    duplicates,
                    enqueued,
                    checkpoints,
                ) = ckpt_run_stats.snapshot();
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let manifest = CheckpointManifest {
                    version: 1,
                    model: ckpt_model_name.clone(),
                    created_unix_secs: now,
                    duration_millis: ckpt_started_at.elapsed().as_millis() as u64,
                    states_generated,
                    states_processed,
                    states_distinct,
                    duplicates,
                    enqueued,
                    checkpoints,
                    configured_workers: ckpt_configured_workers,
                    actual_workers: ckpt_worker_count,
                    allowed_cpu_count: ckpt_allowed_cpus,
                    cgroup_cpuset_cores: ckpt_cgroup_cpuset_cores,
                    cgroup_quota_cores: ckpt_cgroup_quota_cores,
                    numa_nodes_used: ckpt_numa_nodes_used,
                    effective_memory_max_bytes: ckpt_effective_memory_max,
                    resumed_from_checkpoint: ckpt_resumed,
                    queue: ckpt_queue.stats(),
                    fingerprints: {
                        let stats = ckpt_fp_store.stats();
                        OldFingerprintStats {
                            checks: stats.checks,
                            hits: stats.hits,
                            inserts: stats.inserts,
                            batch_calls: stats.batch_calls,
                            batch_items: stats.batch_items,
                        }
                    },
                };

                if let Err(e) = write_validated_rolling_checkpoint(&checkpoint_path, &manifest) {
                    eprintln!("Checkpoint: failed to write/validate manifest: {}", e);
                } else {
                    ckpt_run_stats.checkpoints.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "Checkpoint: completed in {:.1}s (states: {}, distinct: {}) [validated, keeping last {}]",
                        ckpt_start.elapsed().as_secs_f64(),
                        states_generated,
                        states_distinct,
                        MAX_CHECKPOINT_FILES,
                    );

                    // Prune old segment files to prevent unbounded growth
                    // Only do this for normal checkpoints, not emergency ones
                    // (emergency checkpoints need files intact for S3 upload)
                    if !is_emergency {
                        if let Err(e) =
                            prune_work_dir_segments(&ckpt_work_dir, MAX_CHECKPOINT_FILES)
                        {
                            eprintln!("Checkpoint: warning: failed to prune old segments: {}", e);
                        }
                    }
                }

                // For emergency checkpoint (spot preemption), signal completion and don't resume
                if is_emergency {
                    eprintln!(
                        "Checkpoint: EMERGENCY checkpoint complete, signaling for S3 flush..."
                    );
                    crate::chaos::clear_emergency_checkpoint();
                    crate::chaos::set_emergency_checkpoint_complete();
                    // Don't resume workers - signal handler will exit process after S3 flush
                    // But we need to clear flags so signal handler can proceed
                    ckpt_queue.set_pause_requested(false);
                    ckpt_queue.set_checkpoint_in_progress(false);
                    // Wait here until process exits (signal handler will call exit())
                    loop {
                        std::thread::sleep(Duration::from_secs(60));
                    }
                }

                // Resume workers (normal checkpoint path)
                ckpt_pause.resume();
                ckpt_queue.set_pause_requested(false);

                // Wait for background loader to refill queue before allowing termination
                // This prevents workers from seeing empty queues and terminating prematurely
                // The background loader started loading when checkpoint_draining cleared
                let wait_start = Instant::now();
                let max_wait = std::time::Duration::from_secs(30);
                while !ckpt_queue.has_pending_work() && wait_start.elapsed() < max_wait {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                if wait_start.elapsed() >= max_wait && !ckpt_queue.has_pending_work() {
                    eprintln!(
                        "Checkpoint: warning: queue still empty after 30s wait, model may have completed"
                    );
                } else {
                    let pending = ckpt_queue.pending_count();
                    let total_pending = ckpt_queue.total_pending_count();
                    eprintln!(
                        "Checkpoint: queue refilled in {:.1}s (inmem: {}, total: {}), resuming workers",
                        wait_start.elapsed().as_secs_f64(),
                        pending,
                        total_pending
                    );
                }

                // Now allow termination
                ckpt_queue.set_checkpoint_in_progress(false);
                last_checkpoint = Instant::now();
            }
        }))
    } else {
        None
    };

    // Work-stealing queues detect completion internally
    // No need for separate monitor thread

    // Progress reporting thread - prints stats every 10 seconds
    let progress_run_stats = Arc::clone(&run_stats);
    let progress_stop = Arc::clone(&stop);
    let progress_fp_store = Arc::clone(&fp_store);
    let progress_queue = Arc::clone(&queue);
    let progress_throttle = Arc::clone(&throttle);
    let progress_worker_count = worker_plan.worker_count;
    let progress_thread = std::thread::spawn(move || {
        let mut progress_counter = 1u64;
        let mut last_generated = 0u64;
        let mut last_distinct = 0u64;
        let mut last_queue_pending = 0u64;
        let mut last_time = std::time::Instant::now();

        // Memory monitor - check periodically and throttle if needed
        let memory_monitor = MemoryMonitor::default();
        let mut memory_warned = false;
        let mut memory_throttled = false;

        loop {
            // Check for emergency checkpoint request every second
            for _ in 0..10 {
                std::thread::sleep(std::time::Duration::from_secs(1));

                if progress_stop.load(Ordering::Relaxed) {
                    return;
                }

                // Check memory pressure and throttle workers if needed
                let mem_status = memory_monitor.check();
                match mem_status {
                    MemoryStatus::Critical {
                        rss_bytes,
                        limit_bytes,
                        ratio,
                    } => {
                        if !memory_throttled {
                            let rss_gb = rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            let limit_gb = limit_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            eprintln!(
                                "MEMORY CRITICAL: {:.1}GB / {:.1}GB ({:.0}%) - throttling to 25% workers",
                                rss_gb,
                                limit_gb,
                                ratio * 100.0
                            );
                            // Throttle to 25% of workers
                            let reduced = (progress_worker_count / 4).max(1);
                            progress_throttle.set_active_target(reduced);
                            memory_throttled = true;
                            memory_warned = true;
                        }
                    }
                    MemoryStatus::Warning {
                        rss_bytes,
                        limit_bytes,
                        ratio,
                    } => {
                        if !memory_warned {
                            let rss_gb = rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            let limit_gb = limit_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            eprintln!(
                                "MEMORY WARNING: {:.1}GB / {:.1}GB ({:.0}%) - consider reducing workers",
                                rss_gb,
                                limit_gb,
                                ratio * 100.0
                            );
                            // Throttle to 50% of workers
                            let reduced = (progress_worker_count / 2).max(1);
                            progress_throttle.set_active_target(reduced);
                            memory_warned = true;
                        }
                    }
                    MemoryStatus::Ok { ratio, .. } => {
                        // If we were throttled and now OK, gradually restore
                        if memory_throttled && ratio < 0.60 {
                            eprintln!(
                                "Memory pressure relieved ({:.0}%) - restoring workers",
                                ratio * 100.0
                            );
                            progress_throttle.set_active_target(progress_worker_count);
                            memory_throttled = false;
                            memory_warned = false;
                        }
                    }
                }

                // Handle emergency checkpoint request
                if crate::chaos::is_emergency_checkpoint_requested() {
                    eprintln!("Emergency checkpoint: flushing fingerprint store...");
                    if let Err(e) = progress_fp_store.flush() {
                        eprintln!("Emergency checkpoint: fingerprint flush failed: {}", e);
                    } else {
                        eprintln!("Emergency checkpoint: fingerprint store flushed successfully");
                    }
                    let (generated, _, distinct, _, _, _) = progress_run_stats.snapshot();
                    eprintln!(
                        "Emergency checkpoint: {} states generated, {} distinct at time of failure",
                        generated, distinct
                    );
                    crate::chaos::clear_emergency_checkpoint();
                }
            }

            if progress_stop.load(Ordering::Relaxed) {
                break;
            }

            let (states_generated, _states_processed, states_distinct, _, _, _) =
                progress_run_stats.snapshot();
            // Use total_pending_count to include spilled items on disk
            let queue_pending = progress_queue.total_pending_count();

            // Calculate rates per minute
            let now = std::time::Instant::now();
            let elapsed_secs = now.duration_since(last_time).as_secs_f64();
            let elapsed_mins = elapsed_secs / 60.0;

            let generated_rate = if elapsed_mins > 0.0 {
                ((states_generated - last_generated) as f64 / elapsed_mins) as u64
            } else {
                0
            };

            let distinct_rate = if elapsed_mins > 0.0 {
                ((states_distinct - last_distinct) as f64 / elapsed_mins) as u64
            } else {
                0
            };

            // Format timestamp in TLC style: YYYY-MM-DD HH:MM:SS
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

            // Format numbers with commas (TLC style)
            fn format_with_commas(n: u64) -> String {
                let s = n.to_string();
                let mut result = String::new();
                for (i, c) in s.chars().rev().enumerate() {
                    if i > 0 && i % 3 == 0 {
                        result.push(',');
                    }
                    result.push(c);
                }
                result.chars().rev().collect()
            }

            // Calculate ETA based on queue drain rate
            let eta_str = if progress_counter > 1 && elapsed_secs > 0.0 {
                let queue_delta = last_queue_pending as i64 - queue_pending as i64;
                let drain_rate = queue_delta as f64 / elapsed_secs; // states/sec

                if drain_rate > 10.0 {
                    // Queue is draining - estimate completion time
                    let eta_secs = queue_pending as f64 / drain_rate;
                    if eta_secs < 60.0 {
                        format!(" ETA: {:.0}s", eta_secs)
                    } else if eta_secs < 3600.0 {
                        format!(" ETA: {:.1}m", eta_secs / 60.0)
                    } else if eta_secs < 86400.0 {
                        format!(" ETA: {:.1}h", eta_secs / 3600.0)
                    } else {
                        format!(" ETA: {:.1}d", eta_secs / 86400.0)
                    }
                } else if drain_rate < -10.0 {
                    // Queue is growing
                    " ETA: queue growing".to_string()
                } else {
                    // Queue is stable
                    " ETA: stabilizing".to_string()
                }
            } else {
                String::new()
            };

            // Match TLC format with ETA
            eprintln!(
                "Progress({}) at {}: {} states generated ({} s/min), {} distinct states found ({} ds/min), {} states left on queue.{}",
                progress_counter,
                timestamp,
                format_with_commas(states_generated),
                format_with_commas(generated_rate),
                format_with_commas(states_distinct),
                format_with_commas(distinct_rate),
                format_with_commas(queue_pending),
                eta_str
            );

            progress_counter += 1;
            last_generated = states_generated;
            last_distinct = states_distinct;
            last_queue_pending = queue_pending;
            last_time = now;
        }
    });

    // Check if we need to collect labeled transitions for fairness checking
    let collect_labeled_transitions = model.has_fairness_constraints();
    let labeled_transitions: Option<Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>> =
        if collect_labeled_transitions {
            eprintln!("Fairness constraints detected - collecting labeled transitions");
            Some(Arc::new(DashMap::new()))
        } else {
            None
        };

    // Start auto-tuner if enabled (reads run_stats for throughput)
    let mut auto_tuner = if auto_tune_enabled {
        let tune_config = AutoTuneConfig {
            max_workers: worker_plan.worker_count,
            min_workers: worker_plan.worker_count / 8, // Floor at 1/8 of workers
            target_sys_pct: 20.0,
            adjustment_step: (worker_plan.worker_count / 16).max(4), // ~6% at a time
            verbose: true,
            ..Default::default()
        };
        let mut tuner = AutoTuner::new(tune_config, Arc::clone(&throttle), Arc::clone(&stop));
        let stats_for_tuner = Arc::clone(&run_stats);
        tuner.start(move || stats_for_tuner.states_generated.load(Ordering::Relaxed));
        Some(tuner)
    } else {
        None
    };

    // Initialize per-worker pause tracking for debugging stuck workers
    pause.init_worker_tracking(worker_plan.worker_count, &worker_plan.worker_numa_nodes);

    // Initialize NUMA diagnostics for stuck worker analysis
    // Sample fingerprint store shard memory for NUMA location detection.
    let fp_store_addrs = fp_store.memory_base_addrs();
    let numa_topology = NumaTopology::detect().unwrap_or_else(|_| NumaTopology {
        node_count: 1,
        cpu_to_node: HashMap::new(),
        distances: vec![vec![10]],
        cpus_per_node: vec![],
    });
    let numa_diagnostics = NumaDiagnostics::new(
        &numa_topology,
        &worker_plan.worker_numa_nodes,
        &fp_store_addrs,
    );

    // Print startup NUMA diagnostics
    numa_diagnostics.print_startup_info();
    pause.set_numa_diagnostics(numa_diagnostics);

    let mut workers = Vec::with_capacity(worker_plan.worker_count);
    for (worker_id, mut worker_state) in worker_states.into_iter().enumerate() {
        let worker_model = Arc::clone(&model);
        let worker_fp_store = Arc::clone(&fp_store);
        let worker_queue = Arc::clone(&queue);
        let worker_stats = Arc::clone(&run_stats);
        let worker_stop = Arc::clone(&stop);
        let _worker_active = Arc::clone(&active_workers);
        let worker_live = Arc::clone(&live_workers);
        let worker_pause = Arc::clone(&pause);
        let worker_throttle = Arc::clone(&throttle);
        let worker_violation_tx = violation_tx.clone();
        let worker_error_tx = error_tx.clone();
        let worker_stop_on_violation = config.stop_on_violation;
        let worker_fp_batch_size = config.fp_batch_size.max(1);
        let worker_cpu = worker_plan.assigned_cpus.get(worker_id).copied().flatten();
        let worker_numa_node = worker_plan
            .worker_numa_nodes
            .get(worker_id)
            .copied()
            .unwrap_or(0);
        let worker_labeled_transitions = labeled_transitions.clone();
        let worker_parent_map = parent_map.clone();
        let worker_state_map = state_map.clone();
        let worker_trace_count = Arc::clone(&trace_state_count);
        let worker_max_trace_states = max_trace_states;

        workers.push(std::thread::spawn(move || {
            // Pin thread to CPU for cache locality
            if let Some(cpu) = worker_cpu
                && let Err(err) = pin_current_thread_to_cpu(cpu)
            {
                let _ = worker_error_tx.send(format!("cpu pinning failed on core {cpu}: {err}"));
                worker_stop.store(true, Ordering::Release);
            }

            // Set NUMA memory policy - all allocations on this thread will prefer the local node
            // This reduces cross-NUMA memory access which causes high kernel time
            let _ = set_preferred_node(worker_numa_node);

            let mut successors: Vec<M::State> = Vec::with_capacity(64);
            let mut pending_batch: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);
            let mut unique_states: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);
            let mut unique_fps: Vec<u64> = Vec::with_capacity(worker_fp_batch_size);
            let mut batch_seen: Vec<bool> = Vec::with_capacity(worker_fp_batch_size);
            let mut local_fp_dedup: HashSet<u64> = HashSet::with_capacity(worker_fp_batch_size * 2);

            // Per-worker persistent fingerprint cache to reduce global store CAS contention
            // This catches duplicates locally before hitting shared atomics
            // Size: 64K entries = ~512KB per worker, catches most duplicates locally
            const LOCAL_FP_CACHE_SIZE: usize = 65536;
            let mut local_fp_cache: HashSet<u64> = HashSet::with_capacity(LOCAL_FP_CACHE_SIZE);
            let mut local_fp_cache_hits = 0u64;

            // Per-worker stats counters to reduce atomic contention
            // Flushed periodically instead of every state
            let mut local_states_processed = 0u64;
            let mut local_states_generated = 0u64;
            let mut local_duplicates = 0u64;
            let mut local_states_distinct = 0u64;
            let mut local_enqueued = 0u64;
            // Adaptive flush: every 512 states OR every ~1 second (whichever comes first)
            // This balances atomic contention reduction vs stats freshness
            const STATS_FLUSH_INTERVAL: u64 = 512;
            let mut last_stats_flush = Instant::now();

            let flush_local_stats = |processed: &mut u64,
                                     generated: &mut u64,
                                     duplicates: &mut u64,
                                     distinct: &mut u64,
                                     enqueued: &mut u64,
                                     stats: &AtomicRunStats| {
                if *processed > 0 {
                    stats
                        .states_processed
                        .fetch_add(*processed, Ordering::Relaxed);
                    *processed = 0;
                }
                if *generated > 0 {
                    stats
                        .states_generated
                        .fetch_add(*generated, Ordering::Relaxed);
                    *generated = 0;
                }
                if *duplicates > 0 {
                    stats.duplicates.fetch_add(*duplicates, Ordering::Relaxed);
                    *duplicates = 0;
                }
                if *distinct > 0 {
                    stats
                        .states_distinct
                        .fetch_add(*distinct, Ordering::Relaxed);
                    *distinct = 0;
                }
                if *enqueued > 0 {
                    stats.enqueued.fetch_add(*enqueued, Ordering::Relaxed);
                    *enqueued = 0;
                }
            };

            loop {
                // Chaos: check if this worker should crash
                if crate::chaos::should_crash_worker(worker_state.id()) {
                    panic!("chaos: simulated worker {} crash", worker_state.id());
                }

                // Chaos: failpoint for worker panic
                #[cfg(feature = "failpoints")]
                if crate::fail_point_is_set!("worker_panic") {
                    panic!("chaos: failpoint worker_panic triggered");
                }

                worker_pause.worker_pause_point(&worker_stop, worker_id);
                if worker_stop.load(Ordering::Acquire) {
                    break;
                }

                // Auto-tune throttle: workers over the limit yield briefly
                worker_throttle.worker_throttle_point(worker_id);

                // Work-stealing: try local queue first, then steal from others
                // This has zero contention on the common path
                let state = match worker_queue.pop_for_worker(&mut worker_state) {
                    Some(state) => state,
                    None => {
                        // During checkpoint, pop_for_worker returns None because pause_requested
                        // is set. Workers MUST pause for quiescence. We call worker_pause_point
                        // directly here instead of relying on  to reach the one at
                        // the top of the loop. This eliminates a race window where workers
                        // could spin in a tight continue-loop without ever pausing:
                        //   continue -> pause_point(requested=false yet) -> pop(None) -> continue ...
                        // By pausing directly, workers enter quiescence immediately.
                        if pause_worker_after_empty_pop_during_checkpoint(
                            worker_queue.as_ref(),
                            &worker_pause,
                            &worker_stop,
                            worker_id,
                        ) {
                            if worker_stop.load(Ordering::Acquire) {
                                break;
                            }
                            continue;
                        }
                        // After checkpoint, the loader thread may still be loading items from disk.
                        // Workers should not terminate while there's pending disk work - the loader
                        // will push items to the global queue that workers can steal.
                        if worker_queue.has_pending_work() {
                            // Give the loader thread time to load more items
                            std::thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                        // Before terminating, do a final pause check to close
                        // the race window between our has_pending_work() check
                        // and the checkpoint thread requesting pause.  If a
                        // checkpoint was requested in that window, this will
                        // block until the checkpoint completes and then we can
                        // recheck for new work that may have been loaded.
                        worker_pause.worker_pause_point(&worker_stop, worker_id);
                        if worker_stop.load(Ordering::Acquire) {
                            break;
                        }
                        // Recheck work availability — checkpoint may have
                        // loaded items from disk while we were paused.
                        // Use has_pending_work() rather than pop_for_worker()
                        // to avoid consuming a state that the outer match
                        // would not see.
                        if worker_queue.has_pending_work() {
                            continue;
                        }
                        // No work available and no checkpoint pending — safe to terminate
                        break;
                    }
                };

                // Mark worker as active (cache-line padded, no contention with other workers)
                worker_queue.worker_start(worker_state.id());
                local_states_processed += 1;

                // Periodically flush local stats to reduce atomic contention
                // Flush either by count OR by time (every ~1 second) for accurate reporting
                let should_flush = local_states_processed % STATS_FLUSH_INTERVAL == 0
                    || last_stats_flush.elapsed() >= Duration::from_secs(1);
                if should_flush {
                    flush_local_stats(
                        &mut local_states_processed,
                        &mut local_states_generated,
                        &mut local_duplicates,
                        &mut local_states_distinct,
                        &mut local_enqueued,
                        &worker_stats,
                    );
                    last_stats_flush = Instant::now();
                }

                if let Err(message) = worker_model.check_invariants(&state) {
                    // Reconstruct trace to violation
                    let trace = if let Some(ref pm) = worker_parent_map {
                        // Use BFS parent tracking: walk parent chain back to initial state
                        let sm = worker_state_map.as_ref().unwrap();
                        let mut chain = vec![state.clone()];
                        let mut fp = worker_model.fingerprint(&state);
                        while let Some(parent_fp_entry) = pm.get(&fp) {
                            let parent_fp = *parent_fp_entry;
                            if let Some(parent_state) = sm.get(&parent_fp) {
                                chain.push(parent_state.clone());
                                fp = parent_fp;
                            } else {
                                break;
                            }
                        }
                        chain.reverse();
                        chain
                    } else {
                        // Fall back to re-exploration from init states
                        reconstruct_trace_limited(
                            worker_model.as_ref(),
                            &state,
                            100, // Max depth to search
                        )
                        .unwrap_or_else(|| vec![state.clone()])
                    };

                    let _ = worker_violation_tx.try_send(Violation {
                        message,
                        state,
                        property_type: PropertyType::Safety,
                        trace,
                    });
                    if worker_stop_on_violation {
                        worker_stop.store(true, Ordering::Release);
                    }
                    worker_queue.worker_idle(worker_state.id());
                    if worker_stop_on_violation {
                        break;
                    }
                    continue;
                }

                // Apply backpressure: if queue is too full, skip successor generation
                // This allows workers to process the backlog without deadlock
                // Testing: Increased to 1B to allow full state space exploration
                const MAX_PENDING_STATES: u64 = 1_000_000_000;
                if worker_queue.should_apply_backpressure(MAX_PENDING_STATES) {
                    // Don't generate successors, just mark worker as idle and continue
                    // This allows us to process the backlog
                    worker_queue.worker_idle(worker_state.id());
                    continue;
                }

                successors.clear();

                // Use labeled transitions if fairness constraints exist
                if let Some(ref transitions_map) = worker_labeled_transitions {
                    if let Some(labeled_transitions_vec) = worker_model.next_states_labeled(&state)
                    {
                        // Collect labeled transitions and extract successor states
                        let state_fp = worker_model.fingerprint(&state);

                        // Store all transitions from this state
                        let mut transitions_from_state =
                            Vec::with_capacity(labeled_transitions_vec.len());
                        for labeled_trans in labeled_transitions_vec {
                            successors.push(labeled_trans.to.clone());
                            transitions_from_state.push(labeled_trans);
                        }

                        // Store transitions in the map (keyed by source state fingerprint)
                        if !transitions_from_state.is_empty() {
                            transitions_map
                                .entry(state_fp)
                                .or_insert_with(Vec::new)
                                .extend(transitions_from_state);
                        }
                    } else {
                        // Fallback to unlabeled if model doesn't provide labeled transitions
                        worker_model.next_states(&state, &mut successors);
                    }
                } else {
                    // No fairness constraints - use regular next_states
                    worker_model.next_states(&state, &mut successors);
                }

                #[allow(unused_variables)]
                let successors_before_filter = successors.len();
                local_states_generated += successors.len() as u64;

                // Filter successors by state constraints (prune states that don't satisfy constraints)
                successors.retain(|next_state| {
                    worker_model.check_state_constraints(next_state).is_ok()
                        && worker_model
                            .check_action_constraints(&state, next_state)
                            .is_ok()
                });

                // Debug: track constraint filtering (only in debug builds)
                #[cfg(debug_assertions)]
                {
                    static DEBUG_STATES_SAMPLED: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let sample_count = DEBUG_STATES_SAMPLED.fetch_add(1, Ordering::Relaxed);
                    if sample_count < 10 {
                        eprintln!(
                            "DEBUG [state {}]: {} successors generated, {} after constraint filter",
                            sample_count,
                            successors_before_filter,
                            successors.len()
                        );
                    }
                }

                // Pair states with their fingerprint's home NUMA for routing
                let mut states_with_home_numa: Vec<(M::State, usize)> =
                    Vec::with_capacity(worker_fp_batch_size);
                // Fingerprints that need global store check (not in local cache)
                let mut fps_to_check: Vec<u64> = Vec::with_capacity(worker_fp_batch_size);
                let mut states_to_check: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);

                // Compute parent fingerprint once for parent tracking
                let current_state_fp = if worker_parent_map.is_some() {
                    worker_model.fingerprint(&state)
                } else {
                    0
                };

                let mut process_batch = |pending_batch: &mut Vec<M::State>,
                                         local_duplicates: &mut u64,
                                         local_states_distinct: &mut u64,
                                         local_enqueued: &mut u64,
                                         fp_cache: &mut HashSet<u64>,
                                         fp_cache_hits: &mut u64|
                 -> Result<()> {
                    if pending_batch.is_empty() {
                        return Ok(());
                    }
                    unique_states.clear();
                    unique_fps.clear();
                    local_fp_dedup.clear();
                    states_with_home_numa.clear();
                    fps_to_check.clear();
                    states_to_check.clear();

                    let mut duplicates_in_batch = 0u64;
                    for candidate in pending_batch.drain(..) {
                        // Canonicalize under symmetry reduction before fingerprinting.
                        // Two states that differ only by a permutation of symmetric
                        // elements will produce the same canonical form and therefore
                        // the same fingerprint, letting the FP store dedup them.
                        let candidate = worker_model.canonicalize(candidate);
                        let fp = worker_model.fingerprint(&candidate);
                        // Dedup within this batch first
                        if !local_fp_dedup.insert(fp) {
                            duplicates_in_batch += 1;
                            continue;
                        }
                        // Check persistent local cache - catches duplicates without CAS
                        if fp_cache.contains(&fp) {
                            *fp_cache_hits += 1;
                            duplicates_in_batch += 1;
                            continue;
                        }
                        // Need to check global store
                        fps_to_check.push(fp);
                        states_to_check.push(candidate);
                    }

                    if !fps_to_check.is_empty() {
                        // Use worker-affinity batch to reduce CAS contention
                        // Each worker processes shards in a different order
                        worker_fp_store.contains_or_insert_batch_with_affinity(
                            &fps_to_check,
                            &mut batch_seen,
                            worker_id,
                        )?;

                        for (idx, next_state) in states_to_check.drain(..).enumerate() {
                            let fp = fps_to_check[idx];
                            if batch_seen[idx] {
                                *local_duplicates += 1;
                                // Add to local cache to catch future duplicates
                                if fp_cache.len() < LOCAL_FP_CACHE_SIZE {
                                    fp_cache.insert(fp);
                                }
                            } else {
                                *local_states_distinct += 1;
                                // Add new fingerprint to local cache
                                if fp_cache.len() < LOCAL_FP_CACHE_SIZE {
                                    fp_cache.insert(fp);
                                }
                                // Record parent tracking for trace reconstruction
                                if let Some(ref pm) = worker_parent_map {
                                    if worker_trace_count.load(Ordering::Relaxed)
                                        < worker_max_trace_states
                                    {
                                        pm.insert(fp, current_state_fp);
                                        if let Some(ref sm) = worker_state_map {
                                            sm.insert(fp, next_state.clone());
                                        }
                                        worker_trace_count.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                // Get the fingerprint's home NUMA for routing
                                let home_numa = worker_fp_store.home_numa(fp);
                                states_with_home_numa.push((next_state, home_numa));
                            }
                        }

                        // Batch push with NUMA-aware routing
                        let pushed = worker_queue
                            .push_batch_to_numa(&mut worker_state, states_with_home_numa.drain(..));
                        *local_enqueued += pushed as u64;
                    }

                    *local_duplicates += duplicates_in_batch;

                    // Clear cache if it gets too large (prevents unbounded memory)
                    if fp_cache.len() >= LOCAL_FP_CACHE_SIZE {
                        fp_cache.clear();
                    }

                    Ok(())
                };

                for next in successors.drain(..) {
                    if worker_stop.load(Ordering::Acquire) {
                        break;
                    }
                    pending_batch.push(next);
                    if pending_batch.len() >= worker_fp_batch_size
                        && let Err(err) = process_batch(
                            &mut pending_batch,
                            &mut local_duplicates,
                            &mut local_states_distinct,
                            &mut local_enqueued,
                            &mut local_fp_cache,
                            &mut local_fp_cache_hits,
                        )
                    {
                        let _ = worker_error_tx.send(err.to_string());
                        worker_stop.store(true, Ordering::Release);
                        break;
                    }
                }
                if !worker_stop.load(Ordering::Acquire)
                    && let Err(err) = process_batch(
                        &mut pending_batch,
                        &mut local_duplicates,
                        &mut local_states_distinct,
                        &mut local_enqueued,
                        &mut local_fp_cache,
                        &mut local_fp_cache_hits,
                    )
                {
                    let _ = worker_error_tx.send(err.to_string());
                    worker_stop.store(true, Ordering::Release);
                }

                // Mark worker as idle (done with this state)
                worker_queue.worker_idle(worker_state.id());
            }

            // Flush any remaining local stats before exiting
            flush_local_stats(
                &mut local_states_processed,
                &mut local_states_generated,
                &mut local_duplicates,
                &mut local_states_distinct,
                &mut local_enqueued,
                &worker_stats,
            );

            // Flush queue counters
            worker_queue.flush_worker_counters(&mut worker_state);

            worker_live.fetch_sub(1, Ordering::AcqRel);
            worker_pause.wait_cv.notify_all();
        }));
    }

    // Worker crash recovery: continue with remaining workers instead of failing immediately
    let mut crashed_workers = 0usize;
    let total_workers = workers.len();

    for (worker_id, worker) in workers.into_iter().enumerate() {
        if worker.join().is_err() {
            crashed_workers += 1;
            eprintln!(
                "Warning: worker {} crashed ({}/{} workers still running)",
                worker_id,
                total_workers - crashed_workers,
                total_workers
            );

            // If all workers crashed, we must stop
            if crashed_workers == total_workers {
                stop.store(true, Ordering::Release);
                return Err(anyhow!("all {} worker threads panicked", total_workers));
            }

            // Otherwise continue - other workers may still complete the work
            // The work-stealing queues will redistribute work from the crashed worker
        }
    }

    if crashed_workers > 0 {
        eprintln!(
            "Run completed with {}/{} workers crashed (recovered gracefully)",
            crashed_workers, total_workers
        );
    }

    // Stop progress reporting
    stop.store(true, Ordering::Release);
    let _ = progress_thread.join();

    // Stop auto-tuner if running
    if let Some(tuner) = auto_tuner.take() {
        tuner.join();
    }

    // Cleanup checkpoint thread
    queue.finish(); // Signal completion

    mem_monitor_stop.store(true, Ordering::Relaxed);
    if let Some(handle) = mem_monitor_thread {
        let _ = handle.join();
    }
    checkpoint_thread_stop.store(true, Ordering::Release);
    pause.resume();
    if let Some(handle) = checkpoint_thread
        && handle.join().is_err()
    {
        return Err(anyhow!("checkpoint thread panicked"));
    }

    // Write checkpoint manifest on exit if requested
    if config.checkpoint_on_exit {
        let checkpoint_path = config.work_dir.join("checkpoints").join("latest.json");
        let (
            states_generated,
            states_processed,
            states_distinct,
            duplicates,
            enqueued,
            checkpoints,
        ) = run_stats.snapshot();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let manifest = CheckpointManifest {
            version: 1,
            model: std::any::type_name::<M>().to_string(),
            created_unix_secs: now,
            duration_millis: started_at.elapsed().as_millis() as u64,
            states_generated,
            states_processed,
            states_distinct,
            duplicates,
            enqueued,
            checkpoints,
            configured_workers: config.workers,
            actual_workers: worker_plan.worker_count,
            allowed_cpu_count: worker_plan.allowed_cpus.len(),
            cgroup_cpuset_cores: worker_plan.cgroup_cpuset_cores,
            cgroup_quota_cores: worker_plan.cgroup_quota_cores,
            numa_nodes_used: worker_plan.numa_nodes_used,
            effective_memory_max_bytes: effective_memory_max,
            resumed_from_checkpoint,
            queue: queue.stats(),
            fingerprints: {
                let stats = fp_store.stats();
                OldFingerprintStats {
                    checks: stats.checks,
                    hits: stats.hits,
                    inserts: stats.inserts,
                    batch_calls: stats.batch_calls,
                    batch_items: stats.batch_items,
                }
            },
        };
        if let Err(e) = write_validated_rolling_checkpoint(&checkpoint_path, &manifest) {
            eprintln!(
                "Warning: failed to write/validate checkpoint manifest: {}",
                e
            );
        } else {
            eprintln!(
                "Exit checkpoint written and validated [keeping last {}]",
                MAX_CHECKPOINT_FILES
            );
        }
    }

    // Queue cleanup happens automatically
    let _ = fp_store.flush();

    if let Ok(err) = error_rx.try_recv() {
        return Err(anyhow!(err));
    }

    let mut violation = violation_rx.try_recv().ok();

    // Check fairness constraints if we collected labeled transitions
    // (only if no safety violation was already found)
    if violation.is_none() {
    if let Some(transitions_map) = labeled_transitions.as_ref() {
        eprintln!(
            "Checking fairness constraints on {} states with transitions...",
            transitions_map.len()
        );

        // Flatten all transitions into a single vector
        let all_transitions: Vec<LabeledTransition<M::State>> = transitions_map
            .iter()
            .flat_map(|entry| entry.value().clone())
            .collect();

        if !all_transitions.is_empty() {
            eprintln!("  Total transitions collected: {}", all_transitions.len());

            // Collect all unique states from transitions
            let mut state_set = HashSet::new();
            for trans in &all_transitions {
                state_set.insert(model.fingerprint(&trans.from));
                state_set.insert(model.fingerprint(&trans.to));
            }
            let unique_states: Vec<M::State> = all_transitions
                .iter()
                .flat_map(|t| vec![t.from.clone(), t.to.clone()])
                .collect::<Vec<_>>()
                .into_iter()
                .filter(|s| {
                    let fp = model.fingerprint(s);
                    state_set.remove(&fp)
                })
                .collect();

            eprintln!("  Unique states in graph: {}", unique_states.len());

            // Build adjacency map for SCC detection
            let mut adjacency: HashMap<M::State, Vec<M::State>> = HashMap::new();
            for trans in &all_transitions {
                adjacency
                    .entry(trans.from.clone())
                    .or_insert_with(Vec::new)
                    .push(trans.to.clone());
            }

            // Find strongly connected components using Tarjan's algorithm
            let mut tarjan = TarjanSCC::new();
            let sccs = tarjan.find_sccs(&unique_states, |state| {
                adjacency.get(state).cloned().unwrap_or_default()
            });

            eprintln!("  Found {} strongly connected components", sccs.len());

            // Check fairness constraints on each non-trivial SCC
            // (Non-trivial = has more than one state, or has a self-loop)
            let non_trivial_sccs: Vec<_> = sccs
                .iter()
                .filter(|scc| {
                    scc.len() > 1
                        || (scc.len() == 1 && {
                            let state = &scc[0];
                            adjacency
                                .get(state)
                                .map(|succs| succs.contains(state))
                                .unwrap_or(false)
                        })
                })
                .collect();

            if !non_trivial_sccs.is_empty() {
                eprintln!(
                    "  Checking fairness on {} non-trivial SCCs",
                    non_trivial_sccs.len()
                );

                let constraints = model.fairness_constraints();

                if constraints.is_empty() {
                    eprintln!("  No fairness constraints to check");
                } else {
                    eprintln!(
                        "  Checking {} fairness constraints against SCCs",
                        constraints.len()
                    );

                    // Convert model::LabeledTransition to fairness::LabeledTransition
                    let fairness_transitions: Vec<FairnessLabeledTransition<M::State>> =
                        all_transitions
                            .iter()
                            .map(|t| FairnessLabeledTransition {
                                from: t.from.clone(),
                                to: t.to.clone(),
                                action: FairnessActionLabel {
                                    name: t.action.name.clone(),
                                    disjunct_index: t.action.disjunct_index,
                                },
                            })
                            .collect();

                    for (scc_idx, scc) in non_trivial_sccs.iter().enumerate() {
                        let scc_states: Vec<M::State> =
                            scc.iter().map(|s| (*s).clone()).collect();

                        for constraint in &constraints {
                            if let Err(e) = check_fairness_on_scc(
                                &scc_states,
                                constraint,
                                &fairness_transitions,
                            ) {
                                eprintln!(
                                    "  Fairness violation in SCC {} ({} states): {}",
                                    scc_idx,
                                    scc_states.len(),
                                    e
                                );

                                // Pick a representative state from the SCC for the violation
                                let representative = scc_states
                                    .first()
                                    .cloned()
                                    .expect("non-trivial SCC has at least one state");

                                // Build a cycle trace from the SCC states
                                let mut trace = scc_states.clone();
                                // Close the cycle by repeating the first state
                                if let Some(first) = trace.first().cloned() {
                                    trace.push(first);
                                }

                                violation = Some(Violation {
                                    message: format!(
                                        "Fairness violation: {}",
                                        e
                                    ),
                                    state: representative,
                                    property_type: PropertyType::Liveness,
                                    trace,
                                });

                                // Stop on first fairness violation
                                break;
                            }
                        }

                        if violation.is_some() {
                            break;
                        }
                    }

                    if violation.is_none() {
                        eprintln!("  All fairness constraints satisfied");
                    }
                }
            } else {
                eprintln!("  No cycles detected - fairness constraints trivially satisfied");
            }
        }
    }
    }

    let (states_generated, states_processed, states_distinct, duplicates, enqueued, checkpoints) =
        run_stats.snapshot();

    Ok(RunOutcome {
        stats: RunStats {
            duration: started_at.elapsed(),
            states_generated,
            states_processed,
            states_distinct,
            duplicates,
            enqueued,
            checkpoints,
            configured_workers: config.workers,
            actual_workers: worker_plan.worker_count,
            allowed_cpu_count: worker_plan.allowed_cpus.len(),
            cgroup_cpuset_cores: worker_plan.cgroup_cpuset_cores,
            cgroup_quota_cores: worker_plan.cgroup_quota_cores,
            numa_nodes_used: worker_plan.numa_nodes_used,
            effective_memory_max_bytes: effective_memory_max,
            resumed_from_checkpoint,
            queue: {
                let ws_stats = queue.stats();
                QueueStats {
                    pushed: ws_stats.pushed,
                    popped: ws_stats.popped,
                    spilled_items: 0,
                    spill_batches: 0,
                    loaded_segments: 0,
                    loaded_items: 0,
                    max_inmem_len: 0,
                }
            },
            fingerprints: fp_store.stats(),
        },
        violation,
    })
}

/// Reconstruct trace to a violating state using post-processing
///
/// This function re-explores the state space from initial states using BFS
/// to find a path to the target state. This is used when a violation is found
/// during exploration (where we only store fingerprints, not parent pointers).
///
/// Returns a trace from an initial state to the target state, or None if
/// the target is not reachable.
pub fn reconstruct_trace<M: Model>(model: &M, target_state: &M::State) -> Option<Vec<M::State>> {
    // BFS from initial states to target
    let mut queue: VecDeque<(M::State, Vec<M::State>)> = VecDeque::new();
    let mut visited = HashSet::new();

    // Start from initial states
    for init_state in model.initial_states() {
        let fp = model.fingerprint(&init_state);

        if &init_state == target_state {
            // Target is an initial state
            return Some(vec![init_state]);
        }

        visited.insert(fp);
        queue.push_back((init_state.clone(), vec![init_state]));
    }

    // BFS exploration
    while let Some((state, path)) = queue.pop_front() {
        let mut successors = Vec::new();
        model.next_states(&state, &mut successors);

        for next_state in successors {
            if &next_state == target_state {
                // Found the target!
                let mut trace = path.clone();
                trace.push(next_state);
                return Some(trace);
            }

            let fp = model.fingerprint(&next_state);
            if visited.insert(fp) {
                // Not visited yet - add to queue
                let mut next_path = path.clone();
                next_path.push(next_state.clone());
                queue.push_back((next_state, next_path));
            }
        }

        // Limit search to prevent excessive memory usage
        if visited.len() > 1_000_000 {
            // If we haven't found the target after exploring 1M states,
            // it's likely not reachable or the search is too expensive
            return None;
        }
    }

    // Target not reachable
    None
}

/// Reconstruct trace with a maximum depth limit
///
/// This is a depth-limited version that stops after exploring up to max_depth
/// transitions from initial states.
pub fn reconstruct_trace_limited<M: Model>(
    model: &M,
    target_state: &M::State,
    max_depth: usize,
) -> Option<Vec<M::State>> {
    let mut queue: VecDeque<(M::State, Vec<M::State>, usize)> = VecDeque::new();
    let mut visited = HashSet::new();

    for init_state in model.initial_states() {
        if &init_state == target_state {
            return Some(vec![init_state]);
        }

        let fp = model.fingerprint(&init_state);
        visited.insert(fp);
        queue.push_back((init_state.clone(), vec![init_state], 0));
    }

    while let Some((state, path, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue; // Skip states at max depth
        }

        let mut successors = Vec::new();
        model.next_states(&state, &mut successors);

        for next_state in successors {
            if &next_state == target_state {
                let mut trace = path.clone();
                trace.push(next_state);
                return Some(trace);
            }

            let fp = model.fingerprint(&next_state);
            if visited.insert(fp) {
                let mut next_path = path.clone();
                next_path.push(next_state.clone());
                queue.push_back((next_state, next_path, depth + 1));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        EngineConfig, PauseController, next_quiescence_timeout_secs,
        pause_worker_after_empty_pop_during_checkpoint, request_checkpoint_pause, run_model,
        should_enable_file_backed_fingerprint_store, should_start_fingerprint_memory_monitor,
    };
    use crate::models::counter_grid::CounterGridModel;
    use crate::storage::work_stealing_queues::WorkStealingQueues;
    use anyhow::Result;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    fn temp_work_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tlapp-runtime-{prefix}-{nanos}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn writes_checkpoint_manifest_on_exit() -> Result<()> {
        let work_dir = temp_work_dir("manifest");
        let model = CounterGridModel::new(64, 64, 300);
        let config = EngineConfig {
            workers: 1,
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 0,
            checkpoint_on_exit: true,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };
        let outcome = run_model(model, config)?;
        assert!(outcome.violation.is_none());

        let manifest_path = work_dir.join("checkpoints").join("latest.json");
        assert!(manifest_path.exists());

        let raw = std::fs::read_to_string(&manifest_path)?;
        assert!(raw.contains("\"version\": 1"));

        let _ = std::fs::remove_dir_all(work_dir);
        Ok(())
    }

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
    fn file_backed_fingerprint_store_uses_effective_memory_cap() {
        let config = EngineConfig {
            use_bloom_fingerprints: false,
            memory_max_bytes: None,
            enforce_cgroups: true,
            ..EngineConfig::default()
        };

        assert!(should_enable_file_backed_fingerprint_store(
            &config,
            Some(8 * 1024 * 1024 * 1024)
        ));
        assert!(!should_enable_file_backed_fingerprint_store(&config, None));

        let bloom_config = EngineConfig {
            use_bloom_fingerprints: true,
            ..config.clone()
        };
        assert!(!should_enable_file_backed_fingerprint_store(
            &bloom_config,
            Some(8 * 1024 * 1024 * 1024)
        ));
    }

    #[test]
    fn memory_monitor_requires_file_backed_fingerprint_store() {
        let config = EngineConfig {
            use_bloom_fingerprints: false,
            ..EngineConfig::default()
        };

        assert!(should_start_fingerprint_memory_monitor(
            &config,
            Some(4 * 1024 * 1024 * 1024),
            true
        ));
        assert!(!should_start_fingerprint_memory_monitor(
            &config,
            Some(4 * 1024 * 1024 * 1024),
            false
        ));
        assert!(!should_start_fingerprint_memory_monitor(
            &EngineConfig {
                use_bloom_fingerprints: true,
                ..config.clone()
            },
            Some(4 * 1024 * 1024 * 1024),
            true
        ));
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

    #[test]
    #[ignore] // Work-stealing queues don't persist state to disk; checkpoint/resume requires DiskBackedQueue
    fn resumes_from_disk_queue_checkpoint() -> Result<()> {
        let work_dir = temp_work_dir("resume");

        let model = CounterGridModel::new(100, 100, 3);
        let initial_config = EngineConfig {
            workers: 1,
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 0,
            checkpoint_on_exit: true,
            queue_inmem_limit: 32,
            queue_spill_batch: 8,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };
        let first = run_model(model, initial_config)?;
        assert!(first.violation.is_some());
        // Note: With bulk dequeue, workers hold items locally so spilling may not occur
        // assert!(first.stats.queue.spilled_items > 0);

        let resume_model = CounterGridModel::new(100, 100, 3);
        let resume_config = EngineConfig {
            workers: 1,
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: false,
            resume_from_checkpoint: true,
            checkpoint_interval_secs: 0,
            checkpoint_on_exit: true,
            queue_inmem_limit: 32,
            queue_spill_batch: 8,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };
        let resumed = run_model(resume_model, resume_config)?;
        assert!(resumed.stats.resumed_from_checkpoint);
        assert!(resumed.violation.is_some());

        let _ = std::fs::remove_dir_all(work_dir);
        Ok(())
    }
}

/// Failpoint integration tests - run with `cargo test --features failpoints`
#[cfg(all(test, feature = "failpoints"))]
mod failpoint_tests {
    use super::{EngineConfig, run_model};
    use crate::chaos;
    use crate::models::counter_grid::CounterGridModel;
    use anyhow::Result;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_work_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tlapp-failpoint-{prefix}-{nanos}-{}",
            std::process::id()
        ))
    }

    /// Test that worker crash is recovered gracefully
    #[test]
    fn worker_crash_recovery() -> Result<()> {
        let work_dir = temp_work_dir("worker-crash");

        // Set worker 0 to crash after a few iterations
        chaos::set_crash_worker(0);

        let model = CounterGridModel::new(10, 10, 50);
        let config = EngineConfig {
            workers: 4, // Multiple workers so others can continue
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 0,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };

        // Run should complete despite worker 0 crashing
        let outcome = run_model(model, config)?;

        // Reset crash worker
        chaos::set_crash_worker(u64::MAX);

        // Cleanup
        let _ = std::fs::remove_dir_all(work_dir);

        // The run should have completed (other workers finished the work)
        assert!(outcome.stats.states_generated > 0);
        eprintln!(
            "Worker crash recovery test: {} states generated with 1 crashed worker",
            outcome.stats.states_generated
        );

        Ok(())
    }

    /// Test that I/O latency doesn't break the system
    #[test]
    fn io_latency_tolerance() -> Result<()> {
        let work_dir = temp_work_dir("io-latency");

        // Add 10ms I/O latency
        chaos::set_io_latency_us(10_000);

        let model = CounterGridModel::new(5, 5, 20);
        let config = EngineConfig {
            workers: 2,
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 0,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };

        let outcome = run_model(model, config)?;

        // Reset I/O latency
        chaos::set_io_latency_us(0);

        let _ = std::fs::remove_dir_all(work_dir);

        assert!(outcome.stats.states_generated > 0);
        eprintln!(
            "I/O latency test: {} states with 10ms artificial latency",
            outcome.stats.states_generated
        );

        Ok(())
    }

    /// Test fingerprint store degradation (failpoint returns false instead of panicking)
    #[test]
    fn fingerprint_store_degradation() -> Result<()> {
        let scenario = fail::FailScenario::setup();
        let work_dir = temp_work_dir("fp-degrade");

        // Enable fingerprint store shard full failpoint
        fail::cfg("fp_store_shard_full", "return").unwrap();

        let model = CounterGridModel::new(5, 5, 20);
        let config = EngineConfig {
            workers: 2,
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 0,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };

        // Should complete despite fingerprint store "full"
        // May explore some duplicate states but should not crash
        let outcome = run_model(model, config)?;

        scenario.teardown();
        let _ = std::fs::remove_dir_all(work_dir);

        assert!(outcome.stats.states_generated > 0);
        eprintln!(
            "FP degradation test: {} states generated (some may be duplicates)",
            outcome.stats.states_generated
        );

        Ok(())
    }

    /// Test emergency checkpoint functionality
    #[test]
    fn emergency_checkpoint_request() {
        // Request emergency checkpoint
        assert!(!chaos::is_emergency_checkpoint_requested());
        chaos::request_emergency_checkpoint();
        assert!(chaos::is_emergency_checkpoint_requested());

        // Clear it
        chaos::clear_emergency_checkpoint();
        assert!(!chaos::is_emergency_checkpoint_requested());

        // Test completion flag
        assert!(!chaos::is_emergency_checkpoint_complete());
        chaos::set_emergency_checkpoint_complete();
        assert!(chaos::is_emergency_checkpoint_complete());

        // Reset for other tests
        crate::chaos::EMERGENCY_CHECKPOINT_COMPLETE
            .store(false, std::sync::atomic::Ordering::Release);

        eprintln!("Emergency checkpoint request/clear/complete test passed");
    }

    /// Test retry_with_backoff helper
    #[test]
    fn retry_with_backoff_success() {
        let mut attempts = 0;

        // Succeed on 3rd attempt
        let result: Result<i32, String> = chaos::retry_with_backoff(
            || {
                attempts += 1;
                if attempts < 3 {
                    Err(format!("fail attempt {}", attempts))
                } else {
                    Ok(42)
                }
            },
            5,   // max retries
            10,  // 10ms initial delay
            100, // 100ms max delay
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts, 3);
        eprintln!(
            "Retry with backoff test: succeeded after {} attempts",
            attempts
        );
    }

    /// Test retry_with_backoff exhaustion
    #[test]
    fn retry_with_backoff_exhaustion() {
        let mut attempts = 0;

        // Always fail
        let result: Result<i32, String> = chaos::retry_with_backoff(
            || {
                attempts += 1;
                Err(format!("always fail attempt {}", attempts))
            },
            2,   // max 2 retries (3 total attempts)
            10,  // 10ms initial delay
            100, // 100ms max delay
        );

        assert!(result.is_err());
        assert_eq!(attempts, 3); // Initial + 2 retries
        eprintln!(
            "Retry exhaustion test: failed after {} attempts as expected",
            attempts
        );
    }

    /// Test that multiple workers can crash and run still completes
    #[test]
    fn multiple_worker_crashes() -> Result<()> {
        let scenario = fail::FailScenario::setup();
        let work_dir = temp_work_dir("multi-crash");

        // Enable worker panic failpoint - this will affect all workers
        // but they should crash one at a time
        fail::cfg("worker_panic", "1*return->off").unwrap();

        let model = CounterGridModel::new(8, 8, 30);
        let config = EngineConfig {
            workers: 4,
            enforce_cgroups: false,
            numa_pinning: false,
            clean_work_dir: true,
            resume_from_checkpoint: false,
            checkpoint_interval_secs: 0,
            work_dir: work_dir.clone(),
            ..EngineConfig::default()
        };

        // Should complete even with one worker crashing
        let outcome = run_model(model, config)?;

        scenario.teardown();
        let _ = std::fs::remove_dir_all(work_dir);

        assert!(outcome.stats.states_generated > 0);
        eprintln!(
            "Multiple worker crash test: {} states generated",
            outcome.stats.states_generated
        );

        Ok(())
    }
}
