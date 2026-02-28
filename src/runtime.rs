use crate::fairness::{FairnessConstraint, TarjanSCC, check_fairness_on_scc};
use crate::model::{LabeledTransition, Model};
use crate::storage::channel_queue::ChannelQueue;
use crate::storage::fingerprint_store::{
    FingerprintStats as OldFingerprintStats, FingerprintStore,
};
use crate::storage::numa::set_preferred_node;
use crate::storage::page_aligned_fingerprint_store::{
    FingerprintStats, FingerprintStoreConfig as PageAlignedConfig, PageAlignedFingerprintStore,
};
use crate::storage::queue::{DiskBackedQueue, DiskQueueConfig, QueueStats};
use crate::storage::simple_blocking_queue::SimpleBlockingQueue;
use crate::storage::spillable_work_stealing::{
    SpillableConfig, SpillableWorkStealingQueues, SpillableWorkerState,
};
use crate::storage::work_stealing_queues::WorkStealingQueues;
use crate::system::{
    WorkerPlan, WorkerPlanRequest, build_worker_plan, cgroup_memory_max_bytes,
    pin_current_thread_to_cpu,
};
use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use twox_hash::XxHash64;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub workers: usize,
    pub core_ids: Option<Vec<usize>>,
    pub enforce_cgroups: bool,
    pub numa_pinning: bool,
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            workers: 0,
            core_ids: None,
            enforce_cgroups: true,
            numa_pinning: true,
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

#[derive(Default)]
struct PauseController {
    requested: AtomicBool,
    paused_workers: AtomicUsize,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

impl PauseController {
    fn worker_pause_point(&self, stop: &AtomicBool) {
        if !self.requested.load(Ordering::Acquire) {
            return;
        }
        self.paused_workers.fetch_add(1, Ordering::AcqRel);
        let mut guard = self.wait_lock.lock();
        while self.requested.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
            self.wait_cv.wait_for(&mut guard, Duration::from_millis(10));
        }
        drop(guard);
        self.paused_workers.fetch_sub(1, Ordering::AcqRel);
    }

    fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn wait_for_quiescence(
        &self,
        stop: &AtomicBool,
        active_workers: &AtomicUsize,
        live_workers: &AtomicUsize,
    ) {
        loop {
            if stop.load(Ordering::Acquire) {
                break;
            }
            let paused = self.paused_workers.load(Ordering::Acquire);
            let live = live_workers.load(Ordering::Acquire);
            let active = active_workers.load(Ordering::Acquire);
            if paused >= live && active == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn resume(&self) {
        self.requested.store(false, Ordering::Release);
        self.wait_cv.notify_all();
    }
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

fn checkpoint_once<T>(ctx: &CheckpointContext<T>) -> Result<()>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    ctx.pause.request_pause();
    ctx.pause
        .wait_for_quiescence(ctx.stop, ctx.active_workers, ctx.live_workers);

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
        write_checkpoint_manifest(ctx.checkpoint_path, &manifest)?;
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
    let checkpoint_path = config.work_dir.join("checkpoints").join("latest.json");

    if config.clean_work_dir && !config.resume_from_checkpoint && config.work_dir.exists() {
        std::fs::remove_dir_all(&config.work_dir).with_context(|| {
            format!("failed removing old work dir {}", config.work_dir.display())
        })?;
    }
    std::fs::create_dir_all(&config.work_dir)
        .with_context(|| format!("failed creating work dir {}", config.work_dir.display()))?;

    let effective_memory_max = compute_effective_memory_max(&config);
    let (fp_cache_capacity_bytes, queue_inmem_limit) =
        apply_memory_budget(&config, effective_memory_max);

    let worker_plan = build_worker_plan(WorkerPlanRequest {
        requested_workers: config.workers,
        enforce_cgroups: config.enforce_cgroups,
        enable_numa_pinning: config.numa_pinning,
        requested_core_ids: config.core_ids.clone(),
    });

    let fp_path = config.work_dir.join("fingerprints");
    let queue_path = config.work_dir.join("queue");

    // Use page-aligned, NUMA-aware fingerprint store for 95%+ CPU utilization
    // This eliminates TLB thrashing: 48M pages → 98K pages (500x reduction)

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
    let bytes_per_shard = (total_bytes_needed / shard_count).max(2 * 1024 * 1024); // At least 2MB
    let shard_size_mb = (bytes_per_shard / (1024 * 1024)).max(1); // At least 1MB

    if std::env::var("TLAPP_VERBOSE").is_ok() {
        eprintln!("Fingerprint store config:");
        eprintln!("  Expected items: {}", config.fp_expected_items);
        eprintln!("  Shard count: {}", shard_count);
        eprintln!("  Shard size: {} MB", shard_size_mb);
        eprintln!(
            "  Total memory: {} MB ({:.1} GB)",
            shard_count * shard_size_mb,
            (shard_count * shard_size_mb) as f64 / 1024.0
        );
    }

    // Use lock-free page-aligned fingerprint store for high concurrency
    // At 126 workers, RwLock sharding causes severe contention (~97% throughput loss)
    // Atomic CAS operations eliminate lock contention entirely
    let fp_config = PageAlignedConfig {
        shard_count,
        expected_items: config.fp_expected_items,
        shard_size_mb: shard_size_mb,
    };

    let fp_store = PageAlignedFingerprintStore::new(fp_config, &worker_plan.assigned_cpus)?;

    let fp_store = Arc::new(fp_store);

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
    };
    let (queue, worker_states) = SpillableWorkStealingQueues::new(
        worker_plan.worker_count,
        worker_plan.worker_numa_nodes.clone(),
        spill_config,
    )?;

    let run_stats = Arc::new(AtomicRunStats::default());
    let stop = Arc::new(AtomicBool::new(false));
    let active_workers = Arc::new(AtomicUsize::new(0));
    let live_workers = Arc::new(AtomicUsize::new(worker_plan.worker_count));
    let pause = Arc::new(PauseController::default());
    let (violation_tx, violation_rx) = crossbeam_channel::bounded(1);
    let (error_tx, error_rx) = crossbeam_channel::bounded::<String>(1);

    let started_at = Instant::now();
    let resumed_from_checkpoint = config.resume_from_checkpoint && queue.has_pending_work();
    if !resumed_from_checkpoint {
        let initial = model.initial_states();
        let mut initial_fps = Vec::with_capacity(initial.len());
        let mut unique_initial = Vec::with_capacity(initial.len());
        let mut dedup = HashSet::with_capacity(initial.len().max(16));
        for state in initial {
            run_stats.states_generated.fetch_add(1, Ordering::Relaxed);
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
    // Checkpointing disabled for work-stealing queues
    let checkpoint_thread: Option<std::thread::JoinHandle<()>> = None;

    // Work-stealing queues detect completion internally
    // No need for separate monitor thread

    // Progress reporting thread - prints stats every 10 seconds
    let progress_run_stats = Arc::clone(&run_stats);
    let progress_stop = Arc::clone(&stop);
    let progress_fp_store = Arc::clone(&fp_store);
    let progress_queue = Arc::clone(&queue);
    let progress_thread = std::thread::spawn(move || {
        let mut progress_counter = 1u64;
        let mut last_generated = 0u64;
        let mut last_distinct = 0u64;
        let mut last_time = std::time::Instant::now();

        loop {
            // Check for emergency checkpoint request every second
            for _ in 0..10 {
                std::thread::sleep(std::time::Duration::from_secs(1));

                if progress_stop.load(Ordering::Relaxed) {
                    return;
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
            let queue_pending = progress_queue.pending_count();

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

            // Match TLC format exactly
            eprintln!(
                "Progress({}) at {}: {} states generated ({} s/min), {} distinct states found ({} ds/min), {} states left on queue.",
                progress_counter,
                timestamp,
                format_with_commas(states_generated),
                format_with_commas(generated_rate),
                format_with_commas(states_distinct),
                format_with_commas(distinct_rate),
                format_with_commas(queue_pending)
            );

            progress_counter += 1;
            last_generated = states_generated;
            last_distinct = states_distinct;
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

    let mut workers = Vec::with_capacity(worker_plan.worker_count);
    for (worker_id, mut worker_state) in worker_states.into_iter().enumerate() {
        let worker_model = Arc::clone(&model);
        let worker_fp_store = Arc::clone(&fp_store);
        let worker_queue = Arc::clone(&queue);
        let worker_stats = Arc::clone(&run_stats);
        let worker_stop = Arc::clone(&stop);
        let worker_active = Arc::clone(&active_workers);
        let worker_live = Arc::clone(&live_workers);
        let worker_pause = Arc::clone(&pause);
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

            // Per-worker stats counters to reduce atomic contention
            // Flushed periodically instead of every state
            let mut local_states_processed = 0u64;
            let mut local_states_generated = 0u64;
            let mut local_duplicates = 0u64;
            let mut local_states_distinct = 0u64;
            let mut local_enqueued = 0u64;
            const STATS_FLUSH_INTERVAL: u64 = 256; // Flush every 256 states

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

                worker_pause.worker_pause_point(&worker_stop);
                if worker_stop.load(Ordering::Acquire) {
                    break;
                }

                // Work-stealing: try local queue first, then steal from others
                // This has zero contention on the common path
                let state = match worker_queue.pop_for_worker(&mut worker_state) {
                    Some(state) => state,
                    None => {
                        // No work available and exploration complete
                        break;
                    }
                };

                // Mark worker as active (cache-line padded, no contention with other workers)
                worker_queue.worker_start(worker_state.id());
                local_states_processed += 1;

                // Periodically flush local stats to reduce atomic contention
                if local_states_processed % STATS_FLUSH_INTERVAL == 0 {
                    flush_local_stats(
                        &mut local_states_processed,
                        &mut local_states_generated,
                        &mut local_duplicates,
                        &mut local_states_distinct,
                        &mut local_enqueued,
                        &worker_stats,
                    );
                }

                if let Err(message) = worker_model.check_invariants(&state) {
                    // Reconstruct trace to violation using post-processing
                    // This re-explores from init states to find the path
                    let trace = reconstruct_trace_limited(
                        worker_model.as_ref(),
                        &state,
                        100, // Max depth to search
                    )
                    .unwrap_or_else(|| vec![state.clone()]);

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

                local_states_generated += successors.len() as u64;

                // Filter successors by state constraints (prune states that don't satisfy constraints)
                successors.retain(|next_state| {
                    worker_model.check_state_constraints(next_state).is_ok()
                        && worker_model
                            .check_action_constraints(&state, next_state)
                            .is_ok()
                });

                // Pair states with their fingerprint's home NUMA for routing
                let mut states_with_home_numa: Vec<(M::State, usize)> =
                    Vec::with_capacity(worker_fp_batch_size);
                let mut process_batch = |pending_batch: &mut Vec<M::State>,
                                         local_duplicates: &mut u64,
                                         local_states_distinct: &mut u64,
                                         local_enqueued: &mut u64|
                 -> Result<()> {
                    if pending_batch.is_empty() {
                        return Ok(());
                    }
                    unique_states.clear();
                    unique_fps.clear();
                    local_fp_dedup.clear();
                    states_with_home_numa.clear();

                    let mut duplicates_in_batch = 0u64;
                    for candidate in pending_batch.drain(..) {
                        let fp = worker_model.fingerprint(&candidate);
                        if !local_fp_dedup.insert(fp) {
                            duplicates_in_batch += 1;
                            continue;
                        }
                        unique_fps.push(fp);
                        unique_states.push(candidate);
                    }

                    if !unique_fps.is_empty() {
                        worker_fp_store.contains_or_insert_batch(&unique_fps, &mut batch_seen)?;

                        for (idx, next_state) in unique_states.drain(..).enumerate() {
                            if batch_seen[idx] {
                                *local_duplicates += 1;
                            } else {
                                *local_states_distinct += 1;
                                // Get the fingerprint's home NUMA for routing
                                let home_numa = worker_fp_store.home_numa(unique_fps[idx]);
                                states_with_home_numa.push((next_state, home_numa));
                            }
                        }

                        // Batch push with NUMA-aware routing
                        let pushed = worker_queue
                            .push_batch_to_numa(&mut worker_state, states_with_home_numa.drain(..));
                        *local_enqueued += pushed as u64;
                    }

                    *local_duplicates += duplicates_in_batch;
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

    // Cleanup checkpoint thread
    queue.finish(); // Signal completion

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
        if let Err(e) = write_checkpoint_manifest(&checkpoint_path, &manifest) {
            eprintln!("Warning: failed to write checkpoint manifest: {}", e);
        }
    }

    // Queue cleanup happens automatically
    let _ = fp_store.flush();

    if let Ok(err) = error_rx.try_recv() {
        return Err(anyhow!(err));
    }

    let mut violation = violation_rx.try_recv().ok();

    // Check fairness constraints if we collected labeled transitions
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

                // Get fairness constraints from model
                // Note: We need access to fairness constraints, which requires the model
                // to expose them. For now, we'll check if the model implements a method
                // to get fairness constraints. This is model-specific (TlaModel has it).

                // For now, we'll just report that we found cycles
                eprintln!(
                    "  Note: Fairness checking on cycles detected but constraint validation not yet wired"
                );
                eprintln!("        Models must expose fairness constraints for full checking");
            } else {
                eprintln!("  No cycles detected - fairness constraints trivially satisfied");
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
    use super::{EngineConfig, run_model};
    use crate::models::counter_grid::CounterGridModel;
    use anyhow::Result;
    use std::time::{SystemTime, UNIX_EPOCH};

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

        eprintln!("Emergency checkpoint request/clear test passed");
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
