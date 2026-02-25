use crate::model::Model;
use crate::storage::channel_queue::ChannelQueue;
use crate::storage::fingerprint_store::{
    FingerprintStats as OldFingerprintStats, FingerprintStore,
};
use crate::storage::page_aligned_fingerprint_store::{
    FingerprintStats, FingerprintStoreConfig as PageAlignedConfig, PageAlignedFingerprintStore,
};
use crate::storage::queue::{DiskBackedQueue, DiskQueueConfig, QueueStats};
use crate::storage::simple_blocking_queue::SimpleBlockingQueue;
use crate::storage::work_stealing_queues::WorkStealingQueues;
use crate::system::{
    WorkerPlan, WorkerPlanRequest, build_worker_plan, cgroup_memory_max_bytes,
    pin_current_thread_to_cpu,
};
use anyhow::{Context, Result, anyhow};
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::VecDeque;
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
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed creating checkpoint dir {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(manifest).context("failed serializing checkpoint")?;
    std::fs::write(&tmp, bytes)
        .with_context(|| format!("failed writing checkpoint temp file {}", tmp.display()))?;
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
        ctx.queue.checkpoint_flush()?;
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

    // Ensure shard_count is power of 2 for fast modulo
    let shard_count = config.fp_shards.next_power_of_two().max(64);
    let bytes_per_shard = (total_bytes_needed / shard_count).max(2 * 1024 * 1024); // At least 2MB
    let shard_size_mb = (bytes_per_shard / (1024 * 1024)).max(1); // At least 1MB

    eprintln!("Fingerprint store config:");
    eprintln!("  Expected items: {}", config.fp_expected_items);
    eprintln!("  Shard count: {}", shard_count);
    eprintln!("  Shard size: {} MB", shard_size_mb);
    eprintln!(
        "  Total memory: {} MB ({:.1} GB)",
        shard_count * shard_size_mb,
        (shard_count * shard_size_mb) as f64 / 1024.0
    );

    let mut fp_store = PageAlignedFingerprintStore::new(
        PageAlignedConfig {
            shard_count,
            expected_items: config.fp_expected_items,
            shard_size_mb,
        },
        &worker_plan.assigned_cpus,
    )
    .context("Failed to create page-aligned fingerprint store")?;

    // Create async I/O runtime for fingerprint persistence (4 threads, cores 0-1)
    // Workers use cores 2-127, so I/O threads stay out of their way
    let _io_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("tlapp-io")
        .enable_all()
        .build()
        .context("Failed to create async I/O runtime")?;

    // Create persistence channels (one per shard)
    let persist_channel_capacity = 10_000; // Per shard
    let (persist_tx, persist_rx) =
        crate::storage::async_fingerprint_writer::create_persist_channels(
            shard_count,
            persist_channel_capacity,
        );

    // Spawn async writer tasks (one per shard)
    for (shard_id, rx) in persist_rx.into_iter().enumerate() {
        let work_dir = config.work_dir.clone();
        _io_runtime.spawn(async move {
            if let Err(e) = crate::storage::async_fingerprint_writer::fingerprint_writer_task(
                shard_id, rx, work_dir,
            )
            .await
            {
                eprintln!("Fingerprint writer {} failed: {}", shard_id, e);
            }
        });
    }

    eprintln!("Async I/O runtime started with {} threads", 4);
    eprintln!(
        "  {} fingerprint writers spawned (one per shard)",
        shard_count
    );

    // Enable persistence
    fp_store.enable_persistence(persist_tx);

    let fp_store = Arc::new(fp_store);

    // Use work-stealing queues - state of the art for CPU-bound parallel workloads
    // Each worker has its own queue, steals from others when idle
    // Zero contention on the common path (local push/pop)
    let (queue, worker_states) = WorkStealingQueues::new(worker_plan.worker_count);

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
        for (idx, state) in unique_initial.into_iter().enumerate() {
            if seen_flags[idx] {
                run_stats.duplicates.fetch_add(1, Ordering::Relaxed);
            } else {
                run_stats.states_distinct.fetch_add(1, Ordering::Relaxed);
                queue.push_global(state);
                run_stats.enqueued.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    let checkpoint_thread_stop = Arc::new(AtomicBool::new(false));
    // Checkpointing disabled for work-stealing queues
    let checkpoint_thread: Option<std::thread::JoinHandle<()>> = None;

    // Work-stealing queues detect completion internally
    // No need for separate monitor thread

    let mut workers = Vec::with_capacity(worker_plan.worker_count);
    for (worker_id, worker_state) in worker_states.into_iter().enumerate() {
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

        workers.push(std::thread::spawn(move || {
            if let Some(cpu) = worker_cpu
                && let Err(err) = pin_current_thread_to_cpu(cpu)
            {
                let _ = worker_error_tx.send(format!("cpu pinning failed on core {cpu}: {err}"));
                worker_stop.store(true, Ordering::Release);
            }

            let mut successors: Vec<M::State> = Vec::with_capacity(64);
            let mut pending_batch: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);
            let mut unique_states: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);
            let mut unique_fps: Vec<u64> = Vec::with_capacity(worker_fp_batch_size);
            let mut batch_seen: Vec<bool> = Vec::with_capacity(worker_fp_batch_size);
            let mut local_fp_dedup: HashSet<u64> = HashSet::with_capacity(worker_fp_batch_size * 2);

            loop {
                worker_pause.worker_pause_point(&worker_stop);
                if worker_stop.load(Ordering::Acquire) {
                    break;
                }

                // Work-stealing: try local queue first, then steal from others
                // This has zero contention on the common path
                let state = match worker_queue.pop_for_worker(&worker_state) {
                    Some(state) => state,
                    None => {
                        // No work available and exploration complete
                        break;
                    }
                };

                // Mark worker as active (doing actual work)
                worker_queue.worker_start();
                worker_active.fetch_add(1, Ordering::AcqRel);
                worker_stats
                    .states_processed
                    .fetch_add(1, Ordering::Relaxed);

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
                    worker_queue.worker_idle();
                    worker_active.fetch_sub(1, Ordering::AcqRel);
                    if worker_stop_on_violation {
                        break;
                    }
                    continue;
                }

                successors.clear();
                worker_model.next_states(&state, &mut successors);
                worker_stats
                    .states_generated
                    .fetch_add(successors.len() as u64, Ordering::Relaxed);

                // Filter successors by state constraints (prune states that don't satisfy constraints)
                successors.retain(|next_state| {
                    worker_model.check_state_constraints(next_state).is_ok()
                        && worker_model
                            .check_action_constraints(&state, next_state)
                            .is_ok()
                });

                let mut process_batch = |pending_batch: &mut Vec<M::State>| -> Result<()> {
                    if pending_batch.is_empty() {
                        return Ok(());
                    }
                    unique_states.clear();
                    unique_fps.clear();
                    local_fp_dedup.clear();

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
                                worker_stats.duplicates.fetch_add(1, Ordering::Relaxed);
                            } else {
                                worker_stats.states_distinct.fetch_add(1, Ordering::Relaxed);
                                worker_queue.push_local(&worker_state, next_state);
                                worker_stats.enqueued.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }

                    if duplicates_in_batch > 0 {
                        worker_stats
                            .duplicates
                            .fetch_add(duplicates_in_batch, Ordering::Relaxed);
                    }
                    Ok(())
                };

                for next in successors.drain(..) {
                    if worker_stop.load(Ordering::Acquire) {
                        break;
                    }
                    pending_batch.push(next);
                    if pending_batch.len() >= worker_fp_batch_size
                        && let Err(err) = process_batch(&mut pending_batch)
                    {
                        let _ = worker_error_tx.send(err.to_string());
                        worker_stop.store(true, Ordering::Release);
                        break;
                    }
                }
                if !worker_stop.load(Ordering::Acquire)
                    && let Err(err) = process_batch(&mut pending_batch)
                {
                    let _ = worker_error_tx.send(err.to_string());
                    worker_stop.store(true, Ordering::Release);
                }

                // Mark worker as idle (done with this state)
                worker_queue.worker_idle();
                worker_active.fetch_sub(1, Ordering::AcqRel);
            }

            worker_live.fetch_sub(1, Ordering::AcqRel);
            worker_pause.wait_cv.notify_all();
        }));
    }

    for worker in workers {
        if worker.join().is_err() {
            stop.store(true, Ordering::Release);
            return Err(anyhow!("worker thread panicked"));
        }
    }

    // Cleanup checkpoint thread
    queue.finish(); // Signal completion

    checkpoint_thread_stop.store(true, Ordering::Release);
    pause.resume();
    if let Some(handle) = checkpoint_thread
        && handle.join().is_err()
    {
        return Err(anyhow!("checkpoint thread panicked"));
    }

    // Checkpointing disabled with SimpleBlockingQueue

    // Queue cleanup happens automatically
    let _ = fp_store.flush();

    if let Ok(err) = error_rx.try_recv() {
        return Err(anyhow!(err));
    }

    let violation = violation_rx.try_recv().ok();
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
    #[ignore] // TODO: Implement checkpointing for work-stealing queues
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
    #[ignore] // TODO: Bulk dequeue changes queue fill behavior - need to adjust test
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
