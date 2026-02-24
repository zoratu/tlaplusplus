use crate::model::Model;
use crate::storage::fingerprint_store::{
    FingerprintStats, FingerprintStore, FingerprintStoreConfig,
};
use crate::storage::queue::{DiskBackedQueue, DiskQueueConfig, QueueStats};
use crate::system::{
    WorkerPlan, WorkerPlanRequest, build_worker_plan, cgroup_memory_max_bytes,
    pin_current_thread_to_cpu,
};
use anyhow::{Context, Result, anyhow};
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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

#[derive(Debug)]
pub struct Violation<S> {
    pub message: String,
    pub state: S,
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
    fingerprints: FingerprintStats,
}

#[inline]
fn fingerprint<T: Hash>(value: &T) -> u64 {
    let mut hasher = XxHash64::default();
    value.hash(&mut hasher);
    hasher.finish()
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
            fingerprints: ctx.fp_store.stats(),
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

    let fp_store = Arc::new(FingerprintStore::new(FingerprintStoreConfig {
        path: fp_path,
        shard_count: config.fp_shards,
        expected_items: config.fp_expected_items,
        false_positive_rate: config.fp_false_positive_rate,
        hot_entries_per_shard: config.fp_hot_entries_per_shard,
        cache_capacity_bytes: fp_cache_capacity_bytes,
        flush_every_ms: config.fp_flush_every_ms,
    })?);

    let queue: Arc<DiskBackedQueue<M::State>> = Arc::new(DiskBackedQueue::new(DiskQueueConfig {
        spill_dir: queue_path,
        inmem_limit: queue_inmem_limit,
        spill_batch: config.queue_spill_batch,
        spill_channel_bound: config.queue_spill_channel_bound,
        load_existing_segments: config.resume_from_checkpoint,
    })?);

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
            let fp = fingerprint(&state);
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
                queue.push(state)?;
                run_stats.enqueued.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    let checkpoint_thread_stop = Arc::new(AtomicBool::new(false));
    let checkpoint_thread = if config.checkpoint_interval_secs > 0 {
        let cp_stop = Arc::clone(&checkpoint_thread_stop);
        let cp_run_stats = Arc::clone(&run_stats);
        let cp_queue = Arc::clone(&queue);
        let cp_fp_store = Arc::clone(&fp_store);
        let cp_pause = Arc::clone(&pause);
        let cp_active_workers = Arc::clone(&active_workers);
        let cp_live_workers = Arc::clone(&live_workers);
        let cp_error_tx = error_tx.clone();
        let cp_stop_signal = Arc::clone(&stop);
        let cp_worker_plan = worker_plan.clone();
        let cp_config = config.clone();
        let cp_checkpoint_path = checkpoint_path.clone();
        let cp_model_name = model.name().to_string();

        Some(std::thread::spawn(move || {
            while !cp_stop.load(Ordering::Acquire) {
                let interval = Duration::from_secs(cp_config.checkpoint_interval_secs.max(1));
                let sleep_step = Duration::from_millis(250);
                let started = Instant::now();
                while started.elapsed() < interval {
                    if cp_stop.load(Ordering::Acquire) {
                        break;
                    }
                    std::thread::sleep(sleep_step);
                }
                if cp_stop.load(Ordering::Acquire) {
                    break;
                }

                if let Err(err) = checkpoint_once(&CheckpointContext {
                    checkpoint_path: &cp_checkpoint_path,
                    model_name: &cp_model_name,
                    started_at,
                    run_stats: &cp_run_stats,
                    queue: &cp_queue,
                    fp_store: &cp_fp_store,
                    pause: &cp_pause,
                    active_workers: &cp_active_workers,
                    live_workers: &cp_live_workers,
                    stop: &cp_stop_signal,
                    worker_plan: &cp_worker_plan,
                    config: &cp_config,
                    effective_memory_max,
                    resumed_from_checkpoint,
                }) {
                    let _ = cp_error_tx.send(format!("checkpoint failed: {err}"));
                    cp_stop_signal.store(true, Ordering::Release);
                    break;
                }
            }
        }))
    } else {
        None
    };

    let mut workers = Vec::with_capacity(worker_plan.worker_count);
    for worker_id in 0..worker_plan.worker_count {
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
        let worker_poll_sleep = config.poll_sleep_ms;
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

                match worker_queue.pop() {
                    Ok(Some(state)) => {
                        worker_active.fetch_add(1, Ordering::AcqRel);
                        worker_stats
                            .states_processed
                            .fetch_add(1, Ordering::Relaxed);

                        if let Err(message) = worker_model.check_invariants(&state) {
                            let _ = worker_violation_tx.try_send(Violation { message, state });
                            if worker_stop_on_violation {
                                worker_stop.store(true, Ordering::Release);
                            }
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

                        let mut process_batch = |pending_batch: &mut Vec<M::State>| -> Result<()> {
                            if pending_batch.is_empty() {
                                return Ok(());
                            }
                            unique_states.clear();
                            unique_fps.clear();
                            local_fp_dedup.clear();

                            let mut duplicates_in_batch = 0u64;
                            for candidate in pending_batch.drain(..) {
                                let fp = fingerprint(&candidate);
                                if !local_fp_dedup.insert(fp) {
                                    duplicates_in_batch += 1;
                                    continue;
                                }
                                unique_fps.push(fp);
                                unique_states.push(candidate);
                            }

                            if !unique_fps.is_empty() {
                                worker_fp_store
                                    .contains_or_insert_batch(&unique_fps, &mut batch_seen)?;

                                for (idx, next_state) in unique_states.drain(..).enumerate() {
                                    if batch_seen[idx] {
                                        worker_stats.duplicates.fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        worker_stats
                                            .states_distinct
                                            .fetch_add(1, Ordering::Relaxed);
                                        worker_queue.push(next_state).with_context(
                                            || "queue push failed while processing successor batch",
                                        )?;
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

                        worker_active.fetch_sub(1, Ordering::AcqRel);
                    }
                    Ok(None) => {
                        if worker_queue.is_drained()
                            && worker_active.load(Ordering::Acquire) == 0
                            && worker_queue.is_drained()
                        {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(worker_poll_sleep));
                    }
                    Err(err) => {
                        let _ = worker_error_tx.send(format!("queue pop failed in worker: {err}"));
                        worker_stop.store(true, Ordering::Release);
                        break;
                    }
                }
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

    checkpoint_thread_stop.store(true, Ordering::Release);
    pause.resume();
    if let Some(handle) = checkpoint_thread
        && handle.join().is_err()
    {
        return Err(anyhow!("checkpoint thread panicked"));
    }

    if config.checkpoint_on_exit {
        checkpoint_once(&CheckpointContext {
            checkpoint_path: &checkpoint_path,
            model_name: model.name(),
            started_at,
            run_stats: &run_stats,
            queue: &queue,
            fp_store: &fp_store,
            pause: &pause,
            active_workers: &active_workers,
            live_workers: &live_workers,
            stop: &stop,
            worker_plan: &worker_plan,
            config: &config,
            effective_memory_max,
            resumed_from_checkpoint,
        })?;
    }

    queue.shutdown()?;
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
            queue: queue.stats(),
            fingerprints: fp_store.stats(),
        },
        violation,
    })
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
        assert!(first.stats.queue.spilled_items > 0);

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
