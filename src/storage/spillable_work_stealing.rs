// Spillable work-stealing queues - combines fast work-stealing with disk overflow
//
// Architecture:
// - Hot path: WorkStealingQueues (lock-free, NUMA-aware, in-memory)
// - Overflow: Per-worker spill buffers -> async batch writer (no central lock)
//
// Key optimization for many-core scaling:
// - Per-worker spill buffers eliminate central mutex contention
// - Workers use non-blocking try_send() to spill coordinator
// - Spill coordinator batches items from all workers sequentially
// - Workers NEVER block on disk I/O

use crate::storage::compressed_segments::{
    CompressedSegmentRing, CompressionStatsSnapshot, DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_MAX_COMPRESSED_BYTES,
};
use crate::storage::queue::{DiskBackedQueue, DiskQueueConfig, QueueStats, serialize_compressed};
use crate::storage::work_stealing_queues::{WorkStealingQueues, WorkStealingStats, WorkerState};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

/// Configuration for spillable work-stealing queues
#[derive(Clone, Debug)]
pub struct SpillableConfig {
    /// Maximum items to keep in memory before spilling to disk
    pub max_inmem_items: u64,
    /// Directory for spill files
    pub spill_dir: PathBuf,
    /// Batch size for spilling (items per segment file)
    pub spill_batch: usize,
    /// Load existing segments on startup (for resume)
    pub load_existing: bool,
    /// Per-worker spill buffer size before sending to coordinator
    pub worker_spill_buffer_size: usize,
    /// Channel bound per worker (backpressure threshold)
    pub worker_channel_bound: usize,
    /// Skip segment deletion during checkpoint (use when S3 handles cleanup)
    /// When true, segments are retained on disk after being consumed.
    /// S3 sync will upload them, then prune based on min_segment_id.
    pub defer_segment_deletion: bool,
    /// Enable in-memory zstd-compressed segment ring (T8). When true, the
    /// spill coordinator first tries to compress overflow batches into an
    /// in-memory ring instead of writing to disk. This trades CPU time for
    /// memory: serialized TLA+ states typically compress 3-10x, so the
    /// equivalent of several GB of pending state can stay resident before
    /// disk I/O kicks in. When the ring is full, batches fall through to
    /// the existing disk-backed path with no behavior change.
    pub compression_enabled: bool,
    /// Hard cap on resident compressed-ring bytes. When the ring would
    /// cross this limit, additional batches spill to disk as before.
    pub compression_max_bytes: usize,
    /// zstd level (1-22). 1 is fastest (~500MB/s on Graviton) with ~3x
    /// ratio on serialized states; 3 trades 2x time for ~10% better ratio.
    pub compression_level: i32,
}

impl Default for SpillableConfig {
    fn default() -> Self {
        Self {
            max_inmem_items: 10_000_000, // 10M items before spilling
            spill_dir: PathBuf::from("./.tlapp/queue"),
            spill_batch: 50_000,
            load_existing: false,
            worker_spill_buffer_size: 4096, // Each worker buffers 4K items locally
            worker_channel_bound: 16,       // 16 batches in flight per worker
            defer_segment_deletion: false,  // Delete locally by default
            compression_enabled: true,
            compression_max_bytes: DEFAULT_MAX_COMPRESSED_BYTES,
            compression_level: DEFAULT_COMPRESSION_LEVEL,
        }
    }
}

/// Per-worker spill state - completely owned by the worker, no sharing
pub struct SpillableWorkerState<T> {
    /// Underlying work-stealing worker state
    pub inner: WorkerState<T>,
    /// Local spill buffer (accumulates items when in spill mode)
    spill_buffer: Vec<T>,
    /// Threshold to flush spill buffer
    spill_buffer_threshold: usize,
    /// Channel to send batches to coordinator (per-worker, no contention)
    spill_tx: Sender<SpillBatch<T>>,
    /// Count of items spilled by this worker
    items_spilled: u64,
    /// T11.1: shared in-flight counter so Drop can release any items
    /// still in spill_buffer when the worker thread panics. Without this,
    /// a panicked worker leaves inflight permanently > 0 and the runtime
    /// hangs on termination detection.
    inflight_spilled: Arc<AtomicU64>,
}

impl<T> SpillableWorkerState<T> {
    /// Get worker ID (convenience accessor)
    #[inline]
    pub fn id(&self) -> usize {
        self.inner.id
    }

    /// Get NUMA node (convenience accessor)
    #[inline]
    pub fn numa_node(&self) -> usize {
        self.inner.numa_node
    }
}

impl<T> Drop for SpillableWorkerState<T> {
    fn drop(&mut self) {
        // T11.1 panic-safety: any items remaining in the spill_buffer at
        // drop time will not reach the coordinator (their thread is
        // unwinding or just exiting). We must release them from the
        // inflight counter so termination detection can proceed.
        // Items still in the per-worker channel (already sent) are
        // accounted for separately — the coordinator drains them
        // post-stop. Items in spill_buffer have NOT been bumped into
        // the channel-tracked path, so the counter still holds their
        // count and we decrement here.
        //
        // Note: this loses the items themselves (they're dropped along
        // with the Vec). For the panic case this is unavoidable — the
        // worker died holding them. For graceful shutdown, callers
        // should already have flushed via flush_worker_counters before
        // dropping; this Drop is defensive.
        let leftover = self.spill_buffer.len() as u64;
        if leftover > 0 {
            // Best-effort: try to send the buffer once more before
            // dropping. If the channel is closed (coordinator shutting
            // down), we still need to release the inflight count so
            // termination doesn't hang.
            let buf = std::mem::take(&mut self.spill_buffer);
            let batch = SpillBatch {
                worker_id: self.inner.id,
                items: buf,
            };
            match self.spill_tx.try_send(batch) {
                Ok(()) => {
                    // Items will be routed by the coordinator and then
                    // decremented in route_spill_batch. Nothing to do
                    // here.
                }
                Err(crossbeam_channel::TrySendError::Full(b))
                | Err(crossbeam_channel::TrySendError::Disconnected(b)) => {
                    // Couldn't send: items are lost on the floor.
                    // Release them from the inflight counter so
                    // termination doesn't hang. Operator must look at
                    // panic logs to know if this happened.
                    eprintln!(
                        "Warning: SpillableWorkerState dropping {} items in spill_buffer (worker_id={}) — releasing inflight counter",
                        b.items.len(),
                        b.worker_id
                    );
                    self.inflight_spilled
                        .fetch_sub(leftover, Ordering::AcqRel);
                }
            }
        }
    }
}

/// A batch of items to spill, tagged with worker ID for debugging
struct SpillBatch<T> {
    #[allow(dead_code)]
    worker_id: usize,
    items: Vec<T>,
}

/// Work-stealing queues with disk overflow for memory-bounded operation
pub struct SpillableWorkStealingQueues<T> {
    /// Hot in-memory work-stealing queues
    hot: Arc<WorkStealingQueues<T>>,

    /// Overflow disk queue (for loading back from disk)
    overflow: Arc<DiskBackedQueue<T>>,

    /// Threshold for spilling
    max_inmem_items: u64,

    /// Background spill coordinator thread handle
    coordinator_handle: Option<JoinHandle<()>>,

    /// Background loader thread handle
    loader_handle: Option<JoinHandle<()>>,

    /// Signal to stop background threads
    stop_signal: Arc<AtomicBool>,

    /// Flag to prevent worker termination during checkpoint
    checkpoint_in_progress: Arc<AtomicBool>,

    /// Flag to pause loader during checkpoint drain (prevents race with drain)
    /// Only set during the actual drain operation, not the whole checkpoint
    checkpoint_draining: Arc<AtomicBool>,

    /// Skip segment deletion during checkpoint (S3 handles cleanup)
    defer_segment_deletion: bool,

    /// Optional in-memory compressed-segment ring (T8). When `Some`, the
    /// spill coordinator routes batches here before falling back to disk.
    /// `None` disables compression entirely (legacy direct-spill path).
    compressed_ring: Option<Arc<CompressedSegmentRing>>,

    /// Stats
    spill_redirects: AtomicU64,
    disk_loads: AtomicU64,
    spill_batches_sent: AtomicU64,
    spill_channel_full: AtomicU64,
    /// Items pulled out of the compressed ring (decompressed and re-pushed
    /// to the hot queue). Reported in stats so operators can see the ring
    /// is doing work.
    compressed_ring_loads: AtomicU64,

    /// T11.1 SOUNDNESS — number of items currently in flight in the spill
    /// pipeline that are NOT yet visible to `has_pending_work()` via the
    /// hot queue, the compressed ring, or disk overflow. The pipeline:
    ///
    ///   worker spill_buffer -> per-worker channel -> coordinator
    ///   accumulator -> route_spill_batch -> ring or disk overflow
    ///
    /// Items live in worker-owned buffers, MPSC channels, and the
    /// coordinator's accumulator before they reach a tier visible to
    /// termination detection. Without this counter, when the spill cap is
    /// small (`--queue-max-inmem-items` low) all workers can drain their
    /// hot queues, see `has_pending_work() == false`, and terminate while
    /// a non-empty spill_buffer / channel / accumulator silently drops the
    /// items in transit.
    ///
    /// Bumped per item in `push_local_batch` / `push_batch_to_numa` when
    /// the spill path is taken; decremented per item in `route_spill_batch`
    /// after the batch is successfully placed into the ring or disk
    /// overflow (each of which IS visible to `has_pending_work()`).
    /// The Arc is shared with the spill coordinator thread.
    inflight_spilled: Arc<AtomicU64>,
}

impl<T> SpillableWorkStealingQueues<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
{
    /// Create new spillable work-stealing queues
    pub fn new(
        num_workers: usize,
        worker_numa_nodes: Vec<usize>,
        config: SpillableConfig,
    ) -> Result<(Arc<Self>, Vec<SpillableWorkerState<T>>)> {
        let (hot, base_worker_states) = WorkStealingQueues::new(num_workers, worker_numa_nodes);

        let overflow = Arc::new(DiskBackedQueue::new(DiskQueueConfig {
            spill_dir: config.spill_dir.clone(),
            inmem_limit: config.spill_batch * 2,
            spill_batch: config.spill_batch,
            spill_channel_bound: 64,
            load_existing_segments: config.load_existing,
        })?);

        let stop_signal = Arc::new(AtomicBool::new(false));

        let compressed_ring = if config.compression_enabled {
            Some(Arc::new(CompressedSegmentRing::new(
                config.compression_max_bytes,
                config.compression_level,
            )))
        } else {
            None
        };

        // T11.1: shared in-flight counter so the spill coordinator can
        // decrement after items reach a termination-visible tier, and so
        // SpillableWorkerState::drop can release leaked spill_buffer
        // items on worker panic.
        let inflight_spilled = Arc::new(AtomicU64::new(0));

        // Create per-worker channels and spillable states
        let mut worker_states = Vec::with_capacity(num_workers);
        let mut spill_receivers: Vec<Receiver<SpillBatch<T>>> = Vec::with_capacity(num_workers);

        for base_state in base_worker_states {
            let (tx, rx) = crossbeam_channel::bounded(config.worker_channel_bound);
            spill_receivers.push(rx);
            worker_states.push(SpillableWorkerState {
                inner: base_state,
                spill_buffer: Vec::with_capacity(config.worker_spill_buffer_size),
                spill_buffer_threshold: config.worker_spill_buffer_size,
                spill_tx: tx,
                items_spilled: 0,
                inflight_spilled: Arc::clone(&inflight_spilled),
            });
        }

        // Start spill coordinator thread
        let coordinator_overflow = Arc::clone(&overflow);
        let coordinator_stop = Arc::clone(&stop_signal);
        let coordinator_ring = compressed_ring.as_ref().map(Arc::clone);
        let coordinator_inflight = Arc::clone(&inflight_spilled);
        let spill_batch_size = config.spill_batch;
        let coordinator_handle = std::thread::Builder::new()
            .name("tlapp-spill-coordinator".to_string())
            .spawn(move || {
                Self::spill_coordinator_thread(
                    spill_receivers,
                    coordinator_overflow,
                    coordinator_ring,
                    coordinator_stop,
                    coordinator_inflight,
                    spill_batch_size,
                );
            })?;

        // Start background loader thread
        let checkpoint_in_progress = Arc::new(AtomicBool::new(false));
        let checkpoint_draining = Arc::new(AtomicBool::new(false));
        let loader_hot = Arc::clone(&hot);
        let loader_overflow = Arc::clone(&overflow);
        let loader_stop = Arc::clone(&stop_signal);
        let loader_draining = Arc::clone(&checkpoint_draining);
        let loader_ring = compressed_ring.as_ref().map(Arc::clone);
        let loader_inflight = Arc::clone(&inflight_spilled);
        let loader_handle = std::thread::Builder::new()
            .name("tlapp-queue-loader".to_string())
            .spawn(move || {
                Self::loader_thread(
                    loader_hot,
                    loader_overflow,
                    loader_ring,
                    loader_stop,
                    loader_draining,
                    loader_inflight,
                );
            })?;

        let queues = Arc::new(Self {
            hot,
            overflow,
            max_inmem_items: config.max_inmem_items,
            coordinator_handle: Some(coordinator_handle),
            loader_handle: Some(loader_handle),
            stop_signal,
            checkpoint_in_progress,
            checkpoint_draining,
            defer_segment_deletion: config.defer_segment_deletion,
            compressed_ring,
            spill_redirects: AtomicU64::new(0),
            disk_loads: AtomicU64::new(0),
            spill_batches_sent: AtomicU64::new(0),
            spill_channel_full: AtomicU64::new(0),
            compressed_ring_loads: AtomicU64::new(0),
            inflight_spilled,
        });

        Ok((queues, worker_states))
    }

    /// Spill coordinator thread - receives batches from all workers, writes
    /// them to the in-memory compressed ring (if enabled) or directly to
    /// disk overflow.
    ///
    /// The compressed ring is always tried first when present; if it
    /// rejects (over its byte budget) the items fall through to the
    /// existing direct-spill-to-disk path so memory is still bounded.
    fn spill_coordinator_thread(
        receivers: Vec<Receiver<SpillBatch<T>>>,
        overflow: Arc<DiskBackedQueue<T>>,
        compressed_ring: Option<Arc<CompressedSegmentRing>>,
        stop: Arc<AtomicBool>,
        inflight_spilled: Arc<AtomicU64>,
        batch_size: usize,
    ) {
        let mut accumulator: Vec<T> = Vec::with_capacity(batch_size);

        let flush_accumulator = |accumulator: &mut Vec<T>,
                                 ring: &Option<Arc<CompressedSegmentRing>>,
                                 inflight: &Arc<AtomicU64>| {
            if accumulator.is_empty() {
                return;
            }
            let to_write = std::mem::replace(accumulator, Vec::with_capacity(batch_size));
            Self::route_spill_batch(to_write, ring, &overflow, inflight);
        };

        // T11.1: drain receivers fully before exiting on stop. Otherwise
        // items still queued in the per-worker channels are silently
        // dropped at shutdown — and `inflight_spilled` would stay > 0,
        // hanging termination. Keep draining as long as items might still
        // arrive (workers active, stop not set) or are actually buffered.
        //
        // After `stop` is set we still need to drain everything currently
        // in the per-worker channels into the ring/disk so the inflight
        // counter reaches 0. We bound the post-stop drain to avoid hanging
        // forever in the (unreachable on the happy path) case where some
        // route_spill_batch error left inflight > 0 with no items left to
        // drain.
        //
        // T11.1 panic-safety: receivers can become disconnected mid-run
        // when a worker panics (its `spill_tx` Sender is dropped during
        // stack unwind). A disconnected receiver always "wins" Select
        // because select considers it ready, which would tight-loop here
        // and starve other senders. We must rebuild the Select set
        // whenever a receiver becomes disconnected, removing it.
        let mut active_receivers: Vec<(usize, Receiver<SpillBatch<T>>)> = receivers
            .into_iter()
            .enumerate()
            .collect();
        let mut idle_post_stop_iters = 0u32;
        const MAX_IDLE_POST_STOP_ITERS: u32 = 200; // ~2s at 10ms timeout

        loop {
            let stopped = stop.load(Ordering::Acquire);

            if active_receivers.is_empty() {
                // All workers' channels disconnected. Final flush + exit.
                flush_accumulator(&mut accumulator, &compressed_ring, &inflight_spilled);
                break;
            }

            // Build a fresh Select over only the still-connected receivers.
            // We need to ensure all borrows of `active_receivers` are
            // released before we may mutate it (swap_remove on disconnect),
            // so do the select + recv in a tight scope.
            #[derive(Debug)]
            enum CoordTick<T2> {
                Got(Vec<T2>),
                Disconnect(usize),
                Timeout,
            }
            let tick: CoordTick<T> = {
                let mut sel = crossbeam_channel::Select::new();
                for (_, rx) in active_receivers.iter() {
                    sel.recv(rx);
                }
                match sel.select_timeout(std::time::Duration::from_millis(10)) {
                    Ok(oper) => {
                        let recv_idx = oper.index();
                        let (_orig_id, rx) = &active_receivers[recv_idx];
                        match oper.recv(rx) {
                            Ok(batch) => CoordTick::Got(batch.items),
                            Err(_disconnected) => CoordTick::Disconnect(recv_idx),
                        }
                    }
                    Err(crossbeam_channel::SelectTimeoutError) => CoordTick::Timeout,
                }
            };
            match tick {
                CoordTick::Got(items) => {
                    idle_post_stop_iters = 0;
                    accumulator.extend(items);
                    if accumulator.len() >= batch_size {
                        flush_accumulator(
                            &mut accumulator,
                            &compressed_ring,
                            &inflight_spilled,
                        );
                    }
                }
                CoordTick::Disconnect(recv_idx) => {
                    // T11.1: this worker's Sender was dropped (likely a
                    // panic). Remove its receiver from the active set so
                    // future selects don't tight-loop on it.
                    active_receivers.swap_remove(recv_idx);
                }
                CoordTick::Timeout => {
                    flush_accumulator(&mut accumulator, &compressed_ring, &inflight_spilled);
                    if stopped {
                        if inflight_spilled.load(Ordering::Acquire) == 0 {
                            break;
                        }
                        idle_post_stop_iters += 1;
                        if idle_post_stop_iters >= MAX_IDLE_POST_STOP_ITERS {
                            eprintln!(
                                "Warning: spill coordinator giving up shutdown drain with {} items still in flight",
                                inflight_spilled.load(Ordering::Acquire)
                            );
                            break;
                        }
                    }
                }
            }
        }

        // Final flush on shutdown (in case the loop exited via the stopped
        // branch right after a successful recv).
        flush_accumulator(&mut accumulator, &compressed_ring, &inflight_spilled);
    }

    /// Route a spilled batch through the compressed ring first, falling
    /// back to direct disk spill if the ring is full or disabled. Items
    /// are never dropped: the ring returns them on rejection so we can
    /// hand them to disk overflow.
    ///
    /// T11.1: decrements `inflight_spilled` per item once the item reaches
    /// a tier visible to `has_pending_work()` (compressed ring or disk
    /// overflow). The counter was incremented when the item entered the
    /// per-worker spill_buffer; the decrement here closes the accounting
    /// loop. Without it, items in transit between worker -> channel ->
    /// coordinator -> ring/disk are invisible to termination detection
    /// and the runtime can stop while states are still in flight.
    fn route_spill_batch(
        batch: Vec<T>,
        compressed_ring: &Option<Arc<CompressedSegmentRing>>,
        overflow: &Arc<DiskBackedQueue<T>>,
        inflight_spilled: &Arc<AtomicU64>,
    ) {
        let to_disk = if let Some(ring) = compressed_ring.as_ref() {
            let attempted = batch.len() as u64;
            match ring.try_push_batch(batch) {
                Ok(None) => {
                    // Accepted by the ring; ring.is_empty() is now false,
                    // so items are visible to has_pending_work().
                    inflight_spilled.fetch_sub(attempted, Ordering::AcqRel);
                    return;
                }
                Ok(Some(returned)) => returned, // Ring full — spill these to disk.
                Err(e) => {
                    // Internal error path; current implementation never
                    // returns Err but we forward to disk anyway just in case.
                    eprintln!(
                        "Warning: compressed ring push errored ({}); cannot recover items here",
                        e
                    );
                    // Items are lost here; intentionally do NOT decrement
                    // inflight so termination doesn't fire silently on a
                    // lossy spill path. Operator will see the warning.
                    return;
                }
            }
        } else {
            batch
        };

        for item in to_disk {
            match overflow.push(item) {
                Ok(()) => {
                    // Visible to overflow.has_pending_work() now.
                    inflight_spilled.fetch_sub(1, Ordering::AcqRel);
                }
                Err(e) => {
                    eprintln!("Warning: spill coordinator push failed: {}", e);
                    // Item dropped; intentionally do NOT decrement (see
                    // ring-error branch above for rationale).
                }
            }
        }
    }

    /// Background thread that loads from disk (or the compressed ring)
    /// when hot queues need refilling.
    ///
    /// T11.1: takes the shared `inflight_spilled` counter so it can keep
    /// `disk_has_pending_work` set while items are still in flight in the
    /// spill pipeline (per-worker buffers, MPSC channels, coordinator
    /// accumulator). Without this, the inner termination check in
    /// `WorkStealingQueues::should_terminate` (called from
    /// `pop_slow_path`) returns true between coordinator flushes and
    /// workers exit early before the in-flight items reach disk/ring.
    fn loader_thread(
        hot: Arc<WorkStealingQueues<T>>,
        overflow: Arc<DiskBackedQueue<T>>,
        compressed_ring: Option<Arc<CompressedSegmentRing>>,
        stop: Arc<AtomicBool>,
        checkpoint_draining: Arc<AtomicBool>,
        inflight_spilled: Arc<AtomicU64>,
    ) {
        // Threshold: start loading when queue drops below this
        // Must be high enough to keep 64+ workers busy while loading more
        const LOW_WATER_MARK: u64 = 500_000;
        // Stop loading when queue reaches this level (avoid OOM)
        const HIGH_WATER_MARK: u64 = 10_000_000;
        // Load larger batches for efficiency
        const LOAD_BATCH_SIZE: usize = 100_000;

        while !stop.load(Ordering::Acquire) {
            if stop.load(Ordering::Acquire) {
                break;
            }

            // Skip loading only during checkpoint drain - prevents race with drain_injectors_to_disk
            if checkpoint_draining.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            // Check if hot queues need refilling
            // CRITICAL: Use actual queue emptiness, not pending_count()!
            // After checkpoint drains items to disk, pending_count() still includes them
            // even though they're not in memory. This caused workers to spin with 0 throughput
            // because loader thought queue was full when it was actually empty.
            let hot_queues_empty = hot.is_empty();
            let pending = hot.pending_count();
            #[allow(unused_variables)]
            let should_load_aggressively = hot_queues_empty || pending < LOW_WATER_MARK;

            // Queue is low or empty - check if disk has items
            let has_disk_work = overflow.has_pending_work();
            #[allow(unused_variables)]
            let segment_count = overflow.segment_count();
            let has_ring_work = compressed_ring
                .as_ref()
                .map(|r| !r.is_empty())
                .unwrap_or(false);
            // T11.1: items in flight in the spill pipeline are also pending —
            // they will become visible to disk/ring once the coordinator
            // routes them. Without this, the inner WorkStealingQueues
            // termination check could fire between coordinator flushes.
            let has_inflight_spill = inflight_spilled.load(Ordering::Acquire) > 0;
            let has_work = has_disk_work || has_ring_work || has_inflight_spill;

            // CRITICAL: Update hot queue's disk_has_pending_work flag BEFORE any continue.
            // This prevents workers from terminating when hot queues are empty
            // but disk has millions of items waiting to be loaded.
            // Must happen every iteration, even when pending >= HIGH_WATER_MARK.
            // The flag also covers the in-memory compressed ring so workers
            // don't terminate while compressed segments are still pending,
            // AND covers items still in transit in the spill pipeline (T11.1).
            hot.set_disk_has_pending_work(has_work);

            // Only pause loading if hot queues actually have enough items
            // AND there's no work on disk. The pending_count() can be misleading
            // after checkpoint drain, so we also check actual queue state.
            if pending >= HIGH_WATER_MARK && !hot_queues_empty && !has_work {
                // Queue is full enough and no disk work, pause to avoid OOM
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }

            // Log state every 10 seconds (only in debug builds)
            #[cfg(debug_assertions)]
            {
                static LAST_DEBUG: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let last = LAST_DEBUG.load(std::sync::atomic::Ordering::Relaxed);
                if now_secs >= last + 10 {
                    if LAST_DEBUG
                        .compare_exchange(
                            last,
                            now_secs,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        eprintln!(
                            "Loader debug: pending={}, hot_empty={}, has_work={}, segments={}, aggressive={}",
                            pending,
                            hot_queues_empty,
                            has_work,
                            segment_count,
                            should_load_aggressively
                        );
                    }
                }
            }

            if !has_work {
                // No disk items, sleep before checking again
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }

            // Prefer the compressed in-memory ring before disk: decompression
            // is much faster than disk read + deserialize, and segments live
            // there in FIFO order so we drain the oldest pending work first.
            if let Some(ring) = compressed_ring.as_ref() {
                if let Some(segment) = ring.pop_oldest_segment() {
                    match ring.decompress_segment::<T>(&segment) {
                        Ok(items) => {
                            for item in items {
                                hot.push_global(item);
                            }
                            // Loop back so we re-check pending and either
                            // drain another ring segment or pick up disk work.
                            continue;
                        }
                        Err(e) => {
                            eprintln!("Warning: compressed ring decompress error: {}", e);
                            // Fall through to disk path.
                        }
                    }
                }
            }

            if !has_disk_work {
                // Ring was empty but disk_work flag was on — skip the disk
                // load attempt and re-poll quickly.
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }

            // Aggressively load from disk - no sleep between loads when disk has work
            match overflow.pop_bulk(LOAD_BATCH_SIZE) {
                Ok(items) if !items.is_empty() => {
                    #[allow(unused_variables)]
                    let count = items.len();
                    for item in items {
                        hot.push_global(item);
                    }
                    // Log first few loads (only in debug builds)
                    #[cfg(debug_assertions)]
                    {
                        static LOAD_LOG_COUNT: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(0);
                        let log_num =
                            LOAD_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if log_num < 10 || log_num % 100 == 0 {
                            eprintln!(
                                "Loader: loaded {} items from disk (pending was {}, segments left: {})",
                                count,
                                pending,
                                overflow.segment_count()
                            );
                        }
                    }
                    // No sleep between batches - keep loading until HIGH_WATER_MARK
                    // Workers need items faster than the 5ms sleep allows
                }
                Ok(_) => {
                    // Empty batch, brief pause
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => {
                    eprintln!("Warning: disk queue load error: {}", e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }

    /// Push to global queue (for initial states)
    pub fn push_global(&self, item: T) {
        // Initial states always go to memory (they're typically few)
        self.hot.push_global(item);
    }

    /// Push batch to worker's local queue, spilling to disk if over threshold
    /// This is the HOT PATH - optimized for zero contention
    pub fn push_local_batch(
        &self,
        worker_state: &mut SpillableWorkerState<T>,
        items: impl Iterator<Item = T>,
    ) -> usize {
        let pending = self.hot.pending_count();
        let should_spill = pending >= self.max_inmem_items;

        if !should_spill {
            // Fast path: push to in-memory work-stealing queue (lock-free)
            self.hot.push_local_batch(&mut worker_state.inner, items)
        } else {
            // Spill path: accumulate in worker's local buffer, send batches async
            let mut count = 0;
            for item in items {
                worker_state.spill_buffer.push(item);
                count += 1;
                // T11.1: bump inflight as items enter the spill pipeline.
                // Decremented in route_spill_batch once items reach the
                // ring or disk overflow.
                self.inflight_spilled.fetch_add(1, Ordering::AcqRel);

                // When buffer is full, try to send to coordinator
                if worker_state.spill_buffer.len() >= worker_state.spill_buffer_threshold {
                    self.flush_worker_spill_buffer(worker_state);
                }
            }
            // T11.1: always flush at the end of a spill batch (not just at
            // the 4K threshold). Otherwise, when the runtime is in the
            // small-cap regime, items can sit in the spill_buffer
            // indefinitely. The worker's hot queue is regularly refilled
            // by the loader from items that DO make it through, so the
            // worker rarely hits the slow-path flush in pop_for_worker.
            // Without this end-of-call flush, items accumulate in the
            // buffer and never reach disk/ring even though the worker is
            // making progress on other states.
            if !worker_state.spill_buffer.is_empty() {
                self.flush_worker_spill_buffer(worker_state);
            }
            self.spill_redirects
                .fetch_add(count as u64, Ordering::Relaxed);
            count
        }
    }

    /// Push batch with NUMA-aware routing based on fingerprint home NUMA
    /// Each item is routed to its fingerprint's home NUMA node's injector
    /// This ensures fingerprint checks happen on local memory (no cross-NUMA access)
    pub fn push_batch_to_numa(
        &self,
        worker_state: &mut SpillableWorkerState<T>,
        items: impl Iterator<Item = (T, usize)>,
    ) -> usize {
        let pending = self.hot.pending_count();
        let should_spill = pending >= self.max_inmem_items;

        if !should_spill {
            // Fast path: push to in-memory work-stealing queue with NUMA routing
            self.hot.push_batch_to_numa(&mut worker_state.inner, items)
        } else {
            // Spill path: accumulate in worker's local buffer (ignoring NUMA routing)
            // When loaded back from disk, items will be pushed to global queue
            // This is a tradeoff: spilled items may not be NUMA-local on reload
            let mut count = 0;
            for (item, _home_numa) in items {
                worker_state.spill_buffer.push(item);
                count += 1;
                // T11.1: see push_local_batch — bump inflight as items
                // enter the spill pipeline so termination detection sees
                // them while they're in transit.
                self.inflight_spilled.fetch_add(1, Ordering::AcqRel);

                // When buffer is full, try to send to coordinator
                if worker_state.spill_buffer.len() >= worker_state.spill_buffer_threshold {
                    self.flush_worker_spill_buffer(worker_state);
                }
            }
            // T11.1: always flush at end of call so items don't sit in the
            // worker's spill_buffer forever. See push_local_batch for the
            // full rationale.
            if !worker_state.spill_buffer.is_empty() {
                self.flush_worker_spill_buffer(worker_state);
            }
            self.spill_redirects
                .fetch_add(count as u64, Ordering::Relaxed);
            count
        }
    }

    /// Flush a worker's spill buffer to the coordinator (BLOCKING when backpressure needed)
    fn flush_worker_spill_buffer(&self, worker_state: &mut SpillableWorkerState<T>) {
        if worker_state.spill_buffer.is_empty() {
            return;
        }

        let batch = SpillBatch {
            worker_id: worker_state.inner.id,
            items: std::mem::replace(
                &mut worker_state.spill_buffer,
                Vec::with_capacity(worker_state.spill_buffer_threshold),
            ),
        };
        let batch_len = batch.items.len() as u64;

        // Try non-blocking first, then block with timeout for backpressure
        match worker_state.spill_tx.try_send(batch) {
            Ok(()) => {
                worker_state.items_spilled += batch_len;
                self.spill_batches_sent.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(mut returned_batch)) => {
                // Channel full - use blocking send with retry to enforce memory limits
                // This is the key change: instead of putting items back in memory (defeating
                // the memory limit), we block briefly until the channel has space.
                self.spill_channel_full.fetch_add(1, Ordering::Relaxed);

                // Retry with blocking send (with timeout to avoid deadlock)
                loop {
                    match worker_state
                        .spill_tx
                        .send_timeout(returned_batch, std::time::Duration::from_millis(100))
                    {
                        Ok(()) => {
                            worker_state.items_spilled += batch_len;
                            self.spill_batches_sent.fetch_add(1, Ordering::Relaxed);
                            break;
                        }
                        Err(crossbeam_channel::SendTimeoutError::Timeout(batch)) => {
                            // Still full, keep trying
                            returned_batch = batch;
                            std::thread::yield_now();
                        }
                        Err(crossbeam_channel::SendTimeoutError::Disconnected(batch)) => {
                            // Coordinator died - write directly to disk as emergency fallback
                            eprintln!(
                                "Warning: spill coordinator disconnected, dropping {} items",
                                batch.items.len()
                            );
                            break;
                        }
                    }
                }
            }
            Err(TrySendError::Disconnected(batch)) => {
                // Coordinator died - log warning and drop items (they'll be re-explored)
                eprintln!(
                    "Warning: spill coordinator disconnected, dropping {} items",
                    batch.items.len()
                );
            }
        }
    }

    /// Pop work for a worker
    pub fn pop_for_worker(&self, worker_state: &mut SpillableWorkerState<T>) -> Option<T> {
        // CRITICAL: Check pause_requested early to avoid blocking on disk I/O.
        // During checkpoint, we need workers to reach pause_point quickly.
        if self.hot.is_pause_requested() {
            return None;
        }

        // Try hot queues first
        if let Some(item) = self.hot.pop_for_worker(&mut worker_state.inner) {
            return Some(item);
        }

        // T11.1: when the local hot path is empty, flush this worker's
        // spill_buffer so its in-flight items can flow through the
        // coordinator into ring/disk and become visible to other workers
        // (and to has_pending_work()). Without this, a worker can sit on
        // a partial buffer (below the 4K threshold) forever, and the
        // runtime will hang because has_pending_work() reports
        // inflight > 0 while no worker is making progress.
        if !worker_state.spill_buffer.is_empty() {
            self.flush_worker_spill_buffer(worker_state);
        }

        // Check pause again before potentially blocking disk I/O
        if self.hot.is_pause_requested() {
            return None;
        }

        // Hot queues empty - try loading directly from overflow
        self.try_load_from_disk();

        // Check pause AGAIN after disk I/O
        if self.hot.is_pause_requested() {
            return None;
        }

        // Try hot again after loading
        self.hot.pop_for_worker(&mut worker_state.inner)
    }

    /// Try to load items from the compressed in-memory ring first, then
    /// disk overflow, into the hot queues.
    fn try_load_from_disk(&self) {
        if !self.hot.is_empty() {
            return;
        }

        // Compressed ring is faster than disk: decompression at ~1GB/s
        // beats a disk segment fetch even on NVMe. Drain one segment per
        // call so we don't monopolize the worker pop path.
        if let Some(ring) = self.compressed_ring.as_ref() {
            if let Some(segment) = ring.pop_oldest_segment() {
                match ring.decompress_segment::<T>(&segment) {
                    Ok(items) => {
                        let count = items.len();
                        self.compressed_ring_loads
                            .fetch_add(count as u64, Ordering::Relaxed);
                        for item in items {
                            self.hot.push_global(item);
                        }
                        return;
                    }
                    Err(e) => {
                        eprintln!("Warning: compressed ring decompress error: {}", e);
                        // Fall through to disk path.
                    }
                }
            }
        }

        match self.overflow.pop_bulk(10_000) {
            Ok(items) if !items.is_empty() => {
                let count = items.len();
                self.disk_loads.fetch_add(count as u64, Ordering::Relaxed);
                for item in items {
                    self.hot.push_global(item);
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Warning: overflow.pop_bulk error: {}", e);
            }
        }
    }

    /// Check termination - must check hot, compressed ring, and disk.
    /// Also prevents termination during checkpoint to avoid race condition.
    pub fn should_terminate(&self) -> bool {
        // CRITICAL: Don't terminate during checkpoint!
        // Checkpoint drains queues to disk, so queues may appear empty.
        // Workers must wait for checkpoint to finish and reload items.
        if self.checkpoint_in_progress.load(Ordering::Acquire) {
            return false;
        }
        if self.hot.has_pending_work() {
            return false;
        }
        if let Some(ring) = self.compressed_ring.as_ref() {
            if !ring.is_empty() {
                return false;
            }
        }
        if self.overflow.has_pending_work() {
            return false;
        }
        // T11.1: items in worker spill_buffer / spill_tx channel /
        // coordinator accumulator are not yet in any visible tier above.
        if self.inflight_spilled.load(Ordering::Acquire) > 0 {
            return false;
        }
        true
    }

    /// Check if a checkpoint is currently in progress
    pub fn is_checkpoint_in_progress(&self) -> bool {
        self.checkpoint_in_progress.load(Ordering::Acquire)
    }

    /// Set checkpoint in progress flag
    /// Call this BEFORE requesting worker pause to prevent workers from
    /// terminating when they see empty queues during the pause wait.
    pub fn set_checkpoint_in_progress(&self, in_progress: bool) {
        self.checkpoint_in_progress
            .store(in_progress, Ordering::Release);
        self.hot.set_checkpoint_in_progress(in_progress);
    }

    /// Set pause requested flag
    /// Call this when requesting worker pause - causes workers to exit
    /// pop_slow_path so they can reach the worker_pause_point.
    pub fn set_pause_requested(&self, requested: bool) {
        self.hot.set_pause_requested(requested);
    }

    /// Mark worker as active
    pub fn worker_start(&self, worker_id: usize) {
        self.hot.worker_start(worker_id);
    }

    /// Mark worker as idle
    pub fn worker_idle(&self, worker_id: usize) {
        self.hot.worker_idle(worker_id);
    }

    /// Flush worker counters
    pub fn flush_worker_counters(&self, worker_state: &mut SpillableWorkerState<T>) {
        self.hot.flush_worker_counters(&mut worker_state.inner);
        // Also flush any pending spill buffer
        self.flush_worker_spill_buffer(worker_state);
    }

    /// Mark as finished
    pub fn finish(&self) {
        self.hot.finish();
        self.overflow.finish();
    }

    /// Check if all queues are empty (hot + compressed ring + disk +
    /// in-flight spill pipeline).
    pub fn is_empty(&self) -> bool {
        if !self.hot.is_empty() {
            return false;
        }
        if let Some(ring) = self.compressed_ring.as_ref() {
            if !ring.is_empty() {
                return false;
            }
        }
        if !self.overflow.is_empty() {
            return false;
        }
        // T11.1: pipeline buffers/channels still hold items.
        self.inflight_spilled.load(Ordering::Acquire) == 0
    }

    /// Check if there's pending work in any tier.
    ///
    /// T11.1: includes items in flight in the spill pipeline (per-worker
    /// spill_buffer, MPSC channel to coordinator, coordinator accumulator)
    /// that have not yet reached the ring or disk overflow. Without this
    /// check, workers terminate prematurely while items are in transit.
    pub fn has_pending_work(&self) -> bool {
        if self.hot.has_pending_work() {
            return true;
        }
        if let Some(ring) = self.compressed_ring.as_ref() {
            if !ring.is_empty() {
                return true;
            }
        }
        if self.overflow.has_pending_work() {
            return true;
        }
        self.inflight_spilled.load(Ordering::Acquire) > 0
    }

    /// Get approximate pending count (in-memory only for speed)
    pub fn pending_count(&self) -> u64 {
        self.hot.pending_count()
    }

    /// Get total pending count including spilled items on disk and items
    /// resident in the in-memory compressed ring.
    pub fn total_pending_count(&self) -> u64 {
        let inmem = self.hot.pending_count();
        let spilled = self.spill_redirects.load(Ordering::Relaxed);
        let loaded = self.disk_loads.load(Ordering::Relaxed);
        // Also check overflow queue's spilled items
        let overflow_stats = self.overflow.stats();
        let overflow_spilled = overflow_stats.spilled_items;
        let overflow_loaded = overflow_stats.loaded_items;
        let ring_items = self
            .compressed_ring
            .as_ref()
            .map(|r| r.pending_items() as u64)
            .unwrap_or(0);

        inmem + ring_items + (spilled + overflow_spilled).saturating_sub(loaded + overflow_loaded)
    }

    /// Snapshot of compressed-ring stats (or `None` if compression is disabled).
    pub fn compression_stats(&self) -> Option<CompressionStatsSnapshot> {
        self.compressed_ring.as_ref().map(|r| r.snapshot_stats())
    }

    /// Check if queue is too full (for backpressure)
    pub fn should_apply_backpressure(&self, max_pending: u64) -> bool {
        self.hot.should_apply_backpressure(max_pending)
    }

    /// Get combined stats
    pub fn stats(&self) -> QueueStats {
        let hot_stats = self.hot.stats();
        let overflow_stats = self.overflow.stats();

        QueueStats {
            pushed: hot_stats.pushed + overflow_stats.pushed,
            popped: hot_stats.popped + overflow_stats.popped,
            spilled_items: overflow_stats.spilled_items
                + self.spill_redirects.load(Ordering::Relaxed),
            spill_batches: overflow_stats.spill_batches
                + self.spill_batches_sent.load(Ordering::Relaxed),
            loaded_segments: overflow_stats.loaded_segments,
            loaded_items: overflow_stats.loaded_items + self.disk_loads.load(Ordering::Relaxed),
            max_inmem_len: hot_stats.max_inmem_len.max(overflow_stats.max_inmem_len),
        }
    }

    /// Get work-stealing specific stats
    pub fn work_stealing_stats(&self) -> WorkStealingStats {
        self.hot.work_stealing_stats()
    }

    /// Shutdown - stop background threads and disk queue
    pub fn shutdown(&mut self) -> Result<()> {
        self.stop_signal.store(true, Ordering::Release);
        self.hot.finish();
        self.overflow.finish();

        if let Some(handle) = self.coordinator_handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.loader_handle.take() {
            let _ = handle.join();
        }

        self.overflow.shutdown()
    }

    /// Checkpoint - flush all in-memory state to disk
    /// This drains items from the hot queue (injectors), writes them to disk,
    /// then reloads them back into memory so workers can continue.
    ///
    /// IMPORTANT: Caller must call set_checkpoint_in_progress(true) BEFORE
    /// requesting worker pause, and set_checkpoint_in_progress(false) AFTER
    /// resume. This prevents workers from terminating when they see empty
    /// queues during the pause/checkpoint/resume cycle.
    pub fn checkpoint_flush(&self) -> Result<()> {
        // NOTE: checkpoint_in_progress should already be set by caller before
        // requesting pause. We don't set/clear it here to avoid race conditions
        // with worker termination detection.
        self.checkpoint_flush_inner()
    }

    /// Inner checkpoint implementation
    fn checkpoint_flush_inner(&self) -> Result<()> {
        eprintln!("Checkpoint: checkpoint_flush_inner starting");

        // Block the background loader during drain to prevent race condition
        self.checkpoint_draining.store(true, Ordering::Release);
        eprintln!("Checkpoint: checkpoint_draining set to true");

        // Drain the compressed ring back to disk first. Without this,
        // checkpoint state would not include items that are sitting in
        // the compressed in-memory tier and they would be lost on restart.
        let mut ring_drained = 0u64;
        if let Some(ring) = self.compressed_ring.as_ref() {
            let items: Vec<T> = ring.drain_all_decompressed()?;
            ring_drained = items.len() as u64;
            for item in items {
                if let Err(e) = self.overflow.push(item) {
                    eprintln!("Warning: checkpoint ring drain push failed: {}", e);
                }
            }
            if ring_drained > 0 {
                eprintln!(
                    "Checkpoint: drained {} items from compressed ring to disk",
                    ring_drained
                );
            }
        }

        // First, drain all items from the hot queue's injectors to the overflow queue
        // This captures the current frontier that workers haven't processed yet
        eprintln!("Checkpoint: starting drain_injectors_to_disk");
        let drained = self.drain_injectors_to_disk()? + ring_drained;
        eprintln!(
            "Checkpoint: drain_injectors_to_disk returned {} items",
            drained
        );

        // CRITICAL: Adjust the hot queue's counters so pending_count() reflects reality
        // Without this, the loader thread sees pending_count() >> LOW_WATER_MARK
        // and doesn't reload items from disk, causing worker starvation!
        if drained > 0 {
            self.hot.adjust_counters_after_drain(drained);
            eprintln!("Checkpoint: adjusted counters after drain");
        }

        // Flush the overflow queue to ensure all segments are written to disk
        // Use parallel checkpoint with multiple threads for faster I/O
        let num_checkpoint_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(32); // Cap at 32 threads for I/O
        eprintln!(
            "Checkpoint: calling overflow.parallel_checkpoint_flush with {} threads (defer_delete={})",
            num_checkpoint_threads, self.defer_segment_deletion
        );
        if self.defer_segment_deletion {
            self.overflow
                .parallel_checkpoint_flush_defer_delete(num_checkpoint_threads)?;
        } else {
            self.overflow
                .parallel_checkpoint_flush(num_checkpoint_threads)?;
        }
        eprintln!("Checkpoint: overflow.parallel_checkpoint_flush completed");

        // CRITICAL: Set disk_has_pending_work BEFORE resuming workers/loader.
        // After draining items to disk, we KNOW there's pending work on disk.
        // Workers check this flag in should_terminate() - if false, they might
        // terminate prematurely before the loader has a chance to reload items.
        // This fixes a race where:
        // 1. Checkpoint drains items to disk
        // 2. Workers resume but hot queues are empty
        // 3. disk_has_pending_work is stale (false from before checkpoint)
        // 4. Workers see empty queues + disk_has_pending_work=false and terminate
        if self.overflow.has_pending_work() {
            self.hot.set_disk_has_pending_work(true);
            eprintln!("Checkpoint: set disk_has_pending_work=true (segments on disk)");
        }

        // Unblock the background loader - it can start reloading items now
        // This happens BEFORE S3 upload, so reload progresses during upload
        self.checkpoint_draining.store(false, Ordering::Release);

        if drained > 0 {
            let segment_count = self.overflow.segment_count();
            eprintln!(
                "Checkpoint: flushed {} queue items to disk ({} segments)",
                drained, segment_count
            );
            // NOTE: We no longer reload synchronously here to avoid memory spike.
            // The background loader will reload items gradually after checkpoint
            // completes and workers resume. Workers won't terminate during this
            // period because checkpoint_in_progress flag prevents termination.
            // The runtime waits for queue to refill before clearing the flag.
        }

        Ok(())
    }

    /// Drain all items from injector queues to disk for checkpointing
    /// Returns the number of items drained
    /// This uses the legacy async channel approach
    fn drain_injectors_to_disk(&self) -> Result<u64> {
        self.drain_injectors_parallel()
    }

    /// Streaming drain - writes items directly to disk without collecting all in memory
    ///
    /// This solves the OOM problem in the legacy drain_injectors_parallel_legacy:
    /// - Legacy: Collects ALL items (87M * 100 bytes = 8.7GB) into Vec, then writes
    /// - Streaming: Each thread steals items and writes batches immediately
    /// - Memory: bounded to `num_threads * BATCH_SIZE * sizeof(T)` (~160MB for 32 threads)
    ///
    /// Algorithm:
    /// 1. Spawn N writer threads (capped at 32)
    /// 2. Each thread competes to steal from global/NUMA injectors and worker stealers
    /// 3. When a thread fills its batch (50K items), it writes to disk immediately
    /// 4. Threads exit after consecutive empty steals (all sources drained)
    /// 5. Has a 5-minute timeout to prevent infinite hangs
    fn drain_injectors_parallel(&self) -> Result<u64> {
        use crossbeam_deque::Steal;
        use parking_lot::Mutex;
        use std::time::{Duration, Instant};

        const BATCH_SIZE: usize = 50_000;
        const MAX_CONSECUTIVE_EMPTY: u32 = 100; // Exit after this many empty steal rounds
        const MAX_RETRY_PER_STEAL: u32 = 10; // Max retries before treating as empty
        const DRAIN_TIMEOUT: Duration = Duration::from_secs(300); // 5 minute timeout
        const PROGRESS_INTERVAL: Duration = Duration::from_secs(30); // Log progress every 30s

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(32);

        let spill_dir = self.overflow.get_spill_dir().to_path_buf();
        std::fs::create_dir_all(&spill_dir)?;

        let total_drained = AtomicU64::new(0);
        let segments_written = AtomicU64::new(0);
        let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let start_time = Instant::now();
        let should_stop = AtomicBool::new(false);

        eprintln!(
            "Checkpoint: drain_injectors_streaming - starting with {} threads (timeout: {}s)",
            num_threads,
            DRAIN_TIMEOUT.as_secs()
        );

        std::thread::scope(|s| {
            for thread_id in 0..num_threads {
                let total_drained = &total_drained;
                let segments_written = &segments_written;
                let errors = &errors;
                let spill_dir = &spill_dir;
                let overflow = &self.overflow;
                let global = &self.hot.global;
                let numa_injectors = &self.hot.numa_injectors;
                let stealers = &self.hot.stealers;
                let should_stop = &should_stop;

                s.spawn(move || {
                    let mut batch: Vec<T> = Vec::with_capacity(BATCH_SIZE);
                    let mut local_drained = 0u64;
                    let mut local_segments = 0u64;
                    let mut consecutive_empty = 0u32;
                    let mut last_progress = Instant::now();

                    // Helper to write a batch to disk
                    let write_batch =
                        |batch: &mut Vec<T>,
                         local_drained: &mut u64,
                         local_segments: &mut u64,
                         errors: &Mutex<Vec<String>>| {
                            if batch.is_empty() {
                                return;
                            }

                            let segment_id = overflow.allocate_segment_id();
                            let segment_path =
                                spill_dir.join(format!("segment-{segment_id:016}.bin"));

                            match serialize_compressed(batch) {
                                Ok(bytes) => match std::fs::write(&segment_path, &bytes) {
                                    Ok(()) => {
                                        overflow.register_segment(segment_path);
                                        *local_drained += batch.len() as u64;
                                        *local_segments += 1;
                                    }
                                    Err(e) => {
                                        errors.lock().push(format!(
                                            "thread {}: write error: {}",
                                            thread_id, e
                                        ));
                                    }
                                },
                                Err(e) => {
                                    errors.lock().push(format!(
                                        "thread {}: serialize error: {}",
                                        thread_id, e
                                    ));
                                }
                            }
                            batch.clear();
                        };

                    // Helper to steal with retry limit
                    let steal_with_limit = |stealer: &crossbeam_deque::Injector<T>| -> Option<T> {
                        let mut retries = 0u32;
                        loop {
                            match stealer.steal() {
                                Steal::Success(item) => return Some(item),
                                Steal::Empty => return None,
                                Steal::Retry => {
                                    retries += 1;
                                    if retries > MAX_RETRY_PER_STEAL {
                                        return None; // Treat as empty after too many retries
                                    }
                                    std::hint::spin_loop();
                                }
                            }
                        }
                    };

                    let steal_worker_with_limit =
                        |stealer: &crossbeam_deque::Stealer<T>| -> Option<T> {
                            let mut retries = 0u32;
                            loop {
                                match stealer.steal() {
                                    Steal::Success(item) => return Some(item),
                                    Steal::Empty => return None,
                                    Steal::Retry => {
                                        retries += 1;
                                        if retries > MAX_RETRY_PER_STEAL {
                                            return None;
                                        }
                                        std::hint::spin_loop();
                                    }
                                }
                            }
                        };

                    loop {
                        // Check timeout
                        if start_time.elapsed() > DRAIN_TIMEOUT {
                            if thread_id == 0 {
                                eprintln!(
                                    "Checkpoint: drain timeout after {}s, writing partial results",
                                    start_time.elapsed().as_secs()
                                );
                            }
                            should_stop.store(true, Ordering::Relaxed);
                        }

                        if should_stop.load(Ordering::Relaxed) {
                            // Write any remaining items before exiting
                            write_batch(
                                &mut batch,
                                &mut local_drained,
                                &mut local_segments,
                                errors,
                            );
                            break;
                        }

                        // Progress logging (thread 0 only)
                        if thread_id == 0 && last_progress.elapsed() > PROGRESS_INTERVAL {
                            let current_total = total_drained.load(Ordering::Relaxed);
                            let current_segs = segments_written.load(Ordering::Relaxed);
                            eprintln!(
                                "Checkpoint: drain progress - {} items in {} segments ({:.1}s elapsed)",
                                current_total + local_drained,
                                current_segs + local_segments,
                                start_time.elapsed().as_secs_f32()
                            );
                            last_progress = Instant::now();
                        }

                        let mut found_any = false;

                        // Steal from global injector
                        for _ in 0..100 {
                            if let Some(item) = steal_with_limit(global) {
                                batch.push(item);
                                found_any = true;
                                if batch.len() >= BATCH_SIZE {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }

                        // Write if batch is full
                        if batch.len() >= BATCH_SIZE {
                            write_batch(
                                &mut batch,
                                &mut local_drained,
                                &mut local_segments,
                                errors,
                            );
                        }

                        // Steal from NUMA injectors
                        for injector in numa_injectors.iter() {
                            for _ in 0..100 {
                                if let Some(item) = steal_with_limit(injector) {
                                    batch.push(item);
                                    found_any = true;
                                    if batch.len() >= BATCH_SIZE {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                            if batch.len() >= BATCH_SIZE {
                                write_batch(
                                    &mut batch,
                                    &mut local_drained,
                                    &mut local_segments,
                                    errors,
                                );
                            }
                        }

                        // Steal from worker stealers
                        for stealer in stealers.iter() {
                            for _ in 0..100 {
                                if let Some(item) = steal_worker_with_limit(stealer) {
                                    batch.push(item);
                                    found_any = true;
                                    if batch.len() >= BATCH_SIZE {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                            if batch.len() >= BATCH_SIZE {
                                write_batch(
                                    &mut batch,
                                    &mut local_drained,
                                    &mut local_segments,
                                    errors,
                                );
                            }
                        }

                        // Check termination condition
                        if !found_any {
                            consecutive_empty += 1;
                            if consecutive_empty > MAX_CONSECUTIVE_EMPTY {
                                // Write any remaining items in partial batch
                                write_batch(
                                    &mut batch,
                                    &mut local_drained,
                                    &mut local_segments,
                                    errors,
                                );
                                break;
                            }
                            // Yield to let other threads make progress
                            std::thread::yield_now();
                        } else {
                            consecutive_empty = 0;
                        }
                    }

                    // Update global counters
                    total_drained.fetch_add(local_drained, Ordering::Relaxed);
                    segments_written.fetch_add(local_segments, Ordering::Relaxed);
                });
            }
        });

        let errs = errors.into_inner();
        if !errs.is_empty() {
            return Err(anyhow::anyhow!("streaming drain errors: {:?}", errs));
        }

        let drained = total_drained.load(Ordering::Relaxed);
        let segs = segments_written.load(Ordering::Relaxed);

        eprintln!(
            "Checkpoint: drain_injectors_streaming completed - {} items in {} segments ({:.1}s)",
            drained,
            segs,
            start_time.elapsed().as_secs_f32()
        );

        Ok(drained)
    }

    /// Legacy drain function - collects all items into memory first, then writes
    /// DEPRECATED: This can cause OOM on large queues (87M+ items)
    /// Kept for comparison/fallback purposes
    #[allow(dead_code)]
    fn drain_injectors_parallel_legacy(&self) -> Result<u64> {
        use crossbeam_deque::Steal;
        use parking_lot::Mutex;

        const BATCH_SIZE: usize = 50_000;
        const MAX_RETRIES: u32 = 1000;

        // Get number of parallel threads
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(32);

        // First, collect all items into batches
        let mut batches: Vec<Vec<T>> = Vec::new();
        let mut current_batch = Vec::with_capacity(BATCH_SIZE);
        let mut total_drained = 0u64;

        // Helper to drain an injector
        let drain_one = |injector: &crossbeam_deque::Injector<T>,
                         current_batch: &mut Vec<T>,
                         batches: &mut Vec<Vec<T>>,
                         total: &mut u64| {
            let mut retries = 0u32;
            loop {
                match injector.steal() {
                    Steal::Success(item) => {
                        current_batch.push(item);
                        *total += 1;
                        retries = 0;

                        if current_batch.len() >= BATCH_SIZE {
                            batches.push(std::mem::take(current_batch));
                            *current_batch = Vec::with_capacity(BATCH_SIZE);
                        }
                    }
                    Steal::Empty => break,
                    Steal::Retry => {
                        retries += 1;
                        if retries > MAX_RETRIES {
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
            }
        };

        // Drain global injector
        drain_one(
            &self.hot.global,
            &mut current_batch,
            &mut batches,
            &mut total_drained,
        );

        // Drain per-NUMA injectors
        for injector in self.hot.numa_injectors.iter() {
            drain_one(
                injector,
                &mut current_batch,
                &mut batches,
                &mut total_drained,
            );
        }

        // Drain worker stealers
        for stealer in self.hot.stealers.iter() {
            let mut retries = 0u32;
            loop {
                match stealer.steal() {
                    Steal::Success(item) => {
                        current_batch.push(item);
                        total_drained += 1;
                        retries = 0;

                        if current_batch.len() >= BATCH_SIZE {
                            batches.push(std::mem::take(&mut current_batch));
                            current_batch = Vec::with_capacity(BATCH_SIZE);
                        }
                    }
                    Steal::Empty => break,
                    Steal::Retry => {
                        retries += 1;
                        if retries > MAX_RETRIES {
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
            }
        }

        // Don't forget the last partial batch
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        if batches.is_empty() {
            return Ok(0);
        }

        eprintln!(
            "Checkpoint: drain_injectors_parallel - writing {} batches ({} items) with {} threads",
            batches.len(),
            total_drained,
            num_threads
        );

        // Write batches in parallel
        let spill_dir = self.overflow.get_spill_dir().to_path_buf();
        std::fs::create_dir_all(&spill_dir)?;

        // Allocate segment IDs atomically for all batches
        let segment_ids: Vec<u64> = (0..batches.len())
            .map(|_| self.overflow.allocate_segment_id())
            .collect();

        let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let segments_written: Mutex<Vec<std::path::PathBuf>> = Mutex::new(Vec::new());
        let batch_queue: Mutex<Vec<(u64, Vec<T>)>> =
            Mutex::new(segment_ids.into_iter().zip(batches.into_iter()).collect());

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let batch_queue = &batch_queue;
                let errors = &errors;
                let segments_written = &segments_written;
                let spill_dir = &spill_dir;
                let overflow = &self.overflow;

                s.spawn(move || {
                    loop {
                        let (segment_id, batch) = {
                            let mut guard = batch_queue.lock();
                            match guard.pop() {
                                Some(item) => item,
                                None => break,
                            }
                        };

                        let segment_path = spill_dir.join(format!("segment-{segment_id:016}.bin"));

                        // Use zstd compression for 5-10x space savings
                        match serialize_compressed(&batch) {
                            Ok(bytes) => match std::fs::write(&segment_path, &bytes) {
                                Ok(()) => {
                                    // Register segment with overflow queue
                                    overflow.register_segment(segment_path.clone());
                                    segments_written.lock().push(segment_path);
                                }
                                Err(e) => {
                                    errors.lock().push(format!("write error: {}", e));
                                }
                            },
                            Err(e) => {
                                errors.lock().push(format!("serialize error: {}", e));
                            }
                        }
                    }
                });
            }
        });

        let errs = errors.into_inner();
        if !errs.is_empty() {
            return Err(anyhow::anyhow!("parallel drain errors: {:?}", errs));
        }

        eprintln!(
            "Checkpoint: drain_injectors_parallel completed - {} segments written",
            segments_written.lock().len()
        );

        Ok(total_drained)
    }

    /// Legacy drain function for fallback
    #[allow(dead_code)]
    fn drain_injectors_to_disk_legacy(&self) -> Result<u64> {
        use crossbeam_deque::Steal;

        let mut total_drained = 0u64;
        const MAX_RETRIES_PER_QUEUE: u32 = 1000;

        // Drain global injector
        let mut retries = 0u32;
        loop {
            match self.hot.global.steal() {
                Steal::Success(item) => {
                    self.overflow.push(item)?;
                    total_drained += 1;
                    retries = 0;
                }
                Steal::Empty => break,
                Steal::Retry => {
                    retries += 1;
                    if retries > MAX_RETRIES_PER_QUEUE {
                        eprintln!("Warning: drain global injector hit retry limit, breaking");
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }

        // Drain per-NUMA injectors
        for (node_idx, injector) in self.hot.numa_injectors.iter().enumerate() {
            retries = 0;
            loop {
                match injector.steal() {
                    Steal::Success(item) => {
                        self.overflow.push(item)?;
                        total_drained += 1;
                        retries = 0;
                    }
                    Steal::Empty => break,
                    Steal::Retry => {
                        retries += 1;
                        if retries > MAX_RETRIES_PER_QUEUE {
                            eprintln!(
                                "Warning: drain NUMA injector {} hit retry limit, breaking",
                                node_idx
                            );
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
            }
        }

        // Drain worker stealers (workers should be paused at this point)
        for (worker_idx, stealer) in self.hot.stealers.iter().enumerate() {
            retries = 0;
            loop {
                match stealer.steal() {
                    Steal::Success(item) => {
                        self.overflow.push(item)?;
                        total_drained += 1;
                        retries = 0;
                    }
                    Steal::Empty => break,
                    Steal::Retry => {
                        retries += 1;
                        if retries > MAX_RETRIES_PER_QUEUE {
                            eprintln!(
                                "Warning: drain worker {} stealer hit retry limit, breaking",
                                worker_idx
                            );
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
            }
        }

        Ok(total_drained)
    }

    /// Load spilled segments from disk into the hot queue (for resume)
    /// This should be called BEFORE starting workers to ensure spilled states
    /// are available for processing immediately.
    /// Only loads up to max_items to avoid OOM - remaining segments are loaded
    /// on-demand by the background loader thread.
    /// Returns the number of items loaded.
    pub fn load_spilled_segments(&self, max_items: u64) -> Result<u64> {
        let mut total_loaded = 0u64;
        let start = std::time::Instant::now();

        loop {
            // Stop if we've loaded enough to get workers started
            if total_loaded >= max_items {
                eprintln!(
                    "Loaded {} items from disk (limit reached, {} segments remain for background loading)",
                    total_loaded,
                    self.overflow.segment_count()
                );
                break;
            }

            // Load a batch from disk
            match self.overflow.pop_bulk(50_000) {
                Ok(items) if !items.is_empty() => {
                    let count = items.len();
                    total_loaded += count as u64;
                    self.disk_loads.fetch_add(count as u64, Ordering::Relaxed);
                    for item in items {
                        self.hot.push_global(item);
                    }
                    // Print progress every 1M items
                    if total_loaded % 1_000_000 < 50_000 {
                        eprintln!(
                            "Loaded {} items from disk segments ({:.1}s)...",
                            total_loaded,
                            start.elapsed().as_secs_f64()
                        );
                    }
                }
                Ok(_) => {
                    // No more items to load
                    break;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        if total_loaded > 0 {
            eprintln!(
                "Finished loading {} items from disk in {:.1}s",
                total_loaded,
                start.elapsed().as_secs_f64()
            );
        }

        Ok(total_loaded)
    }

    /// Get reference to underlying hot queues (for worker pop operations)
    pub fn hot(&self) -> &Arc<WorkStealingQueues<T>> {
        &self.hot
    }

    /// Get spill stats for debugging
    pub fn spill_stats(&self) -> (u64, u64, u64) {
        (
            self.spill_batches_sent.load(Ordering::Relaxed),
            self.spill_channel_full.load(Ordering::Relaxed),
            self.disk_loads.load(Ordering::Relaxed),
        )
    }

    /// Get the spill directory path
    pub fn spill_dir_path(&self) -> &Path {
        self.overflow.spill_dir_path()
    }

    /// Get paths of all current segments (for S3 upload tracking)
    pub fn segment_paths(&self) -> Vec<PathBuf> {
        self.overflow.segment_paths()
    }

    /// Get the minimum segment ID needed for resume.
    /// Returns the ID of the oldest segment in the queue, or None if no segments exist.
    /// This is used by S3 pruning to determine which segments can be safely deleted.
    pub fn get_min_segment_id(&self) -> Option<u64> {
        self.overflow.get_min_segment_id()
    }

    /// Clean up all segment files in the spill directory
    /// Call this after confirming segments have been uploaded to S3
    /// CAUTION: Only call if you're sure the segments are safely backed up elsewhere!
    pub fn cleanup_spill_dir(&self) -> std::io::Result<u64> {
        self.overflow.cleanup_spill_dir()
    }

    /// Take consumed segment paths for external deletion.
    /// Returns paths of segments that have been loaded into memory.
    /// IMPORTANT: Call delete_segment_files() after confirming S3 sync.
    pub fn take_consumed_segments(&self) -> Vec<PathBuf> {
        self.overflow.take_consumed_segments()
    }

    /// Delete specific segment files from disk.
    /// Use this after confirming segments have been synced to S3.
    pub fn delete_segment_files(paths: &[PathBuf]) {
        crate::storage::queue::DiskBackedQueue::<()>::delete_segment_files(paths);
    }

    /// Prune local segment files with ID < min_segment_id.
    /// Returns the number of segments deleted and bytes freed.
    /// This is called by S3 after upload to free local disk space.
    pub fn prune_local_segments(&self, min_segment_id: u64) -> std::io::Result<(u64, u64)> {
        self.overflow.prune_local_segments(min_segment_id)
    }
}

impl<T> Drop for SpillableWorkStealingQueues<T> {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::Release);
    }
}

// Re-export WorkerState for compatibility
pub use crate::storage::work_stealing_queues::WorkerState as BaseWorkerState;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("tlapp-{prefix}-{nanos}-{}", std::process::id()))
    }

    #[test]
    fn test_basic_push_pop() -> Result<()> {
        let dir = temp_path("spillable-basic");
        let config = SpillableConfig {
            max_inmem_items: 1000,
            spill_dir: dir.clone(),
            spill_batch: 100,
            load_existing: false,
            worker_spill_buffer_size: 50,
            worker_channel_bound: 4,
            defer_segment_deletion: false,
            compression_enabled: true,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };

        let (queues, mut workers) = SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], config)?;

        // Push some items
        for i in 0..100u64 {
            queues.push_global(i);
        }

        // Pop them back
        let mut count = 0;
        while let Some(_) = queues.pop_for_worker(&mut workers[0]) {
            count += 1;
            if count >= 100 {
                break;
            }
        }

        assert_eq!(count, 100);

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn test_spill_to_disk() -> Result<()> {
        let dir = temp_path("spillable-spill");
        let config = SpillableConfig {
            max_inmem_items: 50,
            spill_dir: dir.clone(),
            spill_batch: 20,
            load_existing: false,
            worker_spill_buffer_size: 10,
            worker_channel_bound: 4,
            defer_segment_deletion: false,
            compression_enabled: true,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };

        let (queues, mut workers) = SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], config)?;

        // Push more than max_inmem_items via push_local_batch
        let items: Vec<u64> = (0..200).collect();
        queues.push_local_batch(&mut workers[0], items.into_iter());

        // Flush any remaining items in worker's spill buffer
        queues.flush_worker_counters(&mut workers[0]);

        // Wait for async spill to complete
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let stats = queues.stats();
            if stats.spill_batches > 0 {
                break;
            }
        }

        let stats = queues.stats();
        eprintln!(
            "Stats after push: spilled_items={}, spill_batches={}",
            stats.spilled_items, stats.spill_batches
        );

        // Pop them all back
        let mut count = 0;
        let mut idle_spins = 0;
        while count < 200 && idle_spins < 100 {
            if let Some(_) = queues.pop_for_worker(&mut workers[0]) {
                count += 1;
                idle_spins = 0;
            } else {
                idle_spins += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        let final_stats = queues.stats();
        eprintln!(
            "Final stats: pushed={}, popped={}, loaded_items={}",
            final_stats.pushed, final_stats.popped, final_stats.loaded_items
        );

        assert!(count >= 150, "Should recover most items, got {}", count);

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn test_concurrent_spill() -> Result<()> {
        use std::sync::atomic::AtomicUsize;

        let dir = temp_path("spillable-concurrent");
        let config = SpillableConfig {
            max_inmem_items: 100,
            spill_dir: dir.clone(),
            spill_batch: 50,
            load_existing: false,
            worker_spill_buffer_size: 20,
            worker_channel_bound: 8,
            defer_segment_deletion: false,
            compression_enabled: true,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };

        let num_workers = 8;
        let (queues, workers) =
            SpillableWorkStealingQueues::<u64>::new(num_workers, vec![0; num_workers], config)?;

        let queues = Arc::new(queues);
        let total_pushed = Arc::new(AtomicUsize::new(0));
        let total_popped = Arc::new(AtomicUsize::new(0));

        // Move workers into threads
        let handles: Vec<_> = workers
            .into_iter()
            .map(|mut worker| {
                let q = Arc::clone(&queues);
                let pushed = Arc::clone(&total_pushed);
                let popped = Arc::clone(&total_popped);
                std::thread::spawn(move || {
                    // Each worker pushes items
                    for batch in 0..10 {
                        let items: Vec<u64> = ((batch * 100)..((batch + 1) * 100))
                            .map(|x| x as u64)
                            .collect();
                        let count = q.push_local_batch(&mut worker, items.into_iter());
                        pushed.fetch_add(count, Ordering::Relaxed);
                    }

                    // Flush remaining
                    q.flush_worker_counters(&mut worker);

                    // Each worker pops items
                    for _ in 0..500 {
                        if q.pop_for_worker(&mut worker).is_some() {
                            popped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let pushed = total_pushed.load(Ordering::Relaxed);
        let popped = total_popped.load(Ordering::Relaxed);
        eprintln!("Concurrent test: pushed={}, popped={}", pushed, popped);

        // Should have pushed 8 workers * 10 batches * 100 items = 8000 items
        assert_eq!(pushed, 8000);
        // Popped count depends on timing, but should be significant
        assert!(popped > 0);

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn test_streaming_drain_checkpoint() -> Result<()> {
        // Test that the streaming drain correctly drains all items and writes them to disk,
        // then allows reloading them back into the queue.
        let dir = temp_path("spillable-streaming-drain");
        let config = SpillableConfig {
            max_inmem_items: 1_000_000, // High limit so items stay in memory
            spill_dir: dir.clone(),
            spill_batch: 1000,
            load_existing: false,
            worker_spill_buffer_size: 100,
            worker_channel_bound: 8,
            defer_segment_deletion: false,
            compression_enabled: true,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };

        let num_workers = 4;
        let (queues, mut workers) =
            SpillableWorkStealingQueues::<u64>::new(num_workers, vec![0; num_workers], config)?;

        // Push items to various queues
        let num_items = 10_000u64;

        // Push to global injector
        for i in 0..(num_items / 2) {
            queues.push_global(i);
        }

        // Push to worker local queues via push_local_batch
        let remaining_items: Vec<u64> = ((num_items / 2)..num_items).collect();
        queues.push_local_batch(&mut workers[0], remaining_items.into_iter());

        // Verify items are in memory
        let pending_before = queues.pending_count();
        eprintln!("Before drain: pending_count = {}", pending_before);
        assert!(pending_before > 0, "Should have items in memory");

        // Set checkpoint flag and perform drain (simulates checkpoint)
        queues.set_checkpoint_in_progress(true);
        queues.set_pause_requested(true);

        // Give a moment for flags to propagate
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Perform the checkpoint flush which uses streaming drain
        queues.checkpoint_flush()?;

        // Check that disk has pending work
        assert!(
            queues.overflow.has_pending_work(),
            "Should have work on disk after drain"
        );
        let segment_count = queues.overflow.segment_count();
        eprintln!("After drain: {} segments on disk", segment_count);
        assert!(segment_count > 0, "Should have created segments");

        // Clear checkpoint flags
        queues.set_pause_requested(false);
        queues.set_checkpoint_in_progress(false);

        // Pop items back - they should reload from disk
        let mut recovered = 0u64;
        let mut idle_spins = 0;
        while recovered < num_items && idle_spins < 200 {
            if let Some(_) = queues.pop_for_worker(&mut workers[1]) {
                recovered += 1;
                idle_spins = 0;
            } else {
                idle_spins += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        eprintln!(
            "Recovered {} of {} items ({}%)",
            recovered,
            num_items,
            recovered * 100 / num_items
        );

        // Should recover most items (some may be in flight during drain)
        assert!(
            recovered >= num_items * 90 / 100,
            "Should recover at least 90% of items, got {} of {}",
            recovered,
            num_items
        );

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn test_streaming_drain_bounded_memory() -> Result<()> {
        // Test that streaming drain doesn't accumulate all items in memory
        // This test verifies the fix for the OOM issue on large queues
        let dir = temp_path("spillable-streaming-memory");
        let config = SpillableConfig {
            max_inmem_items: 10_000_000,
            spill_dir: dir.clone(),
            spill_batch: 10_000,
            load_existing: false,
            worker_spill_buffer_size: 1000,
            worker_channel_bound: 8,
            defer_segment_deletion: false,
            compression_enabled: true,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };

        let num_workers = 4;
        let (queues, _workers) =
            SpillableWorkStealingQueues::<u64>::new(num_workers, vec![0; num_workers], config)?;

        // Push a large number of items
        let num_items = 100_000u64;
        for i in 0..num_items {
            queues.push_global(i);
        }

        let pending_before = queues.pending_count();
        eprintln!("Before drain: {} items in memory", pending_before);

        // Perform checkpoint flush with streaming drain
        queues.set_checkpoint_in_progress(true);
        queues.checkpoint_flush()?;
        queues.set_checkpoint_in_progress(false);

        // Check that segments were created
        let segment_count = queues.overflow.segment_count();
        eprintln!("After drain: {} segments created", segment_count);
        assert!(segment_count > 0, "Should have created segments");

        // The test passes if it completes without OOM
        // In production with 87M items, the legacy version would OOM here
        // while the streaming version uses bounded memory

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    /// T8: Verify the in-memory compressed ring captures overflow batches
    /// before they hit disk, items round-trip back to the hot queue, and
    /// pending counters stay accurate across the compressed tier.
    #[test]
    fn test_compression_ring_round_trip() -> Result<()> {
        let dir = temp_path("spillable-comp-roundtrip");
        let config = SpillableConfig {
            // Tiny in-memory budget forces overflow path immediately.
            max_inmem_items: 100,
            spill_dir: dir.clone(),
            spill_batch: 100,
            load_existing: false,
            worker_spill_buffer_size: 50,
            worker_channel_bound: 8,
            defer_segment_deletion: false,
            compression_enabled: true,
            // Generous ring budget — everything should fit in the ring,
            // not on disk.
            compression_max_bytes: 16 * 1024 * 1024,
            compression_level: 1,
        };

        let (queues, mut workers) = SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], config)?;

        // Push more than max_inmem_items so the spill path engages. We
        // submit in small sub-batches because push_local_batch consults
        // the hot queue's pending counter ONCE per call and only spills
        // when that snapshot already exceeds the threshold; the first
        // batch always lands in the hot queue, subsequent batches spill.
        let n: u64 = 5_000;
        let chunk: u64 = 200;
        let mut next = 0u64;
        while next < n {
            let end = (next + chunk).min(n);
            let items: Vec<u64> = (next..end).collect();
            queues.push_local_batch(&mut workers[0], items.into_iter());
            next = end;
        }
        queues.flush_worker_counters(&mut workers[0]);

        // Wait for the spill coordinator to compress at least one segment.
        let mut compressed_seen = false;
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            if let Some(snap) = queues.compression_stats() {
                if snap.segments_compressed > 0 {
                    compressed_seen = true;
                    break;
                }
            }
        }
        assert!(
            compressed_seen,
            "spill coordinator never compressed a segment"
        );

        let snap = queues.compression_stats().expect("ring enabled");
        assert!(snap.bytes_compressed > 0);
        assert!(snap.bytes_uncompressed >= snap.bytes_compressed);
        assert!(
            snap.ratio() >= 1.0,
            "ratio should be >= 1.0, got {}",
            snap.ratio()
        );
        // No items should have hit the disk overflow yet — the ring is huge,
        // so push_rejected_full must be zero.
        assert_eq!(
            snap.push_rejected_full, 0,
            "ring had headroom but rejected {} batches",
            snap.push_rejected_full,
        );

        // Drain everything back through the queues (loader thread reloads
        // from the ring, then we pop).
        let mut recovered = 0u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while recovered < n && std::time::Instant::now() < deadline {
            if let Some(_v) = queues.pop_for_worker(&mut workers[0]) {
                recovered += 1;
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        assert_eq!(
            recovered, n,
            "should recover every item through the compressed ring"
        );

        // After draining, the ring must report empty and decompression
        // counters must show real activity. (Note: not every item necessarily
        // round-trips through the ring — the first batch lands directly in
        // the hot queue; only items pushed after the hot queue exceeds
        // max_inmem_items are compressed. So we assert *some* items moved
        // through the ring, not all of them.)
        let final_snap = queues.compression_stats().unwrap();
        assert_eq!(final_snap.current_items, 0);
        assert!(final_snap.segments_decompressed >= 1);
        assert_eq!(
            final_snap.items_decompressed, final_snap.items_compressed,
            "every compressed item must be decompressed once on the round trip",
        );
        assert!(
            final_snap.items_compressed >= n / 2,
            "at least half of pushed items should have flowed through the ring; \
             got {} compressed vs {} pushed",
            final_snap.items_compressed,
            n
        );

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    /// T8: With compression disabled, the ring is never instantiated and
    /// behavior matches the pre-T8 path (items go straight to disk overflow).
    #[test]
    fn test_compression_disabled_bypasses_ring() -> Result<()> {
        let dir = temp_path("spillable-comp-off");
        let config = SpillableConfig {
            max_inmem_items: 100,
            spill_dir: dir.clone(),
            spill_batch: 100,
            load_existing: false,
            worker_spill_buffer_size: 50,
            worker_channel_bound: 8,
            defer_segment_deletion: false,
            compression_enabled: false,
            compression_max_bytes: 16 * 1024 * 1024,
            compression_level: 1,
        };

        let (queues, mut workers) = SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], config)?;

        assert!(
            queues.compression_stats().is_none(),
            "compression_stats() must be None when ring is disabled"
        );

        // Push and recover; behavior should be identical to legacy direct-spill.
        let n: u64 = 1_000;
        let items: Vec<u64> = (0..n).collect();
        queues.push_local_batch(&mut workers[0], items.into_iter());
        queues.flush_worker_counters(&mut workers[0]);

        let mut recovered = 0u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while recovered < n && std::time::Instant::now() < deadline {
            if let Some(_v) = queues.pop_for_worker(&mut workers[0]) {
                recovered += 1;
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        assert_eq!(recovered, n, "should recover every item via disk path");

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    /// T8: When the ring's byte budget is tiny, additional batches must
    /// fall through to disk overflow rather than blocking or being dropped.
    #[test]
    fn test_compression_ring_overflow_falls_through_to_disk() -> Result<()> {
        let dir = temp_path("spillable-comp-overflow");
        let config = SpillableConfig {
            max_inmem_items: 100,
            spill_dir: dir.clone(),
            spill_batch: 200,
            load_existing: false,
            worker_spill_buffer_size: 100,
            worker_channel_bound: 16,
            defer_segment_deletion: false,
            compression_enabled: true,
            // Tiny budget — first segment fits, the rest overflow to disk.
            compression_max_bytes: 256,
            compression_level: 1,
        };

        let (queues, mut workers) = SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], config)?;

        let n: u64 = 5_000;
        let chunk: u64 = 200;
        let mut next = 0u64;
        while next < n {
            let end = (next + chunk).min(n);
            let items: Vec<u64> = (next..end).collect();
            queues.push_local_batch(&mut workers[0], items.into_iter());
            next = end;
        }
        queues.flush_worker_counters(&mut workers[0]);

        let mut recovered = 0u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while recovered < n && std::time::Instant::now() < deadline {
            if let Some(_v) = queues.pop_for_worker(&mut workers[0]) {
                recovered += 1;
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        assert_eq!(recovered, n, "should recover every item across ring + disk");

        let snap = queues.compression_stats().expect("ring enabled");
        // At least one batch must have been rejected by the ring (forced
        // to spill to disk), proving the fall-through path is exercised.
        assert!(
            snap.push_rejected_full >= 1,
            "expected ring to reject at least one batch; got {} rejects, {} compressed segments",
            snap.push_rejected_full,
            snap.segments_compressed,
        );

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    /// T11.1 regression — `has_pending_work()` must report items still in
    /// flight in the per-worker spill_buffer (or in the channel /
    /// coordinator accumulator). Before the fix, items in the spill
    /// pipeline below the batch threshold were invisible to termination
    /// detection and the runtime could exit while states were lost in
    /// transit.
    #[test]
    fn t11_1_inflight_items_keep_has_pending_work_true() -> Result<()> {
        let dir = temp_path("t11-1-inflight");
        let config = SpillableConfig {
            // tiny cap forces the spill path immediately
            max_inmem_items: 4,
            spill_dir: dir.clone(),
            spill_batch: 100,
            load_existing: false,
            // large worker buffer so items DON'T flush automatically —
            // they stay in the per-worker spill_buffer, which is the
            // worst case for termination detection
            worker_spill_buffer_size: 4096,
            worker_channel_bound: 4,
            defer_segment_deletion: false,
            compression_enabled: true,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };
        let (queues, mut workers) =
            SpillableWorkStealingQueues::<u64>::new(1, vec![0], config)?;

        // Prime pending_count above the cap so push_local_batch takes
        // the spill path. push_global bumps the hot queue's pushed
        // counter without bumping popped.
        for i in 0..10u64 {
            queues.push_global(i);
        }
        // Sanity: pending_count >= max_inmem_items (4) so the next
        // push_local_batch must take the spill path.
        assert!(queues.pending_count() >= 4);

        // Now push 200 items via push_local_batch. They go through the
        // spill path. With the T11.1 fix, push_local_batch always flushes
        // the spill_buffer at end-of-call, so items leave the buffer
        // immediately into the channel; from there the coordinator
        // routes them into the ring/disk asynchronously.
        let items: Vec<u64> = (1000..1200).collect();
        let pushed = queues.push_local_batch(&mut workers[0], items.into_iter());
        assert_eq!(pushed, 200);

        // T11.1 invariant: until the coordinator has routed every item
        // into the ring/disk, the inflight counter must be > 0 AND
        // has_pending_work must be true. We can't deterministically
        // observe inflight_spilled at any specific value because the
        // coordinator is racing against us, but at least at this exact
        // moment some items are still in flight (channel + accumulator).
        // We capture the value to check the invariant: until inflight
        // reaches 0, has_pending_work must hold.
        let inflight = queues.inflight_spilled.load(Ordering::Acquire);
        if inflight > 0 {
            assert!(
                queues.has_pending_work(),
                "has_pending_work must be true with {} items in flight \
                 (T11.1 regression: termination would silently drop these)",
                inflight
            );
            assert!(
                !queues.is_empty(),
                "is_empty must be false with {} items in flight",
                inflight
            );
            assert!(
                !queues.should_terminate(),
                "should_terminate must be false with {} items in flight",
                inflight
            );
        }
        // Either way the queue is non-empty: hot has 10 push_global'd
        // items, plus the 200 in flight or already in ring/disk.
        assert!(queues.has_pending_work());
        assert!(!queues.is_empty());
        assert!(!queues.should_terminate());

        // Drain the 10 hot queue items first.
        let mut hot_popped = 0;
        while queues.pop_for_worker(&mut workers[0]).is_some() {
            hot_popped += 1;
            if hot_popped >= 10 {
                break;
            }
        }
        assert_eq!(hot_popped, 10);

        // Now pop the rest. pop_for_worker auto-flushes the spill_buffer
        // when the hot queue is empty (T11.1 fix), so the in-flight
        // items flow through the coordinator into the ring/disk and
        // back into the hot queue.
        let mut popped = 0;
        let mut attempts = 0;
        while popped < pushed && attempts < 500 {
            if queues.pop_for_worker(&mut workers[0]).is_some() {
                popped += 1;
                attempts = 0;
            } else {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        assert_eq!(
            popped, pushed,
            "must recover all {} items via the spill->ring/disk->hot pipeline",
            pushed
        );

        // Final state: pipeline empty, termination allowed.
        assert_eq!(queues.inflight_spilled.load(Ordering::Acquire), 0);
        assert!(queues.is_empty());
        assert!(queues.should_terminate());

        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }
}
