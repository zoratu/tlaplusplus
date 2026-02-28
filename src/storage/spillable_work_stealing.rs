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

use crate::storage::queue::{DiskBackedQueue, DiskQueueConfig, QueueStats};
use crate::storage::work_stealing_queues::{WorkStealingQueues, WorkStealingStats, WorkerState};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::PathBuf;
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

/// A batch of items to spill, tagged with worker ID for debugging
struct SpillBatch<T> {
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

    /// Stats
    spill_redirects: AtomicU64,
    disk_loads: AtomicU64,
    spill_batches_sent: AtomicU64,
    spill_channel_full: AtomicU64,
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
            });
        }

        // Start spill coordinator thread
        let coordinator_overflow = Arc::clone(&overflow);
        let coordinator_stop = Arc::clone(&stop_signal);
        let spill_batch_size = config.spill_batch;
        let coordinator_handle = std::thread::Builder::new()
            .name("tlapp-spill-coordinator".to_string())
            .spawn(move || {
                Self::spill_coordinator_thread(
                    spill_receivers,
                    coordinator_overflow,
                    coordinator_stop,
                    spill_batch_size,
                );
            })?;

        // Start background loader thread
        let loader_hot = Arc::clone(&hot);
        let loader_overflow = Arc::clone(&overflow);
        let loader_stop = Arc::clone(&stop_signal);
        let loader_handle = std::thread::Builder::new()
            .name("tlapp-queue-loader".to_string())
            .spawn(move || {
                Self::loader_thread(loader_hot, loader_overflow, loader_stop);
            })?;

        let queues = Arc::new(Self {
            hot,
            overflow,
            max_inmem_items: config.max_inmem_items,
            coordinator_handle: Some(coordinator_handle),
            loader_handle: Some(loader_handle),
            stop_signal,
            spill_redirects: AtomicU64::new(0),
            disk_loads: AtomicU64::new(0),
            spill_batches_sent: AtomicU64::new(0),
            spill_channel_full: AtomicU64::new(0),
        });

        Ok((queues, worker_states))
    }

    /// Spill coordinator thread - receives batches from all workers, writes to disk
    /// This is the ONLY thread that writes to the overflow queue
    fn spill_coordinator_thread(
        receivers: Vec<Receiver<SpillBatch<T>>>,
        overflow: Arc<DiskBackedQueue<T>>,
        stop: Arc<AtomicBool>,
        batch_size: usize,
    ) {
        let mut accumulator: Vec<T> = Vec::with_capacity(batch_size);

        // Use select! to receive from any worker without polling
        let mut sel = crossbeam_channel::Select::new();
        let mut recv_indices: Vec<usize> = Vec::with_capacity(receivers.len());
        for (i, rx) in receivers.iter().enumerate() {
            let idx = sel.recv(rx);
            recv_indices.push(idx);
        }

        while !stop.load(Ordering::Acquire) {
            // Try to receive with timeout
            let result = sel.select_timeout(std::time::Duration::from_millis(10));

            match result {
                Ok(oper) => {
                    // Find which receiver was ready
                    let recv_idx = oper.index();
                    if let Ok(batch) = oper.recv(&receivers[recv_idx]) {
                        // Add items to accumulator
                        accumulator.extend(batch.items);

                        // If accumulator is full, write to disk
                        if accumulator.len() >= batch_size {
                            let to_write =
                                std::mem::replace(&mut accumulator, Vec::with_capacity(batch_size));
                            for item in to_write {
                                if let Err(e) = overflow.push(item) {
                                    eprintln!("Warning: spill coordinator push failed: {}", e);
                                }
                            }
                        }
                    }
                }
                Err(crossbeam_channel::SelectTimeoutError) => {
                    // Timeout - flush any accumulated items
                    if !accumulator.is_empty() {
                        let to_write =
                            std::mem::replace(&mut accumulator, Vec::with_capacity(batch_size));
                        for item in to_write {
                            if let Err(e) = overflow.push(item) {
                                eprintln!("Warning: spill coordinator push failed: {}", e);
                            }
                        }
                    }
                }
            }
        }

        // Final flush on shutdown
        for item in accumulator {
            if let Err(e) = overflow.push(item) {
                eprintln!("Warning: spill coordinator final push failed: {}", e);
            }
        }
    }

    /// Background thread that loads from disk when hot queues are empty
    fn loader_thread(
        hot: Arc<WorkStealingQueues<T>>,
        overflow: Arc<DiskBackedQueue<T>>,
        stop: Arc<AtomicBool>,
    ) {
        while !stop.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(50));

            if stop.load(Ordering::Acquire) {
                break;
            }

            // Check if hot queues need refilling
            if !hot.is_empty() {
                continue;
            }

            // Try to load a batch from disk
            match overflow.pop_bulk(10_000) {
                Ok(items) if !items.is_empty() => {
                    for item in items {
                        hot.push_global(item);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Warning: disk queue load error: {}", e);
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

                // When buffer is full, try to send to coordinator
                if worker_state.spill_buffer.len() >= worker_state.spill_buffer_threshold {
                    self.flush_worker_spill_buffer(worker_state);
                }
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

                // When buffer is full, try to send to coordinator
                if worker_state.spill_buffer.len() >= worker_state.spill_buffer_threshold {
                    self.flush_worker_spill_buffer(worker_state);
                }
            }
            self.spill_redirects
                .fetch_add(count as u64, Ordering::Relaxed);
            count
        }
    }

    /// Flush a worker's spill buffer to the coordinator (NON-BLOCKING)
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

        // NON-BLOCKING send - if channel is full, put items back in memory
        match worker_state.spill_tx.try_send(batch) {
            Ok(()) => {
                worker_state.items_spilled += batch_len;
                self.spill_batches_sent.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(returned_batch)) => {
                // Channel full (backpressure) - put items back in memory
                // This allows temporary overshoot of memory limit but maintains throughput
                self.spill_channel_full.fetch_add(1, Ordering::Relaxed);
                for item in returned_batch.items {
                    worker_state.inner.push(item);
                }
            }
            Err(TrySendError::Disconnected(returned_batch)) => {
                // Coordinator died - keep items in memory
                for item in returned_batch.items {
                    worker_state.inner.push(item);
                }
            }
        }
    }

    /// Pop work for a worker
    pub fn pop_for_worker(&self, worker_state: &mut SpillableWorkerState<T>) -> Option<T> {
        // Try hot queues first
        if let Some(item) = self.hot.pop_for_worker(&mut worker_state.inner) {
            return Some(item);
        }

        // Hot queues empty - try loading directly from overflow
        self.try_load_from_disk();

        // Try hot again after loading
        self.hot.pop_for_worker(&mut worker_state.inner)
    }

    /// Try to load items from disk to hot queues
    fn try_load_from_disk(&self) {
        if !self.hot.is_empty() {
            return;
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

    /// Check termination - must check both hot and disk
    pub fn should_terminate(&self) -> bool {
        if self.hot.has_pending_work() {
            return false;
        }
        if self.overflow.has_pending_work() {
            return false;
        }
        true
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

    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.hot.is_empty() && self.overflow.is_empty()
    }

    /// Check if there's pending work
    pub fn has_pending_work(&self) -> bool {
        self.hot.has_pending_work() || self.overflow.has_pending_work()
    }

    /// Get approximate pending count (in-memory only for speed)
    pub fn pending_count(&self) -> u64 {
        self.hot.pending_count()
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
    pub fn checkpoint_flush(&self) -> Result<()> {
        self.overflow.checkpoint_flush()
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
}
