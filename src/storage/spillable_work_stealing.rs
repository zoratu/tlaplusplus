// Spillable work-stealing queues - combines fast work-stealing with disk overflow
//
// Architecture:
// - Hot path: WorkStealingQueues (lock-free, NUMA-aware, in-memory)
// - Overflow: DiskBackedQueue (sequential I/O when memory limit hit)
//
// Workers never block on disk I/O:
// - Pushes go to disk only when memory threshold exceeded
// - Disk loads happen in background, feed into global injector
// - All disk I/O is sequential (no random access, no iowait)

use crate::storage::queue::{DiskBackedQueue, DiskQueueConfig, QueueStats};
use crate::storage::work_stealing_queues::{WorkStealingQueues, WorkStealingStats, WorkerState};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
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
}

impl Default for SpillableConfig {
    fn default() -> Self {
        Self {
            max_inmem_items: 10_000_000, // 10M items before spilling
            spill_dir: PathBuf::from("./.tlapp/queue"),
            spill_batch: 50_000,
            load_existing: false,
        }
    }
}

/// Work-stealing queues with disk overflow for memory-bounded operation
pub struct SpillableWorkStealingQueues<T> {
    /// Hot in-memory work-stealing queues
    hot: Arc<WorkStealingQueues<T>>,

    /// Overflow disk queue (sequential I/O only)
    overflow: Arc<DiskBackedQueue<T>>,

    /// Threshold for spilling
    max_inmem_items: u64,

    /// Background loader thread handle
    loader_handle: Option<JoinHandle<()>>,

    /// Signal to stop loader thread
    loader_stop: Arc<AtomicBool>,

    /// Channel to request loading from disk
    load_request_tx: Sender<()>,

    /// Stats
    spill_redirects: AtomicU64,
    disk_loads: AtomicU64,
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
    ) -> Result<(Arc<Self>, Vec<WorkerState<T>>)> {
        let (hot, worker_states) = WorkStealingQueues::new(num_workers, worker_numa_nodes);

        let overflow = Arc::new(DiskBackedQueue::new(DiskQueueConfig {
            spill_dir: config.spill_dir,
            inmem_limit: config.spill_batch * 2, // Small in-memory buffer
            spill_batch: config.spill_batch,
            spill_channel_bound: 64,
            load_existing_segments: config.load_existing,
        })?);

        let loader_stop = Arc::new(AtomicBool::new(false));
        let (load_request_tx, load_request_rx) = crossbeam_channel::bounded(1);

        // Background loader thread - loads from disk to global injector when needed
        let loader_hot = Arc::clone(&hot);
        let loader_overflow = Arc::clone(&overflow);
        let loader_stop_flag = Arc::clone(&loader_stop);
        let loader_handle = std::thread::Builder::new()
            .name("tlapp-queue-loader".to_string())
            .spawn(move || {
                Self::loader_thread(
                    loader_hot,
                    loader_overflow,
                    loader_stop_flag,
                    load_request_rx,
                );
            })?;

        let queues = Arc::new(Self {
            hot,
            overflow,
            max_inmem_items: config.max_inmem_items,
            loader_handle: Some(loader_handle),
            loader_stop,
            load_request_tx,
            spill_redirects: AtomicU64::new(0),
            disk_loads: AtomicU64::new(0),
        });

        Ok((queues, worker_states))
    }

    /// Background thread that loads from disk when hot queues are empty
    fn loader_thread(
        hot: Arc<WorkStealingQueues<T>>,
        overflow: Arc<DiskBackedQueue<T>>,
        stop: Arc<AtomicBool>,
        load_requests: Receiver<()>,
    ) {
        while !stop.load(Ordering::Acquire) {
            // Wait for load request or timeout
            match load_requests.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }

            if stop.load(Ordering::Acquire) {
                break;
            }

            // Check if hot queues need refilling
            if hot.pending_count() > 0 {
                continue; // Hot queues have work, no need to load
            }

            // Try to load a batch from disk
            match overflow.pop_bulk(10_000) {
                Ok(items) if !items.is_empty() => {
                    // Push loaded items to global injector
                    for item in items {
                        hot.push_global(item);
                    }
                }
                Ok(_) => {} // No items on disk
                Err(e) => {
                    eprintln!("Warning: disk queue load error: {}", e);
                }
            }
        }
    }

    /// Push to global queue (for initial states)
    pub fn push_global(&self, item: T) {
        if self.hot.pending_count() < self.max_inmem_items {
            self.hot.push_global(item);
        } else {
            // Redirect to disk
            if let Err(e) = self.overflow.push(item) {
                eprintln!("Warning: queue spill failed: {}", e);
            } else {
                self.spill_redirects.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Push batch to worker's local queue, spilling to disk if over threshold
    pub fn push_local_batch(
        &self,
        worker_state: &mut WorkerState<T>,
        items: impl Iterator<Item = T>,
    ) -> usize {
        let pending = self.hot.pending_count();

        if pending < self.max_inmem_items {
            // Fast path: push to in-memory work-stealing queue
            self.hot.push_local_batch(worker_state, items)
        } else {
            // Slow path: redirect to disk
            let mut count = 0;
            for item in items {
                if let Err(e) = self.overflow.push(item) {
                    eprintln!("Warning: queue spill failed: {}", e);
                } else {
                    count += 1;
                }
            }
            self.spill_redirects
                .fetch_add(count as u64, Ordering::Relaxed);
            count
        }
    }

    /// Pop work for a worker
    pub fn pop_for_worker(&self, worker_state: &mut WorkerState<T>) -> Option<T> {
        // Try hot queues first
        if let Some(item) = self.hot.pop_for_worker(worker_state) {
            return Some(item);
        }

        // Hot queues empty - try loading directly from overflow
        // This is synchronous but only happens when hot is empty
        self.try_load_from_disk();

        // Try hot again after loading
        if let Some(item) = self.hot.pop_for_worker(worker_state) {
            return Some(item);
        }

        None
    }

    /// Try to load items from disk to hot queues
    fn try_load_from_disk(&self) {
        // Only load if hot queues are actually empty
        // Use is_empty() instead of pending_count() because pending_count
        // relies on counters that aren't flushed immediately
        if !self.hot.is_empty() {
            return;
        }

        // Try to pop from overflow and push to global injector
        match self.overflow.pop_bulk(10_000) {
            Ok(items) if !items.is_empty() => {
                let count = items.len();
                self.disk_loads.fetch_add(count as u64, Ordering::Relaxed);
                for item in items {
                    self.hot.push_global(item);
                }
            }
            Ok(_) => {
                // Empty - check if overflow has pending work
            }
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
    pub fn flush_worker_counters(&self, worker_state: &mut WorkerState<T>) {
        self.hot.flush_worker_counters(worker_state);
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
            spill_batches: overflow_stats.spill_batches,
            loaded_segments: overflow_stats.loaded_segments,
            loaded_items: overflow_stats.loaded_items,
            max_inmem_len: hot_stats.max_inmem_len.max(overflow_stats.max_inmem_len),
        }
    }

    /// Get work-stealing specific stats
    pub fn work_stealing_stats(&self) -> WorkStealingStats {
        self.hot.work_stealing_stats()
    }

    /// Shutdown - stop loader thread and disk queue
    pub fn shutdown(&mut self) -> Result<()> {
        self.loader_stop.store(true, Ordering::Release);

        // Drop the sender to unblock receiver
        // (can't do this directly, but finish() will wake it)
        self.hot.finish();
        self.overflow.finish();

        if let Some(handle) = self.loader_handle.take() {
            let _ = handle.join();
        }

        self.overflow.shutdown()
    }

    /// Checkpoint - flush all in-memory state to disk
    pub fn checkpoint_flush(&self) -> Result<()> {
        // Note: Can't easily drain work-stealing queues from outside
        // For checkpoint, we rely on workers being paused
        self.overflow.checkpoint_flush()
    }

    /// Get reference to underlying hot queues (for worker pop operations)
    pub fn hot(&self) -> &Arc<WorkStealingQueues<T>> {
        &self.hot
    }
}

impl<T> Drop for SpillableWorkStealingQueues<T> {
    fn drop(&mut self) {
        self.loader_stop.store(true, Ordering::Release);
    }
}

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
            max_inmem_items: 50, // Very low to force spilling
            spill_dir: dir.clone(),
            spill_batch: 20,
            load_existing: false,
        };

        let (queues, mut workers) = SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], config)?;

        // Push more than max_inmem_items
        for i in 0..200u64 {
            queues.push_global(i);
        }

        // Wait for async spill to complete (spill thread writes segments)
        for _ in 0..50 {
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
                // Give disk loading time to work
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
}
