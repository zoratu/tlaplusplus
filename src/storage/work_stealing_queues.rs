// Work-stealing queues - optimized for many-core NUMA scalability
//
// Key optimizations:
// 1. Per-worker counters (no shared atomic contention on hot path)
// 2. Cache-line padded structures to prevent false sharing
// 3. Termination detection without per-state atomic updates
// 4. Local queue operations are completely contention-free
// 5. NUMA-aware hierarchical stealing (prefer same-NUMA workers)
// 6. Adaptive steal limits for many-core scalability (380+ workers)
// 7. Per-NUMA idle counters for O(NUMA_nodes) termination check
// 8. Batch stealing to reduce steal overhead

use crate::storage::queue::QueueStats;
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use crossbeam_utils::CachePadded;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

/// Maximum workers to try stealing from within same NUMA node
/// Prevents O(n) steal attempts when n is large (e.g., 64 workers/node)
const MAX_LOCAL_STEAL_ATTEMPTS: usize = 8;

/// Maximum workers to try stealing from per remote NUMA node
/// Keep low because cross-NUMA stealing is expensive (memory access latency)
/// NUMA distances can be 2-3x higher for remote nodes
const MAX_REMOTE_STEAL_ATTEMPTS: usize = 1;

/// Work-stealing queue system for parallel state exploration
/// Each worker has a local queue and can steal from others when idle
pub struct WorkStealingQueues<T> {
    /// Global injector for initial work and external submissions
    global: Injector<T>,

    /// Stealers for all workers (used by other workers to steal)
    stealers: Vec<Stealer<T>>,

    /// Completion flag
    finished: AtomicBool,

    /// Per-worker active state (cache-line padded to prevent false sharing)
    /// 0 = idle, 1 = active
    worker_active: Vec<CachePadded<AtomicU8>>,

    /// Total workers
    num_workers: usize,

    /// NUMA node for each worker (worker_id -> numa_node)
    #[allow(dead_code)]
    worker_numa_nodes: Vec<usize>,

    /// Workers grouped by NUMA node (numa_node -> [worker_ids])
    numa_worker_groups: Vec<Vec<usize>>,

    /// Per-NUMA active worker count (cache-line padded)
    /// Allows O(NUMA_nodes) termination check instead of O(workers)
    numa_active_counts: Vec<CachePadded<AtomicUsize>>,

    /// Global pushed counter (only updated in batches from workers)
    global_pushed: AtomicU64,
    /// Global popped counter (only updated in batches from workers)
    global_popped: AtomicU64,
}

/// Per-worker state that each worker thread owns
/// Contains local counters to avoid shared atomic contention
pub struct WorkerState<T> {
    /// This worker's local queue (owned, not shared)
    worker: Worker<T>,
    /// Worker ID
    pub id: usize,
    /// NUMA node this worker belongs to
    pub numa_node: usize,
    /// Local pushed counter (flushed periodically)
    local_pushed: u64,
    /// Local popped counter (flushed periodically)
    local_popped: u64,
    /// Flush threshold
    flush_threshold: u64,
}

impl<T> WorkerState<T> {
    /// Push an item to this worker's local queue
    #[inline]
    pub fn push(&mut self, item: T) {
        self.worker.push(item);
        self.local_pushed += 1;
    }
}

impl<T: 'static> WorkStealingQueues<T> {
    /// Create a new NUMA-aware work-stealing queue system
    /// worker_numa_nodes: for each worker_id, which NUMA node it belongs to
    pub fn new(
        num_workers: usize,
        worker_numa_nodes: Vec<usize>,
    ) -> (Arc<Self>, Vec<WorkerState<T>>) {
        let mut worker_states = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);
        let mut worker_active = Vec::with_capacity(num_workers);

        // Build NUMA worker groups for hierarchical stealing
        let num_numa_nodes = worker_numa_nodes.iter().max().map(|n| n + 1).unwrap_or(1);
        let mut numa_worker_groups: Vec<Vec<usize>> = vec![Vec::new(); num_numa_nodes];

        for id in 0..num_workers {
            let numa_node = worker_numa_nodes.get(id).copied().unwrap_or(0);
            numa_worker_groups[numa_node].push(id);

            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            worker_active.push(CachePadded::new(AtomicU8::new(0)));
            worker_states.push(WorkerState {
                worker,
                id,
                numa_node,
                local_pushed: 0,
                local_popped: 0,
                flush_threshold: 256, // Flush every 256 operations
            });
        }

        // Initialize per-NUMA active counters
        let numa_active_counts: Vec<CachePadded<AtomicUsize>> = (0..num_numa_nodes)
            .map(|_| CachePadded::new(AtomicUsize::new(0)))
            .collect();

        let shared = Arc::new(Self {
            global: Injector::new(),
            stealers,
            finished: AtomicBool::new(false),
            worker_active,
            num_workers,
            worker_numa_nodes,
            numa_worker_groups,
            numa_active_counts,
            global_pushed: AtomicU64::new(0),
            global_popped: AtomicU64::new(0),
        });

        (shared, worker_states)
    }

    /// Push work to global queue (for initial states)
    pub fn push_global(&self, item: T) {
        self.global.push(item);
        self.global_pushed.fetch_add(1, Ordering::Relaxed);
    }

    /// Push work to a worker's local queue
    /// The worker_state tracks pushed count locally
    pub fn push_local(&self, worker_state: &WorkerState<T>, item: T) {
        worker_state.worker.push(item);
        // Local counter updated by caller via worker_state
    }

    /// Push multiple items and update local counter
    pub fn push_local_batch(
        &self,
        worker_state: &mut WorkerState<T>,
        items: impl Iterator<Item = T>,
    ) -> usize {
        let mut count = 0;
        for item in items {
            worker_state.worker.push(item);
            count += 1;
        }
        worker_state.local_pushed += count as u64;

        // Periodically flush to global counter
        if worker_state.local_pushed >= worker_state.flush_threshold {
            self.global_pushed
                .fetch_add(worker_state.local_pushed, Ordering::Relaxed);
            worker_state.local_pushed = 0;
        }
        count
    }

    /// Pop work for a specific worker
    /// Uses optimized termination detection without per-state atomic updates
    pub fn pop_for_worker(&self, worker_state: &mut WorkerState<T>) -> Option<T> {
        if self.finished.load(Ordering::Acquire) {
            return None;
        }

        // Fast path: pop from local queue (completely contention-free)
        if let Some(item) = worker_state.worker.pop() {
            worker_state.local_popped += 1;
            return Some(item);
        }

        // Slow path: try global and stealing
        self.pop_slow_path(worker_state)
    }

    #[cold]
    fn pop_slow_path(&self, worker_state: &mut WorkerState<T>) -> Option<T> {
        let mut spin_count = 0u32;
        const MAX_SPINS: u32 = 32;

        loop {
            // Try global queue
            match self.global.steal() {
                Steal::Success(item) => {
                    worker_state.local_popped += 1;
                    return Some(item);
                }
                Steal::Retry => {
                    std::hint::spin_loop();
                    continue;
                }
                Steal::Empty => {}
            }

            // Try stealing from other workers
            if let Some(item) = self.try_steal_from_others(worker_state) {
                return Some(item);
            }

            // Check local queue again
            if let Some(item) = worker_state.worker.pop() {
                worker_state.local_popped += 1;
                return Some(item);
            }

            // Check termination: are all workers idle and no work exists?
            if self.should_terminate(worker_state.id) {
                return None;
            }

            // Exponential backoff
            spin_count += 1;
            if spin_count < MAX_SPINS {
                for _ in 0..(1 << spin_count.min(6)) {
                    std::hint::spin_loop();
                }
            } else {
                spin_count = 0;
                std::thread::yield_now();
            }
        }
    }

    /// Try to steal work from other workers using hierarchical NUMA-aware stealing
    /// 1. First try stealing from workers on the same NUMA node (low latency)
    /// 2. Then try stealing from workers on remote NUMA nodes
    ///
    /// Optimized for many-core systems (380+ workers):
    /// - Limits steal attempts to avoid O(n) scanning
    /// - Uses batch stealing to reduce overhead
    fn try_steal_from_others(&self, worker_state: &mut WorkerState<T>) -> Option<T> {
        let my_numa = worker_state.numa_node;

        // Phase 1: Try to steal from same NUMA node first (low latency)
        // Limit attempts to avoid scanning all 64+ workers per NUMA node
        if let Some(local_workers) = self.numa_worker_groups.get(my_numa) {
            let start = (worker_state.id * 7) % local_workers.len().max(1);
            let max_attempts = local_workers.len().min(MAX_LOCAL_STEAL_ATTEMPTS);
            for i in 0..max_attempts {
                let idx = (start + i) % local_workers.len();
                let target = local_workers[idx];
                if target == worker_state.id {
                    continue;
                }

                // Try batch steal first - steal half the items into our local queue
                match self.stealers[target].steal_batch_and_pop(&worker_state.worker) {
                    Steal::Success(item) => {
                        worker_state.local_popped += 1;
                        return Some(item);
                    }
                    Steal::Empty => continue,
                    Steal::Retry => continue,
                }
            }
        }

        // Phase 2: Try to steal from remote NUMA nodes
        for (node_idx, remote_workers) in self.numa_worker_groups.iter().enumerate() {
            if node_idx == my_numa || remote_workers.is_empty() {
                continue;
            }

            // Try a few workers from this remote node
            let start = (worker_state.id * 7) % remote_workers.len();
            let max_attempts = remote_workers.len().min(MAX_REMOTE_STEAL_ATTEMPTS);
            for i in 0..max_attempts {
                let idx = (start + i) % remote_workers.len();
                let target = remote_workers[idx];

                // For remote NUMA, also use batch stealing
                match self.stealers[target].steal_batch_and_pop(&worker_state.worker) {
                    Steal::Success(item) => {
                        worker_state.local_popped += 1;
                        return Some(item);
                    }
                    Steal::Empty => continue,
                    Steal::Retry => continue,
                }
            }
        }

        None
    }

    /// Optimized termination detection using per-NUMA idle counters
    /// Returns true if all workers are idle and no work exists anywhere
    ///
    /// Optimized for many-core systems:
    /// - O(NUMA_nodes) check instead of O(workers) for idle detection
    /// - Only falls back to O(workers) stealer check if all NUMA nodes are idle
    fn should_terminate(&self, _worker_id: usize) -> bool {
        if self.finished.load(Ordering::Acquire) {
            return true;
        }

        // Fast path: Check per-NUMA active counters (O(NUMA_nodes) instead of O(workers))
        for numa_active in &self.numa_active_counts {
            if numa_active.load(Ordering::Acquire) > 0 {
                return false;
            }
        }

        // All NUMA nodes report idle - check global queue
        if !self.global.is_empty() {
            return false;
        }

        // Check all stealers (only reached when all workers idle)
        // This is O(workers) but only happens at termination
        for stealer in &self.stealers {
            if !stealer.is_empty() {
                return false;
            }
        }

        true
    }

    /// Mark worker as active using per-NUMA counter
    /// Cache-line padded to prevent false sharing between NUMA nodes
    #[inline]
    pub fn worker_start(&self, worker_id: usize) {
        // Track per-worker state for debugging
        self.worker_active[worker_id].store(1, Ordering::Release);
        // Increment per-NUMA counter for fast termination detection (O(1) lookup)
        let numa_node = self.worker_numa_nodes.get(worker_id).copied().unwrap_or(0);
        self.numa_active_counts[numa_node].fetch_add(1, Ordering::AcqRel);
    }

    /// Mark worker as idle and decrement per-NUMA counter
    #[inline]
    pub fn worker_idle(&self, worker_id: usize) {
        self.worker_active[worker_id].store(0, Ordering::Release);
        // Decrement per-NUMA counter (O(1) lookup)
        let numa_node = self.worker_numa_nodes.get(worker_id).copied().unwrap_or(0);
        self.numa_active_counts[numa_node].fetch_sub(1, Ordering::AcqRel);
    }

    /// Flush worker's local counters to global (call before worker exits)
    pub fn flush_worker_counters(&self, worker_state: &mut WorkerState<T>) {
        if worker_state.local_pushed > 0 {
            self.global_pushed
                .fetch_add(worker_state.local_pushed, Ordering::Relaxed);
            worker_state.local_pushed = 0;
        }
        if worker_state.local_popped > 0 {
            self.global_popped
                .fetch_add(worker_state.local_popped, Ordering::Relaxed);
            worker_state.local_popped = 0;
        }
    }

    /// Mark as finished
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
    }

    pub fn is_empty(&self) -> bool {
        self.global.is_empty() && self.stealers.iter().all(|s| s.is_empty())
    }

    pub fn stats(&self) -> QueueStats {
        QueueStats {
            pushed: self.global_pushed.load(Ordering::Relaxed),
            popped: self.global_popped.load(Ordering::Relaxed),
            spilled_items: 0, // Work-stealing queues don't spill
            spill_batches: 0,
            loaded_segments: 0,
            loaded_items: 0,
            max_inmem_len: 0, // Not tracked for work-stealing
        }
    }

    pub fn work_stealing_stats(&self) -> WorkStealingStats {
        let mut active_count = 0;
        for active in self.worker_active.iter() {
            let val: u8 = active.load(Ordering::Relaxed);
            if val != 0 {
                active_count += 1;
            }
        }
        WorkStealingStats {
            pushed: self.global_pushed.load(Ordering::Relaxed),
            popped: self.global_popped.load(Ordering::Relaxed),
            steals: 0, // No longer tracked per-steal for performance
            active_workers: active_count,
        }
    }

    pub fn has_pending_work(&self) -> bool {
        if !self.is_empty() {
            return true;
        }
        for active in self.worker_active.iter() {
            let val: u8 = active.load(Ordering::Relaxed);
            if val != 0 {
                return true;
            }
        }
        false
    }

    /// Check if queue is too full (for backpressure)
    /// Uses approximate count from global counters
    pub fn should_apply_backpressure(&self, max_pending: u64) -> bool {
        let pushed = self.global_pushed.load(Ordering::Relaxed);
        let popped = self.global_popped.load(Ordering::Relaxed);
        pushed.saturating_sub(popped) > max_pending
    }

    /// Get approximate pending count
    pub fn pending_count(&self) -> u64 {
        let pushed = self.global_pushed.load(Ordering::Relaxed);
        let popped = self.global_popped.load(Ordering::Relaxed);
        pushed.saturating_sub(popped)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkStealingStats {
    pub pushed: u64,
    pub popped: u64,
    pub steals: u64,
    pub active_workers: usize,
}
