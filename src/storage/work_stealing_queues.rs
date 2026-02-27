// Work-stealing queues - state of the art for CPU-bound parallel workloads
// Based on the design used in Rayon, Tokio, and Go's scheduler
//
// Key properties:
// 1. Each worker has its own local deque (no contention on common path)
// 2. Workers pop from their own queue LIFO (cache-friendly)
// 3. Workers steal from others FIFO when idle (load balancing)
// 4. Automatic work distribution without centralized coordination

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Work-stealing queue system for parallel state exploration
/// Each worker has a local queue and can steal from others when idle
pub struct WorkStealingQueues<T> {
    /// Global injector for initial work and external submissions
    global: Injector<T>,

    /// Stealers for all workers (used by other workers to steal)
    /// Workers own their Worker<T>, this only stores the Stealers for sharing
    stealers: Vec<Stealer<T>>,

    /// Completion flag
    finished: AtomicBool,

    /// Number of active workers (doing work right now)
    active_workers: AtomicUsize,

    /// Total workers
    num_workers: usize,

    /// Stats
    pushed: AtomicU64,
    popped: AtomicU64,
    steals: AtomicU64,
}

/// Per-worker state that each worker thread owns
pub struct WorkerState<T> {
    /// This worker's local queue (owned, not shared)
    worker: Worker<T>,
    /// Worker ID
    id: usize,
}

impl<T: 'static> WorkStealingQueues<T> {
    /// Create a new work-stealing queue system and return the shared state
    /// plus owned WorkerState for each worker
    pub fn new(num_workers: usize) -> (Arc<Self>, Vec<WorkerState<T>>) {
        let mut worker_states = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        for id in 0..num_workers {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            worker_states.push(WorkerState { worker, id });
        }

        let shared = Arc::new(Self {
            global: Injector::new(),
            stealers,
            finished: AtomicBool::new(false),
            active_workers: AtomicUsize::new(0),
            num_workers,
            pushed: AtomicU64::new(0),
            popped: AtomicU64::new(0),
            steals: AtomicU64::new(0),
        });

        (shared, worker_states)
    }

    /// Push work to global queue (for initial states)
    pub fn push_global(&self, item: T) {
        self.global.push(item);
        self.pushed.fetch_add(1, Ordering::Relaxed);
    }

    /// Push work to a worker's local queue (called by the worker with its own WorkerState)
    pub fn push_local(&self, worker_state: &WorkerState<T>, item: T) {
        worker_state.worker.push(item);
        self.pushed.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop work for a specific worker
    /// This implements the work-stealing algorithm:
    /// 1. Try local queue first (LIFO - cache friendly)
    /// 2. Try global queue
    /// 3. Try stealing from other workers (random)
    /// 4. If all empty and no active workers, we're done
    pub fn pop_for_worker(&self, worker_state: &WorkerState<T>) -> Option<T> {
        if self.finished.load(Ordering::Acquire) {
            return None;
        }

        // Fast path: pop from local queue (LIFO for cache locality)
        if let Some(item) = worker_state.worker.pop() {
            self.popped.fetch_add(1, Ordering::Relaxed);
            return Some(item);
        }

        // Slow path: try global and stealing with exponential backoff
        let mut spin_count = 0u32;
        const MAX_SPINS: u32 = 16; // Spin up to 16 times before yielding

        loop {
            // Try global queue
            match self.global.steal() {
                Steal::Success(item) => {
                    self.popped.fetch_add(1, Ordering::Relaxed);
                    return Some(item);
                }
                Steal::Retry => {
                    // Contention on global - spin briefly
                    std::hint::spin_loop();
                    continue;
                }
                Steal::Empty => {}
            }

            // Try stealing from other workers
            if let Some(item) = self.try_steal_from_others(worker_state.id) {
                return Some(item);
            }

            // Check local queue again (another worker might have pushed)
            if let Some(item) = worker_state.worker.pop() {
                self.popped.fetch_add(1, Ordering::Relaxed);
                return Some(item);
            }

            // Check if we're done
            if self.should_finish() {
                return None;
            }

            // Exponential backoff to reduce contention
            spin_count += 1;
            if spin_count < MAX_SPINS {
                // Spin briefly - much cheaper than thread yield
                for _ in 0..(1 << spin_count.min(6)) {
                    std::hint::spin_loop();
                }
            } else {
                // After max spins, yield to OS scheduler
                // Reset spin count for next round
                spin_count = 0;
                std::thread::yield_now();
            }
        }
    }

    /// Try to steal work from other workers (random strategy)
    fn try_steal_from_others(&self, worker_id: usize) -> Option<T> {
        // Use a pseudo-random starting point to distribute stealing
        let start = (worker_id * 7) % self.num_workers;

        for i in 0..self.num_workers {
            let target = (start + i) % self.num_workers;
            if target == worker_id {
                continue; // Don't steal from ourselves
            }

            match self.stealers[target].steal() {
                Steal::Success(item) => {
                    self.steals.fetch_add(1, Ordering::Relaxed);
                    self.popped.fetch_add(1, Ordering::Relaxed);
                    return Some(item);
                }
                Steal::Empty => continue,
                Steal::Retry => continue, // Contention, try next worker
            }
        }

        None
    }

    /// Check if we should finish (all workers idle and no work available)
    fn should_finish(&self) -> bool
    where
        T: 'static,
    {
        if self.finished.load(Ordering::Acquire) {
            return true;
        }

        // If there are active workers, more work might appear
        if self.active_workers.load(Ordering::Acquire) > 0 {
            return false;
        }

        // All workers idle - check if any work exists
        !self.has_work()
    }

    /// Check if any work exists that we haven't already tried to access
    /// Note: This is called after a worker has already tried:
    /// 1. Its own local queue
    /// 2. The global queue
    /// 3. Stealing from all other workers
    ///
    /// So we just need to check if there are active workers that might generate more work
    fn has_work(&self) -> bool {
        self.active_workers.load(Ordering::Acquire) > 0
    }

    /// Increment active worker count (call when starting work)
    pub fn worker_start(&self) {
        self.active_workers.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement active worker count (call when going idle)
    pub fn worker_idle(&self) {
        self.active_workers.fetch_sub(1, Ordering::AcqRel);
    }

    /// Mark as finished and wake all workers
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
    }

    pub fn is_empty(&self) -> bool
    where
        T: 'static,
    {
        !self.has_work()
    }

    pub fn stats(&self) -> WorkStealingStats {
        WorkStealingStats {
            pushed: self.pushed.load(Ordering::Relaxed),
            popped: self.popped.load(Ordering::Relaxed),
            steals: self.steals.load(Ordering::Relaxed),
            active_workers: self.active_workers.load(Ordering::Relaxed),
        }
    }

    pub fn has_pending_work(&self) -> bool
    where
        T: 'static,
    {
        self.has_work()
    }

    /// Check if queue is too full and we should apply backpressure
    /// This prevents unbounded memory growth by pausing state generation
    /// when we have too many unprocessed states
    pub fn should_apply_backpressure(&self, max_pending: u64) -> bool {
        let pushed = self.pushed.load(Ordering::Relaxed);
        let popped = self.popped.load(Ordering::Relaxed);
        let pending = pushed.saturating_sub(popped);
        pending > max_pending
    }

    /// Get current pending count (pushed - popped)
    pub fn pending_count(&self) -> u64 {
        let pushed = self.pushed.load(Ordering::Relaxed);
        let popped = self.popped.load(Ordering::Relaxed);
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
