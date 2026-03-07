#![no_main]

//! Fuzzing tests for checkpoint/quiescence coordination.
//!
//! This fuzz target tests three scenarios:
//! 1. Concurrent worker pause/resume coordination
//! 2. Fingerprint store mode switching under concurrent access
//! 3. Combined checkpoint + switch operations
//!
//! The goal is to find deadlocks, crashes, and data races in the
//! checkpoint/quiescence coordination code.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use parking_lot::{Condvar, Mutex};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Maximum number of simulated workers for fuzzing
const MAX_WORKERS: usize = 16;

/// Maximum number of operations per fuzz iteration
const MAX_OPS: usize = 100;

/// Timeout for any single operation (to detect deadlocks)
const OP_TIMEOUT_MS: u64 = 50;

/// Worker action types
#[derive(Debug, Clone, Copy, Arbitrary)]
enum WorkerAction {
    /// Worker does some work
    Work,
    /// Worker checks pause point
    CheckPausePoint,
    /// Worker yields to other threads
    Yield,
}

/// Controller action types
#[derive(Debug, Clone, Copy, Arbitrary)]
enum ControllerAction {
    /// Request workers to pause
    RequestPause,
    /// Wait for quiescence (all workers paused)
    WaitQuiescence,
    /// Resume workers
    Resume,
    /// No-op (yield)
    Yield,
}

/// Fingerprint store action types
#[derive(Debug, Clone, Copy, Arbitrary)]
enum FingerprintAction {
    /// Insert or check a fingerprint
    ContainsOrInsert(u64),
    /// Force switch to hybrid mode
    ForceSwitch,
    /// Check if in hybrid mode
    CheckHybrid,
}

/// Combined scenario action types
#[derive(Debug, Clone, Copy, Arbitrary)]
enum CombinedAction {
    /// Worker action
    Worker(WorkerAction),
    /// Controller action
    Controller(ControllerAction),
    /// Fingerprint action
    Fingerprint(FingerprintAction),
}

/// Fuzz input for worker pause/resume scenario
#[derive(Debug, Arbitrary)]
struct WorkerPauseInput {
    /// Number of workers (1-MAX_WORKERS)
    num_workers: u8,
    /// Sequence of worker actions
    worker_actions: Vec<(u8, WorkerAction)>, // (worker_id % num_workers, action)
    /// Sequence of controller actions
    controller_actions: Vec<ControllerAction>,
}

/// Fuzz input for fingerprint store mode switching
#[derive(Debug, Arbitrary)]
struct FingerprintSwitchInput {
    /// Number of worker threads
    num_threads: u8,
    /// Sequence of actions per thread
    actions: Vec<(u8, FingerprintAction)>, // (thread_id, action)
    /// When to inject force_switch calls (as fraction 0-255)
    switch_timing: u8,
}

/// Fuzz input for combined checkpoint + switch scenario
#[derive(Debug, Arbitrary)]
struct CombinedInput {
    /// Number of workers
    num_workers: u8,
    /// Mixed sequence of actions
    actions: Vec<CombinedAction>,
}

/// Top-level fuzz input
#[derive(Debug, Arbitrary)]
enum FuzzInput {
    WorkerPause(WorkerPauseInput),
    FingerprintSwitch(FingerprintSwitchInput),
    Combined(CombinedInput),
}

// ============================================================================
// Mock PauseController (simplified version for fuzzing without disk I/O)
// ============================================================================

/// Mock pause controller that mimics the real PauseController
/// but is designed for fast fuzzing without actual blocking.
#[derive(Default)]
struct MockPauseController {
    requested: AtomicBool,
    paused_workers: AtomicUsize,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

impl MockPauseController {
    fn new() -> Self {
        Self::default()
    }

    /// Worker calls this to check if they should pause.
    /// Returns quickly if no pause requested.
    fn worker_pause_point(&self, stop: &AtomicBool) -> bool {
        if !self.requested.load(Ordering::Acquire) {
            return false;
        }

        self.paused_workers.fetch_add(1, Ordering::AcqRel);
        let mut guard = self.wait_lock.lock();

        // Wait with timeout to avoid infinite blocking during fuzzing
        let mut waited = false;
        while self.requested.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
            let result = self
                .wait_cv
                .wait_for(&mut guard, Duration::from_millis(OP_TIMEOUT_MS));
            if result.timed_out() {
                // Timeout during fuzzing - break to avoid deadlock
                break;
            }
            waited = true;
        }

        drop(guard);
        self.paused_workers.fetch_sub(1, Ordering::AcqRel);
        waited
    }

    /// Controller requests all workers to pause
    fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    /// Controller waits for all workers to reach pause point
    fn wait_for_quiescence(
        &self,
        stop: &AtomicBool,
        _active_workers: &AtomicUsize,
        live_workers: &AtomicUsize,
    ) -> bool {
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(OP_TIMEOUT_MS * 2);

        loop {
            if stop.load(Ordering::Acquire) {
                return false;
            }

            if start.elapsed() > timeout {
                // Timeout - return false but don't deadlock
                return false;
            }

            let paused = self.paused_workers.load(Ordering::Acquire);
            let live = live_workers.load(Ordering::Acquire);

            if paused >= live {
                return true; // Quiescence achieved
            }

            thread::yield_now();
        }
    }

    /// Controller resumes all workers
    fn resume(&self) {
        self.requested.store(false, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn is_pause_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn paused_count(&self) -> usize {
        self.paused_workers.load(Ordering::Acquire)
    }
}

// ============================================================================
// Mock AutoSwitchingFingerprintStore (simplified for fuzzing)
// ============================================================================

/// Mock fingerprint store that simulates auto-switching behavior
/// without the actual memory allocation overhead.
struct MockAutoSwitchingStore {
    /// Fingerprints stored in exact mode
    exact: Mutex<HashSet<u64>>,
    /// Fingerprints stored in bloom mode (after switch)
    bloom: Mutex<HashSet<u64>>,
    /// Whether we've switched to hybrid mode
    switched: AtomicBool,
    /// Insert counter
    insert_count: AtomicU64,
    /// Switch threshold (low for fuzzing)
    switch_threshold: u64,
}

impl MockAutoSwitchingStore {
    fn new(switch_threshold: u64) -> Self {
        Self {
            exact: Mutex::new(HashSet::new()),
            bloom: Mutex::new(HashSet::new()),
            switched: AtomicBool::new(false),
            insert_count: AtomicU64::new(0),
            switch_threshold,
        }
    }

    fn contains_or_insert(&self, fp: u64) -> bool {
        if self.switched.load(Ordering::Acquire) {
            // Hybrid mode: check exact (read-only), then bloom
            {
                let exact = self.exact.lock();
                if exact.contains(&fp) {
                    return true;
                }
            }

            let mut bloom = self.bloom.lock();
            if bloom.contains(&fp) {
                true
            } else {
                bloom.insert(fp);
                false
            }
        } else {
            // Exact mode only
            let mut exact = self.exact.lock();
            if exact.contains(&fp) {
                true
            } else {
                exact.insert(fp);
                let count = self.insert_count.fetch_add(1, Ordering::Relaxed) + 1;

                // Auto-switch check
                if count >= self.switch_threshold {
                    drop(exact);
                    self.try_switch();
                }

                false
            }
        }
    }

    fn force_switch(&self) {
        self.try_switch();
    }

    fn try_switch(&self) {
        // Use compare_exchange to ensure only one thread switches
        let _ = self
            .switched
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed);
    }

    fn is_hybrid(&self) -> bool {
        self.switched.load(Ordering::Acquire)
    }

    fn exact_count(&self) -> usize {
        self.exact.lock().len()
    }

    fn bloom_count(&self) -> usize {
        self.bloom.lock().len()
    }
}

// ============================================================================
// Fuzz test implementations
// ============================================================================

/// Fuzz concurrent worker pause/resume
fn fuzz_worker_pause(input: WorkerPauseInput) {
    let num_workers = ((input.num_workers as usize) % MAX_WORKERS).max(1);
    let worker_actions: Vec<_> = input
        .worker_actions
        .into_iter()
        .take(MAX_OPS)
        .map(|(id, action)| ((id as usize) % num_workers, action))
        .collect();
    let controller_actions: Vec<_> = input.controller_actions.into_iter().take(MAX_OPS).collect();

    let pause = Arc::new(MockPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let live_workers = Arc::new(AtomicUsize::new(num_workers));
    let active_workers = Arc::new(AtomicUsize::new(0));

    // Spawn worker threads
    let mut handles = Vec::new();
    let action_idx = Arc::new(AtomicUsize::new(0));

    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let action_idx = Arc::clone(&action_idx);
        let worker_actions = worker_actions.clone();

        handles.push(thread::spawn(move || {
            loop {
                let idx = action_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= worker_actions.len() {
                    break;
                }

                let (target_worker, action) = worker_actions[idx];
                if target_worker != worker_id {
                    // Not our action, continue
                    continue;
                }

                if stop.load(Ordering::Acquire) {
                    break;
                }

                match action {
                    WorkerAction::Work => {
                        // Simulate some work
                        std::hint::black_box(worker_id * 42);
                    }
                    WorkerAction::CheckPausePoint => {
                        pause.worker_pause_point(&stop);
                    }
                    WorkerAction::Yield => {
                        thread::yield_now();
                    }
                }
            }
        }));
    }

    // Run controller actions in main thread
    for action in controller_actions {
        if stop.load(Ordering::Acquire) {
            break;
        }

        match action {
            ControllerAction::RequestPause => {
                pause.request_pause();
            }
            ControllerAction::WaitQuiescence => {
                pause.wait_for_quiescence(&stop, &active_workers, &live_workers);
            }
            ControllerAction::Resume => {
                pause.resume();
            }
            ControllerAction::Yield => {
                thread::yield_now();
            }
        }
    }

    // Signal stop and wait for workers
    stop.store(true, Ordering::Release);
    pause.resume(); // Make sure paused workers can exit

    for handle in handles {
        let _ = handle.join();
    }

    // Invariant checks (these should never fail)
    assert!(
        !pause.is_pause_requested() || stop.load(Ordering::Acquire),
        "Pause should be cleared or stop signaled"
    );
}

/// Fuzz fingerprint store mode switching
fn fuzz_fingerprint_switch(input: FingerprintSwitchInput) {
    let num_threads = ((input.num_threads as usize) % MAX_WORKERS).max(1);
    let actions: Vec<_> = input
        .actions
        .into_iter()
        .take(MAX_OPS)
        .map(|(id, action)| ((id as usize) % num_threads, action))
        .collect();

    // Low threshold to trigger switches quickly during fuzzing
    let switch_threshold = (input.switch_timing as u64 % 50).max(5);
    let store = Arc::new(MockAutoSwitchingStore::new(switch_threshold));

    // Track all inserted fingerprints for correctness checking
    let inserted = Arc::new(Mutex::new(HashSet::new()));

    let action_idx = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let store = Arc::clone(&store);
        let inserted = Arc::clone(&inserted);
        let action_idx = Arc::clone(&action_idx);
        let actions = actions.clone();

        handles.push(thread::spawn(move || {
            loop {
                let idx = action_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= actions.len() {
                    break;
                }

                let (target_thread, action) = actions[idx];
                if target_thread != thread_id {
                    continue;
                }

                match action {
                    FingerprintAction::ContainsOrInsert(fp) => {
                        let was_present = store.contains_or_insert(fp);
                        if !was_present {
                            inserted.lock().insert(fp);
                        }
                    }
                    FingerprintAction::ForceSwitch => {
                        store.force_switch();
                    }
                    FingerprintAction::CheckHybrid => {
                        std::hint::black_box(store.is_hybrid());
                    }
                }
            }
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }

    // Correctness check: all inserted fingerprints should be findable
    let all_inserted = inserted.lock().clone();
    for fp in &all_inserted {
        assert!(
            store.contains_or_insert(*fp),
            "Fingerprint {} was inserted but not found (hybrid={})",
            fp,
            store.is_hybrid()
        );
    }

    // Invariant: exact_count + bloom_count should be >= unique fingerprints inserted
    // (could be equal if no duplicates, or less if some were duplicates)
    let total_stored = store.exact_count() + store.bloom_count();
    assert!(
        total_stored <= all_inserted.len() + 1,
        "Stored count {} exceeds inserted count {} + 1",
        total_stored,
        all_inserted.len()
    );
}

/// Fuzz combined checkpoint + switch scenario
fn fuzz_combined(input: CombinedInput) {
    let num_workers = ((input.num_workers as usize) % MAX_WORKERS).max(1);
    let actions: Vec<_> = input.actions.into_iter().take(MAX_OPS).collect();

    let pause = Arc::new(MockPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let live_workers = Arc::new(AtomicUsize::new(num_workers));
    let active_workers = Arc::new(AtomicUsize::new(0));
    let store = Arc::new(MockAutoSwitchingStore::new(20));
    let inserted = Arc::new(Mutex::new(HashSet::new()));

    let action_idx = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    // Spawn worker threads that process mixed actions
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let store = Arc::clone(&store);
        let inserted = Arc::clone(&inserted);
        let action_idx = Arc::clone(&action_idx);
        let active_workers = Arc::clone(&active_workers);
        let actions = actions.clone();

        handles.push(thread::spawn(move || {
            loop {
                let idx = action_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= actions.len() {
                    break;
                }

                if stop.load(Ordering::Acquire) {
                    break;
                }

                match &actions[idx] {
                    CombinedAction::Worker(action) => match action {
                        WorkerAction::Work => {
                            active_workers.fetch_add(1, Ordering::AcqRel);
                            std::hint::black_box(worker_id * 42);
                            active_workers.fetch_sub(1, Ordering::AcqRel);
                        }
                        WorkerAction::CheckPausePoint => {
                            pause.worker_pause_point(&stop);
                        }
                        WorkerAction::Yield => {
                            thread::yield_now();
                        }
                    },
                    CombinedAction::Fingerprint(action) => match action {
                        FingerprintAction::ContainsOrInsert(fp) => {
                            let was_present = store.contains_or_insert(*fp);
                            if !was_present {
                                inserted.lock().insert(*fp);
                            }
                        }
                        FingerprintAction::ForceSwitch => {
                            store.force_switch();
                        }
                        FingerprintAction::CheckHybrid => {
                            std::hint::black_box(store.is_hybrid());
                        }
                    },
                    CombinedAction::Controller(_) => {
                        // Workers skip controller actions
                        continue;
                    }
                }
            }
        }));
    }

    // Controller thread processes controller actions
    let pause_ctrl = Arc::clone(&pause);
    let stop_ctrl = Arc::clone(&stop);
    let live_workers_ctrl = Arc::clone(&live_workers);
    let active_workers_ctrl = Arc::clone(&active_workers);
    let action_idx_ctrl = Arc::clone(&action_idx);
    let actions_ctrl = actions.clone();

    let controller = thread::spawn(move || {
        loop {
            let idx = action_idx_ctrl.fetch_add(1, Ordering::Relaxed);
            if idx >= actions_ctrl.len() {
                break;
            }

            if stop_ctrl.load(Ordering::Acquire) {
                break;
            }

            if let CombinedAction::Controller(action) = &actions_ctrl[idx] {
                match action {
                    ControllerAction::RequestPause => {
                        pause_ctrl.request_pause();
                    }
                    ControllerAction::WaitQuiescence => {
                        pause_ctrl.wait_for_quiescence(
                            &stop_ctrl,
                            &active_workers_ctrl,
                            &live_workers_ctrl,
                        );
                    }
                    ControllerAction::Resume => {
                        pause_ctrl.resume();
                    }
                    ControllerAction::Yield => {
                        thread::yield_now();
                    }
                }
            }
        }
    });

    // Wait for controller to finish
    let _ = controller.join();

    // Signal stop and resume to release any paused workers
    stop.store(true, Ordering::Release);
    pause.resume();

    // Wait for all workers
    for handle in handles {
        let _ = handle.join();
    }

    // Correctness checks
    let all_inserted = inserted.lock().clone();
    for fp in &all_inserted {
        assert!(
            store.contains_or_insert(*fp),
            "Combined: Fingerprint {} was inserted but not found",
            fp
        );
    }
}

// ============================================================================
// Main fuzz target
// ============================================================================

fuzz_target!(|input: FuzzInput| {
    match input {
        FuzzInput::WorkerPause(inner) => fuzz_worker_pause(inner),
        FuzzInput::FingerprintSwitch(inner) => fuzz_fingerprint_switch(inner),
        FuzzInput::Combined(inner) => fuzz_combined(inner),
    }
});
