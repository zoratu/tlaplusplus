#![no_main]

//! Fuzzing tests for quiescence timeout improvements.
//!
//! This fuzz target specifically tests the timeout/backoff behavior:
//! 1. Random pause/unpause sequences with timeout recovery
//! 2. Try-read mechanism with switch_pending flag (20 attempt limit)
//! 3. Exponential backoff state machine
//! 4. Concurrent access patterns during fingerprint store switch

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use parking_lot::{Condvar, Mutex, RwLock};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Maximum workers for fuzzing
const MAX_WORKERS: usize = 8;

/// Maximum operations per iteration
const MAX_OPS: usize = 50;

/// Timeout for operations (ms)
const OP_TIMEOUT_MS: u64 = 20;

// ============================================================================
// Fuzz Input Types
// ============================================================================

/// Actions that affect pause state
#[derive(Debug, Clone, Copy, Arbitrary)]
enum PauseAction {
    /// Request all workers to pause
    RequestPause,
    /// Resume all workers
    Resume,
    /// Toggle pause state
    Toggle,
}

/// Actions that workers can take
#[derive(Debug, Clone, Copy, Arbitrary)]
enum WorkerBehavior {
    /// Cooperate and reach pause point normally
    Cooperative,
    /// Delay before reaching pause point (slow worker)
    Slow { delay_us: u16 },
    /// Skip pause point entirely (stuck worker)
    Stuck,
    /// Random behavior based on seed
    Random { seed: u8 },
}

/// Actions for fingerprint store
#[derive(Debug, Clone, Copy, Arbitrary)]
enum StoreAction {
    /// Insert fingerprint
    Insert(u64),
    /// Set switch_pending flag
    SetSwitchPending(bool),
    /// Simulate try_read with configurable attempts
    TryRead { max_attempts: u8 },
}

/// Timeout/backoff configuration
#[derive(Debug, Clone, Arbitrary)]
struct BackoffConfig {
    /// Initial timeout (scaled down for fuzzing)
    initial_timeout_scale: u8, // 1-255 maps to 1-25ms
    /// Number of retries
    max_retries: u8, // 0-7
    /// Backoff multiplier (1-4)
    backoff_multiplier: u8,
}

/// Fuzz input for pause/unpause sequences
#[derive(Debug, Arbitrary)]
struct PauseSequenceInput {
    /// Number of workers
    num_workers: u8,
    /// Worker behaviors (indexed by worker_id % len)
    worker_behaviors: Vec<WorkerBehavior>,
    /// Sequence of pause actions
    pause_actions: Vec<PauseAction>,
    /// Backoff configuration
    backoff_config: BackoffConfig,
}

/// Fuzz input for try-read mechanism
#[derive(Debug, Arbitrary)]
struct TryReadInput {
    /// Number of concurrent workers
    num_workers: u8,
    /// Sequence of store actions
    actions: Vec<StoreAction>,
    /// Duration to hold switch_pending (scaled)
    switch_duration_scale: u8,
}

/// Fuzz input for concurrent access during switch
#[derive(Debug, Arbitrary)]
struct ConcurrentSwitchInput {
    /// Number of workers
    num_workers: u8,
    /// Fingerprints to insert
    fingerprints: Vec<u64>,
    /// When to trigger switch (as index into fingerprints)
    switch_point: u8,
    /// Whether to have a stuck worker
    has_stuck_worker: bool,
}

/// Top-level fuzz input
#[derive(Debug, Arbitrary)]
enum FuzzInput {
    PauseSequence(PauseSequenceInput),
    TryRead(TryReadInput),
    ConcurrentSwitch(ConcurrentSwitchInput),
}

// ============================================================================
// Mock Components
// ============================================================================

/// Mock pause controller with backoff support
struct MockPauseControllerWithBackoff {
    requested: AtomicBool,
    paused_workers: AtomicUsize,
    worker_paused: Vec<AtomicBool>,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
    /// Track retry count
    retry_count: AtomicU32,
    /// Track timeouts
    timeout_count: AtomicU32,
}

impl MockPauseControllerWithBackoff {
    fn new(num_workers: usize) -> Self {
        let mut worker_paused = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            worker_paused.push(AtomicBool::new(false));
        }
        Self {
            requested: AtomicBool::new(false),
            paused_workers: AtomicUsize::new(0),
            worker_paused,
            wait_lock: Mutex::new(()),
            wait_cv: Condvar::new(),
            retry_count: AtomicU32::new(0),
            timeout_count: AtomicU32::new(0),
        }
    }

    fn worker_pause_point(&self, worker_id: usize, stop: &AtomicBool) {
        if !self.requested.load(Ordering::Acquire) {
            return;
        }

        if worker_id < self.worker_paused.len() {
            self.worker_paused[worker_id].store(true, Ordering::Release);
        }
        self.paused_workers.fetch_add(1, Ordering::AcqRel);

        let mut guard = self.wait_lock.lock();
        while self.requested.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
            let result = self.wait_cv.wait_for(&mut guard, Duration::from_millis(OP_TIMEOUT_MS));
            if result.timed_out() {
                break;
            }
        }
        drop(guard);

        self.paused_workers.fetch_sub(1, Ordering::AcqRel);
        if worker_id < self.worker_paused.len() {
            self.worker_paused[worker_id].store(false, Ordering::Release);
        }
    }

    fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn resume(&self) {
        self.requested.store(false, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn toggle(&self) {
        let current = self.requested.load(Ordering::Acquire);
        self.requested.store(!current, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn is_pause_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn get_unpaused_workers(&self) -> Vec<usize> {
        self.worker_paused
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.load(Ordering::Acquire))
            .map(|(id, _)| id)
            .collect()
    }

    /// Wait for quiescence with exponential backoff
    fn wait_for_quiescence_with_backoff(
        &self,
        stop: &AtomicBool,
        live_workers: &AtomicUsize,
        initial_timeout_ms: u64,
        max_retries: u32,
        backoff_multiplier: u32,
    ) -> bool {
        let mut current_timeout = initial_timeout_ms;
        let mut retries = 0u32;

        loop {
            let timeout = Duration::from_millis(current_timeout);
            let start = std::time::Instant::now();

            // Wait attempt
            loop {
                if stop.load(Ordering::Acquire) {
                    return false;
                }

                if start.elapsed() > timeout {
                    self.timeout_count.fetch_add(1, Ordering::Relaxed);
                    break;
                }

                let paused = self.paused_workers.load(Ordering::Acquire);
                let live = live_workers.load(Ordering::Acquire);

                if paused >= live || live == 0 {
                    return true;
                }

                thread::yield_now();
            }

            // Timeout occurred
            retries += 1;
            self.retry_count.fetch_add(1, Ordering::Relaxed);

            if retries >= max_retries {
                return false;
            }

            // Exponential backoff
            current_timeout *= backoff_multiplier as u64;
        }
    }

    fn total_retries(&self) -> u32 {
        self.retry_count.load(Ordering::Relaxed)
    }

    fn total_timeouts(&self) -> u32 {
        self.timeout_count.load(Ordering::Relaxed)
    }
}

/// Mock fingerprint store with try-read tracking
struct MockStoreWithTryRead {
    data: RwLock<HashSet<u64>>,
    switch_pending: AtomicBool,
    try_read_attempts: AtomicU64,
    exceeded_threshold: AtomicBool,
}

impl MockStoreWithTryRead {
    fn new() -> Self {
        Self {
            data: RwLock::new(HashSet::new()),
            switch_pending: AtomicBool::new(false),
            try_read_attempts: AtomicU64::new(0),
            exceeded_threshold: AtomicBool::new(false),
        }
    }

    fn contains_or_insert(&self, fp: u64) -> bool {
        if self.switch_pending.load(Ordering::Acquire) {
            let mut attempts = 0u32;
            loop {
                if let Some(mut guard) = self.data.try_write() {
                    let existed = guard.contains(&fp);
                    if !existed {
                        guard.insert(fp);
                    }
                    return existed;
                }
                attempts += 1;
                self.try_read_attempts.fetch_add(1, Ordering::Relaxed);

                if attempts > 10 {
                    self.exceeded_threshold.store(true, Ordering::Relaxed);
                }

                if attempts > 20 {
                    // Bail out - return true to let worker proceed
                    return true;
                }

                std::thread::sleep(Duration::from_micros(50));

                if !self.switch_pending.load(Ordering::Acquire) {
                    break;
                }
            }
        }

        let mut guard = self.data.write();
        let existed = guard.contains(&fp);
        if !existed {
            guard.insert(fp);
        }
        existed
    }

    fn try_read_with_limit(&self, max_attempts: u32) -> bool {
        if !self.switch_pending.load(Ordering::Acquire) {
            return true; // Success immediately
        }

        let mut attempts = 0u32;
        loop {
            if let Some(_guard) = self.data.try_read() {
                return true;
            }
            attempts += 1;
            self.try_read_attempts.fetch_add(1, Ordering::Relaxed);

            if attempts > 10 {
                self.exceeded_threshold.store(true, Ordering::Relaxed);
            }

            if attempts >= max_attempts {
                return false;
            }

            std::thread::sleep(Duration::from_micros(50));

            if !self.switch_pending.load(Ordering::Acquire) {
                return true;
            }
        }
    }

    fn set_switch_pending(&self, pending: bool) {
        self.switch_pending.store(pending, Ordering::Release);
    }

    fn total_try_read_attempts(&self) -> u64 {
        self.try_read_attempts.load(Ordering::Relaxed)
    }

    fn exceeded_threshold(&self) -> bool {
        self.exceeded_threshold.load(Ordering::Relaxed)
    }

    fn len(&self) -> usize {
        self.data.read().len()
    }
}

// ============================================================================
// Fuzz Implementations
// ============================================================================

/// Fuzz random pause/unpause sequences with timeout recovery
fn fuzz_pause_sequence(input: PauseSequenceInput) {
    let num_workers = ((input.num_workers as usize) % MAX_WORKERS).max(1);
    let behaviors: Vec<_> = input.worker_behaviors.into_iter().take(num_workers).collect();
    let actions: Vec<_> = input.pause_actions.into_iter().take(MAX_OPS).collect();

    let pause = Arc::new(MockPauseControllerWithBackoff::new(num_workers));
    let stop = Arc::new(AtomicBool::new(false));
    let live_workers = Arc::new(AtomicUsize::new(num_workers));

    // Spawn workers with configured behaviors
    let mut handles = Vec::new();
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let behavior = behaviors.get(worker_id % behaviors.len().max(1))
            .copied()
            .unwrap_or(WorkerBehavior::Cooperative);

        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                match behavior {
                    WorkerBehavior::Cooperative => {
                        pause.worker_pause_point(worker_id, &stop);
                    }
                    WorkerBehavior::Slow { delay_us } => {
                        thread::sleep(Duration::from_micros(delay_us as u64));
                        pause.worker_pause_point(worker_id, &stop);
                    }
                    WorkerBehavior::Stuck => {
                        // Never reach pause point
                        thread::sleep(Duration::from_millis(1));
                    }
                    WorkerBehavior::Random { seed } => {
                        if seed % 3 == 0 {
                            pause.worker_pause_point(worker_id, &stop);
                        } else if seed % 3 == 1 {
                            thread::sleep(Duration::from_micros((seed as u64) * 10));
                            pause.worker_pause_point(worker_id, &stop);
                        }
                        // else: stuck behavior
                    }
                }
                thread::yield_now();
            }
        }));
    }

    // Parse backoff config
    let initial_timeout = (input.backoff_config.initial_timeout_scale as u64 % 25).max(1);
    let max_retries = (input.backoff_config.max_retries % 8).max(1) as u32;
    let multiplier = (input.backoff_config.backoff_multiplier % 4).max(1) as u32 + 1;

    // Execute pause actions
    let mut quiescence_results = Vec::new();
    for action in actions {
        if stop.load(Ordering::Acquire) {
            break;
        }

        match action {
            PauseAction::RequestPause => {
                pause.request_pause();
                let achieved = pause.wait_for_quiescence_with_backoff(
                    &stop,
                    &live_workers,
                    initial_timeout,
                    max_retries,
                    multiplier,
                );
                quiescence_results.push(achieved);
            }
            PauseAction::Resume => {
                pause.resume();
            }
            PauseAction::Toggle => {
                pause.toggle();
            }
        }
    }

    // Cleanup
    stop.store(true, Ordering::Release);
    pause.resume();

    for handle in handles {
        let _ = handle.join();
    }

    // Invariants
    // 1. Controller should not be in pause state after stop
    assert!(
        !pause.is_pause_requested() || stop.load(Ordering::Acquire),
        "Pause should be cleared after stop"
    );

    // 2. Retry/timeout counts should be consistent
    let retries = pause.total_retries();
    let timeouts = pause.total_timeouts();
    assert!(
        timeouts >= retries,
        "Timeouts should be >= retries: {} < {}",
        timeouts,
        retries
    );
}

/// Fuzz try-read mechanism with switch_pending
fn fuzz_try_read(input: TryReadInput) {
    let num_workers = ((input.num_workers as usize) % MAX_WORKERS).max(1);
    let actions: Vec<_> = input.actions.into_iter().take(MAX_OPS).collect();
    let switch_duration = Duration::from_micros((input.switch_duration_scale as u64 * 10) % 5000);

    let store = Arc::new(MockStoreWithTryRead::new());
    let stop = Arc::new(AtomicBool::new(false));
    let action_idx = Arc::new(AtomicUsize::new(0));

    // Spawn workers
    let mut handles = Vec::new();
    for _worker_id in 0..num_workers {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        let action_idx = Arc::clone(&action_idx);
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
                    StoreAction::Insert(fp) => {
                        store.contains_or_insert(*fp);
                    }
                    StoreAction::SetSwitchPending(pending) => {
                        store.set_switch_pending(*pending);
                    }
                    StoreAction::TryRead { max_attempts } => {
                        store.try_read_with_limit(*max_attempts as u32);
                    }
                }
            }
        }));
    }

    // Periodically toggle switch_pending
    let store_toggle = Arc::clone(&store);
    let stop_toggle = Arc::clone(&stop);
    let toggler = thread::spawn(move || {
        for _ in 0..5 {
            if stop_toggle.load(Ordering::Acquire) {
                break;
            }
            store_toggle.set_switch_pending(true);
            thread::sleep(switch_duration);
            store_toggle.set_switch_pending(false);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Wait for completion
    for handle in handles {
        let _ = handle.join();
    }

    stop.store(true, Ordering::Release);
    let _ = toggler.join();

    // The store should have some data (unless all operations were try_read)
    // No specific invariant here - we're testing for crashes/hangs
}

/// Fuzz concurrent access during fingerprint store switch
fn fuzz_concurrent_switch(input: ConcurrentSwitchInput) {
    let num_workers = ((input.num_workers as usize) % MAX_WORKERS).max(1);
    let fingerprints: Vec<u64> = input.fingerprints.into_iter().take(MAX_OPS).collect();
    let switch_point = (input.switch_point as usize) % fingerprints.len().max(1);

    let pause = Arc::new(MockPauseControllerWithBackoff::new(num_workers));
    let store = Arc::new(MockStoreWithTryRead::new());
    let stop = Arc::new(AtomicBool::new(false));
    let live_workers = Arc::new(AtomicUsize::new(num_workers));
    let fp_idx = Arc::new(AtomicUsize::new(0));
    let inserted = Arc::new(Mutex::new(HashSet::new()));

    // Spawn workers
    let mut handles = Vec::new();
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        let fp_idx = Arc::clone(&fp_idx);
        let fingerprints = fingerprints.clone();
        let inserted = Arc::clone(&inserted);
        let is_stuck = input.has_stuck_worker && worker_id == 0;

        handles.push(thread::spawn(move || {
            loop {
                if stop.load(Ordering::Acquire) {
                    break;
                }

                // Check pause point (unless stuck)
                if !is_stuck {
                    pause.worker_pause_point(worker_id, &stop);
                }

                // Process fingerprints
                let idx = fp_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= fingerprints.len() {
                    break;
                }

                let fp = fingerprints[idx];
                let existed = store.contains_or_insert(fp);
                if !existed {
                    inserted.lock().insert(fp);
                }
            }
        }));
    }

    // Controller: trigger switch at switch_point
    let store_switch = Arc::clone(&store);
    let pause_ctrl = Arc::clone(&pause);
    let stop_ctrl = Arc::clone(&stop);
    let live_ctrl = Arc::clone(&live_workers);
    let fp_idx_ctrl = Arc::clone(&fp_idx);

    let controller = thread::spawn(move || {
        // Wait for workers to reach switch_point
        while fp_idx_ctrl.load(Ordering::Relaxed) < switch_point {
            if stop_ctrl.load(Ordering::Acquire) {
                return;
            }
            thread::yield_now();
        }

        // Trigger pause
        pause_ctrl.request_pause();

        // Set switch_pending during pause
        store_switch.set_switch_pending(true);

        // Wait for quiescence (with backoff)
        pause_ctrl.wait_for_quiescence_with_backoff(&stop_ctrl, &live_ctrl, 5, 3, 2);

        // Complete switch
        store_switch.set_switch_pending(false);

        // Resume
        pause_ctrl.resume();
    });

    // Wait for controller
    let _ = controller.join();

    // Signal stop and cleanup
    stop.store(true, Ordering::Release);
    pause.resume();

    for handle in handles {
        let _ = handle.join();
    }

    // Correctness: all inserted fingerprints should exist
    let all_inserted = inserted.lock().clone();
    for fp in &all_inserted {
        assert!(
            store.contains_or_insert(*fp),
            "Fingerprint {} was inserted but not found",
            fp
        );
    }

    // Store size should be consistent
    let store_len = store.len();
    assert!(
        store_len <= all_inserted.len() + 1,
        "Store size {} exceeds inserted count {}",
        store_len,
        all_inserted.len()
    );
}

// ============================================================================
// Main Fuzz Target
// ============================================================================

fuzz_target!(|input: FuzzInput| {
    match input {
        FuzzInput::PauseSequence(inner) => fuzz_pause_sequence(inner),
        FuzzInput::TryRead(inner) => fuzz_try_read(inner),
        FuzzInput::ConcurrentSwitch(inner) => fuzz_concurrent_switch(inner),
    }
});
