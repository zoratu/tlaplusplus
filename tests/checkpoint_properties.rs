//! Property-based tests for checkpoint quiescence coordination
//!
//! These tests verify the correctness of the checkpoint coordination mechanism:
//! - Pause/Resume linearizability
//! - Fingerprint store mode consistency across switches
//! - Checkpoint atomicity
//! - Bounded quiescence achievement
//! - Deadlock freedom under concurrent operations

use parking_lot::{Condvar, Mutex};
use proptest::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Test Infrastructure: Simulated PauseController
// ============================================================================

/// A simplified PauseController for testing (mirrors runtime.rs implementation)
#[derive(Default)]
struct TestPauseController {
    requested: AtomicBool,
    paused_workers: AtomicUsize,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

impl TestPauseController {
    fn new() -> Self {
        Self::default()
    }

    /// Called by workers at safe points to check if they should pause
    fn worker_pause_point(&self, stop: &AtomicBool) {
        if !self.requested.load(Ordering::Acquire) {
            return;
        }
        self.paused_workers.fetch_add(1, Ordering::AcqRel);
        let mut guard = self.wait_lock.lock();
        while self.requested.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
            self.wait_cv.wait_for(&mut guard, Duration::from_millis(1));
        }
        drop(guard);
        self.paused_workers.fetch_sub(1, Ordering::AcqRel);
    }

    fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    /// Wait for quiescence - returns true when all live workers are paused or stop is signaled
    fn wait_for_quiescence(
        &self,
        stop: &AtomicBool,
        live_workers: &AtomicUsize,
        timeout: Duration,
    ) -> bool {
        let start = Instant::now();
        loop {
            if stop.load(Ordering::Acquire) {
                return true;
            }
            let paused = self.paused_workers.load(Ordering::Acquire);
            let live = live_workers.load(Ordering::Acquire);
            // Quiescence achieved if all live workers are paused, or no workers are live
            if paused >= live || live == 0 {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            thread::sleep(Duration::from_micros(100));
        }
    }

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
// Test Infrastructure: Simulated Fingerprint Store with Mode Switching
// ============================================================================

/// Simulated fingerprint store with exact-to-bloom mode switching
struct TestFingerprintStore {
    /// Exact store (pre-switch)
    exact: Mutex<HashSet<u64>>,
    /// Bloom store (post-switch) - simplified as HashSet with false positive simulation
    bloom: Mutex<HashSet<u64>>,
    /// Whether we've switched to hybrid mode
    switched: AtomicBool,
    /// Insert counter
    inserts: AtomicU64,
}

impl TestFingerprintStore {
    fn new() -> Self {
        Self {
            exact: Mutex::new(HashSet::new()),
            bloom: Mutex::new(HashSet::new()),
            switched: AtomicBool::new(false),
            inserts: AtomicU64::new(0),
        }
    }

    /// Check if fingerprint exists, insert if not
    /// Returns true if the fingerprint was already present
    fn contains_or_insert(&self, fp: u64) -> bool {
        self.inserts.fetch_add(1, Ordering::Relaxed);

        if self.switched.load(Ordering::Acquire) {
            // Hybrid mode: check exact first (read-only), then bloom
            let exact = self.exact.lock();
            if exact.contains(&fp) {
                return true;
            }
            drop(exact);

            let mut bloom = self.bloom.lock();
            if bloom.contains(&fp) {
                return true;
            }
            bloom.insert(fp);
            false
        } else {
            // Exact mode only
            let mut exact = self.exact.lock();
            if exact.contains(&fp) {
                return true;
            }
            exact.insert(fp);
            false
        }
    }

    /// Check if fingerprint exists (without inserting)
    fn contains(&self, fp: u64) -> bool {
        if self.switched.load(Ordering::Acquire) {
            let exact = self.exact.lock();
            if exact.contains(&fp) {
                return true;
            }
            drop(exact);
            self.bloom.lock().contains(&fp)
        } else {
            self.exact.lock().contains(&fp)
        }
    }

    /// Switch to hybrid mode (exact becomes read-only, bloom accepts new inserts)
    fn switch_to_hybrid(&self) {
        self.switched.store(true, Ordering::Release);
    }

    fn is_hybrid(&self) -> bool {
        self.switched.load(Ordering::Acquire)
    }

    /// Get total number of fingerprints stored
    fn len(&self) -> usize {
        let exact_len = self.exact.lock().len();
        let bloom_len = self.bloom.lock().len();
        exact_len + bloom_len
    }

    /// Get all fingerprints (for verification)
    fn all_fingerprints(&self) -> HashSet<u64> {
        let exact = self.exact.lock();
        let bloom = self.bloom.lock();
        exact.union(&*bloom).copied().collect()
    }
}

// ============================================================================
// Property Generators
// ============================================================================

/// Generate pause/resume operations
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum PauseResumeOp {
    Pause,
    Resume,
    WorkerPausePoint,
}

#[allow(dead_code)]
fn pause_resume_op() -> impl Strategy<Value = PauseResumeOp> {
    prop_oneof![
        Just(PauseResumeOp::Pause),
        Just(PauseResumeOp::Resume),
        Just(PauseResumeOp::WorkerPausePoint),
    ]
}

// ============================================================================
// Property 1: Pause/Resume Linearizability
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: After resume, paused_workers == 0 and all workers can proceed
    /// During pause, all workers eventually reach the pause point
    #[test]
    fn prop_pause_resume_linearizable(
        num_workers in 2..=8usize,
        pause_resume_cycles in 1..=3usize,
    ) {
        let pause_ctrl = Arc::new(TestPauseController::new());
        let stop = Arc::new(AtomicBool::new(false));
        let live_workers = Arc::new(AtomicUsize::new(num_workers));
        let work_done = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(num_workers + 1)); // +1 for coordinator

        // Spawn worker threads that run until stop
        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let pc = Arc::clone(&pause_ctrl);
                let st = Arc::clone(&stop);
                let wd = Arc::clone(&work_done);
                let br = Arc::clone(&barrier);
                let lw = Arc::clone(&live_workers);

                thread::spawn(move || {
                    br.wait(); // Sync start
                    while !st.load(Ordering::Acquire) {
                        // Simulate work
                        wd.fetch_add(1, Ordering::Relaxed);
                        // Check for pause
                        pc.worker_pause_point(&st);
                        // Small delay to allow interleaving
                        thread::yield_now();
                    }
                    // Decrement live workers when exiting
                    lw.fetch_sub(1, Ordering::AcqRel);
                })
            })
            .collect();

        barrier.wait(); // Start all workers

        // Run pause/resume cycles
        for _ in 0..pause_resume_cycles {
            // Request pause
            pause_ctrl.request_pause();

            // Wait for quiescence with timeout
            let quiesced = pause_ctrl.wait_for_quiescence(
                &stop,
                &live_workers,
                Duration::from_secs(5),
            );
            prop_assert!(quiesced, "Quiescence not achieved within timeout");

            // Verify all live workers are paused
            let paused = pause_ctrl.paused_count();
            let live = live_workers.load(Ordering::Acquire);
            prop_assert_eq!(paused, live, "Not all live workers paused: {} paused, {} live", paused, live);

            // Resume
            pause_ctrl.resume();

            // Give workers time to unpause
            thread::sleep(Duration::from_millis(10));

            // Eventually no workers should be paused
            let mut retries = 0;
            while pause_ctrl.paused_count() > 0 && retries < 100 {
                thread::sleep(Duration::from_millis(1));
                retries += 1;
            }
            prop_assert_eq!(pause_ctrl.paused_count(), 0, "Workers still paused after resume");
        }

        // Stop all workers
        stop.store(true, Ordering::Release);
        pause_ctrl.resume(); // Wake any paused workers

        for handle in handles {
            handle.join().expect("Worker thread panicked");
        }
    }
}

// ============================================================================
// Property 2: Fingerprint Store Mode Consistency
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: All fingerprints remain findable after mode switch
    /// No false negatives allowed
    #[test]
    fn prop_fp_store_mode_consistency(
        fingerprints in prop::collection::vec(any::<u64>(), 1..200),
        switch_after in 0..200usize,
    ) {
        let store = TestFingerprintStore::new();
        let switch_point = switch_after.min(fingerprints.len());

        // Track which fingerprints we've inserted
        let mut inserted = HashSet::new();

        // Insert fingerprints, switching mode after switch_point
        for (i, &fp) in fingerprints.iter().enumerate() {
            if i == switch_point {
                store.switch_to_hybrid();
            }
            if !inserted.contains(&fp) {
                let was_present = store.contains_or_insert(fp);
                prop_assert!(!was_present, "False positive: {} claimed present before insert", fp);
                inserted.insert(fp);
            } else {
                // Duplicate insert - should return true
                let was_present = store.contains_or_insert(fp);
                prop_assert!(was_present, "False negative: {} not found after insert", fp);
            }
        }

        // Verify: all inserted fingerprints are still findable
        for &fp in &inserted {
            prop_assert!(
                store.contains(fp),
                "False negative after switch: {} not found (switched at {})",
                fp,
                switch_point
            );
        }

        // Verify count matches
        prop_assert_eq!(
            store.len(),
            inserted.len(),
            "Store size mismatch: expected {}, got {}",
            inserted.len(),
            store.len()
        );
    }
}

// ============================================================================
// Property 3: Checkpoint Atomicity
// ============================================================================

/// Simulated state for checkpoint atomicity testing
struct CheckpointState {
    states: Mutex<Vec<u64>>,
    checkpointed: Mutex<Vec<u64>>,
    checkpoint_count: AtomicUsize,
}

impl CheckpointState {
    fn new() -> Self {
        Self {
            states: Mutex::new(Vec::new()),
            checkpointed: Mutex::new(Vec::new()),
            checkpoint_count: AtomicUsize::new(0),
        }
    }

    fn add_state(&self, state: u64) {
        self.states.lock().push(state);
    }

    fn checkpoint(&self) {
        let mut states = self.states.lock();
        let mut checkpointed = self.checkpointed.lock();

        // Move all current states to checkpoint
        checkpointed.append(&mut *states);
        self.checkpoint_count.fetch_add(1, Ordering::Relaxed);
    }

    fn all_states(&self) -> Vec<u64> {
        let states = self.states.lock();
        let checkpointed = self.checkpointed.lock();
        let mut all = states.clone();
        all.extend(checkpointed.iter().copied());
        all
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: No state lost during checkpoint, no duplicates after resume
    /// Workers run in a continuous loop until stopped
    #[test]
    fn prop_checkpoint_atomic(
        num_workers in 2..=4usize,
    ) {
        let pause_ctrl = Arc::new(TestPauseController::new());
        let stop = Arc::new(AtomicBool::new(false));
        let live_workers = Arc::new(AtomicUsize::new(num_workers));
        let checkpoint_state = Arc::new(CheckpointState::new());
        let states_generated = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(num_workers + 1));

        // Spawn worker threads that run until stop
        let handles: Vec<_> = (0..num_workers)
            .map(|worker_id| {
                let pc = Arc::clone(&pause_ctrl);
                let st = Arc::clone(&stop);
                let cs = Arc::clone(&checkpoint_state);
                let sg = Arc::clone(&states_generated);
                let lw = Arc::clone(&live_workers);
                let br = Arc::clone(&barrier);

                thread::spawn(move || {
                    br.wait();
                    let mut state_counter = 0u64;

                    // Workers run until stop is signaled
                    while !st.load(Ordering::Acquire) {
                        // Generate state with unique ID
                        let state_id = (worker_id as u64) << 32 | state_counter;
                        cs.add_state(state_id);
                        sg.fetch_add(1, Ordering::Relaxed);
                        state_counter += 1;

                        // Pause point - workers will block here during checkpoint
                        pc.worker_pause_point(&st);

                        thread::yield_now();
                    }
                    // Decrement live workers when exiting
                    lw.fetch_sub(1, Ordering::AcqRel);
                })
            })
            .collect();

        barrier.wait();

        // Let workers generate some states
        thread::sleep(Duration::from_millis(5));

        // Perform checkpoint
        pause_ctrl.request_pause();
        let quiesced = pause_ctrl.wait_for_quiescence(&stop, &live_workers, Duration::from_secs(2));
        prop_assert!(quiesced, "Failed to achieve quiescence for checkpoint");

        // Do the checkpoint
        checkpoint_state.checkpoint();

        // Resume
        pause_ctrl.resume();

        // Let workers run a bit more
        thread::sleep(Duration::from_millis(5));

        // Stop workers
        stop.store(true, Ordering::Release);
        pause_ctrl.resume();

        for handle in handles {
            handle.join().expect("Worker panicked");
        }

        // Verify: no states lost, no duplicates
        let all_states = checkpoint_state.all_states();
        let unique_states: HashSet<u64> = all_states.iter().copied().collect();

        // Check no duplicates
        prop_assert_eq!(
            all_states.len(),
            unique_states.len(),
            "Duplicate states detected"
        );

        // Check total matches generated count
        let generated = states_generated.load(Ordering::Relaxed) as usize;
        prop_assert_eq!(
            all_states.len(),
            generated,
            "State count mismatch: {} in store, {} generated",
            all_states.len(),
            generated
        );
    }
}

// ============================================================================
// Property 4: Quiescence Eventual (Bounded)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: Quiescence is achieved within bounded time
    /// Time bound: max(worker_delays) + epsilon
    #[test]
    fn prop_quiescence_eventual(
        num_workers in 2..=8usize,
        worker_delays_ms in prop::collection::vec(0..10u64, 8),
    ) {
        let pause_ctrl = Arc::new(TestPauseController::new());
        let stop = Arc::new(AtomicBool::new(false));
        let live_workers = Arc::new(AtomicUsize::new(num_workers));
        let barrier = Arc::new(Barrier::new(num_workers + 1));

        // Get delays for our workers
        let delays: Vec<u64> = worker_delays_ms.iter().take(num_workers).copied().collect();
        let max_delay = *delays.iter().max().unwrap_or(&0);

        // Spawn workers with varying delays before reaching pause point
        let handles: Vec<_> = delays
            .into_iter()
            .map(|delay| {
                let pc = Arc::clone(&pause_ctrl);
                let st = Arc::clone(&stop);
                let lw = Arc::clone(&live_workers);
                let br = Arc::clone(&barrier);

                thread::spawn(move || {
                    br.wait();
                    while !st.load(Ordering::Acquire) {
                        // Simulate work with delay
                        thread::sleep(Duration::from_millis(delay));
                        // Pause point
                        pc.worker_pause_point(&st);
                    }
                    lw.fetch_sub(1, Ordering::AcqRel);
                })
            })
            .collect();

        barrier.wait();

        // Request pause
        let start = Instant::now();
        pause_ctrl.request_pause();

        // Time bound: max delay + generous epsilon for scheduling
        let timeout = Duration::from_millis(max_delay + 500);
        let quiesced = pause_ctrl.wait_for_quiescence(&stop, &live_workers, timeout);

        let elapsed = start.elapsed();

        // Stop workers
        stop.store(true, Ordering::Release);
        pause_ctrl.resume();

        for handle in handles {
            let _ = handle.join();
        }

        prop_assert!(
            quiesced,
            "Quiescence not achieved within bounded time: elapsed={:?}, bound={:?}",
            elapsed,
            timeout
        );
    }
}

// ============================================================================
// Property 5: No Deadlock Under Concurrent Switch
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    /// Property: System doesn't deadlock when checkpoint and FP switch happen concurrently
    #[test]
    fn prop_no_deadlock_concurrent_switch(
        num_workers in 2..=4usize,
        switch_delay_ms in 0..10u64,
        checkpoint_delay_ms in 0..10u64,
    ) {
        let pause_ctrl = Arc::new(TestPauseController::new());
        let fp_store = Arc::new(TestFingerprintStore::new());
        let stop = Arc::new(AtomicBool::new(false));
        let live_workers = Arc::new(AtomicUsize::new(num_workers));
        let barrier = Arc::new(Barrier::new(num_workers + 3)); // workers + checkpoint + switch + main
        let deadlock_detected = Arc::new(AtomicBool::new(false));

        // Spawn worker threads that run until stop
        let worker_handles: Vec<_> = (0..num_workers)
            .map(|worker_id| {
                let pc = Arc::clone(&pause_ctrl);
                let fps = Arc::clone(&fp_store);
                let st = Arc::clone(&stop);
                let lw = Arc::clone(&live_workers);
                let br = Arc::clone(&barrier);

                thread::spawn(move || {
                    br.wait();
                    let mut counter = 0u64;
                    while !st.load(Ordering::Acquire) {
                        // Insert fingerprints
                        let fp = (worker_id as u64) * 1_000_000 + counter;
                        fps.contains_or_insert(fp);
                        counter += 1;

                        // Pause point
                        pc.worker_pause_point(&st);
                        thread::yield_now();
                    }
                    lw.fetch_sub(1, Ordering::AcqRel);
                })
            })
            .collect();

        // Checkpoint thread
        let checkpoint_handle = {
            let pc = Arc::clone(&pause_ctrl);
            let st = Arc::clone(&stop);
            let lw = Arc::clone(&live_workers);
            let br = Arc::clone(&barrier);
            let dd = Arc::clone(&deadlock_detected);

            thread::spawn(move || {
                br.wait();
                thread::sleep(Duration::from_millis(checkpoint_delay_ms));

                if !st.load(Ordering::Acquire) {
                    pc.request_pause();
                    let quiesced = pc.wait_for_quiescence(&st, &lw, Duration::from_secs(5));
                    if !quiesced && !st.load(Ordering::Acquire) {
                        dd.store(true, Ordering::Release);
                    }
                    pc.resume();
                }
            })
        };

        // Switch thread (triggers FP mode switch)
        let switch_handle = {
            let fps = Arc::clone(&fp_store);
            let st = Arc::clone(&stop);
            let br = Arc::clone(&barrier);

            thread::spawn(move || {
                br.wait();
                thread::sleep(Duration::from_millis(switch_delay_ms));

                if !st.load(Ordering::Acquire) {
                    fps.switch_to_hybrid();
                }
            })
        };

        barrier.wait();

        // Give checkpoint and switch threads time to run
        thread::sleep(Duration::from_millis(50));

        // Wait for checkpoint and switch to complete
        let _ = checkpoint_handle.join();
        let _ = switch_handle.join();

        // Stop workers
        stop.store(true, Ordering::Release);
        pause_ctrl.resume();

        for handle in worker_handles {
            let _ = handle.join();
        }

        prop_assert!(
            !deadlock_detected.load(Ordering::Acquire),
            "Deadlock detected: quiescence not achieved"
        );
    }
}

// ============================================================================
// Additional Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Concurrent inserts don't lose fingerprints
    #[test]
    fn prop_concurrent_fp_insert_no_loss(
        num_threads in 2..=4usize,
        fps_per_thread in 10..50usize,
    ) {
        let store = Arc::new(TestFingerprintStore::new());
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let s = Arc::clone(&store);
                let b = Arc::clone(&barrier);

                thread::spawn(move || {
                    b.wait();
                    let base = (thread_id * fps_per_thread) as u64;
                    for i in 0..fps_per_thread {
                        let fp = base + i as u64;
                        s.contains_or_insert(fp);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // All fingerprints should be present
        let total_expected = num_threads * fps_per_thread;
        let all_fps = store.all_fingerprints();

        prop_assert_eq!(
            all_fps.len(),
            total_expected,
            "Lost fingerprints: expected {}, got {}",
            total_expected,
            all_fps.len()
        );
    }

    /// Property: Pause requests are immediately visible to all workers
    #[test]
    fn prop_pause_visibility(
        num_workers in 2..=8usize,
    ) {
        let pause_ctrl = Arc::new(TestPauseController::new());
        let barrier = Arc::new(Barrier::new(num_workers + 1));
        let pause_seen = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let pc = Arc::clone(&pause_ctrl);
                let br = Arc::clone(&barrier);
                let ps = Arc::clone(&pause_seen);
                let st = Arc::clone(&stop);

                thread::spawn(move || {
                    br.wait();
                    // Spin until we see pause request (bounded)
                    let mut iters = 0;
                    while !pc.is_pause_requested() && !st.load(Ordering::Acquire) && iters < 10000 {
                        thread::yield_now();
                        iters += 1;
                    }
                    if pc.is_pause_requested() {
                        ps.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        barrier.wait();

        // Small delay then request pause
        thread::sleep(Duration::from_millis(1));
        pause_ctrl.request_pause();

        // Give workers time to see the pause
        thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Release);

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let seen = pause_seen.load(Ordering::Relaxed);
        prop_assert_eq!(
            seen,
            num_workers,
            "Not all workers saw pause: {} of {} saw it",
            seen,
            num_workers
        );
    }
}

// ============================================================================
// Stress Tests (Non-Property Based)
// ============================================================================

#[test]
fn stress_rapid_pause_resume() {
    let pause_ctrl = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let live_workers = Arc::new(AtomicUsize::new(4));
    let cycles_completed = Arc::new(AtomicUsize::new(0));

    // Spawn workers
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let pc = Arc::clone(&pause_ctrl);
            let st = Arc::clone(&stop);
            let lw = Arc::clone(&live_workers);

            thread::spawn(move || {
                while !st.load(Ordering::Acquire) {
                    pc.worker_pause_point(&st);
                    thread::yield_now();
                }
                lw.fetch_sub(1, Ordering::AcqRel);
            })
        })
        .collect();

    // Rapid pause/resume cycles
    for _ in 0..50 {
        pause_ctrl.request_pause();
        let quiesced = pause_ctrl.wait_for_quiescence(&stop, &live_workers, Duration::from_secs(1));
        assert!(quiesced, "Failed to quiesce");
        pause_ctrl.resume();
        cycles_completed.fetch_add(1, Ordering::Relaxed);
    }

    stop.store(true, Ordering::Release);
    pause_ctrl.resume();

    for handle in handles {
        handle.join().expect("Worker panicked");
    }

    assert_eq!(cycles_completed.load(Ordering::Relaxed), 50);
}

#[test]
fn stress_fp_switch_under_load() {
    let store = Arc::new(TestFingerprintStore::new());
    let stop = Arc::new(AtomicBool::new(false));
    let num_workers = 2;
    // workers + switcher
    let barrier = Arc::new(Barrier::new(num_workers + 1));

    // Spawn insert workers
    let worker_handles: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let s = Arc::clone(&store);
            let st = Arc::clone(&stop);
            let b = Arc::clone(&barrier);

            thread::spawn(move || {
                b.wait();
                let mut counter = 0u64;
                while !st.load(Ordering::Acquire) && counter < 100 {
                    let fp = (worker_id as u64) << 32 | counter;
                    s.contains_or_insert(fp);
                    counter += 1;
                }
                counter
            })
        })
        .collect();

    // Switcher thread
    let switch_handle = {
        let s = Arc::clone(&store);
        let b = Arc::clone(&barrier);

        thread::spawn(move || {
            b.wait();
            s.switch_to_hybrid();
        })
    };

    // Let workers run and complete
    thread::sleep(Duration::from_millis(50));
    stop.store(true, Ordering::Release);

    switch_handle.join().expect("Switcher panicked");
    let mut total_inserts = 0u64;
    for handle in worker_handles {
        total_inserts += handle.join().expect("Worker panicked");
    }

    assert!(store.is_hybrid(), "Switch didn't happen");
    assert!(total_inserts > 0, "No inserts happened");
    assert!(store.len() > 0, "Store is empty");
}
