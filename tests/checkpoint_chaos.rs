//! Chaos/fault injection tests for checkpoint quiescence
//!
//! These tests exercise error handling and recovery paths during checkpoint
//! operations, especially around the quiescence protocol.
//!
//! Run with: `cargo test --test checkpoint_chaos --features failpoints`
//!
//! For non-failpoint tests: `cargo test --test checkpoint_chaos`

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use tlaplusplus::chaos::{
    self, QuiescenceResult, set_force_quiescence_timeout, set_fp_switch_delay_ms,
    set_worker_pause_delay_ms,
};

fn chaos_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("chaos test lock poisoned")
}

// =============================================================================
// Failpoint Tests (require --features failpoints)
// =============================================================================

#[cfg(feature = "failpoints")]
mod checkpoint_failpoint_tests {
    use super::*;
    use tlaplusplus::{fail_point, fail_point_is_set};

    /// Test: checkpoint write fails, recovery works
    #[test]
    fn test_checkpoint_write_fail_recovery() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Enable checkpoint write failure
        fail::cfg("checkpoint_write_fail", "return").unwrap();

        // Simulate a checkpoint write operation that should fail
        let result: Result<(), anyhow::Error> = (|| {
            fail_point!("checkpoint_write_fail");
            Ok(())
        })();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("checkpoint_write_fail"));

        // Clear the failpoint
        fail::cfg("checkpoint_write_fail", "off").unwrap();

        // Now the operation should succeed
        let result: Result<(), anyhow::Error> = (|| {
            fail_point!("checkpoint_write_fail");
            Ok(())
        })();
        assert!(result.is_ok());

        scenario.teardown();
    }

    /// Test: checkpoint flush fails mid-drain
    #[test]
    fn test_checkpoint_flush_fail_mid_drain() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Enable checkpoint queue flush failure
        fail::cfg("checkpoint_queue_flush_fail", "return").unwrap();

        let result: Result<(), anyhow::Error> = (|| {
            fail_point!("checkpoint_queue_flush_fail");
            Ok(())
        })();

        assert!(result.is_err());

        // Verify failpoint message
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("checkpoint_queue_flush_fail"));

        scenario.teardown();
    }

    /// Test: fingerprint store operation fails during switch
    #[test]
    fn test_fp_store_fail_during_switch() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Enable FP store batch failure
        fail::cfg("fp_store_batch_fail", "return").unwrap();

        let result: Result<(), anyhow::Error> = (|| {
            fail_point!("fp_store_batch_fail");
            Ok(())
        })();

        assert!(result.is_err());

        // Clear and verify recovery
        fail::cfg("fp_store_batch_fail", "off").unwrap();

        let result: Result<(), anyhow::Error> = (|| {
            fail_point!("fp_store_batch_fail");
            Ok(())
        })();
        assert!(result.is_ok());

        scenario.teardown();
    }

    /// Test: quiescence timeout failpoint triggers
    #[test]
    fn test_quiescence_timeout_failpoint() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Enable quiescence timeout
        fail::cfg("quiescence_timeout", "return").unwrap();

        // Use the macro to check if it's set
        let should_timeout = fail_point_is_set!("quiescence_timeout");
        assert!(should_timeout);

        // Clear and verify
        fail::cfg("quiescence_timeout", "off").unwrap();
        let should_timeout = fail_point_is_set!("quiescence_timeout");
        assert!(!should_timeout);

        scenario.teardown();
    }

    /// Test: worker pause delay failpoint
    #[test]
    fn test_worker_pause_delay_failpoint() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Set worker pause delay to 100ms
        fail::cfg("worker_pause_delay", "return(100)").unwrap();

        let start = Instant::now();

        // Apply the delay using the macro
        tlaplusplus::apply_worker_pause_delay!();

        let elapsed = start.elapsed();
        // Should have delayed at least 90ms (allowing some slack)
        assert!(
            elapsed >= Duration::from_millis(90),
            "Expected delay of ~100ms, got {:?}",
            elapsed
        );

        scenario.teardown();
    }

    /// Test: FP switch slow failpoint
    #[test]
    fn test_fp_switch_slow_failpoint() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Set FP switch delay to 50ms
        fail::cfg("fp_switch_slow", "return(50)").unwrap();

        let start = Instant::now();

        // Apply the delay using the macro
        tlaplusplus::apply_fp_switch_delay!();

        let elapsed = start.elapsed();
        // Should have delayed at least 40ms (allowing some slack)
        assert!(
            elapsed >= Duration::from_millis(40),
            "Expected delay of ~50ms, got {:?}",
            elapsed
        );

        scenario.teardown();
    }

    /// Test: multiple failpoints active simultaneously
    #[test]
    fn test_multiple_failpoints_active() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // Enable multiple failpoints
        fail::cfg("checkpoint_write_fail", "return").unwrap();
        fail::cfg("fp_store_batch_fail", "return").unwrap();

        // Both should be active
        assert!(fail_point_is_set!("checkpoint_write_fail"));
        assert!(fail_point_is_set!("fp_store_batch_fail"));

        // Disable one
        fail::cfg("checkpoint_write_fail", "off").unwrap();

        // One should be off, other still on
        assert!(!fail_point_is_set!("checkpoint_write_fail"));
        assert!(fail_point_is_set!("fp_store_batch_fail"));

        scenario.teardown();
    }

    /// Test: failpoint with probability (for flaky failure simulation)
    #[test]
    fn test_probabilistic_failpoint() {
        let _test_guard = chaos_test_guard();
        let scenario = fail::FailScenario::setup();

        // 50% failure rate
        fail::cfg("io_fail", "50%return").unwrap();

        let mut failures = 0;
        let iterations = 100;

        for _ in 0..iterations {
            let result: Result<(), anyhow::Error> = (|| {
                fail_point!("io_fail");
                Ok(())
            })();
            if result.is_err() {
                failures += 1;
            }
        }

        // Should have roughly 50% failures (with some variance)
        // Accept 20-80% range to account for randomness
        assert!(
            failures > 20 && failures < 80,
            "Expected ~50% failures, got {}/{}",
            failures,
            iterations
        );

        scenario.teardown();
    }
}

// =============================================================================
// Concurrent Chaos Tests (no failpoints required)
// =============================================================================

/// Simulated PauseController for testing quiescence
struct TestPauseController {
    requested: AtomicBool,
    paused_workers: AtomicUsize,
}

impl TestPauseController {
    fn new() -> Self {
        Self {
            requested: AtomicBool::new(false),
            paused_workers: AtomicUsize::new(0),
        }
    }

    fn request_pause(&self) {
        self.requested.store(true, Ordering::Release);
    }

    fn resume(&self) {
        self.requested.store(false, Ordering::Release);
    }

    fn is_pause_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn worker_enter_pause(&self) {
        self.paused_workers.fetch_add(1, Ordering::AcqRel);
    }

    fn worker_exit_pause(&self) {
        self.paused_workers.fetch_sub(1, Ordering::AcqRel);
    }

    fn paused_count(&self) -> usize {
        self.paused_workers.load(Ordering::Acquire)
    }

    /// Wait for quiescence with timeout
    fn wait_for_quiescence_with_timeout(
        &self,
        expected_workers: usize,
        timeout: Duration,
        stop: &AtomicBool,
    ) -> QuiescenceResult {
        let start = Instant::now();

        // Check for forced timeout (chaos testing)
        if tlaplusplus::should_force_quiescence_timeout!() {
            return QuiescenceResult::Timeout {
                paused_workers: self.paused_count(),
                total_workers: expected_workers,
                elapsed_ms: 0,
                stuck_workers: (0..expected_workers).collect(),
            };
        }

        loop {
            if stop.load(Ordering::Acquire) {
                return QuiescenceResult::Interrupted;
            }

            let paused = self.paused_count();

            if paused >= expected_workers {
                return QuiescenceResult::Achieved {
                    paused_workers: paused,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                };
            }

            if start.elapsed() >= timeout {
                // Determine which workers are stuck
                let stuck: Vec<usize> = (0..expected_workers)
                    .filter(|_| {
                        // In real impl, would check per-worker pause state
                        // For test, just report how many are missing
                        true
                    })
                    .take(expected_workers - paused)
                    .collect();

                return QuiescenceResult::Timeout {
                    paused_workers: paused,
                    total_workers: expected_workers,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    stuck_workers: stuck,
                };
            }

            thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Test: pause with slow workers - verify quiescence still achieved
#[test]
fn test_pause_with_slow_workers() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));

    // Set up slow worker delay (chaos)
    set_worker_pause_delay_ms(50); // 50ms delay per worker

    // Spawn workers with varying delays
    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            // Worker loop
            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    // Apply artificial delay (some workers are slower)
                    let delay = worker_id as u64 * 20; // 0, 20, 40, 60ms
                    thread::sleep(Duration::from_millis(delay));

                    // Also apply chaos delay
                    chaos::apply_worker_pause_delay();

                    pause.worker_enter_pause();

                    // Wait for resume
                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Request pause
    pause.request_pause();

    // Wait for quiescence with generous timeout (workers are slow)
    let result = pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_secs(5), &stop);

    assert!(
        result.is_achieved(),
        "Quiescence should be achieved, got: {:?}",
        result
    );

    if let QuiescenceResult::Achieved {
        paused_workers,
        elapsed_ms,
    } = result
    {
        assert_eq!(paused_workers, num_workers);
        // Should take at least as long as the slowest worker
        assert!(
            elapsed_ms >= 50,
            "Expected at least 50ms, got {}ms",
            elapsed_ms
        );
    }

    // Resume and cleanup
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }

    // Reset chaos settings
    set_worker_pause_delay_ms(0);
}

/// Test: pause during FP switch - verify no deadlock
#[test]
fn test_pause_during_fp_switch() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));
    let fp_switch_in_progress = Arc::new(AtomicBool::new(false));
    let fp_switch_completed = Arc::new(AtomicBool::new(false));

    // Set FP switch delay
    set_fp_switch_delay_ms(100);

    // Spawn workers
    let mut handles = vec![];
    for _worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    pause.worker_enter_pause();

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Spawn FP switch thread
    let fp_switch_in_progress_clone = Arc::clone(&fp_switch_in_progress);
    let fp_switch_completed_clone = Arc::clone(&fp_switch_completed);
    let stop_clone = Arc::clone(&stop);

    let fp_switch_handle = thread::spawn(move || {
        // Give workers time to start
        thread::sleep(Duration::from_millis(50));

        // Start FP switch
        fp_switch_in_progress_clone.store(true, Ordering::Release);

        // Apply FP switch delay (simulates slow switch)
        chaos::apply_fp_switch_delay();

        fp_switch_completed_clone.store(true, Ordering::Release);
        fp_switch_in_progress_clone.store(false, Ordering::Release);

        // Keep running until stopped
        while !stop_clone.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(10));
        }
    });

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Wait for FP switch to start
    while !fp_switch_in_progress.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(5));
    }

    // Request pause WHILE FP switch is in progress
    pause.request_pause();

    // Wait for quiescence - should not deadlock
    let start = Instant::now();
    let timeout = Duration::from_secs(5);

    let result = pause.wait_for_quiescence_with_timeout(num_workers, timeout, &stop);

    // Should achieve quiescence without deadlock
    assert!(
        result.is_achieved(),
        "Quiescence should be achieved during FP switch, got: {:?}",
        result
    );

    // Verify we didn't hit timeout
    assert!(start.elapsed() < timeout, "Should not have timed out");

    // FP switch should have completed (after its delay)
    thread::sleep(Duration::from_millis(150)); // Wait for switch to complete
    assert!(
        fp_switch_completed.load(Ordering::Acquire),
        "FP switch should complete"
    );

    // Resume and cleanup
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
    fp_switch_handle.join().unwrap();

    // Reset chaos settings
    set_fp_switch_delay_ms(0);
}

/// Test: worker crash during checkpoint - verify doesn't hang forever
#[test]
fn test_worker_crash_during_checkpoint() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));
    let live_workers = Arc::new(AtomicUsize::new(num_workers));

    // Spawn workers - worker 0 will "crash" (exit early)
    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);
        let live_workers = Arc::clone(&live_workers);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            // Worker 0 crashes early (simulates panic during checkpoint)
            if worker_id == 0 {
                thread::sleep(Duration::from_millis(50));
                // Simulate crash - worker exits without entering pause
                live_workers.fetch_sub(1, Ordering::AcqRel);
                eprintln!("Worker {} crashed!", worker_id);
                return;
            }

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    pause.worker_enter_pause();

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }

            live_workers.fetch_sub(1, Ordering::AcqRel);
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Give worker 0 time to crash
    thread::sleep(Duration::from_millis(100));

    // Request pause after worker crash
    pause.request_pause();

    // Wait for quiescence - should succeed with reduced worker count
    let remaining_workers = live_workers.load(Ordering::Acquire);
    assert_eq!(
        remaining_workers,
        num_workers - 1,
        "One worker should have crashed"
    );

    let result = pause.wait_for_quiescence_with_timeout(
        remaining_workers, // Only expect remaining workers
        Duration::from_secs(2),
        &stop,
    );

    assert!(
        result.is_achieved(),
        "Quiescence should be achieved with remaining workers, got: {:?}",
        result
    );

    // Resume and cleanup
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

// =============================================================================
// Quiescence Timeout Recovery Tests
// =============================================================================

/// Test: quiescence timeout detection and diagnostic info
#[test]
fn test_quiescence_timeout_detection() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));

    // Spawn workers - but one will NEVER enter pause (stuck worker simulation)
    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    // Worker 3 is stuck - never enters pause
                    if worker_id == 3 {
                        // Spin without entering pause
                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(10));
                        }
                        continue;
                    }

                    pause.worker_enter_pause();

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Request pause
    pause.request_pause();

    // Wait for quiescence with short timeout
    let result = pause.wait_for_quiescence_with_timeout(
        num_workers,
        Duration::from_millis(500), // Short timeout
        &stop,
    );

    // Should timeout since worker 3 never pauses
    assert!(
        result.is_timeout(),
        "Should have timed out, got: {:?}",
        result
    );

    if let QuiescenceResult::Timeout {
        paused_workers,
        total_workers,
        elapsed_ms,
        stuck_workers,
    } = result
    {
        assert_eq!(paused_workers, 3, "Should have 3 paused workers");
        assert_eq!(total_workers, 4, "Should expect 4 total workers");
        assert!(elapsed_ms >= 500, "Should have elapsed at least 500ms");
        assert!(!stuck_workers.is_empty(), "Should report stuck workers");

        eprintln!("Timeout diagnostic info:");
        eprintln!("  Paused: {}/{}", paused_workers, total_workers);
        eprintln!("  Elapsed: {}ms", elapsed_ms);
        eprintln!("  Stuck workers: {:?}", stuck_workers);
    }

    // Resume and cleanup
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

/// Test: forced quiescence timeout (chaos testing)
#[test]
fn test_forced_quiescence_timeout() {
    let _test_guard = chaos_test_guard();
    // Enable forced timeout
    set_force_quiescence_timeout(true);

    let pause = TestPauseController::new();
    let stop = AtomicBool::new(false);

    // Request pause
    pause.request_pause();

    // Should immediately timeout due to chaos setting
    let result = pause.wait_for_quiescence_with_timeout(4, Duration::from_secs(10), &stop);

    assert!(
        result.is_timeout(),
        "Should have timed out due to chaos setting, got: {:?}",
        result
    );

    // Reset chaos setting
    set_force_quiescence_timeout(false);
}

/// Test: graceful recovery after timeout (retry or skip)
#[test]
fn test_quiescence_timeout_recovery() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));
    let stuck_worker_should_unstick = Arc::new(AtomicBool::new(false));

    // Spawn workers - worker 2 is initially stuck but will unstick on retry
    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);
        let should_unstick = Arc::clone(&stuck_worker_should_unstick);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    // Worker 2 is stuck until told to unstick
                    if worker_id == 2 && !should_unstick.load(Ordering::Acquire) {
                        while pause.is_pause_requested()
                            && !stop.load(Ordering::Acquire)
                            && !should_unstick.load(Ordering::Acquire)
                        {
                            thread::sleep(Duration::from_millis(10));
                        }
                        // If we're told to unstick, fall through to normal pause
                        if !pause.is_pause_requested() {
                            continue;
                        }
                    }

                    pause.worker_enter_pause();

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // First attempt - should timeout
    pause.request_pause();
    let result1 =
        pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_millis(200), &stop);
    assert!(result1.is_timeout(), "First attempt should timeout");

    // Resume and unstick the stuck worker
    pause.resume();
    stuck_worker_should_unstick.store(true, Ordering::Release);
    thread::sleep(Duration::from_millis(50));

    // Retry - should succeed now
    pause.request_pause();
    let result2 =
        pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_secs(2), &stop);
    assert!(
        result2.is_achieved(),
        "Second attempt should succeed after unsticking, got: {:?}",
        result2
    );

    // Cleanup
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

/// Test: checkpoint skip on persistent timeout
#[test]
fn test_checkpoint_skip_on_persistent_timeout() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));
    let checkpoint_skipped = Arc::new(AtomicBool::new(false));

    // Spawn workers - worker 1 is permanently stuck
    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    // Worker 1 is permanently stuck
                    if worker_id == 1 {
                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(10));
                        }
                        continue;
                    }

                    pause.worker_enter_pause();

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Try multiple times (simulating checkpoint retry logic)
    let max_retries = 3;
    let mut attempt = 0;

    loop {
        attempt += 1;
        pause.request_pause();

        let result =
            pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_millis(100), &stop);

        pause.resume();

        match result {
            QuiescenceResult::Achieved { .. } => {
                // Checkpoint would proceed normally
                break;
            }
            QuiescenceResult::Timeout {
                paused_workers,
                total_workers,
                ..
            } => {
                eprintln!(
                    "Checkpoint attempt {} failed: {}/{} workers paused",
                    attempt, paused_workers, total_workers
                );

                if attempt >= max_retries {
                    // Skip checkpoint after max retries
                    checkpoint_skipped.store(true, Ordering::Release);
                    eprintln!("Skipping checkpoint after {} failed attempts", max_retries);
                    break;
                }

                // Brief pause before retry
                thread::sleep(Duration::from_millis(50));
            }
            QuiescenceResult::Interrupted => {
                break;
            }
        }
    }

    // Verify checkpoint was skipped
    assert!(
        checkpoint_skipped.load(Ordering::Acquire),
        "Checkpoint should have been skipped after retries"
    );

    // Cleanup
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

// =============================================================================
// Integration Tests (combines multiple chaos features)
// =============================================================================

/// Test: chaos combination - slow workers + FP switch + checkpoint
#[test]
fn test_chaos_combination() {
    let _test_guard = chaos_test_guard();
    // Enable multiple chaos settings
    set_worker_pause_delay_ms(20);
    set_fp_switch_delay_ms(30);

    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));
    let checkpoints_completed = Arc::new(AtomicUsize::new(0));

    // Spawn workers
    let mut handles = vec![];
    for _worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    // Apply worker pause delay
                    chaos::apply_worker_pause_delay();

                    pause.worker_enter_pause();

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }

                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Perform multiple checkpoint cycles with chaos enabled
    for checkpoint_num in 0..3 {
        // Simulate FP switch delay (would happen during checkpoint sometimes)
        if checkpoint_num == 1 {
            chaos::apply_fp_switch_delay();
        }

        pause.request_pause();

        let result =
            pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_secs(5), &stop);

        assert!(
            result.is_achieved(),
            "Checkpoint {} should succeed with chaos, got: {:?}",
            checkpoint_num,
            result
        );

        checkpoints_completed.fetch_add(1, Ordering::SeqCst);

        pause.resume();

        // Brief pause between checkpoints
        thread::sleep(Duration::from_millis(50));
    }

    // Verify all checkpoints completed
    assert_eq!(checkpoints_completed.load(Ordering::Acquire), 3);

    // Cleanup
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }

    // Reset chaos settings
    set_worker_pause_delay_ms(0);
    set_fp_switch_delay_ms(0);
}

/// Test: I/O latency during checkpoint
#[test]
fn test_io_latency_during_checkpoint() {
    let _test_guard = chaos_test_guard();
    // Enable I/O latency
    chaos::set_io_latency_us(10_000); // 10ms

    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));

    // Single worker for simplicity
    let pause_clone = Arc::clone(&pause);
    let stop_clone = Arc::clone(&stop);

    let worker = thread::spawn(move || {
        loop {
            if stop_clone.load(Ordering::Acquire) {
                break;
            }

            if pause_clone.is_pause_requested() {
                pause_clone.worker_enter_pause();

                while pause_clone.is_pause_requested() && !stop_clone.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(5));
                }

                pause_clone.worker_exit_pause();
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    thread::sleep(Duration::from_millis(50)); // Let worker start

    // Checkpoint with I/O latency
    pause.request_pause();

    let result = pause.wait_for_quiescence_with_timeout(1, Duration::from_secs(2), &stop);

    assert!(result.is_achieved());

    // Simulate I/O operations during checkpoint
    let start = Instant::now();
    for _ in 0..5 {
        chaos::apply_io_latency();
    }
    let io_time = start.elapsed();

    // Should have taken at least 50ms (5 * 10ms)
    assert!(
        io_time >= Duration::from_millis(45),
        "Expected ~50ms I/O time, got {:?}",
        io_time
    );

    pause.resume();
    stop.store(true, Ordering::Release);
    worker.join().unwrap();

    // Reset
    chaos::set_io_latency_us(0);
}

// =============================================================================
// Additional Chaos Tests for Quiescence Timeout Improvements
// =============================================================================

/// Test: exponential backoff behavior during checkpoint timeout
#[test]
fn test_exponential_backoff_checkpoint() {
    let _test_guard = chaos_test_guard();
    let num_workers = 4;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));

    // Worker 0 will be stuck, triggering timeout and backoff
    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    // Worker 0 is stuck - never enters pause
                    if worker_id == 0 {
                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(10));
                        }
                        continue;
                    }

                    pause.worker_enter_pause();
                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }
                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Test exponential backoff with short timeouts
    let initial_timeout_ms = 50;
    let max_retries = 3;

    let mut timeouts = Vec::new();
    let mut current_timeout = initial_timeout_ms;
    let overall_start = Instant::now();

    for attempt in 0..max_retries {
        pause.request_pause();

        let result = pause.wait_for_quiescence_with_timeout(
            num_workers,
            Duration::from_millis(current_timeout),
            &stop,
        );

        // Should timeout because worker 0 is stuck
        assert!(
            result.is_timeout(),
            "Attempt {}: Expected timeout, got {:?}",
            attempt,
            result
        );

        timeouts.push(current_timeout);
        pause.resume();

        // Exponential backoff
        current_timeout *= 2;
        thread::sleep(Duration::from_millis(10)); // Brief pause between retries
    }

    let total_time = overall_start.elapsed();

    // Verify exponential backoff sequence
    assert_eq!(timeouts, vec![50, 100, 200]);

    // Total time should be at least sum of timeouts
    let expected_min_time = Duration::from_millis(50 + 100 + 200);
    assert!(
        total_time >= expected_min_time,
        "Expected at least {:?}, got {:?}",
        expected_min_time,
        total_time
    );

    stop.store(true, Ordering::Release);
    for h in handles {
        h.join().unwrap();
    }
}

/// Test: try-read mechanism bails out after 20 attempts
#[test]
fn test_try_read_20_attempt_bailout() {
    let _test_guard = chaos_test_guard();
    use parking_lot::RwLock;
    use std::collections::HashSet;

    struct TestStore {
        data: RwLock<HashSet<u64>>,
        switch_pending: AtomicBool,
        attempt_count: AtomicUsize,
    }

    impl TestStore {
        fn new() -> Self {
            Self {
                data: RwLock::new(HashSet::new()),
                switch_pending: AtomicBool::new(false),
                attempt_count: AtomicUsize::new(0),
            }
        }

        fn contains_or_insert_with_tracking(&self, fp: u64) -> (bool, usize) {
            if self.switch_pending.load(Ordering::Acquire) {
                let mut attempts = 0;
                loop {
                    if let Some(mut guard) = self.data.try_write() {
                        let existed = guard.contains(&fp);
                        if !existed {
                            guard.insert(fp);
                        }
                        return (existed, attempts);
                    }
                    attempts += 1;
                    self.attempt_count.fetch_add(1, Ordering::Relaxed);

                    if attempts > 20 {
                        // Bail out after 20 attempts (~2ms at 100us/attempt)
                        return (true, attempts); // Return "exists" to proceed quickly
                    }
                    std::thread::sleep(Duration::from_micros(100));
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
            (existed, 0)
        }
    }

    let store = Arc::new(TestStore::new());

    // Hold write lock to force try_write failures
    store.switch_pending.store(true, Ordering::Release);
    let guard = store.data.write();

    // Spawn worker that will hit the 20-attempt limit
    let store_clone = Arc::clone(&store);
    let handle = thread::spawn(move || {
        let start = Instant::now();
        let (result, attempts) = store_clone.contains_or_insert_with_tracking(12345);
        let elapsed = start.elapsed();
        (result, attempts, elapsed)
    });

    // Wait for worker to bail out
    let (result, attempts, elapsed) = handle.join().unwrap();
    drop(guard);

    // Verify behavior
    assert!(result, "Should return true (exists) when bailing out");
    assert!(
        attempts > 20,
        "Should have exceeded 20 attempts, got {}",
        attempts
    );
    assert!(
        elapsed >= Duration::from_millis(2),
        "Should take at least 2ms (20 * 100us), got {:?}",
        elapsed
    );
    assert!(
        elapsed < Duration::from_millis(10),
        "Should complete quickly after bailout, got {:?}",
        elapsed
    );
}

/// Test: diagnostic logging threshold at 10 attempts
#[test]
fn test_diagnostic_logging_at_10_attempts() {
    let _test_guard = chaos_test_guard();
    use parking_lot::RwLock;
    use std::collections::HashSet;

    struct TestStore {
        data: RwLock<HashSet<u64>>,
        switch_pending: AtomicBool,
        exceeded_10_count: AtomicUsize,
        exceeded_20_count: AtomicUsize,
    }

    impl TestStore {
        fn new() -> Self {
            Self {
                data: RwLock::new(HashSet::new()),
                switch_pending: AtomicBool::new(false),
                exceeded_10_count: AtomicUsize::new(0),
                exceeded_20_count: AtomicUsize::new(0),
            }
        }

        fn contains_or_insert(&self, fp: u64) -> bool {
            if self.switch_pending.load(Ordering::Acquire) {
                let mut attempts = 0;
                loop {
                    if let Some(mut guard) = self.data.try_write() {
                        let existed = guard.contains(&fp);
                        if !existed {
                            guard.insert(fp);
                        }
                        return existed;
                    }
                    attempts += 1;

                    if attempts > 10 {
                        self.exceeded_10_count.fetch_add(1, Ordering::Relaxed);
                    }
                    if attempts > 20 {
                        self.exceeded_20_count.fetch_add(1, Ordering::Relaxed);
                        return true; // Bail out
                    }
                    std::thread::sleep(Duration::from_micros(100));
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
    }

    let store = Arc::new(TestStore::new());

    // Test 1: Short block (< 1ms) - should not exceed 10 attempts
    {
        store.switch_pending.store(true, Ordering::Release);
        let store_clone = Arc::clone(&store);

        let handle = thread::spawn(move || store_clone.contains_or_insert(1));

        // Release after 0.5ms
        thread::sleep(Duration::from_micros(500));
        store.switch_pending.store(false, Ordering::Release);

        handle.join().unwrap();
        // May or may not exceed 10 depending on timing
    }

    // Test 2: Long block (> 2ms) - should exceed 10 and trigger diagnostic
    let store2 = Arc::new(TestStore::new());
    {
        store2.switch_pending.store(true, Ordering::Release);
        let _guard = store2.data.write(); // Hold lock

        let store_clone = Arc::clone(&store2);
        let handle = thread::spawn(move || store_clone.contains_or_insert(2));

        // Wait for worker to bail out (will take ~2ms)
        thread::sleep(Duration::from_millis(5));

        handle.join().unwrap();
        drop(_guard);

        let exceeded_10 = store2.exceeded_10_count.load(Ordering::Relaxed);
        let exceeded_20 = store2.exceeded_20_count.load(Ordering::Relaxed);

        assert!(
            exceeded_10 > 0,
            "Should have exceeded 10 attempts when blocked"
        );
        assert!(
            exceeded_20 > 0,
            "Should have exceeded 20 attempts when blocked"
        );
    }
}

/// Test: NUMA-like latency causing uneven pause times
#[test]
fn test_numa_latency_uneven_pause() {
    let _test_guard = chaos_test_guard();
    let num_workers = 8;
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));
    let pause_times = Arc::new(parking_lot::Mutex::new(vec![Duration::ZERO; num_workers]));

    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);
        let pause_times = Arc::clone(&pause_times);

        // Simulate NUMA: workers 0-3 are fast, 4-7 are slow
        let is_slow = worker_id >= 4;

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    let start = Instant::now();

                    // Apply NUMA-like latency
                    if is_slow {
                        thread::sleep(Duration::from_millis(20 + (worker_id as u64 - 4) * 5));
                    }

                    pause.worker_enter_pause();

                    let elapsed = start.elapsed();
                    pause_times.lock()[worker_id] = elapsed;

                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }
                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    // Wait for workers to start
    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    // Request pause and wait
    pause.request_pause();

    // Wait for quiescence (should succeed despite NUMA latency)
    let result = pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_secs(2), &stop);

    assert!(
        result.is_achieved(),
        "Should achieve quiescence despite NUMA latency"
    );

    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }

    // Verify that slow workers took longer
    let times = pause_times.lock();
    let fast_max = times[0..4].iter().max().unwrap();
    let slow_min = times[4..8].iter().filter(|t| **t > Duration::ZERO).min();

    if let Some(slow_min) = slow_min {
        assert!(
            slow_min >= fast_max,
            "Slow workers should take longer: fast_max={:?}, slow_min={:?}",
            fast_max,
            slow_min
        );
    }
}

/// Test: multiple stuck workers with detailed reporting
#[test]
fn test_multiple_stuck_workers_reporting() {
    let _test_guard = chaos_test_guard();
    let num_workers = 6;
    let stuck_workers = vec![1, 3, 5]; // Workers 1, 3, 5 are stuck
    let pause = Arc::new(TestPauseController::new());
    let stop = Arc::new(AtomicBool::new(false));
    let workers_started = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    for worker_id in 0..num_workers {
        let pause = Arc::clone(&pause);
        let stop = Arc::clone(&stop);
        let workers_started = Arc::clone(&workers_started);
        let is_stuck = stuck_workers.contains(&worker_id);

        handles.push(thread::spawn(move || {
            workers_started.fetch_add(1, Ordering::SeqCst);

            while !stop.load(Ordering::Acquire) {
                if pause.is_pause_requested() {
                    if is_stuck {
                        // Stuck workers never enter pause
                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(10));
                        }
                        continue;
                    }

                    pause.worker_enter_pause();
                    while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }
                    pause.worker_exit_pause();
                }
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }

    while workers_started.load(Ordering::Acquire) < num_workers {
        thread::sleep(Duration::from_millis(10));
    }

    pause.request_pause();

    // Wait with short timeout
    let result =
        pause.wait_for_quiescence_with_timeout(num_workers, Duration::from_millis(200), &stop);

    // Should timeout
    assert!(result.is_timeout(), "Should timeout with stuck workers");

    if let QuiescenceResult::Timeout {
        paused_workers,
        total_workers,
        stuck_workers: reported_stuck,
        ..
    } = result
    {
        // 3 workers should be paused (0, 2, 4) - the non-stuck ones
        assert_eq!(paused_workers, 3, "3 cooperative workers should be paused");
        assert_eq!(total_workers, num_workers);

        // Stuck workers should be reported - the count should match
        // Note: The mock controller doesn't track per-worker pause status,
        // so it reports workers 0..N where N is the number of stuck workers.
        // We verify the count is correct rather than the specific IDs.
        assert_eq!(
            reported_stuck.len(),
            stuck_workers.len(),
            "Should report {} stuck workers, got {:?}",
            stuck_workers.len(),
            reported_stuck
        );
    }

    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

/// Test: chaos injection via global flags
#[test]
fn test_chaos_global_flags_integration() {
    let _test_guard = chaos_test_guard();
    // Test interaction of multiple chaos flags

    // 1. Set worker pause delay
    chaos::set_worker_pause_delay_ms(10);
    assert_eq!(chaos::get_worker_pause_delay_ms(), 10);

    // 2. Set FP switch delay
    chaos::set_fp_switch_delay_ms(20);
    assert_eq!(chaos::get_fp_switch_delay_ms(), 20);

    // 3. Set forced quiescence timeout
    chaos::set_force_quiescence_timeout(true);
    assert!(chaos::should_force_quiescence_timeout());

    // 4. Apply delays and verify timing
    let start = Instant::now();
    chaos::apply_worker_pause_delay();
    let worker_delay_time = start.elapsed();

    let start = Instant::now();
    chaos::apply_fp_switch_delay();
    let fp_delay_time = start.elapsed();

    assert!(
        worker_delay_time >= Duration::from_millis(8),
        "Worker delay should be ~10ms, got {:?}",
        worker_delay_time
    );
    assert!(
        fp_delay_time >= Duration::from_millis(18),
        "FP delay should be ~20ms, got {:?}",
        fp_delay_time
    );

    // Reset all chaos flags
    chaos::set_worker_pause_delay_ms(0);
    chaos::set_fp_switch_delay_ms(0);
    chaos::set_force_quiescence_timeout(false);

    assert_eq!(chaos::get_worker_pause_delay_ms(), 0);
    assert_eq!(chaos::get_fp_switch_delay_ms(), 0);
    assert!(!chaos::should_force_quiescence_timeout());
}
