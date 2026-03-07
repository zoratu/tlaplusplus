//! Chaos/fault injection tests for checkpoint quiescence
//!
//! These tests exercise error handling and recovery paths during checkpoint
//! operations, especially around the quiescence protocol.
//!
//! Run with: `cargo test --test checkpoint_chaos --features failpoints`
//!
//! For non-failpoint tests: `cargo test --test checkpoint_chaos`

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tlaplusplus::chaos::{
    self, QuiescenceResult, set_force_quiescence_timeout, set_fp_switch_delay_ms,
    set_worker_pause_delay_ms,
};

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
