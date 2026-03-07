//! Chaos testing / fault injection support
//!
//! This module provides fault injection capabilities for testing error handling
//! and recovery paths. Failpoints are only active when compiled with the
//! `failpoints` feature:
//!
//! ```bash
//! cargo test --features failpoints
//! ```
//!
//! ## Available Failpoints
//!
//! ### Checkpoint Operations
//! - `checkpoint_write_fail` - Fail checkpoint manifest write
//! - `checkpoint_flush_fail` - Fail flushing data before checkpoint
//!
//! ### Fingerprint Store
//! - `fp_store_insert_fail` - Fail fingerprint insertion
//! - `fp_store_batch_fail` - Fail batch fingerprint check
//! - `fp_store_shard_full` - Simulate shard capacity exceeded
//!
//! ### Worker Operations
//! - `worker_panic` - Panic in worker thread (value = worker_id)
//! - `worker_slow` - Add artificial delay to worker (value = ms)
//!
//! ### Queue Operations
//! - `queue_spill_fail` - Fail writing spill segment
//! - `queue_load_fail` - Fail loading spill segment
//!
//! ### I/O Operations
//! - `io_slow` - Add artificial I/O latency (value = ms)
//! - `io_fail` - Fail I/O operation
//!
//! ### Quiescence Operations
//! - `quiescence_timeout` - Force timeout in wait_for_quiescence
//! - `fp_switch_slow` - Add delay during switch_to_hybrid (value = ms)
//! - `worker_pause_delay` - Delay worker entering pause (value = ms)
//!
//! ## Usage in Tests
//!
//! ```rust,ignore
//! use fail::FailScenario;
//!
//! #[test]
//! fn test_checkpoint_recovery() {
//!     let scenario = FailScenario::setup();
//!     fail::cfg("checkpoint_write_fail", "return").unwrap();
//!
//!     // Run code that should handle checkpoint failure
//!     let result = checkpoint_write(...);
//!     assert!(result.is_err());
//!
//!     scenario.teardown();
//! }
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Global flag to enable chaos mode (for non-failpoint chaos)
pub static CHAOS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Flag to request emergency checkpoint before failure
pub static EMERGENCY_CHECKPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Flag indicating emergency checkpoint has completed
pub static EMERGENCY_CHECKPOINT_COMPLETE: AtomicBool = AtomicBool::new(false);

/// Worker ID to crash (usize::MAX = none)
pub static CHAOS_CRASH_WORKER: AtomicU64 = AtomicU64::new(u64::MAX);

/// Artificial I/O latency in microseconds
pub static CHAOS_IO_LATENCY_US: AtomicU64 = AtomicU64::new(0);

/// Memory allocation failure probability (0-100)
pub static CHAOS_ALLOC_FAIL_PCT: AtomicU64 = AtomicU64::new(0);

/// Enable chaos testing mode
pub fn enable_chaos() {
    CHAOS_ENABLED.store(true, Ordering::Release);
}

/// Disable chaos testing mode
pub fn disable_chaos() {
    CHAOS_ENABLED.store(false, Ordering::Release);
}

/// Check if chaos mode is enabled
pub fn is_chaos_enabled() -> bool {
    CHAOS_ENABLED.load(Ordering::Acquire)
}

/// Request an emergency checkpoint before the system fails
/// Call this when detecting an impending failure to preserve state
pub fn request_emergency_checkpoint() {
    if !EMERGENCY_CHECKPOINT_REQUESTED.swap(true, Ordering::AcqRel) {
        eprintln!("Emergency checkpoint requested - attempting to preserve state before failure");
    }
}

/// Check if emergency checkpoint was requested
pub fn is_emergency_checkpoint_requested() -> bool {
    EMERGENCY_CHECKPOINT_REQUESTED.load(Ordering::Acquire)
}

/// Clear emergency checkpoint flag (after checkpoint completes)
pub fn clear_emergency_checkpoint() {
    EMERGENCY_CHECKPOINT_REQUESTED.store(false, Ordering::Release);
}

/// Signal that emergency checkpoint has completed
pub fn set_emergency_checkpoint_complete() {
    EMERGENCY_CHECKPOINT_COMPLETE.store(true, Ordering::Release);
}

/// Check if emergency checkpoint has completed
pub fn is_emergency_checkpoint_complete() -> bool {
    EMERGENCY_CHECKPOINT_COMPLETE.load(Ordering::Acquire)
}

/// Wait for emergency checkpoint to complete (with timeout)
/// Returns true if completed, false if timed out
pub fn wait_for_emergency_checkpoint(timeout_secs: u64) -> bool {
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(timeout_secs);

    while !is_emergency_checkpoint_complete() {
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    true
}

/// Set which worker should crash (u64::MAX = none)
pub fn set_crash_worker(worker_id: u64) {
    CHAOS_CRASH_WORKER.store(worker_id, Ordering::Release);
}

/// Check if this worker should crash
pub fn should_crash_worker(worker_id: usize) -> bool {
    CHAOS_CRASH_WORKER.load(Ordering::Acquire) == worker_id as u64
}

/// Set artificial I/O latency
pub fn set_io_latency_us(latency_us: u64) {
    CHAOS_IO_LATENCY_US.store(latency_us, Ordering::Release);
}

/// Get current I/O latency setting
pub fn get_io_latency_us() -> u64 {
    CHAOS_IO_LATENCY_US.load(Ordering::Acquire)
}

/// Apply I/O latency if configured
pub fn apply_io_latency() {
    let latency = get_io_latency_us();
    if latency > 0 {
        std::thread::sleep(std::time::Duration::from_micros(latency));
    }
}

/// Set allocation failure probability (0-100)
pub fn set_alloc_fail_probability(pct: u64) {
    CHAOS_ALLOC_FAIL_PCT.store(pct.min(100), Ordering::Release);
}

/// Check if allocation should fail (based on probability)
pub fn should_fail_alloc() -> bool {
    let pct = CHAOS_ALLOC_FAIL_PCT.load(Ordering::Acquire);
    if pct == 0 {
        return false;
    }
    // Simple pseudo-random based on TSC
    let tsc = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    (tsc % 100) < pct
}

/// Macro to insert a fail point that returns an error
#[macro_export]
macro_rules! fail_point {
    ($name:expr) => {{
        #[cfg(feature = "failpoints")]
        {
            if let Some(_) = ::fail::eval($name, |_| {}) {
                return Err(::anyhow::anyhow!(concat!("failpoint: ", $name)));
            }
        }
    }};
    ($name:expr, $e:expr) => {{
        #[cfg(feature = "failpoints")]
        {
            if let Some(v) = ::fail::eval($name, |v| v) {
                return Err($e(v));
            }
        }
    }};
}

/// Macro to check if a fail point is enabled (for conditional logic)
#[macro_export]
macro_rules! fail_point_is_set {
    ($name:expr) => {{
        #[cfg(feature = "failpoints")]
        {
            ::fail::eval($name, |_| true).is_some()
        }
        #[cfg(not(feature = "failpoints"))]
        {
            false
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chaos_flags_work() {
        assert!(!is_chaos_enabled());
        enable_chaos();
        assert!(is_chaos_enabled());
        disable_chaos();
        assert!(!is_chaos_enabled());
    }

    #[test]
    fn crash_worker_tracking() {
        assert!(!should_crash_worker(0));
        assert!(!should_crash_worker(5));
        set_crash_worker(5);
        assert!(!should_crash_worker(0));
        assert!(should_crash_worker(5));
        set_crash_worker(u64::MAX);
        assert!(!should_crash_worker(5));
    }

    #[test]
    fn io_latency_setting() {
        assert_eq!(get_io_latency_us(), 0);
        set_io_latency_us(1000);
        assert_eq!(get_io_latency_us(), 1000);
        set_io_latency_us(0);
    }

    #[test]
    fn alloc_fail_probability() {
        set_alloc_fail_probability(0);
        assert!(!should_fail_alloc());

        set_alloc_fail_probability(100);
        // Should always fail at 100%
        assert!(should_fail_alloc());

        set_alloc_fail_probability(0);
    }
}

// =============================================================================
// Quiescence Chaos Testing Support
// =============================================================================

/// Worker pause delay in milliseconds (0 = no delay)
pub static CHAOS_WORKER_PAUSE_DELAY_MS: AtomicU64 = AtomicU64::new(0);

/// FP switch delay in milliseconds (0 = no delay)
pub static CHAOS_FP_SWITCH_DELAY_MS: AtomicU64 = AtomicU64::new(0);

/// Force quiescence timeout (for testing timeout recovery)
pub static CHAOS_FORCE_QUIESCENCE_TIMEOUT: AtomicBool = AtomicBool::new(false);

/// Set worker pause delay (simulates slow workers)
pub fn set_worker_pause_delay_ms(delay_ms: u64) {
    CHAOS_WORKER_PAUSE_DELAY_MS.store(delay_ms, Ordering::Release);
}

/// Get worker pause delay
pub fn get_worker_pause_delay_ms() -> u64 {
    CHAOS_WORKER_PAUSE_DELAY_MS.load(Ordering::Acquire)
}

/// Apply worker pause delay if configured
pub fn apply_worker_pause_delay() {
    let delay = get_worker_pause_delay_ms();
    if delay > 0 {
        std::thread::sleep(std::time::Duration::from_millis(delay));
    }
}

/// Set FP switch delay (simulates slow hybrid switch)
pub fn set_fp_switch_delay_ms(delay_ms: u64) {
    CHAOS_FP_SWITCH_DELAY_MS.store(delay_ms, Ordering::Release);
}

/// Get FP switch delay
pub fn get_fp_switch_delay_ms() -> u64 {
    CHAOS_FP_SWITCH_DELAY_MS.load(Ordering::Acquire)
}

/// Apply FP switch delay if configured
pub fn apply_fp_switch_delay() {
    let delay = get_fp_switch_delay_ms();
    if delay > 0 {
        std::thread::sleep(std::time::Duration::from_millis(delay));
    }
}

/// Force quiescence timeout (for testing timeout recovery)
pub fn set_force_quiescence_timeout(force: bool) {
    CHAOS_FORCE_QUIESCENCE_TIMEOUT.store(force, Ordering::Release);
}

/// Check if quiescence should force timeout
pub fn should_force_quiescence_timeout() -> bool {
    CHAOS_FORCE_QUIESCENCE_TIMEOUT.load(Ordering::Acquire)
}

/// Macro to apply worker pause delay with failpoint support
#[macro_export]
macro_rules! apply_worker_pause_delay {
    () => {{
        // Check failpoint first (if feature enabled)
        #[cfg(feature = "failpoints")]
        {
            // fail::eval returns Option<T> where T is the return type of the closure
            // The closure receives the failpoint value as Option<String>
            let delay_applied = ::fail::eval("worker_pause_delay", |v: Option<String>| {
                if let Some(val) = v {
                    if let Ok(ms) = val.parse::<u64>() {
                        std::thread::sleep(std::time::Duration::from_millis(ms));
                        return true;
                    }
                }
                false
            });
            // If failpoint didn't apply delay, apply chaos delay
            if !delay_applied.unwrap_or(false) {
                $crate::chaos::apply_worker_pause_delay();
            }
        }
        #[cfg(not(feature = "failpoints"))]
        {
            $crate::chaos::apply_worker_pause_delay();
        }
    }};
}

/// Macro to apply FP switch delay with failpoint support
#[macro_export]
macro_rules! apply_fp_switch_delay {
    () => {{
        // Check failpoint first (if feature enabled)
        #[cfg(feature = "failpoints")]
        {
            let delay_applied = ::fail::eval("fp_switch_slow", |v: Option<String>| {
                if let Some(val) = v {
                    if let Ok(ms) = val.parse::<u64>() {
                        std::thread::sleep(std::time::Duration::from_millis(ms));
                        return true;
                    }
                }
                false
            });
            // If failpoint didn't apply delay, apply chaos delay
            if !delay_applied.unwrap_or(false) {
                $crate::chaos::apply_fp_switch_delay();
            }
        }
        #[cfg(not(feature = "failpoints"))]
        {
            $crate::chaos::apply_fp_switch_delay();
        }
    }};
}

/// Macro to check for quiescence timeout override
#[macro_export]
macro_rules! should_force_quiescence_timeout {
    () => {{
        #[cfg(feature = "failpoints")]
        {
            if ::fail::eval("quiescence_timeout", |_| true).is_some() {
                true
            } else {
                $crate::chaos::should_force_quiescence_timeout()
            }
        }
        #[cfg(not(feature = "failpoints"))]
        {
            $crate::chaos::should_force_quiescence_timeout()
        }
    }};
}

/// Result of quiescence wait operation
#[derive(Debug, Clone)]
pub enum QuiescenceResult {
    /// All workers successfully paused
    Achieved {
        paused_workers: usize,
        elapsed_ms: u64,
    },
    /// Timeout waiting for workers to pause
    Timeout {
        paused_workers: usize,
        total_workers: usize,
        elapsed_ms: u64,
        stuck_workers: Vec<usize>,
    },
    /// Interrupted by stop signal
    Interrupted,
}

impl QuiescenceResult {
    /// Check if quiescence was achieved
    pub fn is_achieved(&self) -> bool {
        matches!(self, QuiescenceResult::Achieved { .. })
    }

    /// Check if operation timed out
    pub fn is_timeout(&self) -> bool {
        matches!(self, QuiescenceResult::Timeout { .. })
    }
}

/// Retry an operation with exponential backoff
/// Returns Ok on success, or the last error after all retries exhausted
pub fn retry_with_backoff<T, E, F>(
    mut op: F,
    max_retries: usize,
    initial_delay_ms: u64,
    max_delay_ms: u64,
) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    let mut delay_ms = initial_delay_ms;
    let mut last_error = None;

    for attempt in 0..=max_retries {
        match op() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt < max_retries {
                    eprintln!(
                        "Operation failed (attempt {}/{}): {}. Retrying in {}ms...",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay_ms
                    );
                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                    delay_ms = (delay_ms * 2).min(max_delay_ms);
                }
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap())
}

#[cfg(all(test, feature = "failpoints"))]
mod failpoint_tests {
    use super::*;

    #[test]
    fn failpoint_integration() {
        let scenario = fail::FailScenario::setup();

        // Test that failpoint can be set and triggers
        fail::cfg("test_failpoint", "return").unwrap();

        let result: Result<(), anyhow::Error> = (|| {
            fail_point!("test_failpoint");
            Ok(())
        })();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test_failpoint"));

        scenario.teardown();
    }
}
