//! Progress reporting + memory-pressure throttling thread.
//!
//! Spawns a single supervisor thread that:
//! 1. Polls memory once per second; throttles `WorkerThrottle` on
//!    Warning/Critical, restores on Ok.
//! 2. Polls the chaos emergency-checkpoint flag and flushes the
//!    fingerprint store + emits a status banner when set.
//! 3. Every 10s, computes generated/distinct rates per minute and
//!    pending queue size and prints the TLC-compatible Progress line
//!    with an ETA (seconds / minutes / hours / days).

use crate::autotune::WorkerThrottle;
use crate::storage::spillable_work_stealing::SpillableWorkStealingQueues;
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use crate::system::{MemoryMonitor, MemoryStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::AtomicRunStats;

/// Format a u64 with TLC-style comma group separators (1,234,567).
///
/// Pure helper, extracted from the progress-thread closure so it can be
/// covered by unit tests. Used to format every numeric column in the
/// `Progress(...)` line.
pub(super) fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Compute the ETA suffix shown on a Progress(...) line, given the
/// queue drain rate in states/sec.
///
/// - drain_rate > 10:   queue is draining; ETA in seconds/minutes/hours/days
///   based on `queue_pending / drain_rate`.
/// - drain_rate < -10:  queue is growing; surfaced as "ETA: queue growing".
/// - otherwise:          rate is in the ±10 dead-band; "ETA: stabilizing".
///
/// Pure helper, extracted from the progress-thread closure so it can be
/// covered by unit tests.
pub(super) fn format_eta_suffix(queue_pending: u64, drain_rate: f64) -> String {
    if drain_rate > 10.0 {
        let eta_secs = queue_pending as f64 / drain_rate;
        if eta_secs < 60.0 {
            format!(" ETA: {:.0}s", eta_secs)
        } else if eta_secs < 3600.0 {
            format!(" ETA: {:.1}m", eta_secs / 60.0)
        } else if eta_secs < 86400.0 {
            format!(" ETA: {:.1}h", eta_secs / 3600.0)
        } else {
            format!(" ETA: {:.1}d", eta_secs / 86400.0)
        }
    } else if drain_rate < -10.0 {
        " ETA: queue growing".to_string()
    } else {
        " ETA: stabilizing".to_string()
    }
}

/// Compute the per-minute rate (states/min) given a delta and an elapsed
/// duration in seconds. Returns 0 when `elapsed_secs <= 0` to avoid
/// divide-by-zero on the first tick.
///
/// Pure helper, extracted from the progress-thread closure so it can be
/// covered by unit tests.
pub(super) fn rate_per_minute(delta: u64, elapsed_secs: f64) -> u64 {
    let elapsed_mins = elapsed_secs / 60.0;
    if elapsed_mins > 0.0 {
        (delta as f64 / elapsed_mins) as u64
    } else {
        0
    }
}

/// Mutable per-supervisor state carried across ticks.
///
/// Holds the values the periodic report needs to remember from the
/// previous tick so it can compute deltas (states/min, queue drain rate
/// for ETA). The 1-second sub-loop in `spawn_progress_thread` does NOT
/// mutate this — only `progress_tick` does.
///
/// Pulled out of the supervisor closure so the report logic can be
/// unit-tested without spinning a real OS thread or waiting 10s.
#[derive(Debug, Clone)]
pub(super) struct ProgressState {
    pub progress_counter: u64,
    pub last_generated: u64,
    pub last_distinct: u64,
    pub last_queue_pending: u64,
    pub last_time: Instant,
}

impl ProgressState {
    pub fn new(now: Instant) -> Self {
        Self {
            progress_counter: 1,
            last_generated: 0,
            last_distinct: 0,
            last_queue_pending: 0,
            last_time: now,
        }
    }
}

/// One periodic supervisor step.
///
/// Returns the rendered Progress line (without trailing newline) so callers
/// can assert against it; the live supervisor loop forwards the result to
/// stderr. `now` is injected so tests can replay deterministic clocks.
///
/// Updates `state` in place: bumps `progress_counter`, snapshots the
/// `generated`/`distinct`/`queue_pending` deltas, and records `now` as the
/// new `last_time` for the next tick's drain-rate computation.
pub(super) fn progress_tick(
    state: &mut ProgressState,
    states_generated: u64,
    states_distinct: u64,
    queue_pending: u64,
    now: Instant,
) -> String {
    let elapsed_secs = now.duration_since(state.last_time).as_secs_f64();
    let generated_rate =
        rate_per_minute(states_generated.saturating_sub(state.last_generated), elapsed_secs);
    let distinct_rate =
        rate_per_minute(states_distinct.saturating_sub(state.last_distinct), elapsed_secs);

    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

    // ETA only after we have at least one prior tick to base a delta on.
    let eta_str = if state.progress_counter > 1 && elapsed_secs > 0.0 {
        let queue_delta = state.last_queue_pending as i64 - queue_pending as i64;
        let drain_rate = queue_delta as f64 / elapsed_secs;
        format_eta_suffix(queue_pending, drain_rate)
    } else {
        String::new()
    };

    let line = format!(
        "Progress({}) at {}: {} states generated ({} s/min), {} distinct states found ({} ds/min), {} states left on queue.{}",
        state.progress_counter,
        timestamp,
        format_with_commas(states_generated),
        format_with_commas(generated_rate),
        format_with_commas(states_distinct),
        format_with_commas(distinct_rate),
        format_with_commas(queue_pending),
        eta_str
    );

    state.progress_counter += 1;
    state.last_generated = states_generated;
    state.last_distinct = states_distinct;
    state.last_queue_pending = queue_pending;
    state.last_time = now;

    line
}

pub(super) fn spawn_progress_thread<S>(
    run_stats: Arc<AtomicRunStats>,
    stop: Arc<AtomicBool>,
    fp_store: Arc<UnifiedFingerprintStore>,
    queue: Arc<SpillableWorkStealingQueues<S>>,
    throttle: Arc<WorkerThrottle>,
    worker_count: usize,
) -> std::thread::JoinHandle<()>
where
    S: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + Clone + 'static,
{
    let progress_run_stats = run_stats;
    let progress_stop = stop;
    let progress_fp_store = fp_store;
    let progress_queue = queue;
    let progress_throttle = throttle;
    let progress_worker_count = worker_count;

    std::thread::spawn(move || {
        let mut state = ProgressState::new(Instant::now());

        // Memory monitor - check periodically and throttle if needed
        let memory_monitor = MemoryMonitor::default();
        let mut memory_warned = false;
        let mut memory_throttled = false;

        loop {
            // Check for emergency checkpoint request every second
            for _ in 0..10 {
                std::thread::sleep(Duration::from_secs(1));

                if progress_stop.load(Ordering::Relaxed) {
                    return;
                }

                // Check memory pressure and throttle workers if needed
                let mem_status = memory_monitor.check();
                match mem_status {
                    MemoryStatus::Critical {
                        rss_bytes,
                        limit_bytes,
                        ratio,
                    } => {
                        if !memory_throttled {
                            let rss_gb = rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            let limit_gb = limit_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            eprintln!(
                                "MEMORY CRITICAL: {:.1}GB / {:.1}GB ({:.0}%) - throttling to 25% workers",
                                rss_gb,
                                limit_gb,
                                ratio * 100.0
                            );
                            // Throttle to 25% of workers
                            let reduced = (progress_worker_count / 4).max(1);
                            progress_throttle.set_active_target(reduced);
                            memory_throttled = true;
                            memory_warned = true;
                        }
                    }
                    MemoryStatus::Warning {
                        rss_bytes,
                        limit_bytes,
                        ratio,
                    } => {
                        if !memory_warned {
                            let rss_gb = rss_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            let limit_gb = limit_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
                            eprintln!(
                                "MEMORY WARNING: {:.1}GB / {:.1}GB ({:.0}%) - consider reducing workers",
                                rss_gb,
                                limit_gb,
                                ratio * 100.0
                            );
                            // Throttle to 50% of workers
                            let reduced = (progress_worker_count / 2).max(1);
                            progress_throttle.set_active_target(reduced);
                            memory_warned = true;
                        }
                    }
                    MemoryStatus::Ok { ratio, .. } => {
                        // If we were throttled and now OK, gradually restore
                        if memory_throttled && ratio < 0.60 {
                            eprintln!(
                                "Memory pressure relieved ({:.0}%) - restoring workers",
                                ratio * 100.0
                            );
                            progress_throttle.set_active_target(progress_worker_count);
                            memory_throttled = false;
                            memory_warned = false;
                        }
                    }
                }

                // Handle emergency checkpoint request
                if crate::chaos::is_emergency_checkpoint_requested() {
                    eprintln!("Emergency checkpoint: flushing fingerprint store...");
                    if let Err(e) = progress_fp_store.flush() {
                        eprintln!("Emergency checkpoint: fingerprint flush failed: {}", e);
                    } else {
                        eprintln!("Emergency checkpoint: fingerprint store flushed successfully");
                    }
                    let (generated, _, distinct, _, _, _) = progress_run_stats.snapshot();
                    eprintln!(
                        "Emergency checkpoint: {} states generated, {} distinct at time of failure",
                        generated, distinct
                    );
                    crate::chaos::clear_emergency_checkpoint();
                }
            }

            if progress_stop.load(Ordering::Relaxed) {
                break;
            }

            let (states_generated, _states_processed, states_distinct, _, _, _) =
                progress_run_stats.snapshot();
            // Use total_pending_count to include spilled items on disk
            let queue_pending = progress_queue.total_pending_count();

            let line = progress_tick(
                &mut state,
                states_generated,
                states_distinct,
                queue_pending,
                Instant::now(),
            );
            eprintln!("{}", line);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ProgressState, format_eta_suffix, format_with_commas, progress_tick, rate_per_minute,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn format_with_commas_handles_small_numbers() {
        assert_eq!(format_with_commas(0), "0");
        assert_eq!(format_with_commas(1), "1");
        assert_eq!(format_with_commas(99), "99");
        assert_eq!(format_with_commas(999), "999");
    }

    #[test]
    fn format_with_commas_inserts_separator_at_thousands_boundary() {
        // Boundary kills off-by-one mutations on the `i % 3 == 0` test.
        assert_eq!(format_with_commas(1_000), "1,000");
        assert_eq!(format_with_commas(12_345), "12,345");
        assert_eq!(format_with_commas(1_234_567), "1,234,567");
        assert_eq!(format_with_commas(1_000_000_000), "1,000,000,000");
    }

    #[test]
    fn format_with_commas_handles_u64_max() {
        // 18,446,744,073,709,551,615 — exercise the all-the-way-up path.
        assert_eq!(format_with_commas(u64::MAX), "18,446,744,073,709,551,615");
    }

    #[test]
    fn rate_per_minute_zero_elapsed_returns_zero() {
        // Avoids divide-by-zero on the first progress tick.
        assert_eq!(rate_per_minute(1_000, 0.0), 0);
        assert_eq!(rate_per_minute(0, 0.0), 0);
    }

    #[test]
    fn rate_per_minute_basic_arithmetic() {
        // 600 states in 60s → 600 states/min.
        assert_eq!(rate_per_minute(600, 60.0), 600);
        // 100 states in 30s → 200 states/min.
        assert_eq!(rate_per_minute(100, 30.0), 200);
        // 0 delta → 0 rate.
        assert_eq!(rate_per_minute(0, 60.0), 0);
    }

    #[test]
    fn eta_suffix_dead_band_returns_stabilizing() {
        // |drain_rate| <= 10 (inclusive both ends → "stabilizing").
        assert_eq!(format_eta_suffix(1_000, 0.0), " ETA: stabilizing");
        assert_eq!(format_eta_suffix(1_000, 10.0), " ETA: stabilizing");
        assert_eq!(format_eta_suffix(1_000, -10.0), " ETA: stabilizing");
        assert_eq!(format_eta_suffix(1_000, 5.5), " ETA: stabilizing");
    }

    #[test]
    fn eta_suffix_growing_below_minus_10() {
        assert_eq!(format_eta_suffix(0, -11.0), " ETA: queue growing");
        assert_eq!(format_eta_suffix(1_000, -1_000.0), " ETA: queue growing");
    }

    #[test]
    fn eta_suffix_seconds_unit_under_60s() {
        // 50 items / 100 per-sec drain = 0.5s → "1s" (rounded to 0 decimals).
        // Use 110 / 110 = 1 for an unambiguous "1s".
        assert_eq!(format_eta_suffix(110, 110.0), " ETA: 1s");
    }

    #[test]
    fn eta_suffix_minutes_unit_between_60s_and_3600s() {
        // 1200 items / 20 per-sec = 60s — boundary; the `< 60.0` test means
        // exactly-60 falls into the minutes arm.
        let s = format_eta_suffix(1200, 20.0);
        assert!(s.starts_with(" ETA: ") && s.ends_with("m"), "got {s:?}");
    }

    #[test]
    fn eta_suffix_hours_unit_between_3600s_and_86400s() {
        // drain_rate must be > 10 to enter the draining branch.
        // 144_000 items / 20 per-sec = 7200s = 2h.
        let s = format_eta_suffix(144_000, 20.0);
        assert!(s.ends_with("h"), "expected hour units, got {s:?}");
    }

    #[test]
    fn eta_suffix_days_unit_above_86400s() {
        // 2_000_000 items / 20 per-sec = 100_000s ≈ 1.16d. drain_rate > 10
        // gates entry to the draining branch.
        let s = format_eta_suffix(2_000_000, 20.0);
        assert!(s.ends_with("d"), "expected day units, got {s:?}");
    }

    #[test]
    fn eta_suffix_just_above_drain_threshold() {
        // drain_rate = 10.1 (just above the 10.0 threshold) — must enter
        // the draining branch, not the stabilizing one.
        let s = format_eta_suffix(0, 10.1);
        assert_ne!(s, " ETA: stabilizing");
    }

    /// First tick must NOT emit an ETA (we have no prior delta to base it on)
    /// and must bump `progress_counter` from 1 to 2 so subsequent ticks do.
    #[test]
    fn progress_tick_first_tick_emits_no_eta_and_advances_counter() {
        let t0 = Instant::now();
        let mut state = ProgressState::new(t0);
        assert_eq!(state.progress_counter, 1);

        let line = progress_tick(&mut state, 1_000, 800, 200, t0 + Duration::from_secs(10));
        // No ETA suffix on the very first tick — the gating condition is
        // `progress_counter > 1`, which is false here.
        assert!(
            !line.contains(" ETA:"),
            "first tick should not include ETA, got: {line}"
        );
        // Counter advanced; deltas snapshot for next tick.
        assert_eq!(state.progress_counter, 2);
        assert_eq!(state.last_generated, 1_000);
        assert_eq!(state.last_distinct, 800);
        assert_eq!(state.last_queue_pending, 200);
        assert!(line.starts_with("Progress(1) at "));
    }

    /// Two consecutive ticks with no progress (queue size unchanged) must
    /// surface the "stabilizing" branch — drain rate is 0, |drain| <= 10.
    #[test]
    fn progress_tick_no_progress_yields_stabilizing_eta() {
        let t0 = Instant::now();
        let mut state = ProgressState::new(t0);
        // Prime: first tick, no ETA expected.
        let _ = progress_tick(&mut state, 0, 0, 5_000, t0 + Duration::from_secs(10));
        // Second tick: queue size unchanged, so drain_rate = 0 → stabilizing.
        let line = progress_tick(&mut state, 0, 0, 5_000, t0 + Duration::from_secs(20));
        assert!(
            line.contains(" ETA: stabilizing"),
            "no-drain tick should report stabilizing, got: {line}"
        );
        assert_eq!(state.progress_counter, 3);
    }

    /// A queue that's draining fast (>10 states/sec) must emit a finite-time
    /// ETA in the seconds/minutes/hours/days unit. Boundary kept simple.
    #[test]
    fn progress_tick_draining_queue_emits_finite_eta() {
        let t0 = Instant::now();
        let mut state = ProgressState::new(t0);
        // Prime tick.
        let _ = progress_tick(&mut state, 0, 0, 1_000, t0 + Duration::from_secs(10));
        // Second tick: 100 items remaining, dropped 900 in 10s → 90/sec drain.
        let line = progress_tick(&mut state, 0, 0, 100, t0 + Duration::from_secs(20));
        assert!(
            line.contains(" ETA: ") && !line.contains(" ETA: stabilizing")
                && !line.contains(" ETA: queue growing"),
            "draining queue should report finite ETA, got: {line}"
        );
    }

    /// A queue that's growing (queue_delta is negative, drain_rate < -10) must
    /// surface the "queue growing" branch.
    #[test]
    fn progress_tick_growing_queue_reports_growing_eta() {
        let t0 = Instant::now();
        let mut state = ProgressState::new(t0);
        // Prime: queue_pending starts at 0.
        let _ = progress_tick(&mut state, 0, 0, 0, t0 + Duration::from_secs(10));
        // Second: queue jumped to 1000 over 10s → drain_rate = -100/sec.
        let line = progress_tick(&mut state, 5_000, 4_000, 1_000, t0 + Duration::from_secs(20));
        assert!(
            line.contains(" ETA: queue growing"),
            "growing queue should report growing, got: {line}"
        );
    }

    /// Each tick monotonically advances `progress_counter` and the
    /// rendered Progress(N) prefix tracks it. Kills mutations that
    /// forget to bump the counter on the second-and-subsequent ticks.
    #[test]
    fn progress_tick_counter_in_render_advances_on_each_call() {
        let t0 = Instant::now();
        let mut state = ProgressState::new(t0);
        let l1 = progress_tick(&mut state, 0, 0, 0, t0 + Duration::from_secs(10));
        assert!(l1.starts_with("Progress(1) at "));
        let l2 = progress_tick(&mut state, 1, 1, 1, t0 + Duration::from_secs(20));
        assert!(l2.starts_with("Progress(2) at "));
        let l3 = progress_tick(&mut state, 2, 2, 2, t0 + Duration::from_secs(30));
        assert!(l3.starts_with("Progress(3) at "));
        assert_eq!(state.progress_counter, 4);
    }
}
