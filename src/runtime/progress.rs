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
        let mut progress_counter = 1u64;
        let mut last_generated = 0u64;
        let mut last_distinct = 0u64;
        let mut last_queue_pending = 0u64;
        let mut last_time = Instant::now();

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

            // Calculate rates per minute
            let now = Instant::now();
            let elapsed_secs = now.duration_since(last_time).as_secs_f64();
            let elapsed_mins = elapsed_secs / 60.0;

            let generated_rate = if elapsed_mins > 0.0 {
                ((states_generated - last_generated) as f64 / elapsed_mins) as u64
            } else {
                0
            };

            let distinct_rate = if elapsed_mins > 0.0 {
                ((states_distinct - last_distinct) as f64 / elapsed_mins) as u64
            } else {
                0
            };

            // Format timestamp in TLC style: YYYY-MM-DD HH:MM:SS
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

            // Format numbers with commas (TLC style)
            fn format_with_commas(n: u64) -> String {
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

            // Calculate ETA based on queue drain rate
            let eta_str = if progress_counter > 1 && elapsed_secs > 0.0 {
                let queue_delta = last_queue_pending as i64 - queue_pending as i64;
                let drain_rate = queue_delta as f64 / elapsed_secs; // states/sec

                if drain_rate > 10.0 {
                    // Queue is draining - estimate completion time
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
                    // Queue is growing
                    " ETA: queue growing".to_string()
                } else {
                    // Queue is stable
                    " ETA: stabilizing".to_string()
                }
            } else {
                String::new()
            };

            // Match TLC format with ETA
            eprintln!(
                "Progress({}) at {}: {} states generated ({} s/min), {} distinct states found ({} ds/min), {} states left on queue.{}",
                progress_counter,
                timestamp,
                format_with_commas(states_generated),
                format_with_commas(generated_rate),
                format_with_commas(states_distinct),
                format_with_commas(distinct_rate),
                format_with_commas(queue_pending),
                eta_str
            );

            progress_counter += 1;
            last_generated = states_generated;
            last_distinct = states_distinct;
            last_queue_pending = queue_pending;
            last_time = now;
        }
    })
}
