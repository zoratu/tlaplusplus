// Auto-tuning for optimal CPU utilization
//
// Monitors CPU sys% and dynamically adjusts active worker count to minimize
// kernel time while maximizing throughput.
//
// Key insight: High sys% indicates lock/atomic contention. Reducing workers
// reduces contention, often INCREASING throughput.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// CPU utilization sample from /proc/stat
#[derive(Clone, Debug, Default)]
struct CpuSample {
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    total: u64,
}

impl CpuSample {
    fn from_proc_stat() -> Option<Self> {
        let contents = std::fs::read_to_string("/proc/stat").ok()?;
        let line = contents.lines().next()?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 8 || parts[0] != "cpu" {
            return None;
        }

        let user: u64 = parts[1].parse().ok()?;
        let nice: u64 = parts[2].parse().ok()?;
        let system: u64 = parts[3].parse().ok()?;
        let idle: u64 = parts[4].parse().ok()?;
        let iowait: u64 = parts[5].parse().ok()?;
        let irq: u64 = parts[6].parse().ok()?;
        let softirq: u64 = parts[7].parse().ok()?;
        let total = user + nice + system + idle + iowait + irq + softirq;

        Some(Self {
            user,
            nice,
            system,
            idle,
            iowait,
            irq,
            softirq,
            total,
        })
    }

    /// Calculate utilization percentages between two samples
    fn diff(&self, prev: &Self) -> (f64, f64) {
        let total_diff = self.total.saturating_sub(prev.total) as f64;
        if total_diff < 1.0 {
            return (0.0, 0.0);
        }

        let user_diff = (self.user + self.nice).saturating_sub(prev.user + prev.nice) as f64;
        let sys_diff = (self.system + self.irq + self.softirq)
            .saturating_sub(prev.system + prev.irq + prev.softirq) as f64;

        let usr_pct = (user_diff / total_diff) * 100.0;
        let sys_pct = (sys_diff / total_diff) * 100.0;

        (usr_pct, sys_pct)
    }
}

/// Auto-tuner configuration
#[derive(Clone, Debug)]
pub struct AutoTuneConfig {
    /// Maximum workers to use (start with all)
    pub max_workers: usize,
    /// Minimum workers to use (floor)
    pub min_workers: usize,
    /// Target sys% (reduce workers if above this, default 20%)
    pub target_sys_pct: f64,
    /// Sample interval
    pub sample_interval: Duration,
    /// Adjustment step (workers to add/remove)
    pub adjustment_step: usize,
    /// Cooldown between adjustments
    pub adjustment_cooldown: Duration,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            max_workers: 0, // Will be set from actual worker count
            min_workers: 1,
            target_sys_pct: 20.0,
            sample_interval: Duration::from_secs(2),
            adjustment_step: 16, // Adjust by 16 workers at a time
            adjustment_cooldown: Duration::from_secs(5),
            verbose: true,
        }
    }
}

/// Worker throttle - controls how many workers are active
pub struct WorkerThrottle {
    /// Current target active workers
    active_target: AtomicUsize,
    /// Actual max workers
    max_workers: usize,
    /// Counter for workers to check their turn
    worker_counter: AtomicU64,
}

impl WorkerThrottle {
    pub fn new(max_workers: usize) -> Self {
        Self {
            active_target: AtomicUsize::new(max_workers),
            max_workers,
            worker_counter: AtomicU64::new(0),
        }
    }

    /// Get current active target
    pub fn active_target(&self) -> usize {
        self.active_target.load(Ordering::Relaxed)
    }

    /// Set new active target
    pub fn set_active_target(&self, target: usize) {
        self.active_target
            .store(target.min(self.max_workers), Ordering::Relaxed);
    }

    /// Should this worker be active? Uses round-robin to select which workers pause.
    /// Workers with ID >= active_target will yield.
    #[inline]
    pub fn should_worker_yield(&self, worker_id: usize) -> bool {
        let target = self.active_target.load(Ordering::Relaxed);
        worker_id >= target
    }

    /// Worker pause point - yields if over throttle limit
    #[inline]
    pub fn worker_throttle_point(&self, worker_id: usize) {
        if self.should_worker_yield(worker_id) {
            // Brief sleep to reduce CPU usage without completely blocking
            std::thread::sleep(Duration::from_micros(100));
        }
    }
}

/// Auto-tuner thread that monitors CPU and adjusts worker count
pub struct AutoTuner {
    config: AutoTuneConfig,
    throttle: Arc<WorkerThrottle>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    /// Best observed throughput and worker count
    best_throughput: AtomicU64,
    best_workers: AtomicUsize,
}

impl AutoTuner {
    pub fn new(
        config: AutoTuneConfig,
        throttle: Arc<WorkerThrottle>,
        stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            config,
            throttle,
            stop,
            handle: None,
            best_throughput: AtomicU64::new(0),
            best_workers: AtomicUsize::new(0),
        }
    }

    /// Start the auto-tuning thread with a function to read throughput
    pub fn start<F>(&mut self, get_throughput: F)
    where
        F: Fn() -> u64 + Send + 'static,
    {
        let config = self.config.clone();
        let throttle = Arc::clone(&self.throttle);
        let stop = Arc::clone(&self.stop);

        let handle = std::thread::Builder::new()
            .name("autotune".to_string())
            .spawn(move || {
                Self::tuner_loop(config, throttle, stop, get_throughput);
            })
            .expect("Failed to spawn autotune thread");

        self.handle = Some(handle);
    }

    fn tuner_loop<F>(
        config: AutoTuneConfig,
        throttle: Arc<WorkerThrottle>,
        stop: Arc<AtomicBool>,
        get_throughput: F,
    ) where
        F: Fn() -> u64,
    {
        let mut prev_sample = CpuSample::from_proc_stat();
        let mut prev_throughput = get_throughput();
        let mut last_adjustment = Instant::now();
        let mut best_throughput_rate = 0.0f64;
        let mut best_workers = config.max_workers;

        // Warmup period - give workers time to stabilize
        std::thread::sleep(Duration::from_secs(3));

        while !stop.load(Ordering::Acquire) {
            std::thread::sleep(config.sample_interval);

            if stop.load(Ordering::Acquire) {
                break;
            }

            // Sample CPU and throughput
            let current_sample = match CpuSample::from_proc_stat() {
                Some(s) => s,
                None => continue,
            };

            let current_throughput = get_throughput();
            let throughput_rate = (current_throughput - prev_throughput) as f64
                / config.sample_interval.as_secs_f64();

            let (_usr_pct, sys_pct) = if let Some(ref prev) = prev_sample {
                current_sample.diff(prev)
            } else {
                (0.0, 0.0)
            };

            let current_target = throttle.active_target();

            // Track best throughput
            if throughput_rate > best_throughput_rate {
                best_throughput_rate = throughput_rate;
                best_workers = current_target;
            }

            let time_since_adjustment = last_adjustment.elapsed();
            let can_adjust = time_since_adjustment >= config.adjustment_cooldown;

            if can_adjust {
                // Don't reduce workers if throughput has dropped significantly from best
                let throughput_ok = throughput_rate > best_throughput_rate * 0.85;

                // Hysteresis: only reduce if sys% is significantly above target
                // This prevents flapping between adjacent worker counts
                let reduce_threshold = config.target_sys_pct + 5.0;

                if sys_pct > reduce_threshold && throughput_ok {
                    // High sys% but throughput still good - reduce workers
                    let new_target = current_target
                        .saturating_sub(config.adjustment_step)
                        .max(config.min_workers);

                    if new_target < current_target {
                        throttle.set_active_target(new_target);
                        last_adjustment = Instant::now();
                        if config.verbose {
                            eprintln!(
                                "AutoTune: sys={:.1}% > target {:.1}%, reducing workers {} -> {}",
                                sys_pct, config.target_sys_pct, current_target, new_target
                            );
                        }
                    }
                } else if sys_pct < config.target_sys_pct - 5.0
                    && current_target < config.max_workers
                {
                    // Low sys% (with hysteresis) - can try adding workers
                    let new_target =
                        (current_target + config.adjustment_step).min(config.max_workers);

                    throttle.set_active_target(new_target);
                    last_adjustment = Instant::now();
                    if config.verbose {
                        eprintln!(
                            "AutoTune: sys={:.1}% low, trying more workers {} -> {}",
                            sys_pct, current_target, new_target
                        );
                    }
                } else if !throughput_ok && current_target < best_workers {
                    // Throughput dropped significantly - revert toward best worker count
                    let new_target = (current_target + config.adjustment_step).min(best_workers);
                    throttle.set_active_target(new_target);
                    last_adjustment = Instant::now();
                    if config.verbose {
                        eprintln!(
                            "AutoTune: throughput dropped, reverting workers {} -> {} (best={})",
                            current_target, new_target, best_workers
                        );
                    }
                }
            }

            prev_sample = Some(current_sample);
            prev_throughput = current_throughput;
        }

        if config.verbose {
            eprintln!(
                "AutoTune: finished. Best: {:.0} states/s with {} workers",
                best_throughput_rate, best_workers
            );
        }
    }

    /// Wait for tuner thread to finish
    pub fn join(mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
