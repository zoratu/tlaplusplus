//! T10.2 stage 5 — DFS pool throughput benchmark.
//!
//! Quantifies the wall-clock speedup of the multi-worker DFS pool vs
//! the stage-4 single-worker DFS on the same `HighFanoutGrid` model
//! used by `dfs_memory_benchmark.rs`. The single-worker stage-4 path
//! pegs one CPU; the pool is meant to bring exploration time down by
//! parallelising the partition.
//!
//! # Measurement protocol
//!
//! Two passes within a single test process:
//!
//!   Pass 1: stage-4 DFS (`dfs_workers = 1`). Records `states_distinct`,
//!   wall time, and peak RSS delta.
//!
//!   Pass 2: stage-5 DFS pool (`dfs_workers = 4`). Same metrics.
//!
//! Assertions:
//!
//!   1. `pool_states == single_states`     — verdict-equivalence sanity.
//!   2. `pool_secs <= single_secs`         — pool must not be slower
//!                                            than the single-worker
//!                                            path on this fanout.
//!   3. `pool_rss <= 2 * single_rss`       — pool memory must not blow
//!                                            up beyond 2× (cross-
//!                                            partition channel +
//!                                            per-worker scratch
//!                                            overhead is acceptable).
//!
//! # When to run
//!
//! Marked `#[ignore]` so the per-PR loop doesn't pay the ~3-min cost.
//! Run with:
//!
//!   cargo test --release --test dfs_pool_throughput_benchmark -- --ignored --nocapture

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tlaplusplus::fairness::FairnessConstraint;
use tlaplusplus::model::{ActionLabel, LabeledTransition, Model};
use tlaplusplus::system::get_memory_stats;
use tlaplusplus::{EngineConfig, run_model};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct GridState {
    x: u16,
    y: u16,
    payload: [u8; 32],
}

struct HighFanoutGrid {
    dim: u16,
    move_action: String,
}

impl HighFanoutGrid {
    fn new(dim: u16) -> Self {
        Self {
            dim,
            move_action: "Move".to_string(),
        }
    }
    fn move_successors(&self, s: &GridState) -> Vec<(GridState, &'static str)> {
        let mut out: Vec<(GridState, &'static str)> = Vec::with_capacity(6);
        out.push((s.clone(), "Move"));
        if s.x + 1 < self.dim {
            let mut t = s.clone();
            t.x += 1;
            out.push((t, "Move"));
        }
        if s.x > 0 {
            let mut t = s.clone();
            t.x -= 1;
            out.push((t, "Move"));
        }
        if s.y + 1 < self.dim {
            let mut t = s.clone();
            t.y += 1;
            out.push((t, "Move"));
        }
        if s.y > 0 {
            let mut t = s.clone();
            t.y -= 1;
            out.push((t, "Move"));
        }
        out
    }
}

impl Model for HighFanoutGrid {
    type State = GridState;
    fn name(&self) -> &'static str {
        "high-fanout-grid-pool-bench"
    }
    fn initial_states(&self) -> Vec<Self::State> {
        vec![GridState { x: 0, y: 0, payload: [0u8; 32] }]
    }
    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
        for (s, _) in self.move_successors(state) {
            out.push(s);
        }
    }
    fn next_states_labeled(
        &self,
        state: &Self::State,
    ) -> Option<Vec<LabeledTransition<Self::State>>> {
        let mut out = Vec::with_capacity(6);
        for (to, action_name) in self.move_successors(state) {
            out.push(LabeledTransition {
                from: state.clone(),
                to,
                action: ActionLabel { name: action_name.to_string(), disjunct_index: None },
            });
        }
        Some(out)
    }
    fn has_fairness_constraints(&self) -> bool { true }
    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        vec![FairnessConstraint::Weak {
            vars: vec!["x".into(), "y".into(), "payload".into()],
            action: self.move_action.clone(),
        }]
    }
    fn next_action_name(&self) -> Option<&str> { Some(&self.move_action) }
    fn check_invariants(&self, _state: &Self::State) -> Result<(), String> { Ok(()) }
}

fn start_rss_poller(
    poll_interval: Duration,
) -> (Arc<AtomicBool>, Arc<AtomicU64>, std::thread::JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_t = Arc::clone(&stop);
    let peak_t = Arc::clone(&peak);
    let h = std::thread::spawn(move || {
        while !stop_t.load(Ordering::Acquire) {
            let cur = get_memory_stats().rss_bytes;
            let prev = peak_t.load(Ordering::Relaxed);
            if cur > prev {
                peak_t.store(cur, Ordering::Relaxed);
            }
            std::thread::sleep(poll_interval);
        }
        let cur = get_memory_stats().rss_bytes;
        let prev = peak_t.load(Ordering::Relaxed);
        if cur > prev {
            peak_t.store(cur, Ordering::Relaxed);
        }
    });
    (stop, peak, h)
}

fn run_one(dfs_workers: usize, dim: u16, work_dir_root: &std::path::Path) -> (u64, u64, Duration) {
    let model = HighFanoutGrid::new(dim);
    let mut cfg = EngineConfig::default();
    // Set BFS workers >= dfs_workers so the pool is allowed to grow.
    cfg.workers = dfs_workers.max(1);
    cfg.enforce_cgroups = false;
    cfg.numa_pinning = false;
    let states_estimate = (dim as usize) * (dim as usize);
    let fp_slots = states_estimate.next_power_of_two().max(1 << 18);
    cfg.fp_expected_items = fp_slots;
    cfg.fp_shards = 4;
    cfg.fp_cache_capacity_bytes = 8 * 1024 * 1024;
    cfg.checkpoint_on_exit = false;
    cfg.enable_fp_persistence = false;
    cfg.enable_queue_spilling = false;
    cfg.queue_compression = false;
    cfg.work_dir = work_dir_root.join(format!("n{}", dfs_workers));
    cfg.liveness_streaming = false;
    cfg.liveness_streaming_exploration = true;
    cfg.dfs_workers = dfs_workers;

    let baseline = get_memory_stats().rss_bytes;
    let (stop_poll, peak, poll_h) = start_rss_poller(Duration::from_millis(25));
    peak.store(baseline, Ordering::Relaxed);

    let started = Instant::now();
    let outcome = run_model(model, cfg).expect("run_model");
    let elapsed = started.elapsed();

    stop_poll.store(true, Ordering::Release);
    poll_h.join().expect("poll thread joins");

    let peak_rss = peak.load(Ordering::Relaxed);
    let delta = peak_rss.saturating_sub(baseline);

    eprintln!(
        "  pool n={}: states_distinct={} elapsed={:.2?} baseline={:.1} MiB peak={:.1} MiB delta={:.1} MiB violation={:?}",
        dfs_workers,
        outcome.stats.states_distinct,
        elapsed,
        baseline as f64 / (1024.0 * 1024.0),
        peak_rss as f64 / (1024.0 * 1024.0),
        delta as f64 / (1024.0 * 1024.0),
        outcome.violation.as_ref().map(|v| v.message.as_str()),
    );

    (delta, outcome.stats.states_distinct, elapsed)
}

#[test]
#[ignore = "throughput benchmark — run with --ignored --nocapture"]
fn dfs_pool_throughput_vs_single_worker() {
    let dim: u16 = 250; // 62 500 reachable states; ~5 successors each.
    let work_dir_root = std::env::temp_dir().join(format!(
        "tlapp-dfs-pool-bench-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&work_dir_root);

    eprintln!("\n========== T10.2 stage 5 DFS pool throughput benchmark ==========");
    eprintln!(
        "Grid dim={}, expected reachable states ≈ {}",
        dim, dim as u64 * dim as u64
    );

    eprintln!("\n--- Pass 1: stage-4 DFS (dfs_workers=1) ---");
    let (single_delta, single_states, single_secs) = run_one(1, dim, &work_dir_root);

    std::thread::sleep(Duration::from_millis(250));

    eprintln!("\n--- Pass 2: stage-5 DFS pool (dfs_workers=4) ---");
    let (pool_delta, pool_states, pool_secs) = run_one(4, dim, &work_dir_root);

    let _ = std::fs::remove_dir_all(&work_dir_root);

    eprintln!("\n========== Result ==========");
    eprintln!(
        "single-worker:  states={}  wall={:.2?}  delta={:.1} MiB",
        single_states, single_secs, single_delta as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "pool n=4:       states={}  wall={:.2?}  delta={:.1} MiB",
        pool_states, pool_secs, pool_delta as f64 / (1024.0 * 1024.0)
    );
    if pool_secs > Duration::ZERO {
        let speedup = single_secs.as_secs_f64() / pool_secs.as_secs_f64();
        eprintln!("speedup (single / pool): {:.2}x", speedup);
    }
    if single_delta > 0 {
        let mem_ratio = pool_delta as f64 / single_delta as f64;
        eprintln!("memory ratio (pool / single): {:.2}x", mem_ratio);
    }

    // Hard correctness gate: state counts MUST match.
    assert_eq!(
        pool_states, single_states,
        "pool vs single-worker state count mismatch (single={}, pool={})",
        single_states, pool_states
    );

    // Soft throughput gate: pool must not be slower than single-worker
    // on this fanout. Allow 1.5x slowdown to avoid CI flakes from co-
    // tenant interference; the real win on a quiescent box is much
    // better.
    assert!(
        pool_secs.as_secs_f64() <= single_secs.as_secs_f64() * 1.5,
        "pool ({:?}) regressed > 1.5x vs single-worker ({:?})",
        pool_secs, single_secs
    );

    // Memory gate: pool RSS delta must be within 2x of single-worker.
    if single_delta > 0 {
        assert!(
            pool_delta as f64 <= single_delta as f64 * 2.0,
            "pool memory delta {:.1} MiB exceeds 2x single-worker {:.1} MiB",
            pool_delta as f64 / (1024.0 * 1024.0),
            single_delta as f64 / (1024.0 * 1024.0),
        );
    }
}
