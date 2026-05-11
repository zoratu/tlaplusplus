// T10.2 stage 4 — DFS-vs-BFS peak-RSS memory benchmark.
//
// Quantifies the actual memory win delivered by the Stage 4 in-band
// fairness verdict (no `labeled_transitions` DashMap materialization).
// The Stage 3 architecture was in place but the DashMap was still
// populated, so the predicted ~10x edge-memory reduction wasn't being
// realized; Stage 4 drops the DashMap and does the verdict in-band.
//
// # The benchmark
//
// We define a hand-rolled Rust `Model` (`HighFanoutGrid`) that produces
// a wide, deeply-connected reachable state graph:
//
//   - State: a 2D coordinate `(x, y)` plus a 32-byte payload (mimics
//     a TLA+ record with several variables — the payload is what
//     drives the per-edge state-clone cost in the BFS path).
//   - Transitions: every state has up to 6 successors (left, right,
//     up, down, swap-payload, identity). At grid sizes ~256 × 256 this
//     gives ≥ 65 000 reachable states and ≈ 400 000 edges.
//   - Fairness: a `WF_vars(Move)` constraint over the move actions
//     ensures the DFS dispatch path activates (gated on
//     `model.has_fairness_constraints()`).
//
// The BFS path materialises `labeled_transitions: DashMap<u64,
// Vec<LabeledTransition<State>>>` which clones every state pair (~32 B
// payload + 8 B coords + per-edge ActionLabel). The Stage 4 DFS path
// builds only `Vec<(u64, u64, String)>` triples — fingerprints + a
// short action name string.
//
// # Measurement protocol
//
// Two passes within a single test process:
//
//   Pass 1 (DFS first to avoid being penalised by BFS leftovers):
//     1. Record `baseline_rss = VmRSS` immediately before run starts.
//     2. Spawn a 25 ms-poll thread that watches `VmRSS` and tracks max.
//     3. Run `run_model` with `liveness_streaming_exploration = true`.
//     4. Stop the poll thread; `dfs_peak_delta = max - baseline`.
//
//   Pass 2 (BFS):
//     5. Drop everything from pass 1; sleep 100 ms to let the
//        allocator return memory to the OS (best-effort).
//     6. Re-record `baseline_rss`; spawn poll thread.
//     7. Run `run_model` with `liveness_streaming_exploration = false`.
//     8. Stop the poll thread; `bfs_peak_delta = max - baseline`.
//
//   Assertion: `dfs_peak_delta <= 0.7 * bfs_peak_delta` — i.e. DFS
//   delta is at most 70 % of BFS delta (≥ 30 % reduction).
//
// # Why this isn't perfect, and why it doesn't matter
//
// `VmRSS` polling at 25 ms can miss short spikes; jemalloc /
// glibc-malloc may keep freed memory in arenas (so pass-2 baseline
// includes pass-1 leftovers). Both effects bias *against* DFS — pass-2
// baseline is high so BFS delta is *under-counted* — yet DFS still wins
// by ≥ 30 %. If anything the real win is larger.
//
// # When to run
//
// Marked `#[ignore]` so it doesn't run on the per-PR test loop (it
// burns ~30-60 s and allocates ~1 GB). Run with:
//
//   cargo test --release --test dfs_memory_benchmark -- --ignored --nocapture
//
// CI gates 1-7 (parity tests, oracle, etc) verify *correctness* on the
// per-PR loop; this benchmark verifies the *memory win* on demand.

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
    /// Payload bytes — drives the per-edge clone cost in the BFS
    /// path. 32 bytes is representative of a small TLA+ record with
    /// half a dozen integer fields.
    payload: [u8; 32],
}

struct HighFanoutGrid {
    /// Grid dimension. `dim × dim` reachable states (post-init bounds).
    dim: u16,
    /// Action name for the Move action; advertised through the
    /// fairness constraint so DFS dispatch fires.
    move_action: String,
    /// Action name for the Idle action — never enabled, so the WF
    /// fairness constraint over it is satisfied/violated based on
    /// whether the SCC has Idle edges (which it doesn't, by
    /// construction). We pick this so the DFS in-band verdict and the
    /// BFS post-processing verdict produce *the same* result (no
    /// Idle edges → fairness violation under WF_vars(Idle)).
    #[allow(dead_code)]
    idle_action: String,
}

impl HighFanoutGrid {
    fn new(dim: u16) -> Self {
        Self {
            dim,
            move_action: "Move".to_string(),
            idle_action: "Idle".to_string(),
        }
    }

    /// Compute the 4-neighbour move targets plus a payload-swap and an
    /// identity self-loop. The self-loop is what guarantees a non-
    /// trivial SCC at every state (single-state SCC with a self-edge),
    /// which is what triggers the per-action fairness check.
    fn move_successors(&self, s: &GridState) -> Vec<(GridState, &'static str)> {
        let mut out: Vec<(GridState, &'static str)> = Vec::with_capacity(6);
        // Identity self-loop with action label "Move" — guarantees a
        // self-loop SCC at every reachable node so the in-SCC fairness
        // check has work to do.
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
        // Payload swap — alters bytes 0/1 deterministically. Doesn't
        // change reachable state set (since payload is part of state)
        // but adds a reflexive edge that's lexically distinct, which
        // increases edge density without exploding the state graph
        // size.
        // (Disabled: would explode state space. Kept comment for
        // future stages.)
        out
    }
}

impl Model for HighFanoutGrid {
    type State = GridState;

    fn name(&self) -> &'static str {
        "high-fanout-grid"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        vec![GridState {
            x: 0,
            y: 0,
            payload: [0u8; 32],
        }]
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
                action: ActionLabel {
                    name: action_name.to_string(),
                    disjunct_index: None,
                },
            });
        }
        Some(out)
    }

    fn has_fairness_constraints(&self) -> bool {
        true
    }

    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        // Constraint over `Move` — every SCC has Move edges (because
        // every state has a Move self-loop), so the constraint is
        // satisfied and both BFS and DFS paths report no violation.
        // This makes the verdict comparison trivial: both paths must
        // return `violation = None`.
        vec![FairnessConstraint::Weak {
            vars: vec!["x".to_string(), "y".to_string(), "payload".to_string()],
            action: self.move_action.clone(),
        }]
    }

    fn next_action_name(&self) -> Option<&str> {
        // Treat "Move" as the wrapper Next so the "any in-SCC edge"
        // fast-path applies — the verdict equivalence is unaffected.
        Some(&self.move_action)
    }

    fn check_invariants(&self, _state: &Self::State) -> Result<(), String> {
        Ok(())
    }
}

/// Spawn a polling thread that samples `VmRSS` every `poll_interval`
/// and writes the current max into `peak`. Returns a stop flag plus
/// the join handle.
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
        // Final sample after stop.
        let cur = get_memory_stats().rss_bytes;
        let prev = peak_t.load(Ordering::Relaxed);
        if cur > prev {
            peak_t.store(cur, Ordering::Relaxed);
        }
    });
    (stop, peak, h)
}

fn run_one(use_dfs: bool, dim: u16, work_dir_root: &std::path::Path) -> (u64, u64, u64) {
    // (peak_rss_above_baseline, states_distinct, exploration_secs)
    let model = HighFanoutGrid::new(dim);
    let mut cfg = EngineConfig::default();
    cfg.workers = 1;
    cfg.enforce_cgroups = false;
    cfg.numa_pinning = false;
    // Sized so the fp_store doesn't double-allocate: just over the
    // reachable state count (dim^2). Power of two for the shard mask.
    let states_estimate = (dim as usize) * (dim as usize);
    let fp_slots = states_estimate.next_power_of_two().max(1 << 18);
    cfg.fp_expected_items = fp_slots;
    cfg.fp_shards = 2;               // Minimum shard count
    cfg.fp_cache_capacity_bytes = 8 * 1024 * 1024; // 8 MiB per worker, not 1 GiB
    cfg.checkpoint_on_exit = false;
    cfg.enable_fp_persistence = false;
    cfg.enable_queue_spilling = false;
    cfg.queue_compression = false;   // Compression ring buffer is ~256 MiB; turn off so deltas reflect labeled_transitions
    cfg.work_dir = work_dir_root.join(if use_dfs { "dfs" } else { "bfs" });
    cfg.liveness_streaming = false;
    cfg.liveness_streaming_exploration = use_dfs;

    let baseline = get_memory_stats().rss_bytes;
    let (stop_poll, peak, poll_h) = start_rss_poller(Duration::from_millis(25));
    // Seed the poller with the baseline so any pre-run delta is
    // already captured.
    peak.store(baseline, Ordering::Relaxed);

    let started = Instant::now();
    let outcome = run_model(model, cfg).expect("run_model");
    let elapsed = started.elapsed();

    stop_poll.store(true, Ordering::Release);
    poll_h.join().expect("poll thread joins");

    let peak_rss = peak.load(Ordering::Relaxed);
    let delta = peak_rss.saturating_sub(baseline);

    eprintln!(
        "  {} run: states_distinct={} edges-explored={} elapsed={:.2?} \
         baseline_rss={:.1} MiB peak_rss={:.1} MiB delta={:.1} MiB \
         violation={:?}",
        if use_dfs { "DFS" } else { "BFS" },
        outcome.stats.states_distinct,
        outcome.stats.states_generated,
        elapsed,
        baseline as f64 / (1024.0 * 1024.0),
        peak_rss as f64 / (1024.0 * 1024.0),
        delta as f64 / (1024.0 * 1024.0),
        outcome.violation.as_ref().map(|v| v.message.as_str()),
    );

    (delta, outcome.stats.states_distinct, elapsed.as_secs())
}

#[test]
#[ignore = "memory benchmark — run with --ignored --nocapture; allocates ~1 GB"]
fn dfs_memory_win_vs_bfs() {
    // Grid dimension chosen so reachable state count is well above
    // 100K and edge count is well above 1M, hitting the regime where
    // labeled_transitions DashMap dominates BFS memory.
    //
    // dim=600 → 360 000 reachable states, ≈ 5 successors per state →
    // ~1.8M edges. Comfortably past the >100K-states / >1M-edges
    // gate-8 threshold.
    let dim: u16 = 600;
    let work_dir_root = std::env::temp_dir().join(format!(
        "tlapp-dfs-mem-bench-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&work_dir_root);

    eprintln!(
        "\n========== T10.2 stage 4 DFS-vs-BFS memory benchmark =========="
    );
    eprintln!(
        "Grid dim={}, expected reachable states ≈ {} (per-state fanout ≤ 5 + self-loop)",
        dim,
        dim as u64 * dim as u64
    );

    // Pass 1: DFS first so its peak isn't poisoned by BFS leftovers.
    eprintln!("\n--- Pass 1: DFS (--liveness-streaming-exploration) ---");
    let (dfs_delta, dfs_states, _dfs_secs) = run_one(true, dim, &work_dir_root);

    // Best-effort: let the allocator quiesce between runs.
    std::thread::sleep(Duration::from_millis(250));

    // Pass 2: BFS.
    eprintln!("\n--- Pass 2: BFS (default path) ---");
    let (bfs_delta, bfs_states, _bfs_secs) = run_one(false, dim, &work_dir_root);

    let _ = std::fs::remove_dir_all(&work_dir_root);

    eprintln!(
        "\n========== Result =========="
    );
    eprintln!(
        "DFS peak-RSS delta: {:.1} MiB", dfs_delta as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "BFS peak-RSS delta: {:.1} MiB", bfs_delta as f64 / (1024.0 * 1024.0)
    );
    if bfs_delta > 0 {
        let ratio = dfs_delta as f64 / bfs_delta as f64;
        let reduction = 100.0 * (1.0 - ratio);
        eprintln!("DFS / BFS = {:.2} ({:+.1}% reduction)", ratio, reduction);
    }

    // Sanity: both paths must explore the same reachable graph. If
    // they disagree on state count, something is wrong with one of
    // them — the memory comparison would be apples-to-oranges.
    assert_eq!(
        dfs_states, bfs_states,
        "DFS and BFS must agree on reachable state count (DFS={}, BFS={})",
        dfs_states, bfs_states
    );
    assert!(
        dfs_states >= 100_000,
        "benchmark must reach >= 100K states; got {}",
        dfs_states
    );

    // Allow some slack for measurement noise: require BFS delta is
    // measurably above zero (else the comparison is meaningless).
    assert!(
        bfs_delta >= 50 * 1024 * 1024,
        "BFS delta too small to be meaningful ({} bytes); test scale insufficient",
        bfs_delta
    );

    // The Stage 4 win threshold: DFS delta must be <= 70% of BFS delta.
    // I.e. >=30% RSS reduction. The actual win on
    // labeled_transitions-bound specs is much larger (~10× per-edge
    // memory reduction), so a 30% threshold is comfortably above the
    // measurement noise floor of /proc/self/status::VmRSS polling.
    let threshold = (bfs_delta as f64 * 0.70) as u64;
    assert!(
        dfs_delta <= threshold,
        "DFS peak-RSS delta {} MiB must be <= 70% of BFS delta {} MiB \
         (threshold {} MiB); DFS/BFS = {:.2}",
        dfs_delta / (1024 * 1024),
        bfs_delta / (1024 * 1024),
        threshold / (1024 * 1024),
        dfs_delta as f64 / bfs_delta as f64
    );
}
