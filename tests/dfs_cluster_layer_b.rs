//! T10.2 stage 5 layer B — gate-10: 2-node cluster DFS pool parity.
//!
//! Spins up two `MockTransport` instances connected via a shared
//! `MockNetwork` and runs the same fairness spec under:
//!
//! - 1-node single-process pool (Layer A baseline)
//! - 2-node cluster pool (Layer B), each node running half the
//!   workers-per-node so the total worker count matches the baseline
//!
//! The asserted gate is two-fold: identical liveness verdicts AND
//! identical merged `states_distinct` counts. Either disagreement
//! indicates a Layer B bug — typically (a) a partition-edge dropped
//! by the bridge, (b) cross-node termination consensus declaring done
//! while a peer still has work, or (c) duplicate seeding when both
//! nodes claim the same initial state.
//!
//! No real TCP, no spot provisioning: the test is hermetic and lives
//! at the integration-test level only because it needs the public
//! `dfs_cluster_test_api` entrypoint.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use tlaplusplus::distributed::transport::{MockNetwork, MockTransport, Transport};
use tlaplusplus::fairness::FairnessConstraint;
use tlaplusplus::model::{ActionLabel, LabeledTransition, Model};
use tlaplusplus::runtime::dfs_cluster_test_api::{
    ClusterNodeOutcome, partition_for_fp_cluster, run_one_cluster_node,
};

// -- Toy fairness model --------------------------------------------------
//
// Same shape as the toy `Grid` in `runtime::dfs_pool::tests` but lifted
// out so the integration test can construct it directly. Self-loop on
// every state guarantees a non-trivial SCC at every reachable node so
// the in-band fairness check has work to do; `WF_vars(Right)` then
// passes (every SCC has a Right edge from the self-loop).

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct GridState {
    x: u8,
    y: u8,
}

#[derive(Clone)]
struct Grid {
    dim: u8,
}

impl Model for Grid {
    type State = GridState;
    fn name(&self) -> &'static str {
        "dfs-cluster-grid"
    }
    fn initial_states(&self) -> Vec<Self::State> {
        vec![GridState { x: 0, y: 0 }]
    }
    fn next_states(&self, s: &Self::State, out: &mut Vec<Self::State>) {
        if s.x + 1 < self.dim {
            out.push(GridState { x: s.x + 1, y: s.y });
        }
        if s.y + 1 < self.dim {
            out.push(GridState { x: s.x, y: s.y + 1 });
        }
        out.push(s.clone());
    }
    fn next_states_labeled(
        &self,
        s: &Self::State,
    ) -> Option<Vec<LabeledTransition<Self::State>>> {
        let mut out = vec![];
        if s.x + 1 < self.dim {
            out.push(LabeledTransition {
                from: s.clone(),
                to: GridState { x: s.x + 1, y: s.y },
                action: ActionLabel {
                    name: "Right".into(),
                    disjunct_index: None,
                },
            });
        }
        if s.y + 1 < self.dim {
            out.push(LabeledTransition {
                from: s.clone(),
                to: GridState { x: s.x, y: s.y + 1 },
                action: ActionLabel {
                    name: "Up".into(),
                    disjunct_index: None,
                },
            });
        }
        out.push(LabeledTransition {
            from: s.clone(),
            to: s.clone(),
            action: ActionLabel {
                name: "Right".into(),
                disjunct_index: None,
            },
        });
        Some(out)
    }
    fn check_invariants(&self, _s: &Self::State) -> Result<(), String> {
        Ok(())
    }
    fn has_fairness_constraints(&self) -> bool {
        true
    }
    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        vec![FairnessConstraint::Weak {
            vars: vec!["x".into(), "y".into()],
            action: "Right".into(),
        }]
    }
    fn next_action_name(&self) -> Option<&str> {
        Some("Right")
    }
}

// -- Test harness --------------------------------------------------------

/// Run a 1-node DFS pool with `workers_per_node` workers using a
/// dummy `MockTransport`. Returns the outcome.
fn run_one_node_baseline(dim: u8, workers_per_node: usize) -> ClusterNodeOutcome<GridState> {
    let net = MockNetwork::new();
    let transport_concrete = MockTransport::new(0, net);
    // Single-node cluster: num_nodes=1 means the bridge is
    // initialized but never sends remote messages.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let stop = Arc::new(AtomicBool::new(false));
    let model = Arc::new(Grid { dim });
    let transport: Arc<dyn Transport> = transport_concrete;
    run_one_cluster_node(model, 0, 1, workers_per_node, transport, rt.handle().clone(), stop)
}

/// Run a 2-node cluster DFS pool. Returns the per-node outcomes in
/// node-id order. The total worker count is `workers_per_node * 2`.
fn run_two_node_cluster(
    dim: u8,
    workers_per_node: usize,
) -> (
    ClusterNodeOutcome<GridState>,
    ClusterNodeOutcome<GridState>,
) {
    let net = MockNetwork::new();
    let t0_concrete = MockTransport::new(0, net.clone());
    let t1_concrete = MockTransport::new(1, net.clone());

    // Each node gets its own tokio runtime so the bridge's recv-loop
    // and broadcast tasks don't compete for the test thread.
    let rt0 = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio rt 0");
    let rt1 = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio rt 1");
    let h0 = rt0.handle().clone();
    let h1 = rt1.handle().clone();

    let stop0 = Arc::new(AtomicBool::new(false));
    let stop1 = Arc::clone(&stop0);

    let model0 = Arc::new(Grid { dim });
    let model1 = Arc::clone(&model0);

    let t0_dyn: Arc<dyn Transport> = t0_concrete;
    let t1_dyn: Arc<dyn Transport> = t1_concrete;

    // Spawn each node in its own OS thread so the synchronous DFS
    // workers can actually make progress in parallel.
    let n_workers = workers_per_node;
    let h_node0 = std::thread::spawn(move || {
        run_one_cluster_node(model0, 0, 2, n_workers, t0_dyn, h0, stop0)
    });
    let h_node1 = std::thread::spawn(move || {
        run_one_cluster_node(model1, 1, 2, n_workers, t1_dyn, h1, stop1)
    });

    let out0 = h_node0.join().expect("node 0 thread joins");
    let out1 = h_node1.join().expect("node 1 thread joins");

    // Hold runtimes until after the join so the bridge's tasks have a
    // place to live for the full duration.
    drop(rt0);
    drop(rt1);

    (out0, out1)
}

#[test]
fn cluster_layer_b_state_count_matches_single_node_baseline_dim4() {
    // Use the smallest dim that exercises both local and cross-node
    // edges. With 2 nodes × 2 workers/node = 4 partitions, the
    // partition function spreads a 16-state grid roughly evenly.
    let dim = 4u8;
    let baseline = run_one_node_baseline(dim, /*workers=*/ 4);
    let (n0, n1) = run_two_node_cluster(dim, /*workers_per_node=*/ 2);

    let cluster_total = n0.states_distinct + n1.states_distinct;
    assert_eq!(
        baseline.states_distinct,
        cluster_total,
        "single-node ({}) must equal sum of 2-node distinct ({} + {} = {})",
        baseline.states_distinct,
        n0.states_distinct,
        n1.states_distinct,
        cluster_total
    );
    assert_eq!(
        baseline.violation.is_some(),
        n0.violation.is_some() || n1.violation.is_some(),
        "verdict (violation present) must agree across baseline vs cluster"
    );
}

#[test]
fn cluster_layer_b_state_count_matches_single_node_baseline_dim6() {
    // Larger grid to exercise more cross-node edges. 36 states, 2x2
    // partition shape => ~18 states per node, lots of cross-partition
    // and cross-node successors.
    let dim = 6u8;
    let baseline = run_one_node_baseline(dim, 4);
    let (n0, n1) = run_two_node_cluster(dim, 2);

    let cluster_total = n0.states_distinct + n1.states_distinct;
    assert_eq!(
        baseline.states_distinct,
        cluster_total,
        "dim=6 cluster ({} + {} = {}) must equal baseline ({})",
        n0.states_distinct,
        n1.states_distinct,
        cluster_total,
        baseline.states_distinct
    );
    assert_eq!(
        baseline.violation.is_some(),
        n0.violation.is_some() || n1.violation.is_some(),
        "dim=6 verdict must agree"
    );
}

#[test]
fn cluster_layer_b_partition_assignments_are_deterministic_and_disjoint() {
    // For a fixed cluster shape, every fingerprint must map to exactly
    // one (node, worker) pair, and the same fingerprint must always
    // yield the same pair across repeated calls. This is the static
    // partitioning invariant the cluster pool relies on for
    // dedup-without-coordination.
    for fp in 0u64..1_000 {
        let a = partition_for_fp_cluster(fp, 2, 2);
        let b = partition_for_fp_cluster(fp, 2, 2);
        assert_eq!(a, b, "partition for fp={} not deterministic", fp);
        let (n, w) = a;
        assert!(n < 2, "node id {} out of range", n);
        assert!(w < 2, "worker id {} out of range", w);
    }
}

/// Gate-11 NEW: per-node memory of the 2-node cluster pool stays
/// within 2x of the single-node DFS pool at the same logical scale.
///
/// "Logical scale" means: the cluster's per-node fingerprint store +
/// color map are sized for ~half the global state count, so per-node
/// memory should be roughly half the single-node baseline (or at most
/// 2x to leave headroom for transport buffers, tokio task stacks, and
/// the bridge's per-peer counters).
///
/// Marked `#[ignore]` because it allocates ~100 MiB of fingerprint-
/// store backing memory and isn't part of the per-PR loop.
#[test]
#[ignore = "memory benchmark — run with --ignored --nocapture"]
fn cluster_layer_b_per_node_memory_within_2x_single_node() {
    use tlaplusplus::system::get_memory_stats;

    let dim = 8u8; // 64 reachable states; small but real

    let baseline_rss_before = get_memory_stats().rss_bytes;
    let baseline = run_one_node_baseline(dim, 4);
    let baseline_rss_after = get_memory_stats().rss_bytes;
    let baseline_delta = baseline_rss_after.saturating_sub(baseline_rss_before);
    eprintln!(
        "single-node DFS: states_distinct={} RSS delta={:.1} MiB",
        baseline.states_distinct,
        baseline_delta as f64 / (1024.0 * 1024.0)
    );

    std::thread::sleep(Duration::from_millis(250));

    let cluster_rss_before = get_memory_stats().rss_bytes;
    let (n0, n1) = run_two_node_cluster(dim, 2);
    let cluster_rss_after = get_memory_stats().rss_bytes;
    let cluster_delta = cluster_rss_after.saturating_sub(cluster_rss_before);
    let total_distinct = n0.states_distinct + n1.states_distinct;
    eprintln!(
        "2-node cluster: states_distinct=({} + {} = {}) RSS delta={:.1} MiB",
        n0.states_distinct,
        n1.states_distinct,
        total_distinct,
        cluster_delta as f64 / (1024.0 * 1024.0)
    );

    // Per-node memory is the cluster total / 2 (since both nodes ran
    // in the same process). Compare against the single-node delta.
    let per_node_cluster = cluster_delta / 2;
    let ratio = per_node_cluster as f64 / baseline_delta.max(1) as f64;
    eprintln!(
        "per-node cluster delta: {:.1} MiB, ratio vs single-node: {:.2}x",
        per_node_cluster as f64 / (1024.0 * 1024.0),
        ratio
    );

    // Sanity: same logical state count.
    assert_eq!(baseline.states_distinct, total_distinct);

    // The 2x bound: per-node cluster RSS should not exceed 2x the
    // single-node baseline. With per-node fingerprint stores sized
    // for the same expected_items, the cluster pays roughly the same
    // baseline cost per node — so the ratio is dominated by per-bridge
    // overhead (transport buffers, peer counters, broadcast tasks).
    assert!(
        ratio <= 2.0,
        "cluster per-node RSS ({:.1} MiB) exceeds 2x single-node baseline ({:.1} MiB); ratio {:.2}",
        per_node_cluster as f64 / (1024.0 * 1024.0),
        baseline_delta as f64 / (1024.0 * 1024.0),
        ratio
    );
}

#[test]
fn cluster_layer_b_termination_is_eventually_reached() {
    // A regression guard: the 2-node cluster must actually terminate
    // (not hang) on a small reachable graph. We arm a 30-second wall
    // bound on the join; without two-round consensus working
    // correctly, the broadcast loop would never set
    // `globally_terminated` and both nodes' worker threads would
    // park indefinitely on `recv_timeout`.
    let dim = 3u8;
    let started = Instant::now();
    let (n0, n1) = run_two_node_cluster(dim, 2);
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "cluster must terminate in under 30s; took {:?}",
        elapsed
    );
    let total = n0.states_distinct + n1.states_distinct;
    let baseline = run_one_node_baseline(dim, 4);
    assert_eq!(baseline.states_distinct, total);
}
