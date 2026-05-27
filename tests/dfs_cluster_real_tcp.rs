//! T10.2 stage 5 layer B — real-TCP loopback cluster DFS pool.
//!
//! Companion to `tests/dfs_cluster_layer_b.rs`. That test runs the
//! multi-node DFS pool over `MockTransport` (in-process channels); this
//! one runs it over the production `ClusterTransport` on two real TCP
//! sockets bound to `127.0.0.1`. It exercises the framing /
//! serialization / accept / connect / recv-loop path that the mock
//! skips entirely — exactly the path that was never tested before a
//! 4-node spot run surfaced the scale-out defects.
//!
//! What this DOES guard: the real-TCP transport wiring stays correct —
//! partition edges serialize/deserialize and route over sockets, the
//! two-round termination consensus converges over real I/O, and the
//! cluster produces a DISJOINT partition (sum of per-node distinct ==
//! single-node baseline) with a matching verdict.
//!
//! What this does NOT guard: cross-process fingerprint determinism.
//! Both "nodes" run in this one test process, so they share ahash's
//! per-process seed — fingerprints would agree here even if the seed
//! were random. That property is guarded separately and directly by
//! `src/model.rs`'s `fingerprint_determinism_tests` (a hardcoded
//! constant captured from a separate process). Together the two tests
//! cover the bug that a 4-node spot run found: non-deterministic
//! `Model::fingerprint` made ownership diverge across processes, so
//! every node re-explored most of the graph.

use serde::{Deserialize, Serialize};
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use tlaplusplus::distributed::ClusterConfig;
use tlaplusplus::distributed::transport::{
    ClusterTransport, MockNetwork, MockTransport, Transport,
};
use tlaplusplus::fairness::FairnessConstraint;
use tlaplusplus::model::{ActionLabel, LabeledTransition, Model};
use tlaplusplus::runtime::dfs_cluster_test_api::{ClusterNodeOutcome, run_one_cluster_node};

// -- Toy fairness model (identical to the MockTransport sibling test) ----

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
        "dfs-cluster-grid-real-tcp"
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

// -- Harness -------------------------------------------------------------

/// Grab a currently-free `127.0.0.1` TCP port by binding to :0 and
/// reading the assigned port, then releasing it. Small TOCTOU window
/// before `ClusterTransport` re-binds, acceptable for a hermetic test.
fn free_loopback_addr() -> SocketAddr {
    let l = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = l.local_addr().expect("local_addr");
    drop(l);
    addr
}

/// Single-node baseline over MockTransport — gives the canonical
/// distinct-state count the cluster must match.
fn run_one_node_baseline(dim: u8, workers: usize) -> ClusterNodeOutcome<GridState> {
    let net = MockNetwork::new();
    let transport: Arc<dyn Transport> = MockTransport::new(0, net);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let stop = Arc::new(AtomicBool::new(false));
    let model = Arc::new(Grid { dim });
    run_one_cluster_node(model, 0, 1, workers, transport, rt.handle().clone(), stop)
}

/// Run a 2-node cluster over real loopback TCP. Returns per-node
/// outcomes in node-id order.
fn run_two_node_cluster_real_tcp(
    dim: u8,
    workers_per_node: usize,
) -> (ClusterNodeOutcome<GridState>, ClusterNodeOutcome<GridState>) {
    let addr0 = free_loopback_addr();
    let addr1 = free_loopback_addr();
    assert_ne!(addr0, addr1, "distinct loopback ports");

    let cfg0 = ClusterConfig {
        node_id: 0,
        listen_addr: addr0,
        peers: vec![addr1],
    };
    let cfg1 = ClusterConfig {
        node_id: 1,
        listen_addr: addr1,
        peers: vec![addr0],
    };

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

    // Both nodes must be LISTENING before either connects out, so build
    // both transports first, then connect.
    let t0 = rt0
        .block_on(async { ClusterTransport::new(cfg0).await })
        .expect("node 0 listener binds");
    let t1 = rt1
        .block_on(async { ClusterTransport::new(cfg1).await })
        .expect("node 1 listener binds");

    rt0.block_on(async { t0.connect_to_peers().await })
        .expect("node 0 connects to peer");
    rt1.block_on(async { t1.connect_to_peers().await })
        .expect("node 1 connects to peer");

    let h0 = rt0.handle().clone();
    let h1 = rt1.handle().clone();

    let stop0 = Arc::new(AtomicBool::new(false));
    let stop1 = Arc::clone(&stop0);

    let model0 = Arc::new(Grid { dim });
    let model1 = Arc::clone(&model0);

    let t0_dyn: Arc<dyn Transport> = t0;
    let t1_dyn: Arc<dyn Transport> = t1;

    let n = workers_per_node;
    let j0 = std::thread::spawn(move || {
        run_one_cluster_node(model0, 0, 2, n, t0_dyn, h0, stop0)
    });
    let j1 = std::thread::spawn(move || {
        run_one_cluster_node(model1, 1, 2, n, t1_dyn, h1, stop1)
    });

    let out0 = j0.join().expect("node 0 thread joins");
    let out1 = j1.join().expect("node 1 thread joins");

    // Keep the runtimes (acceptor + bridge tasks live there) alive until
    // both node threads have finished.
    drop(rt0);
    drop(rt1);

    (out0, out1)
}

#[test]
fn real_tcp_cluster_matches_single_node_baseline_dim4() {
    let dim = 4u8;
    let started = Instant::now();
    let baseline = run_one_node_baseline(dim, 4);
    let (n0, n1) = run_two_node_cluster_real_tcp(dim, 2);
    assert!(
        started.elapsed() < Duration::from_secs(60),
        "real-TCP cluster must terminate well under 60s; took {:?}",
        started.elapsed()
    );

    let cluster_total = n0.states_distinct + n1.states_distinct;
    assert_eq!(
        baseline.states_distinct, cluster_total,
        "real-TCP 2-node cluster ({} + {} = {}) must equal single-node baseline ({}) \
         — a mismatch means dropped/duplicated partition edges or divergent ownership",
        n0.states_distinct, n1.states_distinct, cluster_total, baseline.states_distinct
    );
    // Real work distribution: neither node should explore the whole
    // graph. With a 4-partition shape over 16 states each node owns a
    // strict subset. (This is the property that was broken across
    // processes before the fingerprint fix; here it must hold.)
    assert!(
        n0.states_distinct > 0 && n1.states_distinct > 0,
        "both nodes must do real work: n0={} n1={}",
        n0.states_distinct,
        n1.states_distinct
    );
    assert!(
        n0.states_distinct < baseline.states_distinct
            && n1.states_distinct < baseline.states_distinct,
        "neither node should explore the whole graph (no partitioning): \
         n0={} n1={} baseline={}",
        n0.states_distinct,
        n1.states_distinct,
        baseline.states_distinct
    );
    assert_eq!(
        baseline.violation.is_some(),
        n0.violation.is_some() || n1.violation.is_some(),
        "verdict must agree across baseline vs real-TCP cluster"
    );
}

#[test]
fn real_tcp_cluster_matches_single_node_baseline_dim6() {
    let dim = 6u8;
    let baseline = run_one_node_baseline(dim, 4);
    let (n0, n1) = run_two_node_cluster_real_tcp(dim, 2);

    let cluster_total = n0.states_distinct + n1.states_distinct;
    assert_eq!(
        baseline.states_distinct, cluster_total,
        "dim=6 real-TCP cluster ({} + {} = {}) must equal baseline ({})",
        n0.states_distinct, n1.states_distinct, cluster_total, baseline.states_distinct
    );
    assert!(
        n0.states_distinct < baseline.states_distinct
            && n1.states_distinct < baseline.states_distinct,
        "dim=6: neither node should explore the whole graph: n0={} n1={} baseline={}",
        n0.states_distinct,
        n1.states_distinct,
        baseline.states_distinct
    );
    assert_eq!(
        baseline.violation.is_some(),
        n0.violation.is_some() || n1.violation.is_some(),
        "dim=6 verdict must agree"
    );
}
