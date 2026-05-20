//! T10.2 stage 5 layer B — multi-node cluster bridge for the DFS pool.
//!
//! Layer A (`dfs_pool.rs`) gives every fingerprint exactly one home worker
//! within a single process; cross-partition successors flow over a
//! crossbeam mpsc. Layer B extends that to span multiple cluster nodes:
//! the partition function returns `(node_id, worker_id)` and a successor
//! whose home is on a peer node is shipped via `Transport` as a
//! `PartitionEdge` message. The receiving node's bridge converts it back
//! into the in-process `ExploreMsg` format and pushes it onto the right
//! local worker's inbox.
//!
//! # Two-round termination consensus
//!
//! Each node tracks two cumulative counters:
//!
//! - `remote_sent_total`   — every `PartitionEdge` this node sent
//! - `remote_received_total` — every `PartitionEdge` this node received
//!   (and dispatched to a local worker inbox)
//!
//! The cluster is globally idle exactly when, for every node N:
//!
//!   `sum_over_nodes(remote_sent_total)  ==
//!    sum_over_nodes(remote_received_total)`     [no in-flight messages]
//!
//! AND every node reports its local pool idle.
//!
//! We approximate this with a Mattern-style two-round token:
//!
//! - **Round 1**: each peer reports `(local_idle, sent, received)` via the
//!   periodic `TerminationToken` with the `inflight_partition_edges`
//!   field encoding `sent - received`.
//! - **Round 2 (re-confirmation)**: after receiving round-1 tokens
//!   showing `sum(diffs) == 0` AND `all_idle`, the bridge demands ONE
//!   more round of fresh tokens with the same predicate before flipping
//!   `globally_terminated`.
//!
//! This catches the race where a peer becomes briefly idle, sends a
//! token, then immediately receives a remote `PartitionEdge` and
//! re-activates: the second round's tokens will show non-zero diffs and
//! consensus restarts.
//!
//! # Async-to-sync bridge
//!
//! `Transport::recv()` is async (returns `Pin<Box<dyn Future>>`); the
//! DFS workers are sync threads with crossbeam mpsc inboxes. The bridge
//! owns a tokio runtime handle and spawns a task that:
//!
//! 1. Awaits `Transport::recv()` in a loop.
//! 2. On `PartitionEdge`, decodes the state blob, looks up the local
//!    worker by `dst_worker`, and calls `inbox_tx[dst_worker].send(msg)`
//!    (sync — but never blocks long because the inbox is unbounded).
//! 3. On `TerminationToken`, updates per-peer state for the consensus.
//! 4. On `Stop`, flips the shared `stop` flag.
//! 5. Bumps `remote_received_total` AFTER the inbox `send` succeeds.
//!
//! The recv loop also periodically broadcasts THIS node's own
//! `TerminationToken` so peers can advance their consensus.
//!
//! # Scope vs the design doc's full Layer B
//!
//! The design doc describes additional infrastructure (cross-partition
//! red-DFS via `RedDfsProbe`/`RedDfsResponse`, lazy `RequestStateBlob`
//! for trace reconstruction). We wire the *receive* arms for those so
//! the wire shape stays stable, but the *send* path stays log-only:
//! Layer B's verdict is produced by the existing in-band Tarjan that
//! the pool already runs after exploration completes — it just needs
//! the merged adjacency list, which the bridge ensures is consistent
//! across nodes by reporting per-node triples back to the coordinator.
//!
//! For Layer B's gate-10 deliverable we keep the verdict path
//! single-coordinator: when the cluster terminates, all nodes ship
//! their per-worker `WorkerOutput` to node 0, which runs the same
//! `run_inband_fairness_check` predicate the single-node pool uses.
//! Equivalence with the single-node DFS pool is then by construction.

use crate::distributed::protocol::{
    Message, PartitionEdgeOutcome, RedDfsOutcome,
};
use crate::distributed::transport::Transport;
use anyhow::Result;
use crossbeam_channel::Sender;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

/// Static cluster partition assignment. Same bit-mixing the in-process
/// `partition_for_fp` uses, but spread over `num_nodes * workers_per_node`
/// total partitions and decomposed into `(node_id, worker_id)`.
///
/// Cfg-split (T13.4 Phase 2): under `--features verus` the bit-mix
/// + modulo for `global` delegates to `compute_numa_index_from_hash`
/// (verified `ensures result < total`). The decompose into
/// `(node_id, worker_id)` stays inline — verifying
/// `(global / workers_per_node) < num_nodes` needs nontrivial
/// integer-division reasoning that's a separate slice.
#[cfg(not(feature = "verus"))]
#[inline]
pub(super) fn partition_for_fp_cluster(
    fp: u64,
    num_nodes: u32,
    workers_per_node: usize,
) -> (u32, usize) {
    debug_assert!(num_nodes > 0, "num_nodes must be positive");
    debug_assert!(workers_per_node > 0, "workers_per_node must be positive");
    let total = (num_nodes as usize) * workers_per_node;
    if total == 1 {
        return (0, 0);
    }
    let mixed = (fp >> 32) ^ (fp >> 16) ^ fp;
    let global = (mixed as usize) % total;
    let node_id = (global / workers_per_node) as u32;
    let worker_id = global % workers_per_node;
    (node_id, worker_id)
}

#[cfg(feature = "verus")]
#[inline]
pub(super) fn partition_for_fp_cluster(
    fp: u64,
    num_nodes: u32,
    workers_per_node: usize,
) -> (u32, usize) {
    debug_assert!(num_nodes > 0, "num_nodes must be positive");
    debug_assert!(workers_per_node > 0, "workers_per_node must be positive");
    let total = (num_nodes as usize) * workers_per_node;
    if total == 1 {
        return (0, 0);
    }
    let global = crate::storage::verus_smoke::compute_numa_index_from_hash(fp, total);
    let node_id = (global / workers_per_node) as u32;
    let worker_id = crate::storage::verus_smoke::compute_index_mod(global, workers_per_node);
    (node_id, worker_id)
}

/// Wire payload for a cross-node successor. We piggyback the
/// predecessor fp + action name string inside the `state_blob` so the
/// receiver's `LocalAdjacency` records the edge with the same
/// fingerprint pair / action name string the in-process path would
/// produce. Keeps the wire shape (the protocol's `PartitionEdge`
/// variant) unchanged.
#[derive(Serialize, Deserialize)]
pub(super) struct WirePayload<S> {
    pub state: S,
    pub from_fp: u64,
    pub action: String,
    pub passes_constraints: bool,
}

/// The cluster bridge. One per node; shared across all of that node's
/// DFS pool workers via `Arc`.
pub(super) struct ClusterBridge<S>
where
    S: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned,
{
    /// Underlying transport (real or mock).
    pub(super) transport: Arc<dyn Transport>,
    pub(super) node_id: u32,
    pub(super) num_nodes: u32,
    /// Retained for diagnostic logging and for the future
    /// per-(node, worker) inflight tracking; not currently consumed
    /// by the consensus predicate (which sums across nodes only).
    #[allow(dead_code)]
    pub(super) workers_per_node: usize,
    /// Sync inbox for each local worker. Indexed by local worker id.
    pub(super) inbox_tx: Vec<Sender<super::dfs_pool::ExploreMsg<S>>>,
    /// Cumulative count of `PartitionEdge` messages this node has sent.
    pub(super) remote_sent_total: Arc<AtomicU64>,
    /// Cumulative count of `PartitionEdge` messages this node has
    /// received and dispatched to a local worker inbox.
    pub(super) remote_received_total: Arc<AtomicU64>,
    /// Per-peer cumulative `remote_sent_total` as last reported by
    /// their `TerminationToken`. Indexed by node id; the local slot
    /// (`peer_sent[self.node_id]`) is updated on every broadcast tick
    /// from this node's own counter so the consensus loop can read a
    /// single uniform array.
    pub(super) peer_sent: Arc<Vec<AtomicI64>>,
    /// Per-peer cumulative `remote_received_total` as last reported.
    /// Encoded as `i64` in the array but always non-negative.
    pub(super) peer_received: Arc<Vec<AtomicI64>>,
    /// Per-peer "is the peer locally idle?" bit, as last reported.
    pub(super) peer_idle: Arc<Vec<AtomicBool>>,
    /// Number of consecutive token rounds the consensus predicate has
    /// held (`sum(diffs) == 0` AND every peer + self idle). When this
    /// hits `2`, `globally_terminated` is set.
    pub(super) consensus_rounds: Arc<AtomicU64>,
    /// Set to `true` once two-round consensus succeeds. Both the local
    /// pool's termination check and the recv loop respect this.
    pub(super) globally_terminated: Arc<AtomicBool>,
    /// Whether THIS node is currently locally idle (every worker idle,
    /// inboxes empty, blue-stack empty). Updated by the pool.
    pub(super) self_idle: Arc<AtomicBool>,
    /// Shared stop flag — flipped on safety violation, cluster stop, or
    /// global termination.
    pub(super) stop: Arc<AtomicBool>,
    /// Shared in-process Mattern counter that the local pool's
    /// workers maintain (pool's `inflight` AtomicI64). When the bridge
    /// pushes a received `PartitionEdge` onto a local worker's inbox,
    /// it `fetch_add(1)`s here too so the worker's
    /// `handle_explore_msg`'s `fetch_sub(1)` balances exactly. Without
    /// this mirror the in-process counter would drift negative on
    /// every cross-node receive.
    pub(super) local_inflight: Arc<AtomicI64>,
    /// Tokio handle used to spawn the recv loop and the periodic
    /// broadcast task. Held even when no real network is in use so
    /// the same shape works for `MockTransport` tests.
    pub(super) tokio_handle: tokio::runtime::Handle,
    /// Period for the periodic `TerminationToken` broadcast.
    pub(super) broadcast_interval: Duration,
}

impl<S> ClusterBridge<S>
where
    S: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned,
{
    pub(super) fn new(
        transport: Arc<dyn Transport>,
        node_id: u32,
        num_nodes: u32,
        workers_per_node: usize,
        inbox_tx: Vec<Sender<super::dfs_pool::ExploreMsg<S>>>,
        stop: Arc<AtomicBool>,
        tokio_handle: tokio::runtime::Handle,
        local_inflight: Arc<AtomicI64>,
    ) -> Arc<Self> {
        debug_assert_eq!(inbox_tx.len(), workers_per_node);
        let peer_sent: Vec<AtomicI64> =
            (0..num_nodes).map(|_| AtomicI64::new(0)).collect();
        let peer_received: Vec<AtomicI64> =
            (0..num_nodes).map(|_| AtomicI64::new(0)).collect();
        let peer_idle: Vec<AtomicBool> =
            (0..num_nodes).map(|_| AtomicBool::new(false)).collect();

        Arc::new(ClusterBridge {
            transport,
            node_id,
            num_nodes,
            workers_per_node,
            inbox_tx,
            remote_sent_total: Arc::new(AtomicU64::new(0)),
            remote_received_total: Arc::new(AtomicU64::new(0)),
            peer_sent: Arc::new(peer_sent),
            peer_received: Arc::new(peer_received),
            peer_idle: Arc::new(peer_idle),
            consensus_rounds: Arc::new(AtomicU64::new(0)),
            globally_terminated: Arc::new(AtomicBool::new(false)),
            self_idle: Arc::new(AtomicBool::new(false)),
            stop,
            local_inflight,
            tokio_handle,
            broadcast_interval: Duration::from_millis(20),
        })
    }

    /// Send a successor whose home is on a peer node. Increments
    /// `remote_sent_total` BEFORE the network send so the consensus
    /// predicate can never observe a "received but not yet sent" state.
    /// Returns `Ok(())` when the message has been handed off to the
    /// transport; the receiver's bridge will eventually push it to a
    /// local worker inbox.
    pub(super) fn send_remote(
        &self,
        dst_node: u32,
        dst_worker: usize,
        state: S,
        fp: u64,
        from_fp: u64,
        action: String,
        passes_constraints: bool,
        sender_depth: u32,
    ) -> Result<()> {
        debug_assert_ne!(dst_node, self.node_id, "send_remote called with self node");
        let payload = WirePayload {
            state,
            from_fp,
            action,
            passes_constraints,
        };
        let blob = bincode::serialize(&payload)
            .map_err(|e| anyhow::anyhow!("bincode serialize PartitionEdge payload: {e}"))?;
        // Increment sent counter BEFORE the send so a peer who racks
        // up our token for "received" can't see "received > sent".
        self.remote_sent_total.fetch_add(1, Ordering::AcqRel);
        let msg = Message::PartitionEdge {
            from_node: self.node_id,
            from_worker: 0, // local-worker-id-on-sender; receiver doesn't use it
            owner_node: dst_node,
            owner_worker: dst_worker as u32,
            state_blob: blob,
            state_fp: fp,
            via_action: u16::MAX, // action name lives in the blob
            sender_depth,
        };
        // Hand the actual `Transport::send` off to a tokio task so we
        // never block the sync worker thread on async I/O. The send
        // is fire-and-forget from the worker's perspective; the
        // remote_sent_total counter has already been bumped above so
        // the termination consensus is correct regardless of when the
        // network catches up.
        let transport = Arc::clone(&self.transport);
        let sent_total = Arc::clone(&self.remote_sent_total);
        let self_node = self.node_id;
        self.tokio_handle.spawn(async move {
            if let Err(e) = transport.send(dst_node, &msg).await {
                // Failed send — roll back the counter so consensus can
                // still converge even when a peer is unreachable.
                sent_total.fetch_sub(1, Ordering::AcqRel);
                eprintln!(
                    "[cluster-bridge n={}] PartitionEdge send to node {} failed: {}",
                    self_node, dst_node, e
                );
            }
        });
        Ok(())
    }

    /// Mark whether this node is locally idle (called by the pool
    /// when the local workers all reach their idle predicate).
    pub(super) fn set_self_idle(&self, idle: bool) {
        let prev = self.self_idle.swap(idle, Ordering::AcqRel);
        // Mirror into our own slot in `peer_idle` so the predicate
        // reads a consistent uniform array. Without this mirror, a
        // recv-loop tick that fires between the worker setting
        // `self_idle` and the broadcast tick that copies into the
        // slot would observe `false` for self even when the worker
        // is in fact idle.
        self.peer_idle[self.node_id as usize].store(idle, Ordering::Release);
        if prev != idle {
            // Reset the consensus when our own idle flips — protects
            // the "two consecutive rounds at zero" invariant against
            // the case where we say idle, peer agrees, then a remote
            // edge lands in our inbox and we re-activate.
            self.consensus_rounds.store(0, Ordering::Release);
        }
    }

    pub(super) fn is_globally_terminated(&self) -> bool {
        self.globally_terminated.load(Ordering::Acquire)
    }

    /// Predicate used by the local pool's termination check: every
    /// peer (and self) is idle AND the cluster-wide
    /// `sum(remote_sent_total) == sum(remote_received_total)`.
    ///
    /// The local node's own counters are written into the
    /// `peer_sent[self.node_id]` and `peer_received[self.node_id]`
    /// slots on every broadcast tick, so a single uniform sum across
    /// the array is correct.
    pub(super) fn cluster_quiescent_round1(&self) -> bool {
        if !self.self_idle.load(Ordering::Acquire) {
            return false;
        }
        let mut sum_sent: i64 = 0;
        let mut sum_received: i64 = 0;
        for id in 0..self.num_nodes as usize {
            if !self.peer_idle[id].load(Ordering::Acquire) {
                return false;
            }
            sum_sent = sum_sent.saturating_add(self.peer_sent[id].load(Ordering::Acquire));
            sum_received =
                sum_received.saturating_add(self.peer_received[id].load(Ordering::Acquire));
        }
        sum_sent == sum_received
    }

    /// Spawn the recv loop + periodic broadcast on the tokio runtime.
    /// Returns immediately; the spawned tasks live until `stop` flips
    /// or the transport closes.
    pub(super) fn start(self: Arc<Self>) {
        let recv_self = Arc::clone(&self);
        self.tokio_handle.spawn(async move {
            recv_self.recv_loop().await;
        });

        let bcast_self = Arc::clone(&self);
        self.tokio_handle.spawn(async move {
            bcast_self.broadcast_loop().await;
        });
    }

    /// Drain the transport, route messages to the right local worker
    /// inbox or update consensus state.
    async fn recv_loop(self: Arc<Self>) {
        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!("[cluster-bridge n={}] recv loop started", self.node_id);
        }
        loop {
            if self.stop.load(Ordering::Acquire) {
                return;
            }
            let recv = self.transport.recv().await;
            let Some((from, msg)) = recv else {
                // Transport closed; nothing more to do.
                return;
            };
            match msg {
                Message::PartitionEdge {
                    owner_worker,
                    state_blob,
                    state_fp,
                    ..
                } => {
                    // Decode wire payload back into an ExploreMsg.
                    let payload: WirePayload<S> = match bincode::deserialize(&state_blob) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!(
                                "[cluster-bridge n={}] PartitionEdge decode failed: {}",
                                self.node_id, e
                            );
                            continue;
                        }
                    };
                    let dst_worker = owner_worker as usize;
                    if dst_worker >= self.inbox_tx.len() {
                        eprintln!(
                            "[cluster-bridge n={}] PartitionEdge dst_worker {} out of range \
                             (workers_per_node={}); dropping",
                            self.node_id,
                            dst_worker,
                            self.inbox_tx.len()
                        );
                        continue;
                    }
                    let explore_msg = super::dfs_pool::ExploreMsg {
                        state: payload.state,
                        fp: state_fp,
                        from_fp: payload.from_fp,
                        action: payload.action,
                        passes_constraints: payload.passes_constraints,
                    };
                    // Handing off to the local inbox: bump remote_received
                    // BEFORE the send so a peer can't observe "sent >
                    // received" relative to consensus once we've actually
                    // accepted the message.
                    self.remote_received_total
                        .fetch_add(1, Ordering::AcqRel);
                    // Re-activate the local node: we've just dropped a
                    // message into a worker's inbox, so the cluster
                    // is *not* quiescent. Without this reset, the
                    // sliding-window race [worker says idle ; remote
                    // edge arrives ; bridge token still says idle]
                    // can let consensus advance prematurely.
                    self.self_idle.store(false, Ordering::Release);
                    self.peer_idle[self.node_id as usize]
                        .store(false, Ordering::Release);
                    self.consensus_rounds.store(0, Ordering::Release);
                    // Mirror the in-process Mattern counter so the
                    // worker's `handle_explore_msg`'s
                    // `inflight.fetch_sub(1)` balances exactly. Without
                    // this, processing N cross-node messages would
                    // drift `inflight` to -N and the local termination
                    // predicate (inflight == 0) would never fire again.
                    self.local_inflight.fetch_add(1, Ordering::AcqRel);
                    if let Err(_) = self.inbox_tx[dst_worker].send(explore_msg) {
                        // Worker shut down before we could deliver;
                        // roll back the receive counter and let the
                        // sender's pending edge time out via the
                        // global-stop signal.
                        self.remote_received_total
                            .fetch_sub(1, Ordering::AcqRel);
                        self.local_inflight.fetch_sub(1, Ordering::AcqRel);
                    }
                    // Reply with an ack — informational; the sender's
                    // Mattern counter is the actual termination signal.
                    let ack = Message::PartitionEdgeAck {
                        from_node: self.node_id,
                        owner_node: from,
                        state_fp,
                        outcome: PartitionEdgeOutcome::ClaimedFresh,
                    };
                    let transport = Arc::clone(&self.transport);
                    tokio::spawn(async move {
                        let _ = transport.send(from, &ack).await;
                    });
                }
                Message::PartitionEdgeAck { .. } => {
                    // Layer B uses the cumulative sent/received counters,
                    // not per-message acks, for termination consensus.
                    // The ack is accepted on the wire (no warning) but
                    // intentionally ignored here.
                }
                Message::TerminationToken {
                    initiator,
                    round: _,
                    all_idle,
                    inflight_partition_edges,
                    inflight_red_probes,
                } => {
                    if (initiator as usize) < self.peer_idle.len() {
                        self.peer_idle[initiator as usize]
                            .store(all_idle, Ordering::Release);
                        if let Some(sent) = inflight_partition_edges {
                            self.peer_sent[initiator as usize]
                                .store(sent as i64, Ordering::Release);
                        }
                        if let Some(received) = inflight_red_probes {
                            self.peer_received[initiator as usize]
                                .store(received as i64, Ordering::Release);
                        }
                    }
                    // Update consensus: count rounds where the predicate
                    // holds globally.
                    if self.cluster_quiescent_round1() {
                        let prev = self.consensus_rounds.fetch_add(1, Ordering::AcqRel);
                        if prev + 1 >= 2 {
                            self.globally_terminated
                                .store(true, Ordering::Release);
                        }
                    } else {
                        self.consensus_rounds.store(0, Ordering::Release);
                    }
                }
                Message::Stop { node_id, message } => {
                    eprintln!(
                        "[cluster-bridge n={}] Stop from node {}: {}",
                        self.node_id, node_id, message
                    );
                    self.stop.store(true, Ordering::Release);
                }
                Message::RedDfsProbe { from_node, seed_fp, .. } => {
                    // Layer B's verdict path is the post-exploration
                    // in-band Tarjan; cross-partition red-DFS is
                    // intentionally a no-op here. Reply NotFound so a
                    // future probe sender can advance.
                    let resp = Message::RedDfsResponse {
                        from_node: self.node_id,
                        seed_fp,
                        outcome: RedDfsOutcome::NotFound,
                        trail_extension: Vec::new(),
                    };
                    let transport = Arc::clone(&self.transport);
                    tokio::spawn(async move {
                        let _ = transport.send(from_node, &resp).await;
                    });
                }
                Message::RedDfsResponse { .. } => {}
                Message::RequestStateBlob {
                    from_node, fps, ..
                } => {
                    // Trace reconstruction is out of scope for Layer B;
                    // reply with empty blobs so the requester moves on.
                    let blobs = vec![Vec::<u8>::new(); fps.len()];
                    let resp = Message::StateBlobResponse {
                        from_node: self.node_id,
                        owner_node: from_node,
                        fps,
                        blobs,
                    };
                    let transport = Arc::clone(&self.transport);
                    tokio::spawn(async move {
                        let _ = transport.send(from_node, &resp).await;
                    });
                }
                Message::StateBlobResponse { .. } => {}
                // Independent-exploration messages — cluster bridge
                // doesn't own these. They're harmlessly ignored when
                // the bridge is the only consumer of the transport.
                Message::StealRequest { .. }
                | Message::StealResponse { .. }
                | Message::BloomExchange { .. }
                | Message::Heartbeat { .. }
                | Message::Join { .. }
                | Message::Leave { .. } => {}
            }
        }
    }

    /// Periodically broadcast THIS node's `TerminationToken` so peers
    /// can advance their consensus.
    async fn broadcast_loop(self: Arc<Self>) {
        let mut round = 0u64;
        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "[cluster-bridge n={}] broadcast loop started (interval={:?})",
                self.node_id, self.broadcast_interval
            );
        }
        loop {
            if self.stop.load(Ordering::Acquire)
                || self.globally_terminated.load(Ordering::Acquire)
            {
                return;
            }
            tokio::time::sleep(self.broadcast_interval).await;
            let sent = self.remote_sent_total.load(Ordering::Acquire);
            let received = self.remote_received_total.load(Ordering::Acquire);
            // Reuse the existing protocol fields:
            //   inflight_partition_edges -> remote_sent_total
            //   inflight_red_probes      -> remote_received_total
            // Both are cumulative u64 counters. The receiving bridge's
            // recv loop maps them into peer_sent[]/peer_received[].
            let token = Message::TerminationToken {
                initiator: self.node_id,
                round,
                all_idle: self.self_idle.load(Ordering::Acquire),
                inflight_partition_edges: Some(sent),
                inflight_red_probes: Some(received),
            };
            // Track our own idle + counters in our own slots so
            // cluster_quiescent_round1() includes a consistent self
            // contribution.
            self.peer_idle[self.node_id as usize]
                .store(self.self_idle.load(Ordering::Acquire), Ordering::Release);
            self.peer_sent[self.node_id as usize]
                .store(sent as i64, Ordering::Release);
            self.peer_received[self.node_id as usize]
                .store(received as i64, Ordering::Release);
            for peer_id in 0..self.num_nodes {
                if peer_id == self.node_id {
                    continue;
                }
                if let Err(e) = self.transport.send(peer_id, &token).await {
                    eprintln!(
                        "[cluster-bridge n={}] termination-token send to node {} failed: {}",
                        self.node_id, peer_id, e
                    );
                }
            }
            // Self-tick consensus on every broadcast: even if no peer
            // sends us a token (because they have nothing to say), as
            // long as our own predicate holds AND every recorded peer
            // bit/diff matches, we should still advance.
            let q = self.cluster_quiescent_round1();
            if q {
                let prev = self.consensus_rounds.fetch_add(1, Ordering::AcqRel);
                if prev + 1 >= 2 {
                    self.globally_terminated.store(true, Ordering::Release);
                }
            } else {
                self.consensus_rounds.store(0, Ordering::Release);
            }
            // Diagnostic dump every few rounds, opt-in via TLAPP_VERBOSE
            // so the chaos / soak runs aren't drowned in token traffic.
            if std::env::var("TLAPP_VERBOSE").is_ok() && round % 50 == 0 {
                let peer_idle_dump: Vec<bool> = self
                    .peer_idle
                    .iter()
                    .map(|f| f.load(Ordering::Acquire))
                    .collect();
                let peer_sent_dump: Vec<i64> = self
                    .peer_sent
                    .iter()
                    .map(|d| d.load(Ordering::Acquire))
                    .collect();
                let peer_recv_dump: Vec<i64> = self
                    .peer_received
                    .iter()
                    .map(|d| d.load(Ordering::Acquire))
                    .collect();
                eprintln!(
                    "[cluster-bridge n={}] round={} self_idle={} \
                     peers idle={:?} sent={:?} recv={:?} quiescent={} rounds={} terminated={}",
                    self.node_id,
                    round,
                    self.self_idle.load(Ordering::Acquire),
                    peer_idle_dump,
                    peer_sent_dump,
                    peer_recv_dump,
                    q,
                    self.consensus_rounds.load(Ordering::Acquire),
                    self.globally_terminated.load(Ordering::Acquire),
                );
            }
            round += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_for_fp_cluster_distributes_across_nodes() {
        let mut node_counts = [0u32; 2];
        let mut worker_counts = [0u32; 4]; // 2 workers/node × 2 nodes
        for fp in 0u64..10_000 {
            let (n, w) = partition_for_fp_cluster(fp, 2, 2);
            node_counts[n as usize] += 1;
            worker_counts[(n as usize) * 2 + w] += 1;
        }
        // Each node should get roughly 5K (within 25%).
        for c in node_counts.iter() {
            assert!(
                (*c as i64 - 5000).unsigned_abs() <= 1250,
                "per-node skew too large: {:?}",
                node_counts
            );
        }
        // Each (node, worker) pair should get roughly 2.5K.
        for c in worker_counts.iter() {
            assert!(
                (*c as i64 - 2500).unsigned_abs() <= 800,
                "per-(node,worker) skew too large: {:?}",
                worker_counts
            );
        }
    }

    #[test]
    fn partition_for_fp_cluster_single_node_matches_in_process() {
        // With num_nodes=1, the cluster partition should agree with the
        // in-process function for the same total worker count.
        for fp in 0u64..1_000 {
            let (n, w) = partition_for_fp_cluster(fp, 1, 4);
            assert_eq!(n, 0);
            assert_eq!(w, super::super::dfs_pool::partition_for_fp(fp, 4));
        }
    }

    #[test]
    fn partition_for_fp_cluster_single_total_partition() {
        for fp in 0u64..100 {
            let (n, w) = partition_for_fp_cluster(fp, 1, 1);
            assert_eq!((n, w), (0, 0));
        }
    }
}
