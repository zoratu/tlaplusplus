use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender, TrySendError};

use super::batch::{BatchAccumulator, ReadyBatch};
use super::protocol::Message;
use super::ring::PartitionRing;
use super::transport::ClusterTransport;

/// Inbound state entry received from a remote node.
/// Contains the fingerprint and the zstd-compressed bincode-serialized state bytes.
pub struct InboundState {
    pub fingerprint: u64,
    pub compressed_state: Vec<u8>,
}

/// Distributed fingerprint proxy that intercepts fingerprint checks and routes
/// them to the correct cluster node based on the partition ring.
///
/// Local fingerprints go through the normal `contains_or_insert_batch_with_affinity`
/// path. Remote fingerprints are batched and sent over TCP to the owning node.
///
/// This struct is `Send + Sync` and designed to be shared across worker threads
/// via `Arc<DistributedFingerprintProxy>`.
pub struct DistributedFingerprintProxy {
    node_id: u32,
    ring: PartitionRing,
    transport: Arc<ClusterTransport>,
    /// Per-worker batch accumulators, keyed by worker_id.
    /// Each worker gets its own accumulator to avoid cross-worker contention.
    batch_accumulators: Vec<Mutex<BatchAccumulator>>,
    /// Channel for receiving states from remote nodes that were found to be new.
    inbound_states_tx: Sender<InboundState>,
    inbound_states_rx: Receiver<InboundState>,
    /// Tokio runtime handle for spawning async send tasks from sync worker threads.
    tokio_handle: tokio::runtime::Handle,
    /// Counter for total remote sends (for diagnostics).
    remote_sends: AtomicU64,
    /// Counter for total inbound states received (for diagnostics).
    inbound_received: AtomicU64,
    /// Whether all outbound batches have been flushed (for termination detection).
    all_flushed: AtomicBool,
    /// Per-peer idle flags for distributed termination detection.
    /// Index = peer node_id. True means that peer reported itself as locally idle.
    peer_idle: Vec<AtomicBool>,
    /// Whether this node is locally idle (all workers idle + all local queues empty).
    locally_idle: AtomicBool,
    /// Termination flag: set when all nodes agree on global termination.
    globally_terminated: AtomicBool,
}

impl DistributedFingerprintProxy {
    /// Create a new distributed fingerprint proxy.
    ///
    /// `num_workers` determines how many per-worker batch accumulators to create.
    /// `num_nodes` is the total number of nodes in the cluster (for termination tracking).
    pub fn new(
        node_id: u32,
        ring: PartitionRing,
        transport: Arc<ClusterTransport>,
        tokio_handle: tokio::runtime::Handle,
        num_workers: usize,
        num_nodes: u32,
        batch_size: usize,
        batch_timeout_ms: u64,
    ) -> Self {
        // Bounded channel: 64K inbound states buffered before backpressure
        let (inbound_states_tx, inbound_states_rx) = crossbeam_channel::bounded(65_536);

        let mut batch_accumulators = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            batch_accumulators.push(Mutex::new(BatchAccumulator::new(
                batch_size,
                batch_timeout_ms,
            )));
        }

        let mut peer_idle = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            peer_idle.push(AtomicBool::new(false));
        }

        DistributedFingerprintProxy {
            node_id,
            ring,
            transport,
            batch_accumulators,
            inbound_states_tx,
            inbound_states_rx,
            tokio_handle,
            remote_sends: AtomicU64::new(0),
            inbound_received: AtomicU64::new(0),
            all_flushed: AtomicBool::new(true),
            peer_idle,
            locally_idle: AtomicBool::new(false),
            globally_terminated: AtomicBool::new(false),
        }
    }

    /// Check if a fingerprint belongs to this node.
    #[inline]
    pub fn is_local(&self, fp: u64) -> bool {
        self.ring.home_node(fp) == self.node_id
    }

    /// Return the home node for a fingerprint.
    #[inline]
    pub fn home_node(&self, fp: u64) -> u32 {
        self.ring.home_node(fp)
    }

    /// This node's ID.
    pub fn node_id(&self) -> u32 {
        self.node_id
    }

    /// Send a (fingerprint, compressed_state) pair to the appropriate remote node.
    ///
    /// The entry is buffered in a per-worker batch accumulator. When the batch
    /// reaches the configured size, it is flushed and sent asynchronously over TCP.
    ///
    /// `worker_id` selects the per-worker accumulator to avoid cross-worker lock
    /// contention.
    pub fn send_remote(&self, worker_id: usize, fp: u64, compressed_state: Vec<u8>) {
        let dest_node = self.ring.home_node(fp);
        debug_assert_ne!(
            dest_node, self.node_id,
            "send_remote called for local fingerprint"
        );

        self.all_flushed.store(false, Ordering::Release);

        let acc_idx = worker_id % self.batch_accumulators.len();
        let ready = {
            let mut acc = self.batch_accumulators[acc_idx]
                .lock()
                .expect("batch accumulator lock poisoned");
            acc.push(dest_node, fp, compressed_state)
        };

        if let Some(batch) = ready {
            self.send_batch(batch);
        }
    }

    /// Flush expired batches from a specific worker's accumulator.
    /// Called periodically by workers to ensure partial batches are sent.
    pub fn flush_expired(&self, worker_id: usize) {
        let acc_idx = worker_id % self.batch_accumulators.len();
        let batches = {
            let mut acc = self.batch_accumulators[acc_idx]
                .lock()
                .expect("batch accumulator lock poisoned");
            acc.flush_expired()
        };
        for batch in batches {
            self.send_batch(batch);
        }
    }

    /// Flush all pending batches across all workers. Called during shutdown
    /// or termination detection.
    pub fn flush_all(&self) {
        for acc_mutex in &self.batch_accumulators {
            let batches = {
                let mut acc = acc_mutex.lock().expect("batch accumulator lock poisoned");
                acc.flush_all()
            };
            for batch in batches {
                self.send_batch(batch);
            }
        }
        self.all_flushed.store(true, Ordering::Release);
    }

    /// Total number of entries currently buffered across all workers.
    pub fn pending_count(&self) -> usize {
        self.batch_accumulators
            .iter()
            .map(|acc| {
                acc.lock()
                    .expect("batch accumulator lock poisoned")
                    .pending_count()
            })
            .sum()
    }

    /// Drain inbound states that have been checked by remote nodes and found new.
    /// Returns up to `limit` states. Non-blocking.
    pub fn drain_inbound(&self, limit: usize) -> Vec<InboundState> {
        let mut result = Vec::with_capacity(limit.min(256));
        for _ in 0..limit {
            match self.inbound_states_rx.try_recv() {
                Ok(state) => result.push(state),
                Err(_) => break,
            }
        }
        result
    }

    /// Enqueue an inbound state (called by the inbound message handler when
    /// a remote node sends us a state that we need to explore).
    pub fn enqueue_inbound(&self, fp: u64, compressed_state: Vec<u8>) {
        self.inbound_received.fetch_add(1, Ordering::Relaxed);
        let state = InboundState {
            fingerprint: fp,
            compressed_state,
        };
        // Try non-blocking send; if full, block briefly — this applies
        // backpressure to the inbound handler.
        match self.inbound_states_tx.try_send(state) {
            Ok(()) => {}
            Err(TrySendError::Full(state)) => {
                // Channel full — block until space is available
                let _ = self.inbound_states_tx.send(state);
            }
            Err(TrySendError::Disconnected(_)) => {
                // Receiver dropped — proxy is shutting down
            }
        }
    }

    /// Get the inbound channel sender (for the handler to clone).
    pub fn inbound_sender(&self) -> Sender<InboundState> {
        self.inbound_states_tx.clone()
    }

    /// Reference to the transport for direct message sending.
    pub fn transport(&self) -> &Arc<ClusterTransport> {
        &self.transport
    }

    /// Tokio runtime handle for spawning async tasks.
    pub fn tokio_handle(&self) -> tokio::runtime::Handle {
        self.tokio_handle.clone()
    }

    // --- Distributed termination detection ---

    /// Mark this node as locally idle or active.
    pub fn set_locally_idle(&self, idle: bool) {
        self.locally_idle.store(idle, Ordering::Release);
    }

    /// Record that a peer reported itself as idle.
    pub fn set_peer_idle(&self, peer_node_id: u32, idle: bool) {
        if let Some(flag) = self.peer_idle.get(peer_node_id as usize) {
            flag.store(idle, Ordering::Release);
        }
    }

    /// Check if global termination has been detected.
    pub fn is_globally_terminated(&self) -> bool {
        self.globally_terminated.load(Ordering::Acquire)
    }

    /// Mark global termination.
    pub fn set_globally_terminated(&self) {
        self.globally_terminated.store(true, Ordering::Release);
    }

    /// Check if all nodes (self + all peers) are idle AND all outbound
    /// batches have been flushed. This is a necessary condition for
    /// distributed termination.
    pub fn all_nodes_idle(&self) -> bool {
        if !self.locally_idle.load(Ordering::Acquire) {
            return false;
        }
        if !self.all_flushed.load(Ordering::Acquire) {
            return false;
        }
        if self.pending_count() > 0 {
            return false;
        }
        // Check all peers
        for (i, flag) in self.peer_idle.iter().enumerate() {
            if i as u32 == self.node_id {
                continue; // skip self
            }
            if !flag.load(Ordering::Acquire) {
                return false;
            }
        }
        true
    }

    /// Diagnostics: total remote batches sent.
    pub fn remote_sends(&self) -> u64 {
        self.remote_sends.load(Ordering::Relaxed)
    }

    /// Diagnostics: total inbound states received.
    pub fn inbound_received(&self) -> u64 {
        self.inbound_received.load(Ordering::Relaxed)
    }

    /// Send a ready batch over the transport asynchronously.
    fn send_batch(&self, batch: ReadyBatch) {
        self.remote_sends.fetch_add(1, Ordering::Relaxed);
        let transport = Arc::clone(&self.transport);
        let from_node = self.node_id;
        let dest_node = batch.dest_node;
        let batch_id = batch.batch_id;
        let entries = batch.entries;

        self.tokio_handle.spawn(async move {
            let msg = Message::FingerprintBatch {
                from_node,
                batch_id,
                entries,
            };
            if let Err(e) = transport.send(dest_node, &msg).await {
                eprintln!(
                    "[cluster] failed to send fingerprint batch {} to node {}: {}",
                    batch_id, dest_node, e
                );
            }
        });
    }
}
