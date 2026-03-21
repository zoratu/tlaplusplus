//! Independent-exploration distributed work stealer.
//!
//! Each node runs a fully independent model checker with its own FP store and
//! work queue. The network is ONLY used for:
//!
//! 1. **Work stealing** — when a node's queue empties, steal batches from peers
//! 2. **Bloom filter exchange** — periodically share compact summaries of explored
//!    states to reduce duplicate work across nodes
//!
//! This replaces the fingerprint-partitioning approach (proxy.rs) which sent 2/3
//! of successor states over the network on every expansion, making the hot path
//! network-bound.
//!
//! Key principle: **ZERO network on the hot path**. Workers call `next_states()`,
//! check the LOCAL FP store, enqueue to the LOCAL queue — exactly like single-node
//! mode. Network only fires on two triggers: empty queue (steal) and timer (bloom).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::bloom::BloomFilter;
use super::protocol::Message;
use super::transport::ClusterTransport;

/// Default bloom exchange interval.
const DEFAULT_BLOOM_INTERVAL: Duration = Duration::from_secs(5);

/// Default bloom filter expected items (10M at 1% FPR = ~12MB).
const DEFAULT_BLOOM_EXPECTED_ITEMS: usize = 10_000_000;

/// Default bloom filter false positive rate.
const DEFAULT_BLOOM_FPR: f64 = 0.01;

/// Distributed work stealer for independent-exploration model checking.
///
/// Each node runs a fully independent model checker. This struct provides:
/// - `should_skip_state(fp)`: check remote bloom filters for dedup
/// - `record_explored(fp)`: record a newly explored state in the local bloom
/// - `maybe_exchange_bloom()`: periodically broadcast local bloom and merge remote ones
///
/// Work stealing is handled via channels: the inbound handler serves StealRequests
/// from a donation channel, and pushes StealResponse states to a stolen-work channel.
///
/// All methods are safe to call from multiple worker threads (`Send + Sync`).
pub struct DistributedWorkStealer {
    node_id: u32,
    num_nodes: u32,
    transport: Arc<ClusterTransport>,
    tokio_handle: tokio::runtime::Handle,

    /// Local bloom filter — records fingerprints explored by this node.
    /// Workers call `record_explored(fp)` after inserting into the local FP store.
    local_bloom: BloomFilter,

    /// Merged bloom filters from all remote nodes.
    /// Updated periodically via `exchange_bloom()`.
    remote_bloom: BloomFilter,

    /// Lock protecting bloom exchange serialization (prevents concurrent exchanges).
    bloom_exchange_lock: std::sync::Mutex<()>,

    /// Last time a bloom exchange was performed.
    last_bloom_exchange: RwLock<Instant>,

    /// Interval between bloom exchanges.
    bloom_interval: Duration,

    // --- Termination detection ---
    /// Per-peer idle flags for distributed termination detection.
    peer_idle: Vec<AtomicBool>,
    /// Whether this node is locally idle.
    locally_idle: AtomicBool,
    /// Global termination flag.
    globally_terminated: AtomicBool,
    /// Timestamp of last received work (for termination grace period).
    last_work_received: RwLock<Instant>,

    // --- Stats ---
    pub states_stolen: AtomicU64,
    pub states_donated: AtomicU64,
    pub bloom_dedup_hits: AtomicU64,
    pub bloom_exchanges: AtomicU64,
}

impl DistributedWorkStealer {
    /// Create a new distributed work stealer.
    pub fn new(
        node_id: u32,
        num_nodes: u32,
        transport: Arc<ClusterTransport>,
        tokio_handle: tokio::runtime::Handle,
    ) -> Self {
        // Create bloom filters — local and remote use same parameters
        let local_bloom = BloomFilter::new(DEFAULT_BLOOM_EXPECTED_ITEMS, DEFAULT_BLOOM_FPR);
        let remote_bloom = BloomFilter::new(DEFAULT_BLOOM_EXPECTED_ITEMS, DEFAULT_BLOOM_FPR);

        let mut peer_idle = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            peer_idle.push(AtomicBool::new(false));
        }

        DistributedWorkStealer {
            node_id,
            num_nodes,
            transport,
            tokio_handle,
            local_bloom,
            remote_bloom,
            bloom_exchange_lock: std::sync::Mutex::new(()),
            last_bloom_exchange: RwLock::new(Instant::now()),
            bloom_interval: DEFAULT_BLOOM_INTERVAL,
            peer_idle,
            locally_idle: AtomicBool::new(false),
            globally_terminated: AtomicBool::new(false),
            last_work_received: RwLock::new(Instant::now()),
            states_stolen: AtomicU64::new(0),
            states_donated: AtomicU64::new(0),
            bloom_dedup_hits: AtomicU64::new(0),
            bloom_exchanges: AtomicU64::new(0),
        }
    }

    /// This node's ID.
    pub fn node_id(&self) -> u32 {
        self.node_id
    }

    /// Record that a fingerprint has been explored by this node.
    ///
    /// Called after inserting into the local FP store. This populates the
    /// bloom filter that will be exchanged with peers.
    #[inline]
    pub fn record_explored(&self, fp: u64) {
        self.local_bloom.insert(fp);
    }

    /// Check if ANY remote bloom filter says this state was already explored.
    ///
    /// If `true`, the state has PROBABLY been explored by another node and
    /// can be skipped (probabilistic — may have false positives at the
    /// configured FPR, but never false negatives).
    ///
    /// Called in the worker loop BEFORE `next_states()`.
    #[inline]
    pub fn should_skip_state(&self, fp: u64) -> bool {
        if self.remote_bloom.may_contain(fp) {
            self.bloom_dedup_hits.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Check if it's time for a bloom exchange, and if so, perform one.
    ///
    /// Called periodically from the bloom exchange background task.
    pub fn maybe_exchange_bloom(&self) {
        let elapsed = {
            let last = self.last_bloom_exchange.read().unwrap();
            last.elapsed()
        };
        if elapsed < self.bloom_interval {
            return;
        }

        // Take the exchange lock to prevent concurrent exchanges
        let _lock = match self.bloom_exchange_lock.try_lock() {
            Ok(guard) => guard,
            Err(_) => return, // Another thread is already exchanging
        };

        // Double-check after acquiring lock
        let elapsed = {
            let last = self.last_bloom_exchange.read().unwrap();
            last.elapsed()
        };
        if elapsed < self.bloom_interval {
            return;
        }

        // Snapshot local bloom and broadcast
        let snapshot = self.local_bloom.snapshot();
        let bloom_bytes = match bincode::serialize(&snapshot) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("[cluster] failed to serialize bloom filter: {}", e);
                return;
            }
        };

        let msg = Message::BloomExchange {
            from_node: self.node_id,
            bloom_data: bloom_bytes,
        };

        let transport = Arc::clone(&self.transport);
        self.tokio_handle.spawn(async move {
            if let Err(e) = transport.broadcast(&msg).await {
                eprintln!("[cluster] failed to broadcast bloom filter: {}", e);
            }
        });

        *self.last_bloom_exchange.write().unwrap() = Instant::now();
        self.bloom_exchanges.fetch_add(1, Ordering::Relaxed);
    }

    /// Merge a received bloom snapshot from a remote node.
    pub fn merge_remote_bloom(&self, bloom_data: &[u8]) {
        let snapshot: super::bloom::BloomSnapshot = match bincode::deserialize(bloom_data) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[cluster] failed to deserialize remote bloom: {}", e);
                return;
            }
        };
        self.remote_bloom.merge_snapshot(&snapshot);
    }

    /// Note that work was received (resets termination grace timer).
    pub fn note_work_received(&self) {
        *self.last_work_received.write().unwrap() = Instant::now();
    }

    /// Reference to the transport (for handler spawning).
    pub fn transport(&self) -> &Arc<ClusterTransport> {
        &self.transport
    }

    /// Tokio runtime handle.
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

    /// Check if all nodes (self + all peers) are idle AND no recent work
    /// has been received (2-second grace period).
    pub fn all_nodes_idle(&self) -> bool {
        if !self.locally_idle.load(Ordering::Acquire) {
            return false;
        }
        // Grace period: require 2 seconds since last received work
        {
            let last = self.last_work_received.read().unwrap();
            if last.elapsed() < Duration::from_secs(2) {
                return false;
            }
        }
        // Check all peers
        for (i, flag) in self.peer_idle.iter().enumerate() {
            if i as u32 == self.node_id {
                continue;
            }
            if !flag.load(Ordering::Acquire) {
                return false;
            }
        }
        true
    }

    /// Print diagnostic stats.
    pub fn print_stats(&self) {
        println!(
            "[cluster] node {} stats: stolen={}, donated={}, bloom_dedup={}, bloom_exchanges={}",
            self.node_id,
            self.states_stolen.load(Ordering::Relaxed),
            self.states_donated.load(Ordering::Relaxed),
            self.bloom_dedup_hits.load(Ordering::Relaxed),
            self.bloom_exchanges.load(Ordering::Relaxed),
        );
    }
}
