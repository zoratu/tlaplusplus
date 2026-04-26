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

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
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

/// Default minimum local pending count below which we consider ourselves "low on work"
/// and eligible to steal from a peer.
pub const DEFAULT_STEAL_TRIGGER_LOW_WATER: u64 = 0;

/// Default minimum local pending count above which we are willing to be a steal victim.
/// If our queue is below this, we reply EMPTY to incoming steal requests so we don't
/// shed states we still need to process locally.
pub const DEFAULT_STEAL_VICTIM_THRESHOLD: u64 = 16_384;

/// Default batch size requested per cross-node steal.
pub const DEFAULT_STEAL_BATCH: u32 = 4096;

/// How long a steal request waits for a response before giving up and marking the peer down.
pub const DEFAULT_STEAL_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a peer stays marked down after a failed/timed-out steal.
pub const DEFAULT_PEER_DOWN_COOLDOWN: Duration = Duration::from_secs(30);

/// Wall time we must remain idle locally before initiating a remote steal.
/// (Well above local steal latency to avoid flapping.)
pub const DEFAULT_IDLE_BEFORE_STEAL: Duration = Duration::from_millis(250);

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

    // --- Cross-node steal trigger ---
    /// Nanoseconds since `started_at` of the last time local workers
    /// reported having work to do. Used by the steal trigger to compute
    /// how long the local node has been starved.
    /// Lock-free so the per-state hot path can update cheaply.
    last_local_work_ns: AtomicU64,
    /// Steal requests this node currently has in flight.
    /// `all_nodes_idle()` will refuse to declare termination while > 0.
    pending_steal_requests: AtomicU32,
    /// Per-peer "down until" timestamps. A peer marked down is skipped by
    /// the steal trigger until the timestamp passes. Stored as nanoseconds
    /// since `started_at`.
    peer_down_until_ns: Vec<AtomicU64>,
    /// Reference instant for `peer_down_until_ns`.
    started_at: Instant,
    /// Configurable steal-victim threshold (states must be queued before we donate).
    steal_victim_threshold: AtomicU64,
    /// Configurable batch size per cross-node steal request.
    steal_batch: AtomicU32,
    /// Configurable wall time we must be idle before triggering a remote steal.
    idle_before_steal_ms: AtomicU64,
    /// Configurable peer-down cooldown after a failed steal.
    peer_down_cooldown_ms: AtomicU64,

    // --- Stats ---
    pub states_stolen: AtomicU64,
    pub states_donated: AtomicU64,
    pub bloom_dedup_hits: AtomicU64,
    pub bloom_exchanges: AtomicU64,
    pub steal_requests_sent: AtomicU64,
    pub steal_requests_failed: AtomicU64,
    pub steal_responses_empty: AtomicU64,
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
        let mut peer_down_until_ns = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            peer_idle.push(AtomicBool::new(false));
            peer_down_until_ns.push(AtomicU64::new(0));
        }

        let started_at = Instant::now();
        DistributedWorkStealer {
            node_id,
            num_nodes,
            transport,
            tokio_handle,
            local_bloom,
            remote_bloom,
            bloom_exchange_lock: std::sync::Mutex::new(()),
            last_bloom_exchange: RwLock::new(started_at),
            bloom_interval: DEFAULT_BLOOM_INTERVAL,
            peer_idle,
            locally_idle: AtomicBool::new(false),
            globally_terminated: AtomicBool::new(false),
            last_work_received: RwLock::new(started_at),
            // Initialize to "just had work" so we don't immediately start
            // stealing during cluster spin-up before initial states are loaded.
            last_local_work_ns: AtomicU64::new(0),
            pending_steal_requests: AtomicU32::new(0),
            peer_down_until_ns,
            started_at,
            steal_victim_threshold: AtomicU64::new(DEFAULT_STEAL_VICTIM_THRESHOLD),
            steal_batch: AtomicU32::new(DEFAULT_STEAL_BATCH),
            idle_before_steal_ms: AtomicU64::new(DEFAULT_IDLE_BEFORE_STEAL.as_millis() as u64),
            peer_down_cooldown_ms: AtomicU64::new(DEFAULT_PEER_DOWN_COOLDOWN.as_millis() as u64),
            states_stolen: AtomicU64::new(0),
            states_donated: AtomicU64::new(0),
            bloom_dedup_hits: AtomicU64::new(0),
            bloom_exchanges: AtomicU64::new(0),
            steal_requests_sent: AtomicU64::new(0),
            steal_requests_failed: AtomicU64::new(0),
            steal_responses_empty: AtomicU64::new(0),
        }
    }

    /// Number of nodes in the cluster (self + peers).
    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
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

    /// Is this node currently locally idle?
    /// Used by the termination broadcaster — peers consume this to compute
    /// their own `all_nodes_idle()`. Earlier versions broadcast the global
    /// view, which is circular and prevents convergence.
    pub fn is_locally_idle(&self) -> bool {
        self.locally_idle.load(Ordering::Acquire)
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
    ///
    /// Cross-node steal extension: also returns false if this node has any
    /// in-flight `StealRequest` outstanding. Otherwise we could declare
    /// global termination while a `StealResponse` is in transit, dropping
    /// states.
    pub fn all_nodes_idle(&self) -> bool {
        if !self.locally_idle.load(Ordering::Acquire) {
            return false;
        }
        // No outstanding steal requests of our own.
        if self.pending_steal_requests.load(Ordering::Acquire) > 0 {
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

    // --- Cross-node steal trigger (T6) ---

    /// Mark that local workers have made progress (have/produced work).
    /// Called from the runtime whenever workers process states or push successors.
    /// Lock-free; safe to call on the per-state hot path.
    #[inline]
    pub fn note_local_work(&self) {
        let now_ns = self.started_at.elapsed().as_nanos() as u64;
        self.last_local_work_ns.store(now_ns, Ordering::Relaxed);
    }

    /// Time elapsed since the last call to `note_local_work`.
    pub fn idle_duration(&self) -> Duration {
        let last_ns = self.last_local_work_ns.load(Ordering::Relaxed);
        let now_ns = self.started_at.elapsed().as_nanos() as u64;
        Duration::from_nanos(now_ns.saturating_sub(last_ns))
    }

    /// Should this node consider stealing from a remote peer?
    ///
    /// Returns true when:
    /// - The cluster has more than one node
    /// - Local pending count is below `low_water` (default 0 — we're empty)
    /// - We've been idle locally for at least `idle_before_steal_ms`
    /// - We are not globally terminated
    /// - There is at least one peer that is not currently marked down
    pub fn should_initiate_steal(&self, local_pending: u64) -> bool {
        if self.num_nodes <= 1 {
            return false;
        }
        if self.is_globally_terminated() {
            return false;
        }
        if local_pending > DEFAULT_STEAL_TRIGGER_LOW_WATER {
            return false;
        }
        let idle_ms = self.idle_before_steal_ms.load(Ordering::Relaxed);
        if self.idle_duration() < Duration::from_millis(idle_ms) {
            return false;
        }
        self.has_live_peer()
    }

    /// Are we willing to act as a steal victim right now?
    /// We refuse if we have less than `steal_victim_threshold` queued locally,
    /// to avoid shedding states we still need to crunch.
    pub fn can_donate(&self, local_pending: u64) -> bool {
        local_pending >= self.steal_victim_threshold.load(Ordering::Relaxed)
    }

    /// Per-request batch size to ask for in a remote steal.
    pub fn steal_batch_size(&self) -> u32 {
        self.steal_batch.load(Ordering::Relaxed)
    }

    /// Set steal-victim threshold (states must exceed this before we donate).
    pub fn set_steal_victim_threshold(&self, n: u64) {
        self.steal_victim_threshold.store(n, Ordering::Relaxed);
    }

    /// Set per-request batch size for outgoing steal requests.
    pub fn set_steal_batch_size(&self, n: u32) {
        self.steal_batch.store(n, Ordering::Relaxed);
    }

    /// Set how long we must be idle before triggering a remote steal.
    pub fn set_idle_before_steal(&self, d: Duration) {
        self.idle_before_steal_ms
            .store(d.as_millis() as u64, Ordering::Relaxed);
    }

    /// Mark a peer as down for `peer_down_cooldown_ms` after a failed steal.
    pub fn mark_peer_down(&self, peer_id: u32) {
        let cooldown_ms = self.peer_down_cooldown_ms.load(Ordering::Relaxed);
        let until_ns =
            self.started_at.elapsed().as_nanos() as u64 + (cooldown_ms * 1_000_000).max(1);
        if let Some(slot) = self.peer_down_until_ns.get(peer_id as usize) {
            slot.store(until_ns, Ordering::Release);
        }
        self.steal_requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Is the named peer currently in the down-cooldown window?
    pub fn is_peer_down(&self, peer_id: u32) -> bool {
        let now_ns = self.started_at.elapsed().as_nanos() as u64;
        match self.peer_down_until_ns.get(peer_id as usize) {
            Some(slot) => slot.load(Ordering::Acquire) > now_ns,
            None => false,
        }
    }

    /// Are any peers (other than self) currently live (not down)?
    pub fn has_live_peer(&self) -> bool {
        for i in 0..self.num_nodes {
            if i == self.node_id {
                continue;
            }
            if !self.is_peer_down(i) {
                return true;
            }
        }
        false
    }

    /// List live (not down) peer ids, rotated by a clock-derived offset so
    /// that two simultaneous starvers don't both target the same peer first.
    /// Avoids pulling in a `rand` dep just for a steal-target shuffle.
    pub fn live_peers_shuffled(&self) -> Vec<u32> {
        let peers: Vec<u32> = (0..self.num_nodes)
            .filter(|&i| i != self.node_id && !self.is_peer_down(i))
            .collect();
        if peers.len() <= 1 {
            return peers;
        }
        // Rotate by (clock_ns ^ node_id) % len. Cheap and good enough for
        // load-spreading; we don't need cryptographic randomness here.
        let now_ns = self.started_at.elapsed().as_nanos() as u64;
        let rot = ((now_ns ^ (self.node_id as u64).wrapping_mul(0x9E3779B97F4A7C15)) as usize)
            % peers.len();
        let mut rotated = Vec::with_capacity(peers.len());
        rotated.extend_from_slice(&peers[rot..]);
        rotated.extend_from_slice(&peers[..rot]);
        rotated
    }

    /// Increment the pending-steal counter. Called when sending a `StealRequest`.
    pub fn begin_steal(&self) {
        self.pending_steal_requests.fetch_add(1, Ordering::AcqRel);
        self.steal_requests_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the pending-steal counter. Called when a `StealResponse`
    /// arrives or the request is abandoned (timeout / transport error).
    pub fn end_steal(&self) {
        self.pending_steal_requests.fetch_sub(1, Ordering::AcqRel);
    }

    /// Number of cross-node steal requests this node has issued and not yet completed.
    pub fn pending_steal_count(&self) -> u32 {
        self.pending_steal_requests.load(Ordering::Acquire)
    }

    /// Record that an empty steal response was received (peer had no work to share).
    pub fn note_empty_steal_response(&self) {
        self.steal_responses_empty.fetch_add(1, Ordering::Relaxed);
    }

    /// Print diagnostic stats.
    pub fn print_stats(&self) {
        println!(
            "[cluster] node {} stats: stolen={}, donated={}, bloom_dedup={}, bloom_exchanges={}, \
             steal_req_sent={}, steal_req_failed={}, steal_resp_empty={}",
            self.node_id,
            self.states_stolen.load(Ordering::Relaxed),
            self.states_donated.load(Ordering::Relaxed),
            self.bloom_dedup_hits.load(Ordering::Relaxed),
            self.bloom_exchanges.load(Ordering::Relaxed),
            self.steal_requests_sent.load(Ordering::Relaxed),
            self.steal_requests_failed.load(Ordering::Relaxed),
            self.steal_responses_empty.load(Ordering::Relaxed),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_transport() -> Arc<ClusterTransport> {
        // For unit tests we don't actually exercise the network.
        // Build a transport bound to a random localhost port.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let cfg = super::super::ClusterConfig {
                node_id: 0,
                listen_addr: "127.0.0.1:0".parse().unwrap(),
                peers: vec![],
            };
            ClusterTransport::new(cfg).await.unwrap()
        })
    }

    #[test]
    fn pending_steal_blocks_termination() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let transport = dummy_transport();
        let stealer =
            DistributedWorkStealer::new(0, 2, Arc::clone(&transport), rt.handle().clone());

        stealer.set_locally_idle(true);
        stealer.set_peer_idle(1, true);

        // Force grace period to elapse by rewinding last_work_received.
        *stealer.last_work_received.write().unwrap() = Instant::now() - Duration::from_secs(10);

        assert!(stealer.all_nodes_idle(), "should be terminated baseline");

        stealer.begin_steal();
        assert!(
            !stealer.all_nodes_idle(),
            "in-flight steal must block termination"
        );
        stealer.end_steal();
        assert!(stealer.all_nodes_idle(), "steal completed re-enables term");
    }

    #[test]
    fn peer_down_cooldown_skips_peer() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let transport = dummy_transport();
        let stealer =
            DistributedWorkStealer::new(0, 3, Arc::clone(&transport), rt.handle().clone());

        assert!(stealer.has_live_peer());
        stealer.mark_peer_down(1);
        assert!(stealer.is_peer_down(1));
        assert!(!stealer.is_peer_down(2));
        let live = stealer.live_peers_shuffled();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0], 2);

        // After cooldown elapses, peer should be back up.
        // Set cooldown to 1ms and rewind via short sleep.
        stealer.peer_down_cooldown_ms.store(1, Ordering::Relaxed);
        stealer.mark_peer_down(1);
        std::thread::sleep(Duration::from_millis(5));
        assert!(!stealer.is_peer_down(1));
    }

    #[test]
    fn should_initiate_steal_requires_idle_window() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let transport = dummy_transport();
        let stealer =
            DistributedWorkStealer::new(0, 2, Arc::clone(&transport), rt.handle().clone());

        // Just-noted local work: should NOT initiate.
        stealer.note_local_work();
        assert!(!stealer.should_initiate_steal(0));

        // Force idle window to elapse by stamping last_local_work_ns to 0
        // (start-of-time relative to the stealer).
        stealer.last_local_work_ns.store(0, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(1));
        // We need at least idle_before_steal elapsed; default is 250ms.
        // Override to 1ms for the test.
        stealer.set_idle_before_steal(Duration::from_millis(1));
        assert!(stealer.should_initiate_steal(0));

        // Non-zero pending count blocks the trigger.
        assert!(!stealer.should_initiate_steal(1));
    }

    #[test]
    fn can_donate_respects_threshold() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let transport = dummy_transport();
        let stealer =
            DistributedWorkStealer::new(0, 2, Arc::clone(&transport), rt.handle().clone());

        stealer.set_steal_victim_threshold(1000);
        assert!(!stealer.can_donate(500));
        assert!(stealer.can_donate(1000));
        assert!(stealer.can_donate(5000));
    }

    #[test]
    fn single_node_never_steals() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let transport = dummy_transport();
        let stealer =
            DistributedWorkStealer::new(0, 1, Arc::clone(&transport), rt.handle().clone());

        stealer.last_local_work_ns.store(0, Ordering::Relaxed);
        stealer.set_idle_before_steal(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(2));
        assert!(!stealer.should_initiate_steal(0));
    }
}
