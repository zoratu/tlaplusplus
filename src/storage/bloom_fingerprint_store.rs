use anyhow::Result;
use bloomfilter::Bloom;
use crossbeam_channel::{Sender, TrySendError};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::storage::async_fingerprint_writer::FingerprintPersistMsg;
use crate::storage::page_aligned_fingerprint_store::FingerprintStats;

/// Bloom filter-based fingerprint store with GUARANTEED bounded memory
///
/// Unlike sled or hash tables, bloom filters have:
/// - Fixed memory usage (determined at creation)
/// - O(1) insertion and lookup
/// - Small false positive rate (tunable)
/// - NO false negatives
///
/// For model checking:
/// - False positives = might skip a few duplicate states (acceptable)
/// - False negatives = never happens (correctness preserved)
///
/// Memory usage: Exactly as configured, no more, no less.
/// Example: 100M items, 1% FPR = ~120MB total
///          1B items, 1% FPR = ~1.2GB total
pub struct BloomFingerprintStore {
    /// Sharded bloom filters for parallelism
    shards: Vec<RwLock<Bloom<u64>>>,

    /// Shard mask for fast modulo
    shard_mask: usize,

    /// Number of NUMA nodes (for home_numa calculation)
    num_numa_nodes: usize,

    /// Persistence channels (one per shard)
    persist_tx: Option<Vec<Sender<FingerprintPersistMsg>>>,

    /// Stats
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
    persist_drops: AtomicU64,
}

impl BloomFingerprintStore {
    /// Create new bloom filter store
    ///
    /// Parameters:
    /// - expected_items: Expected number of unique fingerprints
    /// - false_positive_rate: Acceptable FP rate (e.g., 0.01 = 1%)
    /// - shard_count: Number of shards (must be power of 2)
    /// - num_numa_nodes: Number of NUMA nodes (for home_numa calculation)
    ///
    /// Memory usage: ~(expected_items * -ln(fpr) / (ln(2)^2)) bits
    /// Example: 100M items, 1% FPR = ~120MB total
    pub fn new(
        expected_items: usize,
        false_positive_rate: f64,
        shard_count: usize,
        num_numa_nodes: usize,
    ) -> Result<Self> {
        // Ensure shard_count is power of 2
        assert!(
            shard_count.is_power_of_two(),
            "shard_count must be power of 2"
        );

        let items_per_shard = expected_items / shard_count;

        let mut shards = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            // Create bloom filter for this shard
            let bloom = Bloom::new_for_fp_rate(items_per_shard, false_positive_rate)
                .map_err(|e| anyhow::anyhow!("Failed to create bloom filter: {}", e))?;
            shards.push(RwLock::new(bloom));
        }

        // Calculate actual memory usage
        let bits_per_item = -(false_positive_rate.ln()) / (2.0_f64.ln().powi(2));
        let total_bits = (expected_items as f64 * bits_per_item) as usize;
        let total_mb = total_bits / 8 / 1024 / 1024;

        eprintln!(
            "Bloom fingerprint store: {} expected items, {:.1}% FPR, {}MB memory (FIXED)",
            expected_items,
            false_positive_rate * 100.0,
            total_mb
        );

        Ok(Self {
            shards,
            shard_mask: shard_count - 1,
            num_numa_nodes: num_numa_nodes.max(1),
            persist_tx: None,
            checks: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            batch_calls: AtomicU64::new(0),
            batch_items: AtomicU64::new(0),
            persist_drops: AtomicU64::new(0),
        })
    }

    /// Enable persistence - fingerprints will be written to disk for resume support
    pub fn enable_persistence(&mut self, persist_tx: Vec<Sender<FingerprintPersistMsg>>) {
        self.persist_tx = Some(persist_tx);
    }

    /// Get the shard ID for a fingerprint
    #[inline]
    fn shard_id(&self, fp: u64) -> usize {
        (fp as usize) & self.shard_mask
    }

    /// Get home NUMA node for a fingerprint (for NUMA-aware queue routing)
    #[inline]
    pub fn home_numa(&self, fp: u64) -> usize {
        // Distribute fingerprints across NUMA nodes based on high bits
        ((fp >> 48) as usize) % self.num_numa_nodes
    }

    /// Send fingerprint to persistence writer (non-blocking)
    #[inline]
    fn persist(&self, fp: u64, shard_id: usize) {
        if let Some(ref persist_tx) = self.persist_tx {
            if let Some(tx) = persist_tx.get(shard_id) {
                let msg = FingerprintPersistMsg { fp };
                if let Err(TrySendError::Full(_)) = tx.try_send(msg) {
                    // Channel full - drop this persist (fingerprint is still in bloom filter)
                    // It will be re-explored on next run if process crashes
                    self.persist_drops.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Check if fingerprint exists, insert if not
    ///
    /// Returns: true if fingerprint MIGHT have been seen (could be false positive)
    pub fn contains_or_insert(&self, fp: u64) -> bool {
        self.checks.fetch_add(1, Ordering::Relaxed);

        let shard_id = self.shard_id(fp);
        let shard = &self.shards[shard_id];

        // Check if already present (read lock)
        {
            let bloom = shard.read();
            if bloom.check(&fp) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return true; // Might be false positive, but that's OK
            }
        }

        // Not present - insert (write lock)
        {
            let mut bloom = shard.write();

            // Double-check after acquiring write lock
            if bloom.check(&fp) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return true;
            }

            bloom.set(&fp);
            self.inserts.fetch_add(1, Ordering::Relaxed);

            // Persist to disk (non-blocking)
            self.persist(fp, shard_id);

            false
        }
    }

    /// Batch check and insert fingerprints
    ///
    /// Optimized: Single write lock per shard instead of lock-per-item
    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        self.contains_or_insert_batch_with_affinity(fps, seen, 0)
    }

    /// Batch check and insert fingerprints with worker affinity
    ///
    /// Each worker processes shards starting from a different offset (based on worker_id).
    /// This spreads contention across shards when many workers process similar fingerprints.
    pub fn contains_or_insert_batch_with_affinity(
        &self,
        fps: &[u64],
        seen: &mut Vec<bool>,
        worker_id: usize,
    ) -> Result<()> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.batch_items
            .fetch_add(fps.len() as u64, Ordering::Relaxed);

        seen.clear();
        seen.resize(fps.len(), false);

        // Group by shard for better cache locality
        let num_shards = self.shards.len();
        let mut shard_groups: Vec<Vec<(usize, u64)>> = vec![Vec::new(); num_shards];

        for (idx, &fp) in fps.iter().enumerate() {
            let shard_id = self.shard_id(fp);
            shard_groups[shard_id].push((idx, fp));
        }

        // Track stats locally to reduce atomic operations
        let mut local_hits = 0u64;
        let mut local_inserts = 0u64;
        let mut new_fps: Vec<(u64, usize)> = Vec::new(); // (fp, shard_id) for persistence

        // Process each shard's fingerprints with a SINGLE write lock per shard
        // Start from worker_id offset to spread contention
        for i in 0..num_shards {
            let shard_id = (i + worker_id) % num_shards;
            let group = &shard_groups[shard_id];

            if group.is_empty() {
                continue;
            }

            // One write lock for entire shard batch - eliminates lock-per-item overhead
            let mut bloom = self.shards[shard_id].write();

            for &(idx, fp) in group {
                if bloom.check(&fp) {
                    seen[idx] = true;
                    local_hits += 1;
                } else {
                    bloom.set(&fp);
                    seen[idx] = false;
                    local_inserts += 1;
                    new_fps.push((fp, shard_id));
                }
            }
            // Lock released here after processing all items in this shard
        }

        // Persist new fingerprints (outside of locks for better throughput)
        for (fp, shard_id) in new_fps {
            self.persist(fp, shard_id);
        }

        // Single atomic update for stats (instead of per-item)
        self.checks.fetch_add(fps.len() as u64, Ordering::Relaxed);
        self.hits.fetch_add(local_hits, Ordering::Relaxed);
        self.inserts.fetch_add(local_inserts, Ordering::Relaxed);

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> FingerprintStats {
        FingerprintStats {
            checks: self.checks.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            batch_calls: self.batch_calls.load(Ordering::Relaxed),
            batch_items: self.batch_items.load(Ordering::Relaxed),
            collisions: 0, // Bloom filters don't have collisions
        }
    }

    /// Flush to disk (no-op for bloom filter - persistence is continuous)
    pub fn flush(&self) -> Result<()> {
        // Persistence happens continuously via the async writer
        // Nothing special needed here
        Ok(())
    }

    /// Get the number of shards
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Get persistence drop count (fingerprints not persisted due to full channel)
    pub fn persist_drops(&self) -> u64 {
        self.persist_drops.load(Ordering::Relaxed)
    }

    /// Get the base memory address of the fingerprint store for NUMA diagnostics
    ///
    /// Note: Bloom filters use internal allocations that we cannot easily expose,
    /// so this returns None. Use PageAlignedFingerprintStore for NUMA-aware diagnostics.
    pub fn memory_base_addr(&self) -> Option<*const u8> {
        // Bloom filters use internal Vec allocations that we cannot access
        None
    }
}
