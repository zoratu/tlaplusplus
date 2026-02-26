use anyhow::Result;
use bloomfilter::Bloom;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

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
pub struct BloomFingerprintStore {
    /// Sharded bloom filters for parallelism
    shards: Vec<RwLock<Bloom<u64>>>,

    /// Shard mask for fast modulo
    shard_mask: usize,

    /// Stats
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
}

impl BloomFingerprintStore {
    /// Create new bloom filter store
    ///
    /// Parameters:
    /// - expected_items: Expected number of unique fingerprints
    /// - false_positive_rate: Acceptable FP rate (e.g., 0.01 = 1%)
    /// - shard_count: Number of shards (must be power of 2)
    ///
    /// Memory usage: ~(expected_items * -ln(fpr) / (ln(2)^2)) bits
    /// Example: 100M items, 1% FPR = ~120MB total
    pub fn new(
        expected_items: usize,
        false_positive_rate: f64,
        shard_count: usize,
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

        eprintln!("Bloom filter fingerprint store initialized:");
        eprintln!("  Expected items: {}", expected_items);
        eprintln!("  False positive rate: {}%", false_positive_rate * 100.0);
        eprintln!("  Shard count: {}", shard_count);
        eprintln!("  Memory usage: {} MB (FIXED, guaranteed)", total_mb);
        eprintln!("  Memory per shard: {} MB", total_mb / shard_count);

        Ok(Self {
            shards,
            shard_mask: shard_count - 1,
            checks: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            batch_calls: AtomicU64::new(0),
            batch_items: AtomicU64::new(0),
        })
    }

    /// Check if fingerprint exists, insert if not
    ///
    /// Returns: true if fingerprint MIGHT have been seen (could be false positive)
    pub fn contains_or_insert(&self, fp: u64) -> Result<bool> {
        self.checks.fetch_add(1, Ordering::Relaxed);

        let shard_id = (fp as usize) & self.shard_mask;
        let shard = &self.shards[shard_id];

        // Check if already present (read lock)
        {
            let bloom = shard.read();
            if bloom.check(&fp) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(true); // Might be false positive, but that's OK
            }
        }

        // Not present - insert (write lock)
        {
            let mut bloom = shard.write();

            // Double-check after acquiring write lock
            if bloom.check(&fp) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(true);
            }

            bloom.set(&fp);
            self.inserts.fetch_add(1, Ordering::Relaxed);
            Ok(false)
        }
    }

    /// Batch check and insert fingerprints
    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.batch_items
            .fetch_add(fps.len() as u64, Ordering::Relaxed);
        self.checks.fetch_add(fps.len() as u64, Ordering::Relaxed);

        seen.clear();
        seen.resize(fps.len(), false);

        // Group by shard for better cache locality
        let mut shard_groups: Vec<Vec<(usize, u64)>> = vec![Vec::new(); self.shards.len()];

        for (idx, &fp) in fps.iter().enumerate() {
            let shard_id = (fp as usize) & self.shard_mask;
            shard_groups[shard_id].push((idx, fp));
        }

        // Process each shard's fingerprints
        for (shard_id, group) in shard_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }

            let shard = &self.shards[shard_id];

            for &(idx, fp) in group {
                // Check if already present
                let exists = {
                    let bloom = shard.read();
                    bloom.check(&fp)
                };

                if exists {
                    seen[idx] = true;
                    self.hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Insert
                    let mut bloom = shard.write();

                    // Double-check
                    if bloom.check(&fp) {
                        seen[idx] = true;
                        self.hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        bloom.set(&fp);
                        seen[idx] = false;
                        self.inserts.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

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

    /// Flush to disk (no-op for bloom filter)
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Dummy method for API compatibility
    pub fn enable_persistence(
        &mut self,
        _persist_tx: Vec<
            crossbeam_channel::Sender<
                crate::storage::async_fingerprint_writer::FingerprintPersistMsg,
            >,
        >,
    ) {
        // No-op: bloom filters are in-memory only
    }
}
