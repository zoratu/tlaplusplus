// In-memory lock-free fingerprint store - NO DISK I/O!
// This eliminates the disk I/O bottleneck for 100+ core scaling

use bloomfilter::Bloom;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug)]
pub struct FingerprintStoreConfig {
    pub shard_count: usize,
    pub expected_items: usize,
    pub false_positive_rate: f64,
}

impl Default for FingerprintStoreConfig {
    fn default() -> Self {
        Self {
            shard_count: 64,
            expected_items: 50_000_000,
            false_positive_rate: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FingerprintStats {
    pub checks: u64,
    pub hits: u64,
    pub inserts: u64,
    pub batch_calls: u64,
    pub batch_items: u64,
}

#[derive(Default)]
struct FingerprintStatsAtomic {
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
}

struct FingerprintShard {
    // Lock-free concurrent hash map - supports parallel reads/writes
    // No bloom filter needed - DashMap is already fast for lookups
    map: DashMap<u64, ()>,
}

pub struct FingerprintStore {
    shards: Vec<FingerprintShard>,
    shard_mask: usize,
    stats: FingerprintStatsAtomic,
}

impl FingerprintStore {
    pub fn new(config: FingerprintStoreConfig) -> Self {
        let shard_count = config.shard_count.max(1).next_power_of_two();
        let expected_per_shard = (config.expected_items / shard_count).max(10_000);

        let shards = (0..shard_count)
            .map(|_| FingerprintShard {
                map: DashMap::with_capacity(expected_per_shard / 10),
            })
            .collect();

        Self {
            shards,
            shard_mask: shard_count - 1,
            stats: FingerprintStatsAtomic::default(),
        }
    }

    #[inline]
    fn shard_for(&self, fp: u64) -> &FingerprintShard {
        &self.shards[(fp as usize) & self.shard_mask]
    }

    pub fn contains_or_insert(&self, fp: u64) -> bool {
        self.stats.checks.fetch_add(1, Ordering::Relaxed);
        let shard = self.shard_for(fp);

        // Fast path: check DashMap (lock-free!)
        if shard.map.contains_key(&fp) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Insert and check if it existed
        if shard.map.insert(fp, ()).is_some() {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // New fingerprint
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        false
    }

    pub fn contains_or_insert_batch(
        &self,
        fps: &[u64],
        seen: &mut Vec<bool>,
    ) -> anyhow::Result<()> {
        if fps.is_empty() {
            seen.clear();
            return Ok(());
        }

        self.stats.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.stats
            .batch_items
            .fetch_add(fps.len() as u64, Ordering::Relaxed);

        seen.clear();
        seen.resize(fps.len(), false);

        // Group by shard for better cache locality
        let mut shard_groups: Vec<Vec<(usize, u64)>> = vec![Vec::new(); self.shards.len()];

        for (idx, &fp) in fps.iter().enumerate() {
            let shard_id = (fp as usize) & self.shard_mask;
            shard_groups[shard_id].push((idx, fp));
        }

        let mut checks = 0u64;
        let mut hits = 0u64;
        let mut inserts = 0u64;

        // Process each shard's fingerprints
        for (shard_id, group) in shard_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }

            let shard = &self.shards[shard_id];
            let mut bloom_updates = Vec::new();

            for &(idx, fp) in group {
                checks += 1;

                // Check if exists (lock-free read!)
                if shard.map.contains_key(&fp) {
                    seen[idx] = true;
                    hits += 1;
                    continue;
                }

                // Try insert
                if shard.map.insert(fp, ()).is_some() {
                    // Was inserted concurrently
                    seen[idx] = true;
                    hits += 1;
                } else {
                    // New fingerprint
                    inserts += 1;
                    bloom_updates.push(fp);
                }
            }

            // No bloom filter to update - DashMap handles everything
        }

        self.stats.checks.fetch_add(checks, Ordering::Relaxed);
        self.stats.hits.fetch_add(hits, Ordering::Relaxed);
        self.stats.inserts.fetch_add(inserts, Ordering::Relaxed);

        Ok(())
    }

    pub fn stats(&self) -> FingerprintStats {
        FingerprintStats {
            checks: self.stats.checks.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            batch_calls: self.stats.batch_calls.load(Ordering::Relaxed),
            batch_items: self.stats.batch_items.load(Ordering::Relaxed),
        }
    }

    pub fn flush(&self) -> anyhow::Result<()> {
        // No-op for in-memory store
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.map.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
