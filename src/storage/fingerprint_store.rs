use anyhow::{Context, Result};
use bloomfilter::Bloom;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug)]
pub struct FingerprintStoreConfig {
    pub path: PathBuf,
    pub shard_count: usize,
    pub expected_items: usize,
    pub false_positive_rate: f64,
    pub hot_entries_per_shard: usize,
    pub cache_capacity_bytes: u64,
    pub flush_every_ms: Option<u64>,
}

impl Default for FingerprintStoreConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./.tlapp/fingerprints"),
            shard_count: 64,
            expected_items: 50_000_000,
            false_positive_rate: 0.01,
            hot_entries_per_shard: 1_000_000,
            cache_capacity_bytes: 512 * 1024 * 1024,
            flush_every_ms: Some(10_000),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FingerprintStats {
    pub checks: u64,
    pub hits: u64,
    pub inserts: u64,
    pub disk_lookups: u64,
    pub batch_calls: u64,
    pub batch_items: u64,
    pub shard_count: usize,
}

#[derive(Default)]
struct FingerprintStatsAtomic {
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    disk_lookups: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
}

struct FingerprintShard {
    hot: Mutex<HashSet<u64>>,
    bloom: Mutex<Bloom<u64>>,
    tree: sled::Tree,
}

pub struct FingerprintStore {
    db: sled::Db,
    path: PathBuf,
    shards: Vec<FingerprintShard>,
    shard_mask: usize,
    hot_entries_per_shard: usize,
    stats: FingerprintStatsAtomic,
}

impl FingerprintStore {
    pub fn new(config: FingerprintStoreConfig) -> Result<Self> {
        let shard_count = config.shard_count.max(1).next_power_of_two();
        std::fs::create_dir_all(&config.path)
            .with_context(|| format!("failed to create {}", config.path.display()))?;

        let mut db_config = sled::Config::new()
            .path(&config.path)
            .cache_capacity(config.cache_capacity_bytes)
            .mode(sled::Mode::HighThroughput);
        if let Some(ms) = config.flush_every_ms {
            db_config = db_config.flush_every_ms(Some(ms));
        }
        let db = db_config.open().context("failed to open fingerprint db")?;

        let expected_per_shard = (config.expected_items / shard_count).max(100_000);
        let mut shards = Vec::with_capacity(shard_count);
        for idx in 0..shard_count {
            let tree = db
                .open_tree(format!("fp-shard-{idx:04}"))
                .context("failed to open fingerprint shard")?;
            let bloom =
                Bloom::new_for_fp_rate(expected_per_shard, config.false_positive_rate.max(0.0001))
                    .map_err(|err| anyhow::anyhow!(err.to_string()))?;
            shards.push(FingerprintShard {
                hot: Mutex::new(HashSet::with_capacity(
                    config.hot_entries_per_shard.min(expected_per_shard),
                )),
                bloom: Mutex::new(bloom),
                tree,
            });
        }

        Ok(Self {
            db,
            path: config.path,
            shards,
            shard_mask: shard_count - 1,
            hot_entries_per_shard: config.hot_entries_per_shard.max(10_000),
            stats: FingerprintStatsAtomic::default(),
        })
    }

    #[inline]
    fn shard_for(&self, fp: u64) -> &FingerprintShard {
        &self.shards[(fp as usize) & self.shard_mask]
    }

    #[inline]
    fn touch_hot(&self, shard: &FingerprintShard, fp: u64) {
        let mut hot = shard.hot.lock();
        hot.insert(fp);
        if hot.len() > self.hot_entries_per_shard {
            // Exactness is preserved by sled; this cache only avoids repeated lookups.
            hot.clear();
        }
    }

    pub fn contains_or_insert(&self, fp: u64) -> Result<bool> {
        self.stats.checks.fetch_add(1, Ordering::Relaxed);
        let shard = self.shard_for(fp);

        {
            let hot = shard.hot.lock();
            if hot.contains(&fp) {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(true);
            }
        }

        let key = fp.to_be_bytes();
        let bloom_positive = {
            let bloom = shard.bloom.lock();
            bloom.check(&fp)
        };

        if bloom_positive {
            self.stats.disk_lookups.fetch_add(1, Ordering::Relaxed);
            if shard.tree.contains_key(key)? {
                self.touch_hot(shard, fp);
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(true);
            }
        }

        let previous = shard.tree.insert(key, &[1u8])?;
        if previous.is_some() {
            self.touch_hot(shard, fp);
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(true);
        }

        {
            let mut bloom = shard.bloom.lock();
            bloom.set(&fp);
        }
        self.touch_hot(shard, fp);
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        Ok(false)
    }

    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        if fps.is_empty() {
            seen.clear();
            return Ok(());
        }
        if fps.len() == 1 {
            let existed = self.contains_or_insert(fps[0])?;
            seen.clear();
            seen.push(existed);
            self.stats.batch_calls.fetch_add(1, Ordering::Relaxed);
            self.stats.batch_items.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        #[derive(Clone, Copy)]
        struct Item {
            index: usize,
            fp: u64,
            shard: usize,
        }

        let mut items = Vec::with_capacity(fps.len());
        for (index, fp) in fps.iter().copied().enumerate() {
            items.push(Item {
                index,
                fp,
                shard: (fp as usize) & self.shard_mask,
            });
        }
        items.sort_unstable_by_key(|item| item.shard);

        seen.clear();
        seen.resize(fps.len(), false);

        let mut checks = 0u64;
        let mut hits = 0u64;
        let mut inserts = 0u64;
        let mut disk_lookups = 0u64;

        let mut cursor = 0usize;
        while cursor < items.len() {
            let shard_idx = items[cursor].shard;
            let mut end = cursor + 1;
            while end < items.len() && items[end].shard == shard_idx {
                end += 1;
            }
            let shard = &self.shards[shard_idx];
            let group = &items[cursor..end];

            let mut unresolved = Vec::new();
            {
                let hot = shard.hot.lock();
                for item in group {
                    checks += 1;
                    if hot.contains(&item.fp) {
                        seen[item.index] = true;
                        hits += 1;
                    } else {
                        unresolved.push((item.index, item.fp, false));
                    }
                }
            }

            if !unresolved.is_empty() {
                {
                    let bloom = shard.bloom.lock();
                    for (_, fp, bloom_positive) in &mut unresolved {
                        *bloom_positive = bloom.check(fp);
                    }
                }

                let mut hot_adds = Vec::new();
                let mut bloom_adds = Vec::new();
                for (idx, fp, bloom_positive) in unresolved {
                    let key = fp.to_be_bytes();
                    let mut exists = false;

                    if bloom_positive {
                        disk_lookups += 1;
                        if shard.tree.contains_key(key)? {
                            exists = true;
                        }
                    }

                    if !exists {
                        let prev = shard.tree.insert(key, &[1u8])?;
                        if prev.is_some() {
                            exists = true;
                        }
                    }

                    if exists {
                        seen[idx] = true;
                        hits += 1;
                        hot_adds.push(fp);
                    } else {
                        inserts += 1;
                        hot_adds.push(fp);
                        bloom_adds.push(fp);
                    }
                }

                if !hot_adds.is_empty() {
                    let mut hot = shard.hot.lock();
                    for fp in hot_adds {
                        hot.insert(fp);
                    }
                    if hot.len() > self.hot_entries_per_shard {
                        hot.clear();
                    }
                }

                if !bloom_adds.is_empty() {
                    let mut bloom = shard.bloom.lock();
                    for fp in bloom_adds {
                        bloom.set(&fp);
                    }
                }
            }

            cursor = end;
        }

        self.stats.checks.fetch_add(checks, Ordering::Relaxed);
        self.stats.hits.fetch_add(hits, Ordering::Relaxed);
        self.stats.inserts.fetch_add(inserts, Ordering::Relaxed);
        self.stats
            .disk_lookups
            .fetch_add(disk_lookups, Ordering::Relaxed);
        self.stats.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.stats
            .batch_items
            .fetch_add(fps.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn stats(&self) -> FingerprintStats {
        FingerprintStats {
            checks: self.stats.checks.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            disk_lookups: self.stats.disk_lookups.load(Ordering::Relaxed),
            batch_calls: self.stats.batch_calls.load(Ordering::Relaxed),
            batch_items: self.stats.batch_items.load(Ordering::Relaxed),
            shard_count: self.shards.len(),
        }
    }

    pub fn flush(&self) -> Result<usize> {
        self.db.flush().context("failed to flush fingerprint db")
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::{FingerprintStore, FingerprintStoreConfig};
    use anyhow::Result;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("tlapp-{prefix}-{nanos}-{}", std::process::id()))
    }

    #[test]
    fn fingerprint_store_deduplicates() -> Result<()> {
        let path = temp_path("fp");
        let store = FingerprintStore::new(FingerprintStoreConfig {
            path: path.clone(),
            shard_count: 8,
            expected_items: 1000,
            false_positive_rate: 0.01,
            hot_entries_per_shard: 100,
            cache_capacity_bytes: 8 * 1024 * 1024,
            flush_every_ms: Some(500),
        })?;

        assert!(!store.contains_or_insert(42)?);
        assert!(store.contains_or_insert(42)?);
        assert!(!store.contains_or_insert(43)?);
        assert!(store.contains_or_insert(43)?);

        let stats = store.stats();
        assert_eq!(stats.inserts, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.checks, 4);

        let _ = store.flush()?;
        drop(store);
        let _ = std::fs::remove_dir_all(path);
        Ok(())
    }

    #[test]
    fn fingerprint_store_batch_deduplicates() -> Result<()> {
        let path = temp_path("fp-batch");
        let store = FingerprintStore::new(FingerprintStoreConfig {
            path: path.clone(),
            shard_count: 8,
            expected_items: 10_000,
            false_positive_rate: 0.01,
            hot_entries_per_shard: 256,
            cache_capacity_bytes: 8 * 1024 * 1024,
            flush_every_ms: Some(500),
        })?;

        let mut seen = Vec::new();
        store.contains_or_insert_batch(&[11, 12, 11, 13], &mut seen)?;
        assert_eq!(seen, vec![false, false, true, false]);

        store.contains_or_insert_batch(&[11, 12, 13, 14], &mut seen)?;
        assert_eq!(seen, vec![true, true, true, false]);

        let stats = store.stats();
        assert!(stats.batch_calls >= 2);
        assert!(stats.batch_items >= 8);

        let _ = std::fs::remove_dir_all(path);
        Ok(())
    }
}
