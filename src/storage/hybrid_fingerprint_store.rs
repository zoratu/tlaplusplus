use anyhow::Result;
use sled::Db;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::storage::page_aligned_fingerprint_store::FingerprintStats;

/// Disk-backed fingerprint store using sled with bounded memory
///
/// This store uses sled (embedded database) which:
/// - Has a bounded in-memory cache (configurable)
/// - Automatically evicts old entries when cache is full
/// - Persists all fingerprints to disk
/// - Is lock-free and very fast
///
/// Memory usage: Only sled's cache (configurable, default 2GB)
pub struct HybridFingerprintStore {
    /// Sled database (disk-backed with bounded cache)
    db: Db,

    /// Stats
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
}

impl HybridFingerprintStore {
    /// Create new hybrid store
    ///
    /// Parameters:
    /// - work_dir: Directory for sled database
    /// - cache_size_mb: Sled's in-memory cache size in MB
    pub fn new(work_dir: PathBuf, cache_size_mb: usize) -> Result<Self> {
        let db_path = work_dir.join("fingerprints.sled");

        // Configure sled with bounded cache
        let db = sled::Config::new()
            .path(&db_path)
            .cache_capacity((cache_size_mb * 1024 * 1024) as u64)
            .flush_every_ms(Some(5000)) // Flush every 5s
            .mode(sled::Mode::HighThroughput)
            .open()?;

        eprintln!("Sled fingerprint store initialized:");
        eprintln!("  Cache size: {} MB", cache_size_mb);
        eprintln!("  Database path: {}", db_path.display());

        Ok(Self {
            db,
            checks: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            batch_calls: AtomicU64::new(0),
            batch_items: AtomicU64::new(0),
        })
    }

    /// Check if fingerprint exists, insert if not (single)
    pub fn contains_or_insert(&self, fp: u64) -> Result<bool> {
        self.checks.fetch_add(1, Ordering::Relaxed);
        let key = fp.to_be_bytes();

        // Try to insert - returns Some(old_value) if key existed
        match self.db.insert(&key, &[])? {
            Some(_) => {
                // Key existed
                self.hits.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            None => {
                // Key didn't exist, was inserted
                self.inserts.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
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

        for (idx, &fp) in fps.iter().enumerate() {
            let key = fp.to_be_bytes();

            match self.db.insert(&key, &[])? {
                Some(_) => {
                    // Existed
                    seen[idx] = true;
                    self.hits.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    // Didn't exist
                    seen[idx] = false;
                    self.inserts.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Get statistics (compatible with PageAlignedFingerprintStore)
    pub fn stats(&self) -> FingerprintStats {
        FingerprintStats {
            checks: self.checks.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            batch_calls: self.batch_calls.load(Ordering::Relaxed),
            batch_items: self.batch_items.load(Ordering::Relaxed),
            collisions: 0, // Sled doesn't expose collision stats
        }
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }

    /// Dummy method for API compatibility (sled already persists)
    pub fn enable_persistence(
        &mut self,
        _persist_tx: Vec<
            crossbeam_channel::Sender<
                crate::storage::async_fingerprint_writer::FingerprintPersistMsg,
            >,
        >,
    ) {
        // No-op: sled already persists to disk
        eprintln!("Note: enable_persistence() called but sled already persists automatically");
    }
}
