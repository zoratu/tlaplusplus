// Unified fingerprint store - wraps different implementations with a common interface
//
// This allows the runtime to use different fingerprint store backends:
// - PageAligned: In-memory hash table with huge pages (fastest, but unbounded memory)
// - Bloom: Bloom filter with fixed memory (bounded memory, small false positive rate)

use anyhow::Result;
use crossbeam_channel::Sender;

use crate::storage::async_fingerprint_writer::FingerprintPersistMsg;
use crate::storage::bloom_fingerprint_store::BloomFingerprintStore;
use crate::storage::page_aligned_fingerprint_store::{
    FingerprintStats, FingerprintStoreConfig as PageAlignedConfig, PageAlignedFingerprintStore,
};

/// Configuration for the unified fingerprint store
#[derive(Clone, Debug)]
pub struct UnifiedFingerprintConfig {
    /// Use bloom filter mode (bounded memory, ~1% false positive rate)
    pub use_bloom: bool,
    /// Number of shards
    pub shard_count: usize,
    /// Expected number of unique fingerprints
    pub expected_items: usize,
    /// False positive rate for bloom filter mode (ignored for page-aligned mode)
    pub false_positive_rate: f64,
    /// Shard size in MB for page-aligned mode (ignored for bloom mode)
    pub shard_size_mb: usize,
    /// Number of NUMA nodes (for home_numa calculation)
    pub num_numa_nodes: usize,
}

/// Unified fingerprint store that can use different backends
pub enum UnifiedFingerprintStore {
    PageAligned(PageAlignedFingerprintStore),
    Bloom(BloomFingerprintStore),
}

impl UnifiedFingerprintStore {
    /// Create a new unified fingerprint store
    pub fn new(config: UnifiedFingerprintConfig, assigned_cpus: &[Option<usize>]) -> Result<Self> {
        if config.use_bloom {
            // Bloom filter mode - fixed memory
            let store = BloomFingerprintStore::new(
                config.expected_items,
                config.false_positive_rate,
                config.shard_count,
                config.num_numa_nodes,
            )?;
            Ok(Self::Bloom(store))
        } else {
            // Page-aligned mode - in-memory hash table
            let pa_config = PageAlignedConfig {
                shard_count: config.shard_count,
                expected_items: config.expected_items,
                shard_size_mb: config.shard_size_mb,
            };
            let store = PageAlignedFingerprintStore::new(pa_config, assigned_cpus)?;
            Ok(Self::PageAligned(store))
        }
    }

    /// Enable persistence (fingerprints written to disk for resume support)
    pub fn enable_persistence(&mut self, persist_tx: Vec<Sender<FingerprintPersistMsg>>) {
        match self {
            Self::PageAligned(store) => store.enable_persistence(persist_tx),
            Self::Bloom(store) => store.enable_persistence(persist_tx),
        }
    }

    /// Check if fingerprint exists, insert if not
    pub fn contains_or_insert(&self, fp: u64) -> bool {
        match self {
            Self::PageAligned(store) => store.contains_or_insert(fp),
            Self::Bloom(store) => store.contains_or_insert(fp),
        }
    }

    /// Batch check and insert fingerprints
    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        match self {
            Self::PageAligned(store) => store.contains_or_insert_batch(fps, seen),
            Self::Bloom(store) => store.contains_or_insert_batch(fps, seen),
        }
    }

    /// Batch check and insert with worker affinity
    pub fn contains_or_insert_batch_with_affinity(
        &self,
        fps: &[u64],
        seen: &mut Vec<bool>,
        worker_id: usize,
    ) -> Result<()> {
        match self {
            Self::PageAligned(store) => {
                store.contains_or_insert_batch_with_affinity(fps, seen, worker_id)
            }
            Self::Bloom(store) => {
                store.contains_or_insert_batch_with_affinity(fps, seen, worker_id)
            }
        }
    }

    /// Get home NUMA node for a fingerprint
    pub fn home_numa(&self, fp: u64) -> usize {
        match self {
            Self::PageAligned(store) => store.home_numa(fp),
            Self::Bloom(store) => store.home_numa(fp),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> FingerprintStats {
        match self {
            Self::PageAligned(store) => store.stats(),
            Self::Bloom(store) => store.stats(),
        }
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        match self {
            Self::PageAligned(store) => {
                store.flush()?;
                Ok(())
            }
            Self::Bloom(store) => store.flush(),
        }
    }

    /// Get the number of shards
    pub fn shard_count(&self) -> usize {
        match self {
            Self::PageAligned(store) => store.shard_count(),
            Self::Bloom(store) => store.shard_count(),
        }
    }

    /// Check if using bloom filter mode
    pub fn is_bloom(&self) -> bool {
        matches!(self, Self::Bloom(_))
    }
}
