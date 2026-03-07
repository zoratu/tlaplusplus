// Unified fingerprint store - wraps different implementations with a common interface
//
// This allows the runtime to use different fingerprint store backends:
// - PageAligned: In-memory hash table with huge pages (fastest, but unbounded memory)
// - Bloom: Bloom filter with fixed memory (bounded memory, small false positive rate)
// - AutoSwitch: Starts with PageAligned, automatically switches to Bloom under pressure

use anyhow::Result;
use crossbeam_channel::Sender;

use crate::storage::async_fingerprint_writer::FingerprintPersistMsg;
use crate::storage::auto_switching_fingerprint_store::{
    AutoSwitchConfig, AutoSwitchingFingerprintStore,
};
use crate::storage::bloom_fingerprint_store::BloomFingerprintStore;
use crate::storage::page_aligned_fingerprint_store::{
    FingerprintStats, FingerprintStoreConfig as PageAlignedConfig, PageAlignedFingerprintStore,
};

/// Configuration for the unified fingerprint store
#[derive(Clone, Debug)]
pub struct UnifiedFingerprintConfig {
    /// Use bloom filter mode (bounded memory, ~1% false positive rate)
    pub use_bloom: bool,
    /// Enable automatic switching from exact to bloom (default: false)
    pub use_auto_switch: bool,
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
    /// Auto-switch configuration (only used when use_auto_switch is true)
    pub auto_switch_config: Option<AutoSwitchConfigInput>,
}

/// Input configuration for auto-switch mode
#[derive(Clone, Debug, Default)]
pub struct AutoSwitchConfigInput {
    /// State count threshold to trigger switch (default: 1 billion)
    pub state_count_threshold: Option<u64>,
    /// Memory pressure threshold (0.0 - 1.0) to trigger switch (default: 0.85)
    pub memory_threshold: Option<f64>,
    /// False positive rate for bloom filter after switch (default: 0.001)
    pub bloom_false_positive_rate: Option<f64>,
}

/// Unified fingerprint store that can use different backends
pub enum UnifiedFingerprintStore {
    PageAligned(PageAlignedFingerprintStore),
    Bloom(BloomFingerprintStore),
    AutoSwitch(AutoSwitchingFingerprintStore),
}

impl UnifiedFingerprintStore {
    /// Create a new unified fingerprint store
    pub fn new(config: UnifiedFingerprintConfig, assigned_cpus: &[Option<usize>]) -> Result<Self> {
        if config.use_auto_switch {
            // Auto-switching mode - starts exact, switches to bloom under pressure
            let auto_config = config.auto_switch_config.unwrap_or_default();
            let switch_config = AutoSwitchConfig {
                enabled: true,
                state_count_threshold: auto_config.state_count_threshold.unwrap_or(1_000_000_000),
                memory_threshold: auto_config.memory_threshold.unwrap_or(0.85),
                bloom_false_positive_rate: auto_config.bloom_false_positive_rate.unwrap_or(0.001),
                bloom_expected_items: config.expected_items * 10, // Allow for growth
                shard_count: config.shard_count,
                shard_size_mb: config.shard_size_mb,
                num_numa_nodes: config.num_numa_nodes,
                check_interval: 10_000,
            };
            let store = AutoSwitchingFingerprintStore::new(switch_config, assigned_cpus)?;
            Ok(Self::AutoSwitch(store))
        } else if config.use_bloom {
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
            Self::AutoSwitch(store) => store.enable_persistence(persist_tx),
        }
    }

    /// Check if fingerprint exists, insert if not
    pub fn contains_or_insert(&self, fp: u64) -> bool {
        match self {
            Self::PageAligned(store) => store.contains_or_insert(fp),
            Self::Bloom(store) => store.contains_or_insert(fp),
            Self::AutoSwitch(store) => store.contains_or_insert(fp),
        }
    }

    /// Batch check and insert fingerprints
    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        match self {
            Self::PageAligned(store) => store.contains_or_insert_batch(fps, seen),
            Self::Bloom(store) => store.contains_or_insert_batch(fps, seen),
            Self::AutoSwitch(store) => store.contains_or_insert_batch(fps, seen),
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
            Self::AutoSwitch(store) => {
                store.contains_or_insert_batch_with_affinity(fps, seen, worker_id)
            }
        }
    }

    /// Get home NUMA node for a fingerprint
    pub fn home_numa(&self, fp: u64) -> usize {
        match self {
            Self::PageAligned(store) => store.home_numa(fp),
            Self::Bloom(store) => store.home_numa(fp),
            Self::AutoSwitch(store) => store.home_numa(fp),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> FingerprintStats {
        match self {
            Self::PageAligned(store) => store.stats(),
            Self::Bloom(store) => store.stats(),
            Self::AutoSwitch(store) => store.stats(),
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
            Self::AutoSwitch(store) => store.flush(),
        }
    }

    /// Get the number of shards
    pub fn shard_count(&self) -> usize {
        match self {
            Self::PageAligned(store) => store.shard_count(),
            Self::Bloom(store) => store.shard_count(),
            Self::AutoSwitch(store) => store.shard_count(),
        }
    }

    /// Check if using bloom filter mode
    pub fn is_bloom(&self) -> bool {
        matches!(self, Self::Bloom(_))
    }

    /// Check if using auto-switch mode
    pub fn is_auto_switch(&self) -> bool {
        matches!(self, Self::AutoSwitch(_))
    }

    /// Check if auto-switch mode has switched to hybrid
    pub fn is_in_hybrid_mode(&self) -> bool {
        match self {
            Self::AutoSwitch(store) => store.is_hybrid(),
            _ => false,
        }
    }
}
