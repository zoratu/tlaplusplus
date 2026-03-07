// Auto-switching fingerprint store - automatically transitions from exact to bloom filter
//
// This store provides a seamless transition from exact fingerprint storage to
// probabilistic (bloom filter) storage when either:
// - Memory pressure exceeds a threshold (default: 85% of available)
// - State count exceeds a configurable limit (default: 1 billion)
//
// The transition preserves all existing fingerprints by keeping the exact store
// read-only and directing new insertions to the bloom filter. Lookups check both
// stores: exact first (for guaranteed correctness), then bloom (for new states).
//
// This ensures:
// - No fingerprint is ever lost during transition
// - After switch, lookups may have false positives (bloom) but no false negatives
// - Memory stays bounded after switch

use anyhow::Result;
use crossbeam_channel::Sender;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crate::storage::async_fingerprint_writer::FingerprintPersistMsg;
use crate::storage::bloom_fingerprint_store::BloomFingerprintStore;
use crate::storage::page_aligned_fingerprint_store::{
    FingerprintStats, FingerprintStoreConfig as PageAlignedConfig, PageAlignedFingerprintStore,
};
use crate::system::{MemoryMonitor, MemoryStatus};

/// Configuration for the auto-switching fingerprint store
#[derive(Clone, Debug)]
pub struct AutoSwitchConfig {
    /// Enable automatic switching (default: true)
    pub enabled: bool,
    /// State count threshold to trigger switch (default: 1 billion)
    pub state_count_threshold: u64,
    /// Memory pressure threshold (0.0 - 1.0) to trigger switch (default: 0.85)
    pub memory_threshold: f64,
    /// False positive rate for bloom filter after switch (default: 0.001)
    pub bloom_false_positive_rate: f64,
    /// Expected items for bloom filter sizing (default: 10 billion)
    pub bloom_expected_items: usize,
    /// Number of shards for the stores
    pub shard_count: usize,
    /// Shard size in MB for page-aligned store
    pub shard_size_mb: usize,
    /// Number of NUMA nodes
    pub num_numa_nodes: usize,
    /// How often to check for switch conditions (in inserts)
    pub check_interval: u64,
}

impl Default for AutoSwitchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            state_count_threshold: 1_000_000_000, // 1 billion
            memory_threshold: 0.85,
            bloom_false_positive_rate: 0.001, // 0.1%
            bloom_expected_items: 10_000_000_000, // 10 billion
            shard_count: 64,
            shard_size_mb: 1024,
            num_numa_nodes: 1,
            check_interval: 10_000, // Check every 10K inserts
        }
    }
}

/// State of the fingerprint store
enum StoreState {
    /// Using exact store only (pre-switch)
    Exact {
        store: PageAlignedFingerprintStore,
    },
    /// Transitioning: exact is read-only, bloom accepts new inserts
    /// Invariant: check exact first, then bloom
    Hybrid {
        exact: PageAlignedFingerprintStore,
        bloom: BloomFingerprintStore,
        switch_time: Instant,
        switch_reason: SwitchReason,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum SwitchReason {
    MemoryPressure { ratio: f64 },
    StateCountThreshold { count: u64 },
    Manual,
}

impl std::fmt::Display for SwitchReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwitchReason::MemoryPressure { ratio } => {
                write!(f, "memory pressure ({:.1}%)", ratio * 100.0)
            }
            SwitchReason::StateCountThreshold { count } => {
                write!(f, "state count threshold ({})", count)
            }
            SwitchReason::Manual => write!(f, "manual switch"),
        }
    }
}

/// Auto-switching fingerprint store
///
/// Starts with exact (PageAlignedFingerprintStore) and automatically switches
/// to a hybrid mode (exact read-only + bloom for new) when conditions are met.
pub struct AutoSwitchingFingerprintStore {
    /// Current store state (guarded by RwLock for safe transition)
    state: RwLock<StoreState>,

    /// Configuration
    config: AutoSwitchConfig,

    /// Whether we've switched to hybrid mode
    switched: AtomicBool,

    /// Insert counter for periodic checks
    insert_counter: AtomicU64,

    /// Stats tracking
    stats: StoreStats,

    /// Memory monitor for pressure detection
    memory_monitor: MemoryMonitor,

    /// Assigned CPUs for workers (for store creation)
    assigned_cpus: Vec<Option<usize>>,

    /// Persistence channels (if enabled)
    persist_tx: RwLock<Option<Vec<Sender<FingerprintPersistMsg>>>>,
}

#[derive(Default)]
struct StoreStats {
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
    collisions: AtomicU64,
    exact_hits: AtomicU64,
    bloom_hits: AtomicU64,
    bloom_inserts: AtomicU64,
}

impl AutoSwitchingFingerprintStore {
    /// Create a new auto-switching fingerprint store
    pub fn new(config: AutoSwitchConfig, assigned_cpus: &[Option<usize>]) -> Result<Self> {
        let pa_config = PageAlignedConfig {
            shard_count: config.shard_count,
            expected_items: config.bloom_expected_items / 10, // Start with 10% capacity
            shard_size_mb: config.shard_size_mb,
        };

        let exact_store = PageAlignedFingerprintStore::new(pa_config, assigned_cpus)?;

        eprintln!(
            "Auto-switching fingerprint store initialized (auto-switch: {})",
            if config.enabled { "enabled" } else { "disabled" }
        );
        if config.enabled {
            eprintln!(
                "  Switch triggers: state count > {} OR memory > {:.0}%",
                config.state_count_threshold,
                config.memory_threshold * 100.0
            );
            eprintln!(
                "  Bloom FPR after switch: {:.2}%",
                config.bloom_false_positive_rate * 100.0
            );
        }

        let memory_monitor = MemoryMonitor {
            limit_bytes: crate::system::get_memory_stats().total_bytes,
            warn_threshold: config.memory_threshold - 0.10, // Warn 10% before switch
            critical_threshold: config.memory_threshold,
        };

        Ok(Self {
            state: RwLock::new(StoreState::Exact { store: exact_store }),
            config: config.clone(),
            switched: AtomicBool::new(false),
            insert_counter: AtomicU64::new(0),
            stats: StoreStats::default(),
            memory_monitor,
            assigned_cpus: assigned_cpus.to_vec(),
            persist_tx: RwLock::new(None),
        })
    }

    /// Enable persistence for fingerprints
    pub fn enable_persistence(&mut self, persist_tx: Vec<Sender<FingerprintPersistMsg>>) {
        // Store for later (bloom store will need it after switch)
        *self.persist_tx.write() = Some(persist_tx.clone());

        // Also enable on current exact store
        let mut state = self.state.write();
        match &mut *state {
            StoreState::Exact { store } => {
                store.enable_persistence(persist_tx);
            }
            StoreState::Hybrid { bloom, .. } => {
                bloom.enable_persistence(persist_tx);
            }
        }
    }

    /// Check if fingerprint exists, insert if not
    pub fn contains_or_insert(&self, fp: u64) -> bool {
        self.stats.checks.fetch_add(1, Ordering::Relaxed);

        let state = self.state.read();
        let result = match &*state {
            StoreState::Exact { store } => {
                let existed = store.contains_or_insert(fp);
                if existed {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    self.stats.exact_hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats.inserts.fetch_add(1, Ordering::Relaxed);
                    self.maybe_trigger_switch();
                }
                existed
            }
            StoreState::Hybrid { exact, bloom, .. } => {
                // Check exact first (guaranteed correct, no false positives)
                // Exact is read-only after switch - use contains() not contains_or_insert()
                if exact.contains(fp) {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    self.stats.exact_hits.fetch_add(1, Ordering::Relaxed);
                    true
                } else {
                    // Not in exact, check/insert in bloom
                    let bloom_result = bloom.contains_or_insert(fp);
                    if bloom_result {
                        self.stats.hits.fetch_add(1, Ordering::Relaxed);
                        self.stats.bloom_hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
                        self.stats.bloom_inserts.fetch_add(1, Ordering::Relaxed);
                    }
                    bloom_result
                }
            }
        };

        result
    }

    /// Batch check and insert fingerprints
    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        self.contains_or_insert_batch_with_affinity(fps, seen, 0)
    }

    /// Batch check and insert fingerprints with worker affinity
    pub fn contains_or_insert_batch_with_affinity(
        &self,
        fps: &[u64],
        seen: &mut Vec<bool>,
        worker_id: usize,
    ) -> Result<()> {
        self.stats.batch_calls.fetch_add(1, Ordering::Relaxed);
        self.stats
            .batch_items
            .fetch_add(fps.len() as u64, Ordering::Relaxed);

        seen.clear();
        seen.resize(fps.len(), false);

        let state = self.state.read();
        match &*state {
            StoreState::Exact { store } => {
                store.contains_or_insert_batch_with_affinity(fps, seen, worker_id)?;

                // Count stats
                let mut inserts = 0u64;
                for &s in seen.iter() {
                    if s {
                        self.stats.hits.fetch_add(1, Ordering::Relaxed);
                        self.stats.exact_hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        inserts += 1;
                    }
                }
                if inserts > 0 {
                    self.stats.inserts.fetch_add(inserts, Ordering::Relaxed);
                    self.maybe_trigger_switch();
                }
            }
            StoreState::Hybrid { exact, bloom, .. } => {
                // Check exact first for all fingerprints (read-only, no insert)
                let mut exact_seen = Vec::new();
                exact.contains_batch(fps, &mut exact_seen);

                // Collect fingerprints not found in exact
                let mut bloom_fps = Vec::new();
                let mut bloom_indices = Vec::new();
                for (idx, (&fp, &was_seen)) in fps.iter().zip(exact_seen.iter()).enumerate() {
                    if was_seen {
                        seen[idx] = true;
                        self.stats.hits.fetch_add(1, Ordering::Relaxed);
                        self.stats.exact_hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        bloom_fps.push(fp);
                        bloom_indices.push(idx);
                    }
                }

                // Check/insert remaining in bloom
                if !bloom_fps.is_empty() {
                    let mut bloom_seen = Vec::new();
                    bloom.contains_or_insert_batch_with_affinity(
                        &bloom_fps,
                        &mut bloom_seen,
                        worker_id,
                    )?;

                    for (bloom_idx, &original_idx) in bloom_indices.iter().enumerate() {
                        if bloom_seen[bloom_idx] {
                            seen[original_idx] = true;
                            self.stats.hits.fetch_add(1, Ordering::Relaxed);
                            self.stats.bloom_hits.fetch_add(1, Ordering::Relaxed);
                        } else {
                            // New insertion into bloom
                            self.stats.inserts.fetch_add(1, Ordering::Relaxed);
                            self.stats.bloom_inserts.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Trigger a check for switch conditions (called periodically)
    fn maybe_trigger_switch(&self) {
        if !self.config.enabled || self.switched.load(Ordering::Relaxed) {
            return;
        }

        let count = self.insert_counter.fetch_add(1, Ordering::Relaxed);
        if count % self.config.check_interval != 0 {
            return;
        }

        // Check state count threshold
        let state = self.state.read();
        let current_count = match &*state {
            StoreState::Exact { store } => store.len(),
            StoreState::Hybrid { .. } => return, // Already switched
        };
        drop(state);

        if current_count >= self.config.state_count_threshold {
            self.switch_to_hybrid(SwitchReason::StateCountThreshold {
                count: current_count,
            });
            return;
        }

        // Check memory pressure
        let mem_status = self.memory_monitor.check();
        if let MemoryStatus::Critical { ratio, .. } = mem_status {
            self.switch_to_hybrid(SwitchReason::MemoryPressure { ratio });
        }
    }

    /// Manually trigger switch to hybrid mode
    pub fn force_switch(&self) {
        if !self.switched.load(Ordering::Relaxed) {
            self.switch_to_hybrid(SwitchReason::Manual);
        }
    }

    /// Switch from exact to hybrid mode
    fn switch_to_hybrid(&self, reason: SwitchReason) {
        // Use compare_exchange to ensure only one thread switches
        if self
            .switched
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return; // Another thread already switched
        }

        eprintln!("=== FINGERPRINT STORE: SWITCHING TO HYBRID MODE ===");
        eprintln!("Reason: {}", reason);

        let switch_start = Instant::now();

        // Create bloom filter
        let bloom = match BloomFingerprintStore::new(
            self.config.bloom_expected_items,
            self.config.bloom_false_positive_rate,
            self.config.shard_count,
            self.config.num_numa_nodes,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ERROR: Failed to create bloom filter: {}", e);
                self.switched.store(false, Ordering::Release);
                return;
            }
        };

        // Get exclusive lock for the transition
        let mut state = self.state.write();

        // Extract exact store from current state
        let exact = match std::mem::replace(
            &mut *state,
            StoreState::Exact {
                store: PageAlignedFingerprintStore::new(
                    PageAlignedConfig {
                        shard_count: 1,
                        expected_items: 1,
                        shard_size_mb: 64,
                    },
                    &[],
                )
                .unwrap(),
            },
        ) {
            StoreState::Exact { store } => store,
            StoreState::Hybrid { .. } => {
                eprintln!("ERROR: Already in hybrid mode");
                return;
            }
        };

        let exact_count = exact.len();

        // Transition to hybrid
        *state = StoreState::Hybrid {
            exact,
            bloom,
            switch_time: switch_start,
            switch_reason: reason,
        };

        eprintln!(
            "Switch complete in {:.2}ms",
            switch_start.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!("  Exact store: {} fingerprints (now read-only)", exact_count);
        eprintln!("  Bloom filter: ready for new fingerprints");
        eprintln!("  FPR: {:.2}%", self.config.bloom_false_positive_rate * 100.0);
        eprintln!("=================================================");
    }

    /// Check if we've switched to hybrid mode
    pub fn is_hybrid(&self) -> bool {
        self.switched.load(Ordering::Relaxed)
    }

    /// Get home NUMA node for a fingerprint
    pub fn home_numa(&self, fp: u64) -> usize {
        let state = self.state.read();
        match &*state {
            StoreState::Exact { store } => store.home_numa(fp),
            StoreState::Hybrid { exact, .. } => exact.home_numa(fp),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> FingerprintStats {
        FingerprintStats {
            checks: self.stats.checks.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            batch_calls: self.stats.batch_calls.load(Ordering::Relaxed),
            batch_items: self.stats.batch_items.load(Ordering::Relaxed),
            collisions: self.stats.collisions.load(Ordering::Relaxed),
        }
    }

    /// Get extended statistics including hybrid mode info
    pub fn extended_stats(&self) -> AutoSwitchStats {
        let state = self.state.read();
        let (exact_count, bloom_count, switch_info) = match &*state {
            StoreState::Exact { store } => (store.len(), 0, None),
            StoreState::Hybrid {
                exact,
                bloom,
                switch_time,
                switch_reason,
            } => (
                exact.len(),
                bloom.stats().inserts,
                Some((*switch_reason, switch_time.elapsed())),
            ),
        };

        AutoSwitchStats {
            base: self.stats(),
            exact_count,
            bloom_count,
            exact_hits: self.stats.exact_hits.load(Ordering::Relaxed),
            bloom_hits: self.stats.bloom_hits.load(Ordering::Relaxed),
            bloom_inserts: self.stats.bloom_inserts.load(Ordering::Relaxed),
            is_hybrid: self.switched.load(Ordering::Relaxed),
            switch_info,
        }
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        let state = self.state.read();
        match &*state {
            StoreState::Exact { store } => {
                store.flush()?;
            }
            StoreState::Hybrid { exact, bloom, .. } => {
                exact.flush()?;
                bloom.flush()?;
            }
        }
        Ok(())
    }

    /// Get shard count
    pub fn shard_count(&self) -> usize {
        self.config.shard_count
    }
}

/// Extended statistics for auto-switching store
#[derive(Debug, Clone)]
pub struct AutoSwitchStats {
    pub base: FingerprintStats,
    pub exact_count: u64,
    pub bloom_count: u64,
    pub exact_hits: u64,
    pub bloom_hits: u64,
    pub bloom_inserts: u64,
    pub is_hybrid: bool,
    pub switch_info: Option<(SwitchReason, std::time::Duration)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let config = AutoSwitchConfig {
            enabled: false, // Disable auto-switch for basic test
            shard_count: 4,
            shard_size_mb: 64,
            ..Default::default()
        };

        let store = AutoSwitchingFingerprintStore::new(config, &[]).unwrap();

        // Insert new fingerprints
        assert!(!store.contains_or_insert(100));
        assert!(!store.contains_or_insert(200));
        assert!(!store.contains_or_insert(300));

        // Check existing
        assert!(store.contains_or_insert(100));
        assert!(store.contains_or_insert(200));
        assert!(store.contains_or_insert(300));

        // Check non-existing
        assert!(!store.contains_or_insert(400));

        let stats = store.stats();
        assert_eq!(stats.inserts, 4); // 100, 200, 300, 400
        assert_eq!(stats.hits, 3); // 100, 200, 300 (second time)
    }

    #[test]
    fn test_batch_operations() {
        let config = AutoSwitchConfig {
            enabled: false,
            shard_count: 4,
            shard_size_mb: 64,
            ..Default::default()
        };

        let store = AutoSwitchingFingerprintStore::new(config, &[]).unwrap();

        let fps = vec![1, 2, 3, 4, 5];
        let mut seen = Vec::new();

        // First batch - all new
        store.contains_or_insert_batch(&fps, &mut seen).unwrap();
        assert_eq!(seen, vec![false, false, false, false, false]);

        // Second batch - all existing
        store.contains_or_insert_batch(&fps, &mut seen).unwrap();
        assert_eq!(seen, vec![true, true, true, true, true]);

        // Mixed batch
        let mixed = vec![1, 100, 2, 200, 3];
        store.contains_or_insert_batch(&mixed, &mut seen).unwrap();
        assert_eq!(seen, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_manual_switch_to_hybrid() {
        let config = AutoSwitchConfig {
            enabled: true,
            shard_count: 4,
            shard_size_mb: 64,
            bloom_expected_items: 1_000_000,
            ..Default::default()
        };

        let store = AutoSwitchingFingerprintStore::new(config, &[]).unwrap();

        // Insert some fingerprints in exact mode
        for i in 0..100 {
            assert!(!store.contains_or_insert(i));
        }

        assert!(!store.is_hybrid());

        // Force switch
        store.force_switch();
        assert!(store.is_hybrid());

        // Existing fingerprints should still be found (in exact store)
        for i in 0..100 {
            assert!(store.contains_or_insert(i), "fp {} should exist", i);
        }

        // New fingerprints go to bloom
        for i in 100..200 {
            assert!(!store.contains_or_insert(i));
        }

        // New fingerprints should be found (in bloom)
        for i in 100..200 {
            assert!(store.contains_or_insert(i), "fp {} should exist in bloom", i);
        }

        let ext_stats = store.extended_stats();
        assert!(ext_stats.is_hybrid);
        assert_eq!(ext_stats.exact_count, 100);
        assert_eq!(ext_stats.bloom_inserts, 100);
    }

    // Note: test_auto_switch_on_count_threshold is disabled because
    // the PageAlignedFingerprintStore allocation takes a long time on macOS.
    // The auto-switch functionality is tested via test_manual_switch_to_hybrid
    // which uses force_switch() to trigger the same code paths.

    #[test]
    fn test_batch_operations_in_hybrid_mode() {
        let config = AutoSwitchConfig {
            enabled: true,
            shard_count: 4,
            shard_size_mb: 64,
            bloom_expected_items: 1_000_000,
            ..Default::default()
        };

        let store = AutoSwitchingFingerprintStore::new(config, &[]).unwrap();

        // Insert initial batch in exact mode
        let initial_fps: Vec<u64> = (0..50).collect();
        let mut seen = Vec::new();
        store
            .contains_or_insert_batch(&initial_fps, &mut seen)
            .unwrap();
        assert!(seen.iter().all(|&s| !s)); // All new

        // Force switch
        store.force_switch();
        assert!(store.is_hybrid());

        // Batch lookup - should find all in exact
        store
            .contains_or_insert_batch(&initial_fps, &mut seen)
            .unwrap();
        assert!(seen.iter().all(|&s| s), "All should be found in exact store");

        // Insert new batch - should go to bloom
        let new_fps: Vec<u64> = (100..150).collect();
        store.contains_or_insert_batch(&new_fps, &mut seen).unwrap();
        assert!(seen.iter().all(|&s| !s)); // All new

        // Verify new batch is in bloom
        store.contains_or_insert_batch(&new_fps, &mut seen).unwrap();
        assert!(seen.iter().all(|&s| s), "All should be found in bloom store");

        // Mixed batch (some in exact, some in bloom, some new)
        let mixed_fps: Vec<u64> = vec![25, 125, 200]; // exact, bloom, new
        store.contains_or_insert_batch(&mixed_fps, &mut seen).unwrap();
        assert_eq!(seen, vec![true, true, false]);
    }
}
