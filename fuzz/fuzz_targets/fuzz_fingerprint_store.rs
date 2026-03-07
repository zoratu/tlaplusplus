#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tlaplusplus::storage::page_aligned_fingerprint_store::{
    FingerprintStoreConfig, PageAlignedFingerprintStore,
};

/// Operations to perform on fingerprint store
#[derive(Debug, Arbitrary)]
enum FingerprintOp {
    /// Insert a fingerprint, returns true if already present
    Insert(u64),
    /// Check if fingerprint exists
    Contains(u64),
    /// Insert batch of fingerprints (limited size)
    InsertBatch([u64; 8]),
}

/// Sequence of operations to perform
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Operations to perform (limited to avoid timeout)
    ops: [FingerprintOp; 32],
}

fuzz_target!(|input: FuzzInput| {
    // Create store with minimal config for fast fuzzing
    let config = FingerprintStoreConfig {
        shard_count: 4,      // Minimal shards
        expected_items: 1000,
        shard_size_mb: 1,    // Minimal memory
    };

    // Try to create store - may fail on NUMA detection, that's ok
    let store = match PageAlignedFingerprintStore::new(config, &[]) {
        Ok(s) => s,
        Err(_) => return, // Skip if creation fails (e.g., NUMA not available)
    };

    // Track what we've inserted for verification
    let mut inserted = std::collections::HashSet::new();

    for op in &input.ops {
        match op {
            FingerprintOp::Insert(fp) => {
                let existed = store.contains_or_insert(*fp);
                if inserted.contains(fp) {
                    assert!(existed, "Previously inserted fp should exist");
                } else {
                    inserted.insert(*fp);
                }
            }
            FingerprintOp::Contains(fp) => {
                let exists = store.contains(*fp);
                if inserted.contains(fp) {
                    assert!(exists, "Inserted fp should be found");
                }
            }
            FingerprintOp::InsertBatch(fps) => {
                let mut seen = vec![false; fps.len()];
                if store.contains_or_insert_batch(fps, &mut seen).is_ok() {
                    for (fp, existed) in fps.iter().zip(seen.iter()) {
                        if inserted.contains(fp) {
                            assert!(*existed, "Previously inserted fp should exist in batch");
                        } else {
                            inserted.insert(*fp);
                        }
                    }
                }
            }
        }
    }

    // Final verification: all inserted fingerprints should still exist
    for fp in &inserted {
        assert!(store.contains(*fp), "All inserted fps should persist");
    }
});
