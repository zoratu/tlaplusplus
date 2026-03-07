#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tlaplusplus::storage::bloom_fingerprint_store::BloomFingerprintStore;

/// Operations to perform on bloom filter
#[derive(Debug, Arbitrary)]
enum BloomOp {
    /// Insert a fingerprint (returns true if already present)
    ContainsOrInsert(u64),
    /// Insert batch of fingerprints
    ContainsOrInsertBatch([u64; 8]),
}

/// Sequence of operations to perform
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Operations to perform (limited to avoid timeout)
    ops: [BloomOp; 64],
}

fuzz_target!(|input: FuzzInput| {
    // Create bloom filter with minimal parameters for fast fuzzing
    let store = match BloomFingerprintStore::new(
        10_000, // Expected items
        0.01,   // 1% false positive rate
        4,      // 4 shards (power of 2)
        1,      // 1 NUMA node
    ) {
        Ok(s) => s,
        Err(_) => return, // Skip if creation fails
    };

    // Track what we've inserted
    let mut inserted = std::collections::HashSet::new();

    for op in &input.ops {
        match op {
            BloomOp::ContainsOrInsert(fp) => {
                let existed = store.contains_or_insert(*fp);
                // If we inserted it before, bloom filter MUST say it existed (no false negatives)
                if inserted.contains(fp) {
                    assert!(existed, "Bloom filter must not have false negatives");
                } else {
                    // First time seeing this, record it
                    inserted.insert(*fp);
                }
                // False positives are allowed (existed=true for new items)
            }
            BloomOp::ContainsOrInsertBatch(fps) => {
                let mut seen = vec![false; fps.len()];
                if store.contains_or_insert_batch(fps, &mut seen).is_ok() {
                    for (fp, existed) in fps.iter().zip(seen.iter()) {
                        if inserted.contains(fp) {
                            // Must have been seen as existing
                            assert!(*existed, "Bloom filter must not have false negatives in batch");
                        } else {
                            inserted.insert(*fp);
                        }
                    }
                }
            }
        }
    }

    // Final verification: all inserted fingerprints should still be "found"
    for fp in &inserted {
        assert!(
            store.contains_or_insert(*fp),
            "All inserted fps must be found (no false negatives)"
        );
    }
});
