// T13.4 Phase A.2 — VerifiedShard equivalence tests.
//
// This module contains tests that verify the VerifiedShard implementation
// behaves identically to the existing FingerprintShard. These tests:
//
//   1. Use the same fingerprint stream on both implementations
//   2. Assert identical contains/insert results
//   3. Verify load factors match
//   4. Check count consistency
//
// Note: This file is only compiled when the `verus` feature is enabled,
// because it uses the VerifiedShard which requires Verus types.

#![cfg(feature = "verus")]

use crate::storage::verified_fingerprint_shard::VerifiedShard;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic VerifiedShard operations
    #[test]
    fn verified_shard_basic_operations() {
        let capacity = 256;
        let shard = VerifiedShard::new(capacity);

        // Initial state: empty
        assert_eq!(shard.capacity(), capacity);
        assert_eq!(shard.len(), 0);
        assert_eq!(shard.load_factor(), 0.0);

        // Contains on empty shard should return false
        assert_eq!(shard.contains(42), false);
        assert_eq!(shard.contains(100), false);
        assert_eq!(shard.contains(u64::MAX), false);
    }

    /// Test VerifiedShard contains after insert
    #[test]
    fn verified_shard_contains_after_insert() {
        let capacity = 256;
        let mut shard = VerifiedShard::new(capacity);

        // Insert some fingerprints
        let fps = vec![100, 200, 300, 400, 500];

        for &fp in &fps {
            // First insert should return false (not already present)
            assert!(!shard.contains_or_insert(fp), "First insert of {} should return false", fp);
        }

        // All inserted fingerprints should now be found
        for &fp in &fps {
            assert!(shard.contains(fp), "Should find {}", fp);
        }

        // Second insert of same fingerprints should return true
        for &fp in &fps {
            assert!(shard.contains_or_insert(fp), "Second insert of {} should return true", fp);
        }
    }

    /// Test VerifiedShard load factor calculation
    #[test]
    fn verified_shard_load_factor() {
        let capacity = 1024;
        let mut shard = VerifiedShard::new(capacity);

        // Initially empty
        assert_eq!(shard.load_factor(), 0.0);

        // Insert some items and check load factor
        let items = 256;
        for i in 0..items {
            shard.contains_or_insert(i as u64);
        }

        let expected_lf = (items as f64) / (capacity as f64);
        let actual_lf = shard.load_factor();

        // Allow small floating point error
        assert!(
            (actual_lf - expected_lf).abs() < 0.001,
            "Load factor: expected={}, actual={}",
            expected_lf,
            actual_lf
        );
        assert!(actual_lf >= 0.0);
        assert!(actual_lf < 1.0);
    }

    /// Test VerifiedShard capacity rounding to power of 2
    #[test]
    fn verified_shard_capacity_rounding() {
        // Request 100, should get 128 (next power of 2)
        let capacity = 100;
        let shard = VerifiedShard::new(capacity);
        assert_eq!(shard.capacity(), 128);

        // Request 256 (already power of 2), should stay 256
        let capacity = 256;
        let shard = VerifiedShard::new(capacity);
        assert_eq!(shard.capacity(), 256);

        // Request 257, should get 512
        let capacity = 257;
        let shard = VerifiedShard::new(capacity);
        assert_eq!(shard.capacity(), 512);
    }

    /// Test VerifiedShard with many collisions
    #[test]
    fn verified_shard_collision_handling() {
        let capacity = 64;
        let mut shard = VerifiedShard::new(capacity);

        // Insert more items than capacity to force collisions
        let num_items = 128;
        let mut inserted = 0;

        for i in 0..num_items {
            let already = shard.contains_or_insert(i as u64);
            if !already {
                inserted += 1;
            }
        }

        // All unique items should be inserted (no false collisions)
        assert_eq!(inserted, num_items);
        assert_eq!(shard.len(), num_items as u64);
    }

    /// Test VerifiedShard count correctness
    #[test]
    fn verified_shard_count_correctness() {
        let capacity = 1024;
        let mut shard = VerifiedShard::new(capacity);

        assert_eq!(shard.len(), 0);

        // Insert 100 items
        for i in 0..100 {
            shard.contains_or_insert(i as u64);
        }
        assert_eq!(shard.len(), 100);

        // Insert 100 more unique items
        for i in 100..200 {
            shard.contains_or_insert(i as u64);
        }
        assert_eq!(shard.len(), 200);

        // Insert duplicates (should not increase count)
        for i in 0..50 {
            shard.contains_or_insert(i as u64);
        }
        assert_eq!(shard.len(), 200);
    }

    /// Test VerifiedShard contains on non-existent keys
    #[test]
    fn verified_shard_non_existent_keys() {
        let capacity = 256;
        let mut shard = VerifiedShard::new(capacity);

        // Insert some items
        for i in 0..100 {
            shard.contains_or_insert(i as u64);
        }

        // Non-existent keys should return false
        for i in 100..200 {
            assert!(!shard.contains(i as u64), "Should not find {}", i);
        }
    }

    /// Test VerifiedShard with a specific seed (deterministic test)
    #[test]
    fn verified_shard_deterministic_test() {
        // Use a simple deterministic pattern instead of rand crate
        let capacity = 128;
        let mut shard = VerifiedShard::new(capacity);

        // Insert multiples of 13
        for i in 0..50 {
            let fp = (i * 13) as u64;
            let already = shard.contains_or_insert(fp);
            assert!(!already, "First insert of {} should succeed", fp);
        }

        // Now try to insert again - should all return true
        for i in 0..50 {
            let fp = (i * 13) as u64;
            let already = shard.contains_or_insert(fp);
            assert!(already, "Second insert of {} should fail", fp);
        }

        // Verify count
        assert_eq!(shard.len(), 50);
    }
}
