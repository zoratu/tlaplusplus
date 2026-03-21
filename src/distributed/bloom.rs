//! Bloom filter for probabilistic dedup of explored states across cluster nodes.
//!
//! Each node maintains a local bloom filter of fingerprints it has explored.
//! Periodically, nodes exchange their bloom filters so that other nodes can
//! skip states that have likely already been explored elsewhere.
//!
//! This is NOT the primary fingerprint store — that remains the existing
//! lock-free hash table. This is a compact summary for cross-node dedup.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, Ordering};

/// A concurrent bloom filter using atomic byte operations.
///
/// Supports lock-free `insert` and `may_contain` from multiple threads.
/// Serialization snapshots the current state for network exchange.
#[derive(Debug)]
pub struct BloomFilter {
    /// Bit array stored as atomic bytes (each byte holds 8 bits).
    bits: Vec<AtomicU8>,
    /// Total number of bits in the filter (bits.len() * 8).
    num_bits: usize,
    /// Number of hash functions (k).
    num_hashes: usize,
}

/// Serializable snapshot of a bloom filter for network exchange.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BloomSnapshot {
    pub bits: Vec<u8>,
    pub num_bits: usize,
    pub num_hashes: usize,
}

impl BloomFilter {
    /// Create a new bloom filter sized for the given expected item count
    /// and false positive rate.
    ///
    /// Uses optimal sizing formulas:
    /// - m = -n * ln(p) / (ln(2))^2
    /// - k = (m/n) * ln(2)
    ///
    /// The filter size is clamped to a minimum of 1 KB and maximum of 64 MB.
    pub fn new(expected_items: usize, fpr: f64) -> Self {
        let expected_items = expected_items.max(1);
        let fpr = fpr.max(1e-10).min(0.5);

        let ln2 = std::f64::consts::LN_2;
        let ln2_sq = ln2 * ln2;

        // Optimal number of bits
        let m = (-(expected_items as f64) * fpr.ln() / ln2_sq).ceil() as usize;
        // Clamp to [8192, 512M] bits = [1KB, 64MB]
        let m = m.max(8192).min(512 * 1024 * 1024);
        // Round up to byte boundary
        let m = (m + 7) & !7;

        // Optimal number of hash functions
        let k = ((m as f64 / expected_items as f64) * ln2).ceil() as usize;
        let k = k.max(1).min(20);

        let num_bytes = m / 8;
        let mut bits = Vec::with_capacity(num_bytes);
        for _ in 0..num_bytes {
            bits.push(AtomicU8::new(0));
        }

        BloomFilter {
            bits,
            num_bits: m,
            num_hashes: k,
        }
    }

    /// Create a bloom filter with explicit parameters (for testing or
    /// when receiving a remote filter's parameters).
    pub fn with_params(num_bits: usize, num_hashes: usize) -> Self {
        let num_bits = (num_bits + 7) & !7;
        let num_bytes = num_bits / 8;
        let mut bits = Vec::with_capacity(num_bytes);
        for _ in 0..num_bytes {
            bits.push(AtomicU8::new(0));
        }
        BloomFilter {
            bits,
            num_bits,
            num_hashes,
        }
    }

    /// Insert a fingerprint into the filter.
    ///
    /// Uses atomic OR operations — safe to call concurrently from multiple threads.
    #[inline]
    pub fn insert(&self, fp: u64) {
        for i in 0..self.num_hashes {
            let bit_pos = self.hash_position(fp, i);
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            self.bits[byte_idx].fetch_or(1 << bit_idx, Ordering::Relaxed);
        }
    }

    /// Check if a fingerprint may be in the filter.
    ///
    /// Returns `true` if the fingerprint is possibly present (may be false positive).
    /// Returns `false` if the fingerprint is definitely NOT present.
    #[inline]
    pub fn may_contain(&self, fp: u64) -> bool {
        for i in 0..self.num_hashes {
            let bit_pos = self.hash_position(fp, i);
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if self.bits[byte_idx].load(Ordering::Relaxed) & (1 << bit_idx) == 0 {
                return false;
            }
        }
        true
    }

    /// Create a serializable snapshot of the current filter state.
    pub fn snapshot(&self) -> BloomSnapshot {
        let bits: Vec<u8> = self
            .bits
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .collect();
        BloomSnapshot {
            bits,
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
        }
    }

    /// Merge another filter's bits into this one (bitwise OR).
    ///
    /// The other filter must have the same size and number of hash functions.
    /// If sizes differ, this is a no-op (logged as warning).
    pub fn merge_snapshot(&self, snapshot: &BloomSnapshot) {
        if snapshot.num_bits != self.num_bits || snapshot.num_hashes != self.num_hashes {
            eprintln!(
                "[bloom] cannot merge snapshot: size mismatch (local {}b/{}h, remote {}b/{}h)",
                self.num_bits, self.num_hashes, snapshot.num_bits, snapshot.num_hashes
            );
            return;
        }
        for (i, &byte_val) in snapshot.bits.iter().enumerate() {
            if byte_val != 0 {
                self.bits[i].fetch_or(byte_val, Ordering::Relaxed);
            }
        }
    }

    /// Compute the bit position for the i-th hash function applied to fingerprint fp.
    ///
    /// Uses double hashing: h(fp, i) = (h1(fp) + i * h2(fp)) mod num_bits
    /// where h1 and h2 are derived from splitmix64 variants.
    #[inline]
    fn hash_position(&self, fp: u64, i: usize) -> usize {
        let h1 = splitmix64(fp);
        let h2 = splitmix64(fp.wrapping_add(0x517cc1b727220a95));
        (h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.num_bits as u64) as usize
    }
}

/// splitmix64 finalizer — fast, deterministic, good distribution.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query() {
        let bloom = BloomFilter::new(1000, 0.01);
        bloom.insert(42);
        bloom.insert(12345);
        assert!(bloom.may_contain(42));
        assert!(bloom.may_contain(12345));
    }

    #[test]
    fn absent_items_usually_negative() {
        let bloom = BloomFilter::new(1000, 0.01);
        for i in 0..100u64 {
            bloom.insert(i);
        }
        // Check items that were NOT inserted — most should return false
        let mut false_positives = 0;
        for i in 1000..2000u64 {
            if bloom.may_contain(i) {
                false_positives += 1;
            }
        }
        // With 1% FPR and 1000 queries, expect ~10 false positives.
        // Allow up to 50 to account for variance.
        assert!(
            false_positives < 50,
            "too many false positives: {} out of 1000",
            false_positives
        );
    }

    #[test]
    fn snapshot_roundtrip() {
        let bloom = BloomFilter::new(1000, 0.01);
        for i in 0..50u64 {
            bloom.insert(i);
        }
        let snap = bloom.snapshot();

        let bloom2 = BloomFilter::with_params(snap.num_bits, snap.num_hashes);
        bloom2.merge_snapshot(&snap);

        for i in 0..50u64 {
            assert!(
                bloom2.may_contain(i),
                "item {} should be present after merge",
                i
            );
        }
    }

    #[test]
    fn merge_combines_filters() {
        let bloom1 = BloomFilter::new(1000, 0.01);
        let bloom2 = BloomFilter::with_params(bloom1.num_bits, bloom1.num_hashes);

        for i in 0..50u64 {
            bloom1.insert(i);
        }
        for i in 50..100u64 {
            bloom2.insert(i);
        }

        let snap2 = bloom2.snapshot();
        bloom1.merge_snapshot(&snap2);

        // bloom1 should now contain items from both
        for i in 0..100u64 {
            assert!(
                bloom1.may_contain(i),
                "item {} should be present after merge",
                i
            );
        }
    }

    #[test]
    fn size_clamping() {
        // Very small expected items should still create a reasonable filter
        let bloom = BloomFilter::new(1, 0.5);
        assert!(bloom.num_bits >= 8192, "filter should be at least 1KB");

        // Very large expected items should be clamped
        let bloom = BloomFilter::new(10_000_000_000, 0.0001);
        assert!(
            bloom.num_bits <= 512 * 1024 * 1024,
            "filter should be at most 64MB"
        );
    }
}
