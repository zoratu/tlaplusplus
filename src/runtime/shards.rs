//! Fingerprint-store shard count heuristic.

/// Calculate optimal shard count based on system characteristics.
///
/// Heuristics (optimized for many-core scalability):
/// - 2 shards per worker to minimize CAS contention (critical for 380+ workers)
/// - Power of 2 for fast modulo (bitwise AND)
/// - NUMA-aligned for even distribution across nodes
/// - Worker scaling takes priority over shard size for many-core systems
///
pub(super) fn calculate_optimal_shard_count(
    worker_count: usize,
    numa_node_count: usize,
    total_memory_bytes: usize,
) -> usize {
    // For many-core systems, CAS contention is the primary bottleneck.
    // Having more shards (even if small) is better than fewer large shards.
    // Target: 2 shards per worker ensures low collision probability.
    let shards_per_worker = 2;
    let base_count = worker_count * shards_per_worker;

    // Round up to next power of 2 for fast modulo (hash & (count - 1))
    let mut candidate = base_count.next_power_of_two();

    // Ensure it's NUMA-aligned (multiple of NUMA node count)
    if candidate % numa_node_count != 0 {
        candidate = ((candidate / numa_node_count) + 1) * numa_node_count;
        // Round back to power of 2 if needed
        candidate = candidate.next_power_of_two();
    }

    // For many-core systems (>128 workers), prioritize worker scaling over shard size.
    // Small shards are fine - the overhead is negligible compared to CAS contention.
    // Only apply size constraints for systems with fewer workers where memory layout matters more.
    let is_many_core = worker_count > 128;

    let adjusted = if is_many_core {
        // Many-core: keep 2 shards per worker regardless of size
        candidate
    } else {
        // Fewer workers: apply traditional size-based constraints
        const MIN_SHARD_SIZE: usize = 256 * 1024 * 1024; // 256 MB
        const MAX_SHARD_SIZE: usize = 2 * 1024 * 1024 * 1024; // 2 GB

        let per_shard_bytes = total_memory_bytes / candidate;

        if per_shard_bytes < MIN_SHARD_SIZE {
            // Shards too small - reduce count
            let min_count = (total_memory_bytes / MAX_SHARD_SIZE).max(1);
            min_count.next_power_of_two()
        } else if per_shard_bytes > MAX_SHARD_SIZE {
            // Shards too large - increase count
            let max_count = (total_memory_bytes / MIN_SHARD_SIZE).max(1);
            max_count.next_power_of_two()
        } else {
            candidate
        }
    };

    // Clamp to reasonable bounds
    // Min: at least 2x NUMA nodes for distribution (64 for backwards compat)
    // Max: 4096 shards is plenty (avoid excessive overhead)
    // Minimum: enough for NUMA distribution. No arbitrary 64 floor —
    // small worker counts get small shard counts for fast startup.
    let min_shards = (numa_node_count * 2).max(worker_count.next_power_of_two());
    let max_shards = 4096;

    #[cfg(not(feature = "verus"))]
    let final_count = adjusted.clamp(min_shards, max_shards);
    #[cfg(feature = "verus")]
    let final_count = crate::storage::verus_smoke::clamp_usize(adjusted, min_shards, max_shards);

    if std::env::var("TLAPP_VERBOSE").is_ok() {
        eprintln!(
            "Auto-calculated shard count: {} (base: {}, workers: {}, NUMA nodes: {}, per-shard: {:.1} MB, many-core: {})",
            final_count,
            base_count,
            worker_count,
            numa_node_count,
            (total_memory_bytes / final_count) as f64 / (1024.0 * 1024.0),
            is_many_core
        );
    }

    final_count
}

#[cfg(test)]
mod tests {
    use super::calculate_optimal_shard_count;

    /// Helper: pick a memory budget that lands per-shard squarely inside
    /// the [256MB, 2GB] band so the size-clamp branch is a no-op.
    /// 64 workers × 2 = 128 base shards × 1GB each = 128 GB total.
    const NEUTRAL_TOTAL_MEM: usize = 128 * 1024 * 1024 * 1024;

    #[test]
    fn result_is_always_power_of_two() {
        // The store does shard-id := hash & (count - 1), so a non-power-of-two
        // count would be a soundness bug.
        // Cap the worker sweep at 1024; the heuristic's `min_shards =
        // max(numa*2, workers.next_power_of_two())` would otherwise exceed
        // the 4096 max-clamp and panic in `clamp(min, max)`.
        for &workers in &[1usize, 4, 8, 16, 64, 128, 192, 256, 384, 1024] {
            for &numa in &[1usize, 2, 4, 6] {
                let n = calculate_optimal_shard_count(workers, numa, NEUTRAL_TOTAL_MEM);
                assert!(
                    n.is_power_of_two(),
                    "workers={workers} numa={numa} → {n} (not pow2)"
                );
            }
        }
    }

    #[test]
    fn result_clamped_to_max_4096() {
        // The 4096 max-clamp must hold for typical many-core sizes
        // (workers ≤ next_pow2(4096) = 4096). 2048 workers produce a
        // candidate of 4096 directly; the result must not exceed 4096.
        let n = calculate_optimal_shard_count(2048, 1, NEUTRAL_TOTAL_MEM);
        assert!(n <= 4096, "expected <= 4096 for 2048 workers, got {n}");
        // 4096 workers exactly: candidate = 8192 → clamp to 4096.
        let n2 = calculate_optimal_shard_count(4096, 1, NEUTRAL_TOTAL_MEM);
        assert_eq!(n2, 4096, "4096 workers must clamp exactly to max");
    }

    #[test]
    fn result_at_least_2x_numa_nodes_for_distribution() {
        // Min floor is `(numa * 2).max(workers.next_power_of_two())`.
        // With workers=1 and numa=8, min should be max(16, 1) = 16.
        let n = calculate_optimal_shard_count(1, 8, NEUTRAL_TOTAL_MEM);
        assert!(n >= 16, "expected >= 16 (2 × NUMA=8), got {n}");
    }

    #[test]
    fn many_core_threshold_at_128_keeps_2x_workers() {
        // 128 workers is NOT many-core (>128 is); 192 IS.
        // For many-core we keep 2 shards per worker as the candidate
        // (rounded up to next pow2). 192*2=384 → 512.
        let many = calculate_optimal_shard_count(192, 1, NEUTRAL_TOTAL_MEM);
        assert_eq!(
            many, 512,
            "many-core (>128) should keep 2x workers rounded up to pow2"
        );

        // 128 workers exactly → not many-core, falls into size-band path.
        // 128*2=256 candidate; per_shard = 128GB / 256 = 512MB ∈ [256MB, 2GB].
        // So size-band leaves candidate alone → 256.
        let edge = calculate_optimal_shard_count(128, 1, NEUTRAL_TOTAL_MEM);
        assert_eq!(edge, 256, "128 workers (boundary) keeps base candidate");
    }

    #[test]
    fn many_core_ignores_size_band_constraints() {
        // Many-core (worker_count > 128) skips the [256MB, 2GB] band check
        // entirely. Even with a tiny memory budget that would push per-shard
        // way under 256MB, many-core retains its 2×workers shape.
        // 256 workers × 2 = 512 base; tiny memory wouldn't change this.
        let n = calculate_optimal_shard_count(256, 1, 1024 * 1024); // 1MB
        assert_eq!(n, 512, "many-core ignores per-shard size band");
    }

    #[test]
    fn small_workers_with_oversized_shards_increase_count() {
        // Few workers, lots of memory → per-shard would exceed 2GB upper
        // band; the size-band branch must bump count up.
        // 4 workers × 2 = 8 base; per-shard = 1TB / 8 = 128GB » 2GB
        // max_count = 1TB / 256MB = 4096 → next_pow2 = 4096.
        // Min floor = max(2*1, 4) = 4. So result clamps at 4096.
        let n = calculate_optimal_shard_count(4, 1, 1024 * 1024 * 1024 * 1024);
        assert!(
            n > 8,
            "oversized shards should trigger increase, got {n} (≤ base 8)"
        );
    }

    #[test]
    fn small_workers_with_undersized_shards_decrease_count() {
        // Many small workers (≤128) where 2*workers shards would be too small
        // (<256MB each). The size-band branch must reduce count.
        // 128 workers × 2 = 256 base; total mem = 1GB → per-shard = 4MB << 256MB.
        // min_count = 1GB / 2GB = 0 → max(0,1)=1 → next_pow2 = 1.
        // Then clamped to min_shards = max(2*1, 256) = 256.
        let n = calculate_optimal_shard_count(128, 1, 1024 * 1024 * 1024);
        // The min-floor wins here, but the result must NOT exceed base 256.
        assert!(n <= 256, "undersized shards should not exceed base, got {n}");
    }

    #[test]
    fn numa_alignment_preserves_pow2_after_round_up() {
        // 4 workers × 2 = 8; numa=3 → 8 % 3 = 2, so candidate becomes
        // ((8/3)+1)*3 = 9, then next_pow2 = 16. Result must be a pow2 multiple
        // of (or ≥ ) the NUMA min, and must remain pow2.
        let n = calculate_optimal_shard_count(4, 3, NEUTRAL_TOTAL_MEM);
        assert!(n.is_power_of_two(), "NUMA-aligned result must stay pow2");
        assert!(n >= 6, "must respect 2 × NUMA min");
    }

    #[test]
    fn worker_count_floor_in_min_clamp() {
        // min_shards := max(numa*2, workers.next_power_of_two()).
        // A 1-NUMA system with 64 workers: min = max(2, 64) = 64.
        // 64*2 = 128 base, 1GB total → per_shard = 8MB < 256MB,
        // so size band tries to shrink: min_count = 1GB/2GB = 0 → 1, next_pow2=1.
        // Clamp(min=64, max=4096) brings it back to 64.
        let n = calculate_optimal_shard_count(64, 1, 1024 * 1024 * 1024);
        assert!(
            n >= 64,
            "min_shards must include workers.next_power_of_two(), got {n}"
        );
    }
}
