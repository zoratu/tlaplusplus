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

    let final_count = adjusted.clamp(min_shards, max_shards);

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
