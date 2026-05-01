//! Memory budget computations.
//!
//! Pure helpers that translate cgroup limits + user-supplied memory caps
//! into effective per-component budgets (fp store cache size, queue
//! in-memory limit). Also predicates for whether to enable the
//! file-backed fingerprint store and the memory-monitor thread.

use crate::system::cgroup_memory_max_bytes;

use super::EngineConfig;

pub(super) fn compute_effective_memory_max(config: &EngineConfig) -> Option<u64> {
    let cgroup_limit = if config.enforce_cgroups {
        cgroup_memory_max_bytes()
    } else {
        None
    };
    match (config.memory_max_bytes, cgroup_limit) {
        (Some(user), Some(cgroup)) => Some(user.min(cgroup)),
        (Some(user), None) => Some(user),
        (None, Some(cgroup)) => Some(cgroup),
        (None, None) => None,
    }
}

pub(super) fn should_enable_file_backed_fingerprint_store(
    config: &EngineConfig,
    effective_memory_max: Option<u64>,
) -> bool {
    !config.use_bloom_fingerprints && effective_memory_max.is_some()
}

pub(super) fn should_start_fingerprint_memory_monitor(
    config: &EngineConfig,
    effective_memory_max: Option<u64>,
    backing_dir_enabled: bool,
) -> bool {
    should_enable_file_backed_fingerprint_store(config, effective_memory_max) && backing_dir_enabled
}

pub(super) fn apply_memory_budget(
    config: &EngineConfig,
    effective_memory_max: Option<u64>,
) -> (u64, usize) {
    let mut fp_cache = config.fp_cache_capacity_bytes;
    let mut queue_limit = config.queue_inmem_limit;

    if let Some(memory_max) = effective_memory_max {
        // Memory budget split:
        //   25% fingerprint store (sized separately, see fp store init)
        //   10% per-worker fingerprint cache
        //   50% state queue (in-memory portion)
        //   15% OS, worker stacks, other overhead
        let budget_fp_cache = (memory_max.saturating_mul(10) / 100).max(64 * 1024 * 1024);
        fp_cache = fp_cache.min(budget_fp_cache);

        let state_bytes = config.estimated_state_bytes.max(1) as u64;
        let budget_queue_items = (memory_max.saturating_mul(50) / 100) / state_bytes;
        let budget_queue_items = budget_queue_items.max(10_000) as usize;
        queue_limit = queue_limit.min(budget_queue_items);

        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "Memory budget ({:.1} GB): fp_cache={:.0}MB, queue_limit={}M items",
                memory_max as f64 / (1024.0 * 1024.0 * 1024.0),
                budget_fp_cache as f64 / (1024.0 * 1024.0),
                budget_queue_items / 1_000_000
            );
        }
    }

    (fp_cache, queue_limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_backed_fingerprint_store_uses_effective_memory_cap() {
        let config = EngineConfig {
            use_bloom_fingerprints: false,
            memory_max_bytes: None,
            enforce_cgroups: true,
            ..EngineConfig::default()
        };

        assert!(should_enable_file_backed_fingerprint_store(
            &config,
            Some(8 * 1024 * 1024 * 1024)
        ));
        assert!(!should_enable_file_backed_fingerprint_store(&config, None));

        let bloom_config = EngineConfig {
            use_bloom_fingerprints: true,
            ..config.clone()
        };
        assert!(!should_enable_file_backed_fingerprint_store(
            &bloom_config,
            Some(8 * 1024 * 1024 * 1024)
        ));
    }

    #[test]
    fn memory_monitor_requires_file_backed_fingerprint_store() {
        let config = EngineConfig {
            use_bloom_fingerprints: false,
            ..EngineConfig::default()
        };

        assert!(should_start_fingerprint_memory_monitor(
            &config,
            Some(4 * 1024 * 1024 * 1024),
            true
        ));
        assert!(!should_start_fingerprint_memory_monitor(
            &config,
            Some(4 * 1024 * 1024 * 1024),
            false
        ));
        assert!(!should_start_fingerprint_memory_monitor(
            &EngineConfig {
                use_bloom_fingerprints: true,
                ..config.clone()
            },
            Some(4 * 1024 * 1024 * 1024),
            true
        ));
    }
}
