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
        #[cfg(not(feature = "verus"))]
        (Some(user), Some(cgroup)) => Some(user.min(cgroup)),
        #[cfg(feature = "verus")]
        (Some(user), Some(cgroup)) => Some(crate::storage::verus_smoke::min_u64(user, cgroup)),
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
        #[cfg(not(feature = "verus"))]
        { fp_cache = fp_cache.min(budget_fp_cache); }
        #[cfg(feature = "verus")]
        { fp_cache = crate::storage::verus_smoke::min_u64(fp_cache, budget_fp_cache); }

        let state_bytes = config.estimated_state_bytes.max(1) as u64;
        let budget_queue_items = (memory_max.saturating_mul(50) / 100) / state_bytes;
        let budget_queue_items = budget_queue_items.max(10_000) as usize;
        #[cfg(not(feature = "verus"))]
        { queue_limit = queue_limit.min(budget_queue_items); }
        #[cfg(feature = "verus")]
        { queue_limit = crate::storage::verus_smoke::min_usize(queue_limit, budget_queue_items); }

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
    fn compute_effective_memory_max_user_only() {
        // (Some(user), None) — user cap returned verbatim. Use
        // enforce_cgroups=false so cgroup_limit is unconditionally None
        // and the test is platform-independent.
        let user_cap: u64 = 4 * 1024 * 1024 * 1024;
        let config = EngineConfig {
            enforce_cgroups: false,
            memory_max_bytes: Some(user_cap),
            ..EngineConfig::default()
        };
        assert_eq!(compute_effective_memory_max(&config), Some(user_cap));
    }

    #[test]
    fn compute_effective_memory_max_neither_returns_none() {
        // (None, None) — no user cap, cgroup checks skipped.
        let config = EngineConfig {
            enforce_cgroups: false,
            memory_max_bytes: None,
            ..EngineConfig::default()
        };
        assert!(compute_effective_memory_max(&config).is_none());
    }

    #[test]
    fn compute_effective_memory_max_cgroup_disabled_ignores_host() {
        // enforce_cgroups=false → cgroup_limit is always None even on Linux,
        // so the only signal is the user cap.
        let user_cap: u64 = 8 * 1024 * 1024 * 1024;
        let cfg_none = EngineConfig {
            enforce_cgroups: false,
            memory_max_bytes: None,
            ..EngineConfig::default()
        };
        assert_eq!(compute_effective_memory_max(&cfg_none), None);

        let cfg_some = EngineConfig {
            enforce_cgroups: false,
            memory_max_bytes: Some(user_cap),
            ..EngineConfig::default()
        };
        assert_eq!(compute_effective_memory_max(&cfg_some), Some(user_cap));
    }

    #[test]
    fn apply_memory_budget_no_cap_returns_config_defaults() {
        // No effective cap → fp_cache and queue_limit pass through unchanged.
        let config = EngineConfig {
            fp_cache_capacity_bytes: 12345,
            queue_inmem_limit: 67890,
            ..EngineConfig::default()
        };
        let (fp_cache, queue_limit) = apply_memory_budget(&config, None);
        assert_eq!(fp_cache, 12345);
        assert_eq!(queue_limit, 67890);
    }

    #[test]
    fn apply_memory_budget_uses_10_percent_for_fp_cache() {
        // memory_max = 10 GiB → 10% = 1 GiB. With a config fp_cache far
        // larger than the budget cap, the budget must win.
        let memory_max: u64 = 10 * 1024 * 1024 * 1024;
        let expected_fp_cache: u64 = memory_max / 10;
        let config = EngineConfig {
            fp_cache_capacity_bytes: u64::MAX,
            queue_inmem_limit: usize::MAX,
            estimated_state_bytes: 256,
            ..EngineConfig::default()
        };
        let (fp_cache, _) = apply_memory_budget(&config, Some(memory_max));
        assert_eq!(
            fp_cache, expected_fp_cache,
            "fp_cache should be 10% of memory_max"
        );
    }

    #[test]
    fn apply_memory_budget_uses_50_percent_for_queue_items() {
        // memory_max = 10 GiB, state_bytes = 256 → queue items = 50% / 256.
        let memory_max: u64 = 10 * 1024 * 1024 * 1024;
        let state_bytes: u64 = 256;
        let expected_queue: usize = ((memory_max / 2) / state_bytes) as usize;
        let config = EngineConfig {
            fp_cache_capacity_bytes: u64::MAX,
            queue_inmem_limit: usize::MAX,
            estimated_state_bytes: state_bytes as usize,
            ..EngineConfig::default()
        };
        let (_, queue_limit) = apply_memory_budget(&config, Some(memory_max));
        assert_eq!(
            queue_limit, expected_queue,
            "queue_limit should be 50% of memory_max ÷ state_bytes"
        );
    }

    #[test]
    fn apply_memory_budget_keeps_smaller_user_value() {
        // The budget must be a CAP, not a setpoint: if the user's
        // configured value is already smaller, keep it.
        let memory_max: u64 = 10 * 1024 * 1024 * 1024;
        let user_fp_cache: u64 = 32 * 1024 * 1024; // 32 MB ≪ 1 GB budget
        let user_queue: usize = 1_000;
        let config = EngineConfig {
            fp_cache_capacity_bytes: user_fp_cache,
            queue_inmem_limit: user_queue,
            estimated_state_bytes: 256,
            ..EngineConfig::default()
        };
        let (fp_cache, queue_limit) = apply_memory_budget(&config, Some(memory_max));
        assert_eq!(fp_cache, user_fp_cache, "user-supplied fp_cache must win");
        assert_eq!(queue_limit, user_queue, "user-supplied queue_limit must win");
    }

    #[test]
    fn apply_memory_budget_floors_fp_cache_at_64mib() {
        // The 10% formula has a 64 MiB floor (e.g. for tiny memory_max).
        // memory_max = 1 MiB → 10% = ~100 KB, but the floor pushes to 64 MiB.
        let memory_max: u64 = 1024 * 1024;
        let config = EngineConfig {
            fp_cache_capacity_bytes: u64::MAX,
            queue_inmem_limit: usize::MAX,
            estimated_state_bytes: 256,
            ..EngineConfig::default()
        };
        let (fp_cache, _) = apply_memory_budget(&config, Some(memory_max));
        assert_eq!(fp_cache, 64 * 1024 * 1024, "fp_cache floor is 64 MiB");
    }

    #[test]
    fn apply_memory_budget_floors_queue_items_at_10_000() {
        // The queue formula has a 10K-item floor.
        // Tiny memory_max with huge state_bytes → 50% would be near zero.
        let memory_max: u64 = 1024 * 1024;
        let config = EngineConfig {
            fp_cache_capacity_bytes: u64::MAX,
            queue_inmem_limit: usize::MAX,
            estimated_state_bytes: 1024 * 1024, // 1 MB per state
            ..EngineConfig::default()
        };
        let (_, queue_limit) = apply_memory_budget(&config, Some(memory_max));
        assert_eq!(queue_limit, 10_000, "queue_limit floor is 10_000 items");
    }

    #[test]
    fn apply_memory_budget_zero_state_bytes_treated_as_one() {
        // estimated_state_bytes is .max(1) before division — guards against
        // divide-by-zero. 1 GiB / 2 / 1 = 512 Mi items (above floor).
        let memory_max: u64 = 1024 * 1024 * 1024;
        let config = EngineConfig {
            fp_cache_capacity_bytes: u64::MAX,
            queue_inmem_limit: usize::MAX,
            estimated_state_bytes: 0,
            ..EngineConfig::default()
        };
        let (_, queue_limit) = apply_memory_budget(&config, Some(memory_max));
        assert_eq!(queue_limit, ((memory_max / 2) as usize));
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
