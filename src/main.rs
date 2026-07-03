//! `tlaplusplus` binary entry point.
//!
//! This file is intentionally tiny: all CLI code lives in
//! `tlaplusplus::cli` (see `src/cli/mod.rs`). The 11,711-line monolith
//! that used to live here was split into a `cli/` module tree per
//! `docs/main-refactor-plan.md`.

/// Global allocator: jemalloc (Linux only). The parallel work-stealing model
/// checker allocates a state on one worker and frees it on another (after a
/// steal), so the hot pattern is cross-thread frees — which glibc malloc
/// serializes through per-arena locks (~39% of self-time landed in allocator
/// internals in a `perf` profile of MCKVSSafetyMedium at 32 workers).
/// jemalloc's sharded thread caches absorb this: +29% state-exploration
/// throughput, +4% peak RSS. See the PR for the glibc/mimalloc/jemalloc A/B.
#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> anyhow::Result<()> {
    tlaplusplus::cli::run()
}
