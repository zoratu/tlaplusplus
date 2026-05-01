//! `tlaplusplus` binary entry point.
//!
//! This file is intentionally tiny: all CLI code lives in
//! `tlaplusplus::cli` (see `src/cli/mod.rs`). The 11,711-line monolith
//! that used to live here was split into a `cli/` module tree per
//! `docs/main-refactor-plan.md`.

fn main() -> anyhow::Result<()> {
    tlaplusplus::cli::run()
}
