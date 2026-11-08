//! Chaff — feature-gated, intentional semantic regressions used to *measure the
//! sensitivity of our own test suite* (Examplar-style "conceptual mutation").
//!
//! # Why this exists
//!
//! Our differential/corpus gates are good at saying "this build agrees with
//! TLC". They do **not**, on their own, tell us whether a gate would actually
//! *fail* if a known bug class were reintroduced. Historically, real semantic
//! bugs reached `main` because no gate exercised the relevant path (see
//! `corpus/diff_test/list.tsv` regression entries and the project log).
//!
//! A "chaff" is a deliberately-wrong implementation of one bug class, sourced
//! directly from a bug we have actually shipped. `scripts/chaff_matrix.sh`
//! builds with `--features chaff`, activates one chaff at a time via the
//! `TLAPP_CHAFF` env var, runs the gate that is *supposed* to catch it, and
//! asserts the gate fails. A chaff that slips through every gate is a proven
//! hole in the suite.
//!
//! # Production safety
//!
//! Without `--features chaff` (the default), [`active`] is a `#[inline(always)]`
//! function that unconditionally returns `false`, so there is no way to activate
//! a chaff and **production behavior is unchanged**. In a release (optimized)
//! build — what actually ships — each `if chaff::active(..)` folds to the constant
//! `false` and the guarded branch is dropped, so there is no runtime cost; an
//! unoptimized debug build keeps a trivially-predictable dead branch. The call
//! sites are deliberately left unconditional (rather than `#[cfg]`-removed) so the
//! default build still type-checks every chaff branch and the feature cannot
//! bit-rot. The `chaff` feature must never be part of `default` and must never be
//! enabled in a release artifact.
//!
//! # Usage at a bug site
//!
//! ```ignore
//! // Correct behavior is the `else`; the chaff reintroduces the historical bug.
//! if crate::chaff::active("guard_disjunction_split_naive") {
//!     split_including_nested_disjunction(body) // the old, wrong behavior
//! } else {
//!     split_top_level_junctions_only(body)     // the fix
//! }
//! ```
//!
//! `TLAPP_CHAFF` accepts a single chaff name or a `;`-separated list (matching
//! the `FAILPOINTS` swarm convention), so the matrix can also exercise
//! fault-combination coverage later.

#[cfg(feature = "chaff")]
mod imp {
    use std::collections::HashSet;
    use std::sync::OnceLock;

    fn selected() -> &'static HashSet<String> {
        static SEL: OnceLock<HashSet<String>> = OnceLock::new();
        SEL.get_or_init(|| {
            std::env::var("TLAPP_CHAFF")
                .unwrap_or_default()
                .split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
    }

    /// True iff the named chaff is active for this process (`TLAPP_CHAFF`).
    #[inline]
    pub fn active(name: &str) -> bool {
        selected().contains(name)
    }

    /// The set of chaff names requested via `TLAPP_CHAFF`, for diagnostics.
    pub fn requested() -> Vec<String> {
        let mut v: Vec<String> = selected().iter().cloned().collect();
        v.sort();
        v
    }
}

#[cfg(not(feature = "chaff"))]
mod imp {
    /// No-op in every build that does not enable `--features chaff`. The
    /// optimizer deletes the guarded (buggy) branch entirely.
    #[inline(always)]
    pub fn active(_name: &str) -> bool {
        false
    }

    /// Always empty without the feature.
    #[inline(always)]
    pub fn requested() -> Vec<String> {
        Vec::new()
    }
}

pub use imp::{active, requested};

#[cfg(all(test, feature = "chaff"))]
mod tests {
    // These run only under `--features chaff`; they document the contract of
    // the selection parser rather than asserting on process-global env state
    // (which other tests may also touch).
    #[test]
    fn inactive_name_is_never_active_without_env() {
        // A name that is never set must be inactive. We cannot safely mutate
        // the process env here without racing other tests, so we only assert
        // the stable negative for an obviously-unset sentinel.
        assert!(!super::active("__chaff_name_that_is_never_set__"));
    }
}
