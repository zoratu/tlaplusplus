// T13.4 Phase 2 — verus annotations on shipping code, gated by the
// `verus` cargo feature.
//
// What this proves
// ================
//
// `cargo verus check --features verus` from the main tlaplusplus crate
// root runs the Verus verifier on the items inside the `verus!{}` block
// below. Sibling files in `src/storage/` and the rest of the crate are
// marked `#[verifier::external]` via the gated attributes in
// `src/lib.rs` and `src/storage/mod.rs`; this file is the only verus-
// processed module.
//
// What this validates
// ===================
//
// The cargo-verus build flow runs end-to-end against the actual
// tlaplusplus crate (not a parallel demo). The `verus_integration_smoke`
// function below is a verified item INSIDE the shipping crate's source
// tree.
//
// The path to annotating actual `FingerprintShard` / `PageAlignedFingerprintStore`
// methods is open from here: each shipping method that should be
// verified needs an `external_type_specification` bridge (so Verus
// knows about the struct's shape) plus the per-method annotation. The
// verified prototypes in `verification/verus/shard_exec_wired.rs`,
// `shard_methods.rs`, `shard_multi_slot.rs`, `mmap_external_body.rs`,
// and `atomic_ptr_with_epoch.rs` cover every pattern those annotations
// need. See `verification/verus/T13.4-PHASE2-CLOSURE.md` for the full
// hand-off.

use verus_builtin::*;
use verus_builtin_macros::verus;
use vstd::prelude::*;

verus! {
    /// Smoke-test that the cargo-verus integration runs end-to-end on
    /// shipping code. Trivial postcondition (`n == 42`) discharged
    /// from the literal in the body. Replace with real method
    /// annotations once the `external_type_specification` bridges for
    /// `FingerprintShard` / `PageAlignedFingerprintStore` are written.
    pub fn verus_integration_smoke() -> (n: usize)
        ensures n == 42,
    {
        42
    }
}
