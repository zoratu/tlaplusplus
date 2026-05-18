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
// What is annotated
// =================
//
// `compute_numa_index_from_hash` is the body of
// `PageAlignedFingerprintStore::home_numa` extracted as a free function
// over plain values (no struct refs). The annotation discharges:
//
//   - `requires num_numa_nodes > 0` — prevents the `% 0` UB
//   - `ensures c < num_numa_nodes` — the routing invariant the
//     shipping `home_numa` method depends on for its callers (which
//     index into `self.shards` by the returned value)
//
// The shipping `home_numa` body in
// `src/storage/page_aligned_fingerprint_store.rs` is the same three
// lines verbatim; both compute the same value byte-for-byte. A future
// pass can replace the inline body with `compute_numa_index_from_hash(
// fp, self.num_numa_nodes)` to lift the verified bound into the
// shipping call path; for now this file ships the verified shadow as
// proof that real shipping logic is verifiable here.
//
// What's needed for full method annotation
// ========================================
//
// To verify `PageAlignedFingerprintStore::home_numa` directly (as a
// method on the struct, not via this free function), Verus needs an
// `external_type_specification` bridge for `PageAlignedFingerprintStore`
// plus a spec function exposing the `num_numa_nodes` field. That is
// open-ended scaffolding work; the patterns are in
// `verification/verus/T13.4-PHASE2-CLOSURE.md`.

use verus_builtin::*;
use verus_builtin_macros::verus;
use vstd::prelude::*;

verus! {
    /// Compute which NUMA node a fingerprint hashes to. Mirrors the
    /// body of `PageAlignedFingerprintStore::home_numa` in the
    /// shipping file. Verified to satisfy `c < num_numa_nodes`, the
    /// invariant the shipping callers rely on when indexing into the
    /// per-NUMA shard array.
    pub fn compute_numa_index_from_hash(fp: u64, num_numa_nodes: usize) -> (c: usize)
        requires num_numa_nodes > 0,
        ensures c < num_numa_nodes,
    {
        // Mix bits to reduce correlation between NUMA routing and shard
        // selection.
        let mixed = (fp >> 32) ^ (fp >> 16) ^ fp;
        (mixed as usize) % num_numa_nodes
    }
}
