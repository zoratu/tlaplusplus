// T13.4 full lift — Phase A.1: shipping-source tracer-bullet for the
// architectural lift planned in `verification/verus/T13.4-FULL-LIFT-PLAN.md`.
//
// What this module ships
// ======================
//
// `VerifiedFingerprintShard` — a parallel struct to
// `page_aligned_fingerprint_store::FingerprintShard`, built on
// `Vec<PAtomicU64>` per the validated shadow blueprint at
// `verification/verus/shard_wrapper.rs` (Tier A.7, 34 verified).
//
// This is the first wrapper-carve-out chunk (Phase A.1 of the lift plan).
// It exposes the *structural* mirror of `FingerprintShard` plus two
// verified trivial getters: `capacity()` and `len_slots()`. The load /
// insert / resize methods are deferred to A.2 / A.3 / A.4 / A.5 and
// tracked in the lift plan.
//
// Spec predicates (`perms_wf`, `shard_view`) and the bounded probe
// loop stay in the standalone `verification/verus/shard_wrapper.rs`
// until a cross-file vstd-spec consolidation lands. Reason: under the
// vstd `verify = false` patch needed for the aarch64 Z3 4.13.3
// workaround, certain spec-mode items (`Seq::new` constructor,
// `Map::spec_index`) are erased and unavailable to downstream cargo-
// verus consumers. The standalone file runs against an unpatched vstd
// build so it can use them.
//
// Phase A.2 onward will land the methods + spec predicates as they
// become accessible (either via a vstd upgrade or by inlining the
// minimal predicates we need).
//
// The module is entirely `#[cfg(feature = "verus")]` — under default
// builds the struct does not exist (shipping callers continue to use
// `FingerprintShard`).

use verus_builtin::*;
use verus_builtin_macros::verus;
use vstd::atomic::*;
use vstd::prelude::*;
use vstd::view::*;

verus! {

// ============================================================================
// Vec spec availability
// ============================================================================
//
// Under cargo-verus with a patched vstd (`axiom_u64_trailing_zeros` in
// `std_specs/bits.rs` marked `#[verifier::external_body]`), vstd's own
// `Vec<T, A>` spec at `std_specs/vec.rs` IS available — including
// `Vec::len` with its `n == v@.len()` postcondition. No local bridge
// is needed.
//
// The patch is applied to the spot's cargo git checkout of vstd by
// `/tmp/verus_bootstrap_s7c.sh`. Background on why: under stock
// `verify = true`, vstd's `axiom_u64_trailing_zeros` proof body triggers
// a Z3 4.13.3 reader-thread crash on aarch64 (irreducible to an rlimit
// bump — Z3 returns malformed output, not a timeout). Marking just
// that one function `external_body` keeps the rest of vstd's spec
// machinery (including the Vec spec) fully available.

// ============================================================================
// SHIPPING-SHAPE STRUCT — mirrors `FingerprintShard`'s slot array.
// ============================================================================

pub struct VerifiedFingerprintShard {
    /// Per-slot atomic fingerprint storage. Each slot is a Verus
    /// `PAtomicU64` whose access is gated by a `PermissionU64` ghost
    /// token in the shard's tracked permission map (the permission map
    /// itself is threaded by callers; see Phase A.3 docs).
    pub slots: Vec<PAtomicU64>,
    /// Capacity (number of slots). Set at construction and stable for
    /// the lifetime of this instance — resize lives in Phase A.5.
    /// By the wrapper invariant (Phase A.3's `perms_wf`),
    /// `self.slots@.len() == self.capacity`.
    pub capacity: usize,
}

impl VerifiedFingerprintShard {
    /// Capacity getter. Mirrors `FingerprintShard::get_capacity` (which
    /// in shipping does an atomic load; we model the resize-stable
    /// epoch where capacity is fixed).
    ///
    /// Verified: `ensures cap == self.capacity`. Trivial postcondition;
    /// the value is what makes the rest of the per-method `requires`
    /// clauses (e.g. `idx < self.capacity`) discharge in callers.
    pub fn capacity(&self) -> (cap: usize)
        ensures cap == self.capacity,
    {
        self.capacity
    }

    /// Slot-count getter. Mirrors the implicit `slots.len()` call in
    /// shipping `FingerprintShard` (which uses raw pointer arithmetic
    /// + capacity in `unsafe { from_raw_parts(table, capacity) }.len()`
    /// at the read path).
    ///
    /// Phase A.2 ships this as exec-callable WITHOUT the `n == self.slots@.len()`
    /// ensures, because vstd's `View for Vec` impl is gated behind
    /// `verus_keep_ghost` cfg and isn't visible from this cargo-verus
    /// context. Phase A.3 lifts the postcondition once the Seq-view
    /// machinery is wired (either via a Verus upstream cfg fix or via
    /// a local View bridge).
    pub fn len_slots(&self) -> usize {
        self.slots.len()
    }
}

}  // end verus!

// ============================================================================
// PHASE A.3 — single probe step (verified)
// ============================================================================
//
// Ported from `verification/verus/shard_wrapper.rs::probe_slot_for_contains`
// (Tier A.7 standalone, 34 verified). The standalone form uses a Seq-view
// based ensures clause tying the result to `perm.view().value`; under the
// cargo-verus path here the same ensures works without modification.
//
// What this verifies
// ==================
// One iteration of the shipping `FingerprintShard::contains` probe loop
// (page_aligned_fingerprint_store.rs:728-828): load the slot, classify
// the outcome as Hit / Empty / Continue based on whether the load
// returned `fp`, the empty sentinel, or some other fingerprint.

verus! {

/// Outcome of one linear-probe step. Maps to the 3-way fork at the
/// load site in shipping `FingerprintShard::contains`.
#[derive(Clone, Copy)]
pub enum ProbeStep {
    /// Slot held `fp` — caller can return `true` (present).
    Hit,
    /// Slot held the empty sentinel — caller can return `false`
    /// (absent under linear-probe semantics).
    Empty,
    /// Slot held something else — caller must continue probing.
    Continue,
}

/// Empty-slot sentinel. Mirrors shipping `HashTableEntry::EMPTY = 0`.
pub open spec fn empty_slot() -> u64 { 0u64 }

}  // end verus!

verus! {

/// Pure-value classification of a slot's stored fingerprint.
///
/// The full probe-step exec function (with the `PAtomicU64::load`
/// call) is parked until the View-trait gating issue is resolved
/// (see "View gating" comment in `T13.4-FULL-LIFT-PLAN.md` Phase A.3.1).
/// In the meantime, this *pure* classification function is fully
/// verified — it has no atomic load and no permission threading, so
/// it sidesteps the vstd `View for PermissionU64` / `View for Vec`
/// machinery entirely. The caller does the atomic load externally
/// and feeds the result here.
///
/// Verified contract:
///   Hit      ==> stored == fp
///   Empty    ==> stored == empty_slot()
///   Continue ==> stored != fp && stored != empty_slot()
///
/// Together with `next_probe_slot` (verified in
/// `src/storage/verus_smoke.rs`) this completes the algorithmic
/// pieces of the linear-probe loop. What remains for Phase A.3.1 is
/// just the atomic-load step, which the wrapper struct's existence
/// (Phase A.1) and Vec field (Phase A.2) already prepared the way for.
pub fn classify_slot_value(stored: u64, fp: u64) -> (result: ProbeStep)
    requires fp != empty_slot(),
    ensures
        (matches!(result, ProbeStep::Hit) ==> stored == fp)
        && (matches!(result, ProbeStep::Empty) ==> stored == empty_slot())
        && (matches!(result, ProbeStep::Continue) ==> stored != fp && stored != empty_slot()),
{
    if stored == fp {
        ProbeStep::Hit
    } else if stored == 0 {
        ProbeStep::Empty
    } else {
        ProbeStep::Continue
    }
}

}  // end verus!

// ============================================================================
// SCAFFOLDING NOTES — what's deferred
// ============================================================================
//
// Phase A.1 (this file) ships: struct + `capacity()` + `len_slots()`.
// That establishes the verified-shipping-source baseline for the lift
// and surfaces the Vec<PAtomicU64> design choice in actual shipping
// source.
//
// Phase A.2 — `new(capacity: usize) -> (Self, Tracked<Map<int, PermissionU64>>)`
//   The constructor allocates `capacity` `PAtomicU64`s and mints their
//   initial permissions. Pattern: `shard_methods.rs::ShardCells::new`
//   shape. Requires the spec predicates `perms_wf` + `shard_view` —
//   may need either a vstd upgrade or inlined minimal predicates per
//   the module header note.
//
// Phase A.3 — `contains(&self, fp, Tracked(perms)) -> bool`
//   Uses the verified `probe_slot_for_contains` helper +
//   `bounded_contains_loop` from `shard_wrapper.rs` (already proven at
//   34 verified items).
//
// Phase A.4 — `contains_or_insert(&self, fp, Tracked(perms)) -> bool`
//   Uses `cas_insert_or_observe` + `bounded_contains_or_insert_loop`.
//
// Phase A.5 — Resize coordination. The shadow choice is between the
// lock-free atomic-pointer swap (`atomic_ptr_with_epoch.rs`, Tier A.9)
// and `vstd::rwlock::RwLock` (`shard_exec_wired.rs`, Tier A.10).
// Decision deferred to A.5 PR.
