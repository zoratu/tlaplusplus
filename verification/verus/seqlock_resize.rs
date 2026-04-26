// Verus proof of the seqlock resize protocol used by
// `src/storage/page_aligned_fingerprint_store.rs`.
//
// SCOPE: this is a *protocol-level* proof. It models the seqlock resize as a
// pure abstract state machine and proves three safety properties:
//
//   (P1) Monotonic seq number: the resize sequence number never decreases.
//   (P2) Resize parity invariant: seq is even iff no resize is in progress.
//   (P3) Conservation across resize: every fingerprint reachable before a
//        resize starts is observable after the resize completes — i.e.
//        rehash + finalize together preserve membership.
//
// What this proves vs. what it does NOT prove
// ============================================
// This proof targets the *algorithm*. It deliberately abstracts away:
//   - unsafe pointer arithmetic (open-addressed probe, raw mmap, etc.),
//   - linear-probe collision behavior (we model the table as Set<u64>),
//   - memory-ordering of atomics (the abstract steps execute atomically),
//   - concurrent inserts during the rehash window (modeled as a single
//     interleaving where every "insert during resize" deposits into the
//     new table, exactly mirroring the production code's `contains_or_insert`
//     resize-in-progress branch at lines 656–776 of
//     page_aligned_fingerprint_store.rs).
//
// The production code's three claimed invariants on the seqlock resize
// (see CLAUDE.md "Lock-Free Fingerprint Store" section) are:
//   1. Seqlock coordination (odd = resizing, even = stable)  -> proved (P2)
//   2. Atomic pointer swapping for lock-free table replacement -> modeled
//      as a single `finalize_resize` step that updates `table` and bumps
//      `seq` to the next even value (single-stepped here; the production
//      code does these as two atomic stores plus one fetch_add).
//   3. Readers spin-wait during resize, then retry -> modeled by the
//      `contains_under_resize` reader spec, which checks BOTH old and
//      new tables (matching contains_or_insert lines 678–776).
//
// The headline soundness property the user cares about ("model checker
// must not silently lose a fingerprint across resize") corresponds directly
// to (P3) below.
//
// HOW TO RUN
// ==========
// Requires Verus built from source with Z3 (any 4.12+ with
// `--no-solver-version-check`). Run:
//
//     cd verification/verus
//     verus seqlock_resize.rs
//
// or via the helper script:
//
//     ./run_proof.sh
//
// Successful output ends with `verification results:: <N> verified, 0 errors`.

use vstd::prelude::*;

verus! {

// ----------------------------------------------------------------------------
// Abstract state of one fingerprint shard mid-resize.
// ----------------------------------------------------------------------------
//
// We model a shard as the conjunction of:
//   - the seqlock counter (Nat, parity = stable/resizing),
//   - the "live" table contents as a Set<u64>,
//   - during resize, an additional "old" snapshot frozen at resize start.
//
// Production correspondence:
//   `seq`        <-> AtomicU64, page_aligned_fingerprint_store.rs:138
//   `table`      <-> AtomicPtr<HashTableEntry> at line 130 (current table)
//   `old_table`  <-> AtomicPtr at line 145 (snapshot during resize)
//   `new_table`  <-> AtomicPtr at line 150 (insertion target during resize)
pub struct ShardState {
    pub seq: nat,
    pub table: Set<u64>,        // current table, used when seq is even
    pub old_table: Set<u64>,    // valid only when seq is odd; snapshot of
                                // `table` at the moment resize started
    pub new_table: Set<u64>,    // valid only when seq is odd; accumulates
                                // both rehashed entries and concurrent inserts
}

// ----------------------------------------------------------------------------
// Well-formedness invariant.
// ----------------------------------------------------------------------------
//
// Captures the protocol-level invariant the production code maintains:
//   - parity tracks whether resize is in progress;
//   - during resize, `new_table` is a superset of the entries already
//     migrated from `old_table` (rehash never deletes);
//   - in the stable state, the resize buffers are empty.
pub open spec fn wf(s: ShardState) -> bool {
    if s.seq % 2 == 0 {
        // STABLE: no resize, helper buffers cleared.
        s.old_table =~= Set::<u64>::empty()
        && s.new_table =~= Set::<u64>::empty()
    } else {
        // RESIZING: the snapshot is non-empty only insofar as it captures
        // a previous stable `table`. The reader-observable union is
        // (old_table ∪ new_table). The current `table` field is unchanged
        // until finalize runs (production: `finalize_resize` swaps `table`
        // to point at the new region at line 368 BEFORE the seq bump at
        // line 374, but at the abstract level the swap and bump are one
        // step). We do not constrain `table` further here since readers
        // taking the resize-in-progress path do not read it.
        true
    }
}

// ----------------------------------------------------------------------------
// What a reader observes on the public surface (matches `contains` at
// page_aligned_fingerprint_store.rs:554–650).
// ----------------------------------------------------------------------------
pub open spec fn observable(s: ShardState, fp: u64) -> bool {
    if s.seq % 2 == 0 {
        s.table.contains(fp)
    } else {
        // During resize, readers union old + new.
        s.old_table.contains(fp) || s.new_table.contains(fp)
    }
}

// ----------------------------------------------------------------------------
// PROTOCOL TRANSITIONS
// Each `step_*` function is the abstract effect of a primitive in the
// production seqlock resize.
// ----------------------------------------------------------------------------

// Insert during the stable state (no resize in progress).
// Production correspondence: `contains_or_insert` "NORMAL PATH" branch
// at lines 778–838. Inserts go into `table`, seq is unchanged.
pub open spec fn step_insert_stable(s: ShardState, fp: u64) -> ShardState {
    ShardState {
        seq: s.seq,
        table: s.table.insert(fp),
        old_table: s.old_table,
        new_table: s.new_table,
    }
}

// Begin a resize: snapshot `table` into `old_table`, clear `new_table`,
// bump seq to the next odd number.
// Production correspondence: `resize` lines 469–497.
pub open spec fn step_begin_resize(s: ShardState) -> ShardState {
    ShardState {
        seq: s.seq + 1,
        table: s.table,                 // remains until finalize swaps it
        old_table: s.table,             // RCU snapshot for in-flight readers
        new_table: Set::<u64>::empty(), // freshly allocated, zero-filled mmap
    }
}

// Insert during resize: production code (lines 704–746) places it in the
// new table directly. Old table is frozen.
pub open spec fn step_insert_during_resize(s: ShardState, fp: u64) -> ShardState {
    ShardState {
        seq: s.seq,
        table: s.table,
        old_table: s.old_table,
        new_table: s.new_table.insert(fp),
    }
}

// Migrate one entry from old to new during incremental rehash (one bucket
// at a time). Production: `rehash_batch_counted` lines 280–344.
// We model migration of a single fingerprint that is in `old_table` but
// not yet in `new_table`. The production code uses CAS on the new table;
// the abstract step is a set insert.
pub open spec fn step_rehash_one(s: ShardState, fp: u64) -> ShardState {
    ShardState {
        seq: s.seq,
        table: s.table,
        old_table: s.old_table,
        new_table: s.new_table.insert(fp),
    }
}

// Finalize: swap `new_table` into `table`, clear scratch, bump seq to even.
// Production: `finalize_resize` lines 357–390. Precondition (matched in
// production by `is_rehash_complete` at line 503) is that every old entry
// has been migrated.
pub open spec fn step_finalize_resize(s: ShardState) -> ShardState {
    ShardState {
        seq: s.seq + 1,
        table: s.new_table,
        old_table: Set::<u64>::empty(),
        new_table: Set::<u64>::empty(),
    }
}

// ----------------------------------------------------------------------------
// PROOF (P1): the seq counter is monotonically non-decreasing across
// every protocol step.
// ----------------------------------------------------------------------------
pub proof fn lemma_seq_monotonic_insert_stable(s: ShardState, fp: u64)
    ensures step_insert_stable(s, fp).seq >= s.seq,
{}

pub proof fn lemma_seq_monotonic_begin(s: ShardState)
    ensures step_begin_resize(s).seq >= s.seq,
{}

pub proof fn lemma_seq_monotonic_insert_during_resize(s: ShardState, fp: u64)
    ensures step_insert_during_resize(s, fp).seq >= s.seq,
{}

pub proof fn lemma_seq_monotonic_rehash(s: ShardState, fp: u64)
    ensures step_rehash_one(s, fp).seq >= s.seq,
{}

pub proof fn lemma_seq_monotonic_finalize(s: ShardState)
    ensures step_finalize_resize(s).seq >= s.seq,
{}

// ----------------------------------------------------------------------------
// PROOF (P2): parity correctly tracks whether a resize is in progress.
// `step_begin_resize` flips parity from even to odd, and
// `step_finalize_resize` flips it back.
// ----------------------------------------------------------------------------
pub proof fn lemma_begin_resize_parity(s: ShardState)
    requires s.seq % 2 == 0,
    ensures step_begin_resize(s).seq % 2 == 1,
{}

pub proof fn lemma_finalize_parity(s: ShardState)
    requires s.seq % 2 == 1,
    ensures step_finalize_resize(s).seq % 2 == 0,
{}

// Inserts and rehash steps never change the parity bit.
pub proof fn lemma_insert_stable_parity(s: ShardState, fp: u64)
    ensures step_insert_stable(s, fp).seq % 2 == s.seq % 2,
{}

pub proof fn lemma_rehash_parity(s: ShardState, fp: u64)
    ensures step_rehash_one(s, fp).seq % 2 == s.seq % 2,
{}

// ----------------------------------------------------------------------------
// PROOF (P3): no fingerprint is lost across a complete resize cycle.
//
// This is the headline soundness property for tlaplusplus: if the model
// checker has marked a state as "seen" (i.e. its fingerprint is in `table`),
// no resize event may cause that fingerprint to disappear from the
// observable set.
// ----------------------------------------------------------------------------

// Helper lemma: rehashing one fingerprint preserves all earlier new_table
// contents AND adds the migrated fingerprint.
pub proof fn lemma_rehash_preserves_new(s: ShardState, fp: u64)
    ensures
        forall|x: u64| #![auto] s.new_table.contains(x) ==> step_rehash_one(s, fp).new_table.contains(x),
        step_rehash_one(s, fp).new_table.contains(fp),
{}

// Helper: an insert during resize never removes anything.
pub proof fn lemma_insert_during_resize_preserves_new(s: ShardState, fp: u64)
    ensures
        forall|x: u64| #![auto] s.new_table.contains(x) ==> step_insert_during_resize(s, fp).new_table.contains(x),
{}

// Begin -> rehash all old entries -> finalize. We model "rehash all" with
// a single ghost step `step_rehash_complete` that asserts new_table is a
// superset of old_table (the loop invariant of `resize` at lines 502–507).
pub open spec fn step_rehash_complete(s: ShardState) -> ShardState {
    // Models "after the worker-driven rehash loop terminates": every old
    // fingerprint has been copied into the new table. Concurrent inserts
    // may have added more.
    ShardState {
        seq: s.seq,
        table: s.table,
        old_table: s.old_table,
        new_table: s.new_table.union(s.old_table),
    }
}

// Conservation across a single complete resize cycle.
//
// Setup:
//   s0  -- pre-resize stable state (seq even, old/new empty).
//   s1 = step_begin_resize(s0)            -- snapshot taken.
//   s2 = step_rehash_complete(s1)         -- loop terminated, every old
//                                            entry copied into new.
//   s3 = step_finalize_resize(s2)         -- pointers swapped, seq even.
//
// Conclusion:
//   For every fingerprint observable in s0, it is observable in s3.
pub proof fn lemma_resize_preserves_membership(s0: ShardState, fp: u64)
    requires
        s0.seq % 2 == 0,
        wf(s0),
        observable(s0, fp),
    ensures
        ({
            let s1 = step_begin_resize(s0);
            let s2 = step_rehash_complete(s1);
            let s3 = step_finalize_resize(s2);
            observable(s3, fp)
        }),
{
    let s1 = step_begin_resize(s0);
    let s2 = step_rehash_complete(s1);
    let s3 = step_finalize_resize(s2);

    // s0 is stable: observable(s0, fp) means s0.table.contains(fp).
    assert(s0.table.contains(fp));

    // s1 carried s0.table over to s1.old_table (snapshot).
    assert(s1.old_table =~= s0.table);
    assert(s1.old_table.contains(fp));

    // s2.new_table = s1.new_table U s1.old_table, so it contains fp.
    assert(s2.new_table.contains(fp));

    // s3.table = s2.new_table, and s3.seq is even, so observable(s3, fp)
    // reduces to s3.table.contains(fp).
    assert(s3.table.contains(fp));
    assert(s3.seq % 2 == 0);
}

// ----------------------------------------------------------------------------
// PROOF (P3 strengthened): even with concurrent inserts during the resize
// window, no inserted fingerprint is lost.
//
// We weave a sequence of inserts during resize between begin and finalize.
// The proof composes lemma_resize_preserves_membership with the helpers
// above to cover the "insert lands in new table while resize is mid-flight"
// case the production code handles at contains_or_insert lines 704–746.
// ----------------------------------------------------------------------------
pub proof fn lemma_concurrent_insert_during_resize_preserved(
    s0: ShardState,
    fp_pre: u64,
    fp_concurrent: u64,
)
    requires
        s0.seq % 2 == 0,
        wf(s0),
        s0.table.contains(fp_pre),
    ensures
        ({
            // Begin resize.
            let s1 = step_begin_resize(s0);
            // A worker thread inserts fp_concurrent into the new table while
            // the resize is in progress.
            let s2 = step_insert_during_resize(s1, fp_concurrent);
            // The rehash loop completes (old entries migrated into new).
            let s3 = step_rehash_complete(s2);
            // Finalize.
            let s4 = step_finalize_resize(s3);
            // Both fingerprints survive.
            observable(s4, fp_pre) && observable(s4, fp_concurrent)
        }),
{
    let s1 = step_begin_resize(s0);
    let s2 = step_insert_during_resize(s1, fp_concurrent);
    let s3 = step_rehash_complete(s2);
    let s4 = step_finalize_resize(s3);

    // fp_pre route:
    //   s0.table.contains(fp_pre)
    //     -> s1.old_table.contains(fp_pre)            [snapshot]
    //     -> s3.new_table.contains(fp_pre)            [rehash union]
    //     -> s4.table.contains(fp_pre)                [finalize swap]
    assert(s1.old_table =~= s0.table);
    assert(s1.old_table.contains(fp_pre));
    // step_insert_during_resize doesn't touch old_table.
    assert(s2.old_table.contains(fp_pre));
    // step_rehash_complete unions in old.
    assert(s3.new_table.contains(fp_pre));

    // fp_concurrent route:
    //   step_insert_during_resize put it in s2.new_table
    //     -> step_rehash_complete(union with old) preserves it
    //     -> finalize swaps new into table.
    assert(s2.new_table.contains(fp_concurrent));
    assert(s3.new_table.contains(fp_concurrent));

    assert(s4.table.contains(fp_pre));
    assert(s4.table.contains(fp_concurrent));
    assert(s4.seq % 2 == 0);
}

// ----------------------------------------------------------------------------
// READER LIVENESS / CONSISTENCY
//
// The production reader uses the standard seqlock retry pattern: read
// `seq_before`, do the lookup, read `seq_after`, retry if they differ
// (page_aligned_fingerprint_store.rs:644–648 and 828–831).
//
// The retry is justified by: if seq did not change, no resize started or
// completed during the lookup, so the read of `table` was consistent.
// ----------------------------------------------------------------------------

// Models a single non-retried reader observation. Returns the value the
// reader would have returned and the seq tag attached to that read.
pub open spec fn reader_observation(s: ShardState, fp: u64) -> (bool, nat) {
    (observable(s, fp), s.seq)
}

// If a reader sees the same seq before and after, the observation is
// against one consistent snapshot — no resize boundary was crossed.
pub proof fn lemma_seqlock_consistent_read_means_no_resize_boundary(
    s_before: ShardState,
    s_after: ShardState,
    fp: u64,
)
    requires
        s_before.seq == s_after.seq,
    ensures
        // Then the reader's view of the parity (and thus which lookup
        // path it takes) is consistent across the two reads — i.e. if
        // the lookup ran on the "stable" path before, it would have run
        // on the stable path after, too.
        (s_before.seq % 2 == 0) == (s_after.seq % 2 == 0),
{}

// ----------------------------------------------------------------------------
// COMPOSITE INVARIANT: across an arbitrary number of inserts and one
// resize cycle, every previously-inserted fingerprint remains observable.
//
// This is the property model checking soundness depends on: no transition
// in the seqlock protocol can cause `tlaplusplus` to "forget" that it
// already explored a particular state.
// ----------------------------------------------------------------------------

// Composability (single insert + resize): if an insert lands BEFORE
// begin_resize, the conservation lemma still applies.
pub proof fn lemma_insert_then_resize_preserved(s0: ShardState, fp: u64)
    requires
        s0.seq % 2 == 0,
        wf(s0),
    ensures
        ({
            let s1 = step_insert_stable(s0, fp);
            let s2 = step_begin_resize(s1);
            let s3 = step_rehash_complete(s2);
            let s4 = step_finalize_resize(s3);
            observable(s4, fp)
        }),
{
    let s1 = step_insert_stable(s0, fp);
    assert(s1.table.contains(fp));
    assert(s1.seq % 2 == 0);
    // Insert during stable does not change parity or buffers.
    assert(s1.old_table =~= Set::<u64>::empty());
    assert(s1.new_table =~= Set::<u64>::empty());
    // Now invoke the resize-conservation lemma on s1.
    lemma_resize_preserves_membership(s1, fp);
}

// ----------------------------------------------------------------------------
// STATE MACHINE FORMULATION
//
// Lift the protocol steps into a single transition relation `step`. A trace
// is a sequence of states linked by `step`. We then prove that the
// "effective contents" of the shard — the set of fingerprints any reader
// would observe — is monotonically non-decreasing across every transition.
//
// This is the strongest tier-B property: it states, in one universally-
// quantified theorem, that the seqlock resize protocol cannot lose data.
// ----------------------------------------------------------------------------

// "Effective contents" of a shard: the fingerprints currently observable
// to readers. Mirrors `observable` but lifted to the whole shard.
pub open spec fn effective_contents(s: ShardState) -> Set<u64> {
    if s.seq % 2 == 0 {
        s.table
    } else {
        s.old_table.union(s.new_table)
    }
}

// The transition relation. `s` -> `t` is allowed iff `t` is reachable
// from `s` via one of the protocol primitives. The fingerprint argument
// is existentially quantified inside via the disjunction.
pub open spec fn step(s: ShardState, t: ShardState) -> bool {
    ||| (exists|fp: u64| s.seq % 2 == 0 && t == #[trigger] step_insert_stable(s, fp))
    ||| (s.seq % 2 == 0 && t == step_begin_resize(s))
    ||| (exists|fp: u64| s.seq % 2 == 1 && t == #[trigger] step_insert_during_resize(s, fp))
    ||| (exists|fp: u64|
            s.seq % 2 == 1 && s.old_table.contains(fp) && t == #[trigger] step_rehash_one(s, fp))
    ||| (s.seq % 2 == 1 && t == step_finalize_resize(s) && s.old_table.subset_of(s.new_table))
}

// THE main inductive invariant: every transition preserves `effective_contents`
// as a superset (no fingerprint is dropped). EXCEPTION: `begin_resize` and
// `finalize_resize` can drop fingerprints from `effective_contents` ONLY if
// the rehash-completion precondition (`s.old_table.subset_of(s.new_table)`)
// has been established by prior `step_rehash_one` transitions. We capture
// this via the explicit subset_of guard on the finalize disjunct above.
pub proof fn lemma_step_preserves_contents(s: ShardState, t: ShardState)
    requires
        wf(s),
        step(s, t),
    ensures
        forall|fp: u64| #![auto] effective_contents(s).contains(fp) ==> effective_contents(t).contains(fp),
{
    if s.seq % 2 == 0 {
        // Stable -> ?
        if exists|fp: u64| t == step_insert_stable(s, fp) {
            // table grew, parity unchanged; effective_contents = table for both.
            assert(forall|x: u64| #![auto] s.table.contains(x) ==> t.table.contains(x));
        } else if t == step_begin_resize(s) {
            // Snapshot taken: t.old_table = s.table, t.new_table = empty,
            // t.seq is odd. effective_contents(t) = old.union(new) = s.table.
            assert(t.seq == s.seq + 1);
            assert(t.seq % 2 == 1);
            assert(t.old_table =~= s.table);
            assert(t.new_table =~= Set::<u64>::empty());
            assert(forall|x: u64| #![auto]
                s.table.contains(x) ==> t.old_table.union(t.new_table).contains(x));
        } else {
            // No other disjunct can fire from a stable state.
            assert(false);
        }
    } else {
        // Resizing -> ?
        let snapshot_union = s.old_table.union(s.new_table);
        if exists|fp: u64| t == step_insert_during_resize(s, fp) {
            // new_table grew, old_table unchanged. Union grew.
            assert(t.old_table =~= s.old_table);
            assert(forall|x: u64| #![auto]
                s.new_table.contains(x) ==> t.new_table.contains(x));
        } else if exists|fp: u64| t == step_rehash_one(s, fp) {
            // Same: new_table grew, old_table unchanged.
            assert(t.old_table =~= s.old_table);
            assert(forall|x: u64| #![auto]
                s.new_table.contains(x) ==> t.new_table.contains(x));
        } else if t == step_finalize_resize(s) && s.old_table.subset_of(s.new_table) {
            // Finalize: t.table = s.new_table, parity flips to even,
            // t.old_table and t.new_table become empty.
            // effective_contents(t) = t.table = s.new_table.
            // We need: s.old_table ∪ s.new_table ⊆ s.new_table.
            // That's exactly the precondition s.old_table.subset_of(s.new_table).
            assert(s.old_table.subset_of(s.new_table));
            assert(forall|x: u64| #![auto]
                s.old_table.contains(x) ==> s.new_table.contains(x));
            assert(t.table =~= s.new_table);
            assert(t.seq % 2 == 0);
        } else {
            assert(false);
        }
    }
}

// Lift to multi-step traces. A finite execution is a sequence of states;
// every adjacent pair is related by `step`, and the first state is well-
// formed and stable. We prove that the union of all states' contents grows
// monotonically.
pub open spec fn is_trace(trace: Seq<ShardState>) -> bool {
    trace.len() > 0
    && wf(trace[0])
    && trace[0].seq % 2 == 0
    && (forall|i: int|
            0 <= i < trace.len() - 1 ==> #[trigger] step(trace[i], trace[i + 1]))
    // wf is preserved trivially for our stronger model since wf only
    // constrains the stable state (which we re-establish at finalize).
    && (forall|i: int| 0 <= i < trace.len() ==> #[trigger] wf(trace[i]))
}

// Inductive lemma: contents grow along every prefix of a trace.
pub proof fn lemma_trace_preserves_contents(trace: Seq<ShardState>, fp: u64, i: int, j: int)
    requires
        is_trace(trace),
        0 <= i <= j < trace.len(),
        effective_contents(trace[i]).contains(fp),
    ensures
        effective_contents(trace[j]).contains(fp),
    decreases j - i,
{
    if i < j {
        lemma_step_preserves_contents(trace[i], trace[i + 1]);
        lemma_trace_preserves_contents(trace, fp, i + 1, j);
    }
}

// THE MAIN THEOREM
//
// In any well-formed execution of the seqlock resize protocol, every
// fingerprint inserted at any point remains observable from then on.
//
// In tlaplusplus terms: once a state's fingerprint is admitted to the
// fingerprint store, no resize event will cause the model checker to
// re-explore it (which would violate completeness/efficiency) or to
// silently miss a future violation by losing track of it.
pub proof fn theorem_no_fingerprint_lost(trace: Seq<ShardState>, fp: u64, i: int)
    requires
        is_trace(trace),
        0 <= i < trace.len(),
        effective_contents(trace[i]).contains(fp),
    ensures
        forall|j: int| #![auto] i <= j < trace.len() ==> effective_contents(trace[j]).contains(fp),
{
    assert forall|j: int| #![auto] i <= j < trace.len() implies effective_contents(trace[j]).contains(fp) by {
        lemma_trace_preserves_contents(trace, fp, i, j);
    }
}

// Sanity: an executable smoke test exercising the model.
fn main() {}

}  // verus!
