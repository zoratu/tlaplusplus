// T13.5 (V2) — Constructive discharge of reader-liveness axioms.
//
// This file is the constructive sibling of `reader_liveness.rs`. It proves
// the same `theorem_reader_eventually_succeeds` and `theorem_no_starvation`
// statements WITHOUT the three `#[verifier::external_body]` axioms used in
// the original file.
//
// The witnesses are explicit, finite extensions of the prefix:
//
//   * `axiom_writer_eventually_finalizes` is replaced by
//     `lemma_writer_eventually_finalizes`, which constructs the 2-element
//     extension `seq![s, finalize(s)]` where `s = prefix.last()` and
//     `finalize` is `step_finalize_resize_seq`. Since `s` is resizing
//     (odd seq), `finalize(s)` has even seq and is therefore stable.
//
//   * `axiom_reader_can_observe_stutter` is replaced by
//     `lemma_reader_can_observe_stutter`, which constructs the 2-element
//     extension `seq![s, s]` where `s = prefix.last()`. Because
//     `step_stutter_seq(s) == s`, the second element is exactly the
//     stutter successor of the first; `wf_prefix` holds and
//     `reader_attempt_succeeds(s, s)` holds since `s` is stable.
//
//   * `axiom_extension_composes` is replaced by
//     `lemma_extension_composes`, which constructs the 3-element
//     extension `seq![s, finalize(s), finalize(s)]` where `s` is the
//     resizing prefix-tail. The middle step is finalize (odd -> even);
//     the final step is stutter (even -> even, same value). The final
//     adjacent pair satisfies `reader_attempt_succeeds`.
//
// All three lemmas are pure, axiom-free Verus proofs. The original
// file is preserved as the bounded-form fallback.

use vstd::prelude::*;

verus! {

// ----------------------------------------------------------------------------
// ABSTRACT SHARD STATE FOR LIVENESS  (identical to reader_liveness.rs)
// ----------------------------------------------------------------------------

pub struct ShardSeqState {
    pub seq: nat,
}

pub open spec fn is_stable(s: ShardSeqState) -> bool {
    s.seq % 2 == 0
}

pub open spec fn is_resizing(s: ShardSeqState) -> bool {
    s.seq % 2 == 1
}

// ----------------------------------------------------------------------------
// PROTOCOL TRANSITIONS  (identical to reader_liveness.rs)
// ----------------------------------------------------------------------------

pub open spec fn step_begin_resize_seq(s: ShardSeqState) -> ShardSeqState
    recommends is_stable(s),
{
    ShardSeqState { seq: s.seq + 1 }
}

pub open spec fn step_finalize_resize_seq(s: ShardSeqState) -> ShardSeqState
    recommends is_resizing(s),
{
    ShardSeqState { seq: s.seq + 1 }
}

pub open spec fn step_stutter_seq(s: ShardSeqState) -> ShardSeqState {
    s
}

pub open spec fn step_relation(s: ShardSeqState, s_next: ShardSeqState) -> bool {
    s_next == step_begin_resize_seq(s)
    || s_next == step_finalize_resize_seq(s)
    || s_next == step_stutter_seq(s)
}

// ----------------------------------------------------------------------------
// SAFETY: PARITY DISCIPLINE  (identical to reader_liveness.rs)
// ----------------------------------------------------------------------------

pub proof fn lemma_begin_resize_parity_seq(s: ShardSeqState)
    requires is_stable(s),
    ensures is_resizing(step_begin_resize_seq(s)),
{}

pub proof fn lemma_finalize_resize_parity_seq(s: ShardSeqState)
    requires is_resizing(s),
    ensures is_stable(step_finalize_resize_seq(s)),
{}

pub proof fn lemma_stutter_parity_seq(s: ShardSeqState)
    ensures is_stable(step_stutter_seq(s)) == is_stable(s),
{}

pub proof fn lemma_seq_monotonic_seq(s: ShardSeqState)
    ensures
        step_begin_resize_seq(s).seq >= s.seq,
        step_finalize_resize_seq(s).seq >= s.seq,
        step_stutter_seq(s).seq >= s.seq,
{}

pub proof fn lemma_step_relation_monotonic(s: ShardSeqState, s_next: ShardSeqState)
    requires step_relation(s, s_next),
    ensures s_next.seq >= s.seq,
{}

// ----------------------------------------------------------------------------
// READER SEMANTICS  (identical to reader_liveness.rs)
// ----------------------------------------------------------------------------

pub open spec fn reader_attempt_succeeds(s_before: ShardSeqState, s_after: ShardSeqState) -> bool {
    is_stable(s_before) && s_before.seq == s_after.seq
}

pub proof fn lemma_reader_attempt_soundness(s_before: ShardSeqState, s_after: ShardSeqState)
    requires reader_attempt_succeeds(s_before, s_after),
    ensures
        is_stable(s_before),
        is_stable(s_after),
        s_before.seq == s_after.seq,
{}

// ----------------------------------------------------------------------------
// FINITE-PREFIX TRACE  (identical to reader_liveness.rs)
// ----------------------------------------------------------------------------

pub open spec fn wf_prefix(prefix: Seq<ShardSeqState>) -> bool {
    forall|i: int|
        0 <= i && i + 1 < prefix.len()
        ==> step_relation(#[trigger] prefix[i], prefix[i + 1])
}

pub proof fn theorem_prefix_seq_monotonic(prefix: Seq<ShardSeqState>, i: int, j: int)
    requires
        wf_prefix(prefix),
        0 <= i,
        i <= j,
        j < prefix.len(),
    ensures
        prefix[i].seq <= prefix[j].seq,
    decreases j - i,
{
    if i < j {
        assert(0 <= i && i + 1 < prefix.len());
        assert(step_relation(prefix[i], prefix[i + 1]));
        lemma_step_relation_monotonic(prefix[i], prefix[i + 1]);
        theorem_prefix_seq_monotonic(prefix, i + 1, j);
    }
}

pub proof fn theorem_equal_seq_implies_stutter(prefix: Seq<ShardSeqState>, i: int)
    requires
        wf_prefix(prefix),
        0 <= i,
        i + 1 < prefix.len(),
        prefix[i].seq == prefix[i + 1].seq,
    ensures
        prefix[i + 1] == step_stutter_seq(prefix[i]),
{
    assert(step_relation(prefix[i], prefix[i + 1]));
    if prefix[i + 1] == step_begin_resize_seq(prefix[i]) {
        assert(prefix[i + 1].seq == prefix[i].seq + 1);
        assert(false);
    }
    if prefix[i + 1] == step_finalize_resize_seq(prefix[i]) {
        assert(prefix[i + 1].seq == prefix[i].seq + 1);
        assert(false);
    }
}

pub proof fn theorem_reader_success_is_stutter(prefix: Seq<ShardSeqState>, i: int)
    requires
        wf_prefix(prefix),
        0 <= i,
        i + 1 < prefix.len(),
        reader_attempt_succeeds(prefix[i], prefix[i + 1]),
    ensures
        prefix[i + 1] == step_stutter_seq(prefix[i]),
        prefix[i] == prefix[i + 1],
        is_stable(prefix[i]),
        is_stable(prefix[i + 1]),
{
    theorem_equal_seq_implies_stutter(prefix, i);
    assert(step_stutter_seq(prefix[i]) == prefix[i]);
}

pub proof fn theorem_writer_step_parity_in_prefix(prefix: Seq<ShardSeqState>, i: int)
    requires
        wf_prefix(prefix),
        0 <= i,
        i + 1 < prefix.len(),
    ensures
        prefix[i + 1] == step_begin_resize_seq(prefix[i])
            ==> prefix[i + 1].seq == prefix[i].seq + 1,
        prefix[i + 1] == step_finalize_resize_seq(prefix[i])
            ==> prefix[i + 1].seq == prefix[i].seq + 1,
        prefix[i + 1] == step_stutter_seq(prefix[i])
            ==> prefix[i + 1].seq == prefix[i].seq,
{}

pub proof fn theorem_prefix_seq_growth_bounded(prefix: Seq<ShardSeqState>, i: int)
    requires
        wf_prefix(prefix),
        0 <= i,
        i < prefix.len(),
    ensures
        prefix[i].seq <= prefix[0].seq + i,
    decreases i,
{
    if i > 0 {
        theorem_prefix_seq_growth_bounded(prefix, i - 1);
        assert(step_relation(prefix[i - 1], prefix[i]));
        if prefix[i] == step_stutter_seq(prefix[i - 1]) {
            assert(prefix[i].seq == prefix[i - 1].seq);
        } else if prefix[i] == step_begin_resize_seq(prefix[i - 1]) {
            assert(prefix[i].seq == prefix[i - 1].seq + 1);
        } else {
            assert(prefix[i] == step_finalize_resize_seq(prefix[i - 1]));
            assert(prefix[i].seq == prefix[i - 1].seq + 1);
        }
    }
}

// ----------------------------------------------------------------------------
// LIVENESS HYPOTHESES + STATEMENT
// ----------------------------------------------------------------------------

pub open spec fn ends_stable(prefix: Seq<ShardSeqState>) -> bool
    recommends prefix.len() > 0,
{
    prefix.len() > 0 && is_stable(prefix[prefix.len() - 1])
}

pub open spec fn ends_resizing(prefix: Seq<ShardSeqState>) -> bool
    recommends prefix.len() > 0,
{
    prefix.len() > 0 && is_resizing(prefix[prefix.len() - 1])
}

pub open spec fn finalize_extension_exists(prefix: Seq<ShardSeqState>, ext: Seq<ShardSeqState>) -> bool {
    wf_prefix(ext)
    && ext.len() > 0
    && ext[0] == prefix[prefix.len() - 1]
    && is_stable(ext[ext.len() - 1])
}

// ----------------------------------------------------------------------------
// CONSTRUCTIVE DISCHARGE OF THE THREE AXIOMS
// ----------------------------------------------------------------------------
//
// The original `reader_liveness.rs` admits three axioms here. This file
// replaces each one with a concrete-witness lemma. Plain English of the
// witnesses:
//
//   1. From a resizing tail s, the 2-step extension [s, finalize(s)]
//      lands in a stable state.
//   2. From a stable tail s, the 2-step extension [s, s] (a stutter
//      step) is itself a successful reader attempt.
//   3. From a resizing tail s, the 3-step extension
//      [s, finalize(s), finalize(s)] (finalize then stutter) ends in a
//      successful reader attempt across the last two states.
//
// The proofs work entirely with `seq!` literals; `wf_prefix` over a
// 2- or 3-element literal reduces to a small number of `step_relation`
// checks that the SMT solver discharges by case-splitting `step_relation`.

// LEMMA 1 (writer-fairness, constructive). Replaces
// `axiom_writer_eventually_finalizes`.
pub proof fn lemma_writer_eventually_finalizes(prefix: Seq<ShardSeqState>)
    requires
        wf_prefix(prefix),
        prefix.len() > 0,
        ends_resizing(prefix),
    ensures
        exists|ext: Seq<ShardSeqState>| finalize_extension_exists(prefix, ext),
{
    let s = prefix[prefix.len() - 1];
    let ext: Seq<ShardSeqState> = seq![s, step_finalize_resize_seq(s)];

    // wf_prefix(ext): only one adjacent pair (i = 0). The successor is
    // step_finalize_resize_seq(s), which falls under step_relation by
    // its second disjunct.
    assert(ext.len() == 2);
    assert(ext[0] == s);
    assert(ext[1] == step_finalize_resize_seq(s));
    assert forall|i: int| 0 <= i && i + 1 < ext.len()
        implies step_relation(#[trigger] ext[i], ext[i + 1]) by {
        assert(i == 0);
        assert(ext[0] == s);
        assert(ext[1] == step_finalize_resize_seq(ext[0]));
    }
    assert(wf_prefix(ext));

    // ext[0] == prefix.last(): direct from construction.
    assert(ext[0] == prefix[prefix.len() - 1]);

    // is_stable(ext.last()): finalize on an odd seq yields even.
    assert(is_resizing(s));   // from ends_resizing(prefix)
    lemma_finalize_resize_parity_seq(s);
    assert(is_stable(step_finalize_resize_seq(s)));
    assert(ext[ext.len() - 1] == step_finalize_resize_seq(s));

    assert(finalize_extension_exists(prefix, ext));
}

// LEMMA 2 (reader-progress, constructive). Replaces
// `axiom_reader_can_observe_stutter`.
pub proof fn lemma_reader_can_observe_stutter(prefix: Seq<ShardSeqState>)
    requires
        wf_prefix(prefix),
        prefix.len() > 0,
        ends_stable(prefix),
    ensures
        exists|ext: Seq<ShardSeqState>|
            wf_prefix(ext)
            && ext.len() == 2
            && ext[0] == prefix[prefix.len() - 1]
            && ext[1] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds(ext[0], ext[1]),
{
    let s = prefix[prefix.len() - 1];
    let ext: Seq<ShardSeqState> = seq![s, s];

    // wf_prefix(ext): only one adjacent pair (i = 0). ext[1] == s ==
    // step_stutter_seq(s) so the third disjunct of step_relation fires.
    assert(ext.len() == 2);
    assert(ext[0] == s);
    assert(ext[1] == s);
    assert(step_stutter_seq(s) == s);
    assert forall|i: int| 0 <= i && i + 1 < ext.len()
        implies step_relation(#[trigger] ext[i], ext[i + 1]) by {
        assert(i == 0);
        assert(ext[1] == step_stutter_seq(ext[0]));
    }
    assert(wf_prefix(ext));

    // The remaining clauses are direct.
    assert(is_stable(s));   // from ends_stable(prefix)
    assert(reader_attempt_succeeds(ext[0], ext[1]));
}

// LEMMA 3 (composition, constructive). Replaces
// `axiom_extension_composes`. Discharges the resizing branch of the
// main theorem directly with a 3-element witness.
pub proof fn lemma_extension_composes(prefix: Seq<ShardSeqState>)
    requires
        wf_prefix(prefix),
        prefix.len() > 0,
        ends_resizing(prefix),
    ensures
        exists|reader_ext: Seq<ShardSeqState>|
            wf_prefix(reader_ext)
            && reader_ext.len() >= 2
            && reader_ext[0] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds(
                reader_ext[reader_ext.len() - 2],
                reader_ext[reader_ext.len() - 1],
            ),
{
    let s = prefix[prefix.len() - 1];
    let f = step_finalize_resize_seq(s);
    let reader_ext: Seq<ShardSeqState> = seq![s, f, f];

    // wf_prefix(reader_ext) — adjacent pairs are (0,1) and (1,2).
    //   Pair (0,1): reader_ext[1] == finalize(s), second disjunct.
    //   Pair (1,2): reader_ext[2] == f == step_stutter_seq(f), third
    //     disjunct of step_relation.
    assert(reader_ext.len() == 3);
    assert(reader_ext[0] == s);
    assert(reader_ext[1] == f);
    assert(reader_ext[2] == f);
    assert(step_stutter_seq(f) == f);
    assert forall|i: int| 0 <= i && i + 1 < reader_ext.len()
        implies step_relation(#[trigger] reader_ext[i], reader_ext[i + 1]) by {
        if i == 0 {
            assert(reader_ext[1] == step_finalize_resize_seq(reader_ext[0]));
        } else {
            assert(i == 1);
            assert(reader_ext[2] == step_stutter_seq(reader_ext[1]));
        }
    }
    assert(wf_prefix(reader_ext));

    // len >= 2 and first-element clause are direct.
    assert(reader_ext.len() >= 2);
    assert(reader_ext[0] == prefix[prefix.len() - 1]);

    // reader_attempt_succeeds(reader_ext[len-2], reader_ext[len-1]):
    //   len-2 == 1, len-1 == 2; both are `f`.
    //   is_stable(f) holds because s is resizing (odd) and finalize
    //   bumps seq to even. f.seq == f.seq trivially.
    assert(is_resizing(s));
    lemma_finalize_resize_parity_seq(s);
    assert(is_stable(f));
    assert(reader_ext[reader_ext.len() - 2] == f);
    assert(reader_ext[reader_ext.len() - 1] == f);
    assert(reader_attempt_succeeds(reader_ext[reader_ext.len() - 2],
                                   reader_ext[reader_ext.len() - 1]));
}

// ----------------------------------------------------------------------------
// MAIN THEOREMS — now fully proved (no axioms)
// ----------------------------------------------------------------------------

pub proof fn theorem_reader_eventually_succeeds(prefix: Seq<ShardSeqState>)
    requires
        wf_prefix(prefix),
        prefix.len() > 0,
    ensures
        exists|reader_ext: Seq<ShardSeqState>|
            wf_prefix(reader_ext)
            && reader_ext.len() >= 2
            && reader_ext[0] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds(
                reader_ext[reader_ext.len() - 2],
                reader_ext[reader_ext.len() - 1],
            ),
{
    if is_stable(prefix[prefix.len() - 1]) {
        // ends_stable: the 2-step stutter [s, s] is the witness.
        lemma_reader_can_observe_stutter(prefix);
        let ext = choose|ext: Seq<ShardSeqState>|
            wf_prefix(ext)
            && ext.len() == 2
            && ext[0] == prefix[prefix.len() - 1]
            && ext[1] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds(ext[0], ext[1]);
        assert(ext.len() >= 2);
        assert(ext[0] == prefix[prefix.len() - 1]);
        assert(ext[ext.len() - 2] == ext[0]);
        assert(ext[ext.len() - 1] == ext[1]);
        assert(reader_attempt_succeeds(ext[ext.len() - 2], ext[ext.len() - 1]));
    } else {
        // ends_resizing: 3-step finalize+stutter is the witness.
        assert(is_resizing(prefix[prefix.len() - 1]));
        assert(ends_resizing(prefix));
        lemma_extension_composes(prefix);
    }
}

pub proof fn theorem_no_starvation(prefix: Seq<ShardSeqState>)
    requires
        wf_prefix(prefix),
        prefix.len() > 0,
    ensures
        exists|reader_ext: Seq<ShardSeqState>|
            wf_prefix(reader_ext)
            && reader_ext.len() >= 2
            && reader_ext[0] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds(
                reader_ext[reader_ext.len() - 2],
                reader_ext[reader_ext.len() - 1],
            ),
{
    theorem_reader_eventually_succeeds(prefix);
}

fn main() {}

}  // verus!
