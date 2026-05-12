// T13.5 (state_machines! port) — LTL-native restatement of the seqlock
// reader-liveness protocol via Verus's `state_machine!` macro.
//
// Status: this file is the OPTIONAL polish port called out in
// `verification/verus/README.md` ("Genuinely deferred to v1.2.0+":
// "T13.5 `state_machines!` port: ... an LTL-native restatement via
// `state_machines!` is optional polish, no longer load-bearing").
//
// The headline theorem `theorem_no_starvation` over the `Seq<ShardSeqState>`
// finite-prefix model already ships axiom-free in
// `reader_liveness_v2.rs` (17 verified, 0 axioms). This file adds an
// independently-verified RESTATEMENT of the same protocol using the
// `state_machine!` macro, so future state_machines!-native consumers
// (refinement to a tracked-state-machines version of the shipping
// fingerprint shard, etc.) have a structural starting point.
//
// What this file ships
// ====================
//
//   1. A `state_machine!` named `ShardSeq` with three transitions
//      mirroring the v2 file: `begin_resize`, `finalize_resize`,
//      `stutter`. The single field `seq: nat` carries the seqlock
//      counter and matches `ShardSeqState.seq` from v2.
//
//   2. Two `#[invariant]` predicates:
//      - `inv_seq_nonneg`: trivially true at the type level (`nat >= 0`).
//      - `inv_parity_classification`: every reachable state is either
//        stable (seq even) or resizing (seq odd) — the parity-discipline
//        analog of `lemma_begin_resize_parity_seq` /
//        `lemma_finalize_resize_parity_seq`.
//
//   3. Four `#[inductive]` proofs discharging that each transition
//      preserves the invariants.
//
//   4. **Refinement bridge** to `reader_liveness_v2`: a `spec fn interp`
//      from `ShardSeq::State` to `ShardSeqState` (an identity map on
//      the `seq` field) plus a `proof fn` showing that every
//      `state_machine!`-level transition (`ShardSeq::State::next`)
//      corresponds to v2's `step_relation`. This is the LTL-native
//      restatement: the state-machine protocol is sound w.r.t. the
//      v2 finite-prefix protocol, so the v2 `theorem_no_starvation`
//      lifts to the state-machine setting.
//
//   5. A reader-success spec `reader_attempt_succeeds_sm` and a
//      headline corollary `theorem_no_starvation_sm` that restates the
//      v2 result over `ShardSeq::State` traces.
//
// What this file does NOT do
// ===========================
//
//   - It is NOT a strengthening of `reader_liveness_v2.rs`. The v2
//     file's `theorem_no_starvation` is already 0-axiom; this port
//     does not introduce new soundness, only an LTL-native shape.
//   - It does not introduce new axioms (`#[verifier::external_body]`).
//     Per the brief: "DO NOT introduce new external_body axioms."
//
// HOW TO RUN
//     cd verification/verus
//     ./run_proof.sh reader-liveness-sm
//     # OR equivalently:
//     ./run_proof.sh sm

#![allow(unused_imports)]
#![allow(non_snake_case)]
use verus_builtin::*;
use verus_builtin_macros::*;
use verus_state_machines_macros::*;
use vstd::prelude::*;

// ============================================================================
// PROTOCOL STATE MACHINE
// ============================================================================
//
// Single field `seq: nat` mirroring `ShardSeqState.seq` from
// `reader_liveness_v2.rs`. Three transitions mirroring v2:
//
//   - begin_resize: seq even -> seq + 1 (odd)
//   - finalize_resize: seq odd -> seq + 1 (even)
//   - stutter: seq -> seq (no change)
//
// Invariant `parity_classified` says every reachable state has a
// well-defined parity (which is trivial at the type level since seq is
// a `nat`, but the invariant statement gives us the structural hook
// for the refinement bridge below).

state_machine!(
    ShardSeq {
        fields {
            pub seq: nat,
        }

        init!{
            initialize() {
                init seq = 0;
            }
        }

        transition!{
            begin_resize() {
                require pre.seq % 2 == 0;
                update seq = pre.seq + 1;
            }
        }

        transition!{
            finalize_resize() {
                require pre.seq % 2 == 1;
                update seq = pre.seq + 1;
            }
        }

        transition!{
            stutter() {
                // Always allowed; no preconditions.
            }
        }

        // The seq counter, being a nat, is trivially >= 0; the invariant
        // is included for structural completeness and to give the
        // refinement bridge a hook.
        #[invariant]
        pub fn inv_seq_nonneg(&self) -> bool {
            true
        }

        // Every reachable state is classified as stable or resizing by
        // its seq parity. This is trivially true (parity is total over
        // nats), but the inductive proofs below establish it via the
        // transition shapes — the same pattern v2's parity lemmas use.
        #[invariant]
        pub fn inv_parity_classified(&self) -> bool {
            self.seq % 2 == 0 || self.seq % 2 == 1
        }

        #[inductive(initialize)]
        fn initialize_inductive(post: Self) {
            // post.seq == 0; 0 % 2 == 0; satisfies inv_parity_classified.
        }

        #[inductive(begin_resize)]
        fn begin_resize_inductive(pre: Self, post: Self) {
            // pre.seq % 2 == 0 (precondition); post.seq == pre.seq + 1.
            // (pre.seq + 1) % 2 == 1, so inv_parity_classified holds.
        }

        #[inductive(finalize_resize)]
        fn finalize_resize_inductive(pre: Self, post: Self) {
            // pre.seq % 2 == 1 (precondition); post.seq == pre.seq + 1.
            // (pre.seq + 1) % 2 == 0, so inv_parity_classified holds.
        }

        #[inductive(stutter)]
        fn stutter_inductive(pre: Self, post: Self) {
            // post == pre; invariants trivially preserved.
        }
    }
);

// ============================================================================
// REFINEMENT BRIDGE TO `reader_liveness_v2`
// ============================================================================
//
// We re-state v2's `ShardSeqState` and `step_relation` here (instead of
// importing — these proof files don't share a crate). The bridge proofs
// show that every `ShardSeq::State::next` step corresponds to a v2
// `step_relation` step on the interpretation, and that the headline
// `theorem_no_starvation` lifts.

verus! {

// V2 protocol shape, restated.
pub struct ShardSeqStateV2 {
    pub seq: nat,
}

pub open spec fn is_stable_v2(s: ShardSeqStateV2) -> bool {
    s.seq % 2 == 0
}

pub open spec fn is_resizing_v2(s: ShardSeqStateV2) -> bool {
    s.seq % 2 == 1
}

pub open spec fn step_begin_resize_v2(s: ShardSeqStateV2) -> ShardSeqStateV2
    recommends is_stable_v2(s),
{
    ShardSeqStateV2 { seq: s.seq + 1 }
}

pub open spec fn step_finalize_resize_v2(s: ShardSeqStateV2) -> ShardSeqStateV2
    recommends is_resizing_v2(s),
{
    ShardSeqStateV2 { seq: s.seq + 1 }
}

pub open spec fn step_stutter_v2(s: ShardSeqStateV2) -> ShardSeqStateV2 {
    s
}

pub open spec fn step_relation_v2(s: ShardSeqStateV2, s_next: ShardSeqStateV2) -> bool {
    s_next == step_begin_resize_v2(s)
    || s_next == step_finalize_resize_v2(s)
    || s_next == step_stutter_v2(s)
}

// Reader-success predicate (stable + equal seq across attempts).
pub open spec fn reader_attempt_succeeds_v2(
    s_before: ShardSeqStateV2, s_after: ShardSeqStateV2,
) -> bool {
    is_stable_v2(s_before) && s_before.seq == s_after.seq
}

// ============================================================================
// INTERPRETATION FUNCTION
// ============================================================================
//
// `interp` is the identity on the `seq` field — both states have the
// same shape, so the refinement is structural. This is the analog of
// `interp(a) = B::State { number: a.number * 2 }` in
// `verus/examples/state_machines/refinement.rs`, but for our identity
// case the formula is just `interp(s) = ShardSeqStateV2 { seq: s.seq }`.

pub open spec fn interp(s: ShardSeq::State) -> ShardSeqStateV2 {
    ShardSeqStateV2 { seq: s.seq }
}

// ============================================================================
// REFINEMENT THEOREMS
// ============================================================================
//
// `theorem_init_refines`: if `ShardSeq::State::init(post)` holds, then
// `interp(post)` is the v2 initial state (seq == 0, stable).
//
// `theorem_step_refines`: every `ShardSeq::State::next(pre, post)` step
// corresponds to a v2 `step_relation_v2(interp(pre), interp(post))`
// step.

pub proof fn theorem_init_refines(post: ShardSeq::State)
    requires
        ShardSeq::State::init(post),
    ensures
        interp(post).seq == 0,
        is_stable_v2(interp(post)),
{
    // Open the state-machine init predicate. There is exactly one
    // init step (`initialize`), which sets seq to 0.
    reveal(ShardSeq::State::init);
    reveal(ShardSeq::State::init_by);
    let step = choose|step: ShardSeq::Config| ShardSeq::State::init_by(post, step);
    match step {
        ShardSeq::Config::initialize() => {
            assert(post.seq == 0);
        }
        ShardSeq::Config::dummy_to_use_type_params(_) => {
            // The macro generates `dummy_to_use_type_params(_) => false`
            // in init_by, so this branch is unreachable. We can prove
            // false from `init_by(post, dummy)` by revealing init_by.
            assert_by(false, {
                reveal(ShardSeq::State::init_by);
            });
        }
    }
}

// Helper: each transition's parity contract, lifted to the v2 spec.
pub proof fn lemma_begin_resize_refines(pre: ShardSeq::State, post: ShardSeq::State)
    requires
        ShardSeq::State::begin_resize(pre, post),
    ensures
        is_stable_v2(interp(pre)),
        post.seq == pre.seq + 1,
        interp(post) == step_begin_resize_v2(interp(pre)),
{
    reveal(ShardSeq::State::begin_resize);
}

pub proof fn lemma_finalize_resize_refines(pre: ShardSeq::State, post: ShardSeq::State)
    requires
        ShardSeq::State::finalize_resize(pre, post),
    ensures
        is_resizing_v2(interp(pre)),
        post.seq == pre.seq + 1,
        interp(post) == step_finalize_resize_v2(interp(pre)),
{
    reveal(ShardSeq::State::finalize_resize);
}

pub proof fn lemma_stutter_refines(pre: ShardSeq::State, post: ShardSeq::State)
    requires
        ShardSeq::State::stutter(pre, post),
    ensures
        post == pre,
        interp(post) == step_stutter_v2(interp(pre)),
{
    reveal(ShardSeq::State::stutter);
}

// MAIN REFINEMENT THEOREM — every state-machine transition refines a v2 step.
pub proof fn theorem_step_refines(pre: ShardSeq::State, post: ShardSeq::State)
    requires
        ShardSeq::State::next(pre, post),
    ensures
        step_relation_v2(interp(pre), interp(post)),
{
    reveal(ShardSeq::State::next);
    reveal(ShardSeq::State::next_by);
    let step = choose|step: ShardSeq::Step| ShardSeq::State::next_by(pre, post, step);
    match step {
        ShardSeq::Step::begin_resize() => {
            lemma_begin_resize_refines(pre, post);
            assert(interp(post) == step_begin_resize_v2(interp(pre)));
        }
        ShardSeq::Step::finalize_resize() => {
            lemma_finalize_resize_refines(pre, post);
            assert(interp(post) == step_finalize_resize_v2(interp(pre)));
        }
        ShardSeq::Step::stutter() => {
            lemma_stutter_refines(pre, post);
            assert(interp(post) == step_stutter_v2(interp(pre)));
        }
        ShardSeq::Step::dummy_to_use_type_params(_) => {
            // The macro generates `dummy_to_use_type_params(_) => false`
            // in next_by, so this branch is unreachable. We prove false
            // from `next_by(pre, post, dummy)` by revealing next_by.
            assert_by(false, {
                reveal(ShardSeq::State::next_by);
            });
        }
    }
}

// ============================================================================
// FINITE-PREFIX TRACES OVER THE STATE MACHINE
// ============================================================================
//
// The state-machine port carries the same finite-prefix model as v2.
// `wf_prefix_sm` says every adjacent pair in the prefix is a state-
// machine `next` step. We bridge this to v2's `wf_prefix` via the
// refinement theorem above.

pub open spec fn wf_prefix_sm(prefix: Seq<ShardSeq::State>) -> bool {
    forall|i: int|
        0 <= i && i + 1 < prefix.len()
        ==> ShardSeq::State::next(#[trigger] prefix[i], prefix[i + 1])
}

pub open spec fn wf_prefix_v2(prefix: Seq<ShardSeqStateV2>) -> bool {
    forall|i: int|
        0 <= i && i + 1 < prefix.len()
        ==> step_relation_v2(#[trigger] prefix[i], prefix[i + 1])
}

pub open spec fn interp_prefix(prefix: Seq<ShardSeq::State>) -> Seq<ShardSeqStateV2> {
    Seq::new(prefix.len(), |i: int| interp(prefix[i]))
}

// MAIN REFINEMENT — well-formed SM prefix lifts to a well-formed v2 prefix.
pub proof fn theorem_prefix_refines(prefix: Seq<ShardSeq::State>)
    requires
        wf_prefix_sm(prefix),
    ensures
        wf_prefix_v2(interp_prefix(prefix)),
        interp_prefix(prefix).len() == prefix.len(),
{
    let v2_prefix = interp_prefix(prefix);
    assert(v2_prefix.len() == prefix.len());
    assert forall|i: int| 0 <= i && i + 1 < v2_prefix.len()
        implies step_relation_v2(#[trigger] v2_prefix[i], v2_prefix[i + 1]) by {
        assert(v2_prefix[i] == interp(prefix[i]));
        assert(v2_prefix[i + 1] == interp(prefix[i + 1]));
        assert(ShardSeq::State::next(prefix[i], prefix[i + 1]));
        theorem_step_refines(prefix[i], prefix[i + 1]);
    }
}

// ============================================================================
// READER LIVENESS — RESTATED OVER THE STATE MACHINE
// ============================================================================
//
// Reader-success on the SM side is the obvious lift of v2's predicate.
// `theorem_no_starvation_sm` is the headline analog of v2's
// `theorem_no_starvation`: from any well-formed SM prefix, we can
// extend with a finite reader-witness that ends in a successful
// reader attempt.

pub open spec fn reader_attempt_succeeds_sm(
    s_before: ShardSeq::State, s_after: ShardSeq::State,
) -> bool {
    s_before.seq % 2 == 0 && s_before.seq == s_after.seq
}

// Bridge: SM-level reader success implies v2-level reader success.
pub proof fn lemma_reader_success_refines(
    s_before: ShardSeq::State, s_after: ShardSeq::State,
)
    requires
        reader_attempt_succeeds_sm(s_before, s_after),
    ensures
        reader_attempt_succeeds_v2(interp(s_before), interp(s_after)),
{
    assert(interp(s_before).seq == s_before.seq);
    assert(interp(s_after).seq == s_after.seq);
}

// ============================================================================
// CONSTRUCTIVE WITNESSES — direct port of v2's three discharge lemmas.
// ============================================================================
//
// Just like v2, we discharge the three "axioms" by constructing finite
// extension witnesses. The shapes are identical:
//   1. From a resizing tail, [s, finalize(s)] lands stable.
//   2. From a stable tail, [s, s] is a successful reader attempt.
//   3. From a resizing tail, [s, finalize(s), finalize(s)] ends in a
//      successful reader attempt across the last two states.

pub open spec fn ends_stable_sm(prefix: Seq<ShardSeq::State>) -> bool
    recommends prefix.len() > 0,
{
    prefix.len() > 0 && prefix[prefix.len() - 1].seq % 2 == 0
}

pub open spec fn ends_resizing_sm(prefix: Seq<ShardSeq::State>) -> bool
    recommends prefix.len() > 0,
{
    prefix.len() > 0 && prefix[prefix.len() - 1].seq % 2 == 1
}

// Constructive lifting: stable-tail SM extension.
pub proof fn lemma_reader_can_observe_stutter_sm(prefix: Seq<ShardSeq::State>)
    requires
        wf_prefix_sm(prefix),
        prefix.len() > 0,
        ends_stable_sm(prefix),
    ensures
        exists|ext: Seq<ShardSeq::State>| #![auto]
            wf_prefix_sm(ext)
            && ext.len() == 2
            && ext[0] == prefix[prefix.len() - 1]
            && ext[1] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds_sm(ext[0], ext[1]),
{
    let s = prefix[prefix.len() - 1];
    let ext: Seq<ShardSeq::State> = seq![s, s];

    // wf_prefix_sm(ext): only one adjacent pair (i = 0). ShardSeq::State::next
    // holds because `stutter` always fires (no precondition) and post == pre.
    assert(ext.len() == 2);
    assert(ext[0] == s);
    assert(ext[1] == s);
    assert forall|i: int| 0 <= i && i + 1 < ext.len()
        implies ShardSeq::State::next(#[trigger] ext[i], ext[i + 1]) by {
        assert(i == 0);
        // Show stutter step from s to s.
        ShardSeq::show::stutter(s, s);
    }
    assert(wf_prefix_sm(ext));

    // The remaining clauses are direct.
    assert(s.seq % 2 == 0);   // from ends_stable_sm
    assert(reader_attempt_succeeds_sm(ext[0], ext[1]));
}

// Constructive lifting: resizing-tail SM extension (3-step finalize+stutter).
pub proof fn lemma_extension_composes_sm(prefix: Seq<ShardSeq::State>)
    requires
        wf_prefix_sm(prefix),
        prefix.len() > 0,
        ends_resizing_sm(prefix),
    ensures
        exists|reader_ext: Seq<ShardSeq::State>| #![auto]
            wf_prefix_sm(reader_ext)
            && reader_ext.len() >= 2
            && reader_ext[0] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds_sm(
                reader_ext[reader_ext.len() - 2],
                reader_ext[reader_ext.len() - 1],
            ),
{
    let s = prefix[prefix.len() - 1];
    let f = ShardSeq::State { seq: s.seq + 1 };
    let reader_ext: Seq<ShardSeq::State> = seq![s, f, f];

    // s.seq is odd (resizing); f.seq == s.seq + 1 is even (stable).
    assert(s.seq % 2 == 1);
    assert(f.seq == s.seq + 1);
    // The parity of (s.seq + 1) is even when s.seq is odd:
    //    s.seq % 2 == 1 ==> (s.seq + 1) % 2 == 0
    assert(f.seq % 2 == 0);

    // wf_prefix_sm(reader_ext) — adjacent pairs are (0,1) and (1,2):
    //   Pair (0,1): finalize_resize step from s (seq odd) to f (seq+1).
    //   Pair (1,2): stutter step from f to f.
    assert(reader_ext.len() == 3);
    assert(reader_ext[0] == s);
    assert(reader_ext[1] == f);
    assert(reader_ext[2] == f);
    assert forall|i: int| 0 <= i && i + 1 < reader_ext.len()
        implies ShardSeq::State::next(#[trigger] reader_ext[i], reader_ext[i + 1]) by {
        if i == 0 {
            ShardSeq::show::finalize_resize(s, f);
        } else {
            assert(i == 1);
            ShardSeq::show::stutter(f, f);
        }
    }
    assert(wf_prefix_sm(reader_ext));

    // Reader-success at the last two indices.
    assert(reader_ext.len() >= 2);
    assert(reader_ext[0] == prefix[prefix.len() - 1]);
    assert(reader_ext[reader_ext.len() - 2] == f);
    assert(reader_ext[reader_ext.len() - 1] == f);
    assert(reader_attempt_succeeds_sm(reader_ext[reader_ext.len() - 2],
                                      reader_ext[reader_ext.len() - 1]));
}

// ============================================================================
// MAIN THEOREM — restated over the state machine.
// ============================================================================

pub proof fn theorem_reader_eventually_succeeds_sm(prefix: Seq<ShardSeq::State>)
    requires
        wf_prefix_sm(prefix),
        prefix.len() > 0,
    ensures
        exists|reader_ext: Seq<ShardSeq::State>| #![auto]
            wf_prefix_sm(reader_ext)
            && reader_ext.len() >= 2
            && reader_ext[0] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds_sm(
                reader_ext[reader_ext.len() - 2],
                reader_ext[reader_ext.len() - 1],
            ),
{
    let s = prefix[prefix.len() - 1];
    if s.seq % 2 == 0 {
        // ends_stable: the 2-step stutter [s, s] is the witness.
        assert(ends_stable_sm(prefix));
        lemma_reader_can_observe_stutter_sm(prefix);
        let ext = choose|ext: Seq<ShardSeq::State>| #![auto]
            wf_prefix_sm(ext)
            && ext.len() == 2
            && ext[0] == prefix[prefix.len() - 1]
            && ext[1] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds_sm(ext[0], ext[1]);
        assert(ext.len() >= 2);
        assert(ext[0] == prefix[prefix.len() - 1]);
        assert(ext[ext.len() - 2] == ext[0]);
        assert(ext[ext.len() - 1] == ext[1]);
        assert(reader_attempt_succeeds_sm(ext[ext.len() - 2], ext[ext.len() - 1]));
    } else {
        // ends_resizing: 3-step finalize+stutter is the witness.
        assert(s.seq % 2 == 1);
        assert(ends_resizing_sm(prefix));
        lemma_extension_composes_sm(prefix);
    }
}

pub proof fn theorem_no_starvation_sm(prefix: Seq<ShardSeq::State>)
    requires
        wf_prefix_sm(prefix),
        prefix.len() > 0,
    ensures
        exists|reader_ext: Seq<ShardSeq::State>| #![auto]
            wf_prefix_sm(reader_ext)
            && reader_ext.len() >= 2
            && reader_ext[0] == prefix[prefix.len() - 1]
            && reader_attempt_succeeds_sm(
                reader_ext[reader_ext.len() - 2],
                reader_ext[reader_ext.len() - 1],
            ),
{
    theorem_reader_eventually_succeeds_sm(prefix);
}

} // verus!

// ============================================================================
// COVERAGE TABLE — what's now machine-checked beyond reader_liveness_v2.rs.
// ============================================================================
//
// The v2 file ships the constructive headline (`theorem_no_starvation`)
// over a hand-rolled `ShardSeqState` struct. This file ports the same
// protocol shape to Verus's `state_machine!` macro and proves an
// LTL-native restatement, with a refinement bridge to v2.
//
// New verified items (this file):
//
//   - state_machine!(ShardSeq) with init + 3 transitions + 2 invariants
//     + 4 inductive proofs.
//   - theorem_init_refines: SM init refines v2 init shape (seq == 0).
//   - lemma_begin_resize_refines, lemma_finalize_resize_refines,
//     lemma_stutter_refines: per-transition refinement lemmas.
//   - theorem_step_refines: every SM step refines a v2 step_relation.
//   - theorem_prefix_refines: well-formed SM prefix lifts to v2 prefix.
//   - lemma_reader_success_refines: reader-success on SM lifts to v2.
//   - lemma_reader_can_observe_stutter_sm,
//     lemma_extension_composes_sm: constructive witnesses (stutter +
//     finalize-then-stutter) lifted to the SM.
//   - theorem_reader_eventually_succeeds_sm: SM analog of v2's main
//     theorem.
//   - theorem_no_starvation_sm: headline corollary.
//
// What this file does NOT do:
//
//   - Does NOT introduce any axioms. All proofs are constructive.
//   - Does NOT replace `reader_liveness_v2.rs`. v2 remains the
//     canonical 0-axiom proof.
//   - Does NOT integrate with the shipping fingerprint shard. The
//     shipping code is unchanged; this is a Verus-only proof file.

fn main() {}
