# Verus proofs for tlaplusplus

This directory contains formal verification artifacts for tlaplusplus's own correctness, separate from the model checker's TLA+ frontend.

The first artifact (T13, release 1.0.0) is `seqlock_resize.rs` — a machine-checked proof of the seqlock-based dynamic-resize protocol used by the lock-free fingerprint store (`src/storage/page_aligned_fingerprint_store.rs`).

## What is proved

The proof file declares 19 lemmas that Verus discharges via the Z3 SMT solver. Together they establish, on an abstract model of the protocol:

| Property | Lemma | Plain English |
|---|---|---|
| **P1. Monotonic seq** | `lemma_seq_monotonic_*` (5 cases) | The seq counter is non-decreasing across every protocol step (`insert`, `begin_resize`, `insert_during_resize`, `rehash_one`, `finalize_resize`). |
| **P2. Parity discipline** | `lemma_begin_resize_parity`, `lemma_finalize_parity`, `lemma_insert_stable_parity`, `lemma_rehash_parity` | Seq is even iff no resize is in progress. `begin_resize` flips even -> odd. `finalize_resize` flips odd -> even. Inserts and rehashes do not change parity. |
| **P3. Conservation across one resize cycle** | `lemma_resize_preserves_membership` | Every fingerprint observable to a reader before `begin_resize` is observable after `finalize_resize`. |
| **P3'. Concurrent insert during resize survives** | `lemma_concurrent_insert_during_resize_preserved` | A fingerprint inserted into `new_table` while resize is in flight survives the rehash union and the table swap. |
| **P3''. Insert + resize composition** | `lemma_insert_then_resize_preserved` | Inserting first, then resizing, still preserves the inserted fingerprint. |
| **Reader consistency** | `lemma_seqlock_consistent_read_means_no_resize_boundary` | If the seqlock value before and after a read is equal, the parity (and thus the read path) is consistent. (Justifies the seqlock retry loop's correctness.) |
| **Step monotonicity** | `lemma_step_preserves_contents` | Every transition in the protocol's state machine preserves `effective_contents` (the set of fingerprints observable to readers) as a superset, with the *single* exception of `finalize_resize`, which is guarded by the rehash-completion precondition `s.old_table.subset_of(s.new_table)`. |
| **MAIN THEOREM** | `theorem_no_fingerprint_lost` | In any well-formed execution of the protocol, every fingerprint present at any point remains observable from then on. |

The main theorem is the precise machine-checked statement of the soundness property the user cares about: **"Once a state's fingerprint is admitted to the fingerprint store, no resize event can cause the model checker to silently re-explore it or to silently miss a future violation by losing track of it."**

## What is *assumed* (axioms / unproven preconditions)

The proof is at the **protocol abstraction layer**. It deliberately abstracts:

1. **Pointer arithmetic and open-addressed probing.** The shipping code stores fingerprints in a flat array indexed by `fp % capacity` with linear probing on collisions. The proof models the table as a `Set<u64>`. Implication: the proof shows the *protocol* is correct, but does not (yet) prove that the probe sequence implementation actually realises that abstract set faithfully. A bug like "linear probe wraps incorrectly at the table boundary" would not be caught by this proof.

2. **Memory orderings of atomic operations.** The proof treats each protocol step as atomic. The shipping code uses `AcqRel`/`Acquire` pairings. The proof assumes these orderings are sufficient — i.e. that the abstract sequential semantics is the right model for the relaxed-atomic concurrent execution. (For the seqlock pattern this is the standard assumption, justified in the literature; see "C++ Concurrency in Action" Ch. 5.)

3. **Single ongoing resize.** The protocol model assumes at most one resize is in progress at a time. The shipping code enforces this via `resize_lock: Mutex<()>` (line 142). We do not model the mutex explicitly; we instead model "resize is in progress" as the `seq % 2 == 1` parity bit being set.

4. **Mmap allocation produces a zero-filled region.** Shipping uses `MAP_ANONYMOUS` (line 466 comment) which the kernel guarantees to zero-fill. The proof models `step_begin_resize` as setting `new_table` to the empty set, equivalent to all-zero entries.

5. **Rehash completion as an atomic ghost step.** The shipping code migrates entries one bucket at a time (`rehash_batch_counted` lines 280–344), and concurrent inserts can land in `new_table` during the migration. Our `step_rehash_complete` step asserts the union as a single primitive; the inductive `lemma_step_preserves_contents` over `step_rehash_one` covers the per-entry case faithfully.

## Tier A (post-1.0.0): partial coverage shipped

`seqlock_resize_tier_a.rs` extends tier B along the three axes flagged in the v1.1.0 backlog. The file is independently verified (`./run_proof.sh tier-a` => `verification results:: 31 verified, 0 errors`, ~0.7s wall), in addition to tier B.

| Property | Lemma | Status |
|---|---|---|
| **T13.1 — Linear-probe table model.** Replace `Set<u64>` with `Seq<u64>` (open-addressed, 0 = empty sentinel). | `Table = Seq<u64>`, `tab_lookup`, `tab_insert`, `tab_contents`, `probe_index`, `probe_terminus_at`, `probe_terminus` | Shipped. Captures the shipping layout exactly (`HashTableEntry { fp: AtomicU64, ... }` at lines 101-121). |
| **T13.1 — Probe correctness.** Linear-probe insert is faithful to the abstract set. | `lemma_probe_terminus_bounded`, `lemma_probe_terminus_slot`, `lemma_probe_indices_distinct`, `lemma_probe_index_in_range`, `lemma_probe_walk_matches`, `lemma_insert_then_lookup`, `lemma_insert_preserves_contents`, `lemma_insert_adds_fp`, `lemma_lookup_implies_in_contents` | Shipped. `lemma_insert_then_lookup` is the headline: a successful linear-probe insert is observable to a subsequent lookup. The probe-distinctness lemma (`lemma_probe_indices_distinct`) is the modular-arithmetic core; proved via `vstd::arithmetic::div_mod::lemma_fundamental_div_mod` + nonlinear-arith assertions. |
| **T13.2 — CAS soundness (spec-level).** A successful CAS at slot S transitions table contents from C to C ∪ {fp}; failure leaves the table unchanged. | `cas_step`, `lemma_cas_soundness`, `lemma_cas_failure_no_clobber`, `lemma_cas_during_resize_observable_a` | Shipped. The CAS is modeled as a pure function on `Seq<u64>`, matching the abstract semantics of `entry.fp.compare_exchange(0, fp, AcqRel, Acquire)` at lines 723-741. **What is NOT shipped:** Verus tracked pointers on the actual `AtomicU64` field (would require rewriting `FingerprintShard` to thread `Tracked<PointsTo<...>>` through every call). What IS shipped is the spec-level proof that the CAS algorithm preserves table soundness — the prerequisite for a future shipping-code annotation pass. |
| **T13.3 — Bounded-resize termination.** The reader retry loop terminates after at most `R + 1` iterations, where `R` is the number of resizes the writer performs during the reader's lifetime. | `reader_iters_bound`, `reader_step_consistent`, `lemma_reader_terminates`, `lemma_reader_progress`, `lemma_reader_consistent_snapshot` | Shipped (bounded form). **What is NOT shipped:** the unbounded-fairness case ("writers can't starve readers indefinitely") requires temporal-logic reasoning (LTL liveness with a fairness assumption on the writer's resize rate) that Verus does not yet have first-class support for. Documented as a tracked follow-up. |
| **End-to-end conservation.** Every fp present pre-resize is observable post-resize. | `rehash_complete_a`, `lemma_resize_preserves_contents_a`, `lemma_rehash_complete_preserves_old_a`, `lemma_finalize_promotes_new_table_a`, `theorem_no_fingerprint_lost_a`, `theorem_concurrent_insert_survives_a` | Shipped. The `theorem_no_fingerprint_lost_a` lemma is the table-level analog of tier-B's `theorem_no_fingerprint_lost`. |

### Shipping-code coverage now machine-checked at tier A

| Shipping source location | Tier-A spec |
|---|---|
| `page_aligned_fingerprint_store.rs:101-121` (`HashTableEntry` layout, fp=0 sentinel) | `Table = Seq<u64>`, `EMPTY()` |
| `page_aligned_fingerprint_store.rs:280-344` (`rehash_batch_counted`) | `step_rehash_one_a`, `lemma_rehash_preserves_a` |
| `page_aligned_fingerprint_store.rs:357-390` (`finalize_resize`) | `step_finalize_resize_a`, `lemma_finalize_parity_a`, `lemma_finalize_promotes_new_table_a` |
| `page_aligned_fingerprint_store.rs:469-497` (resize start, snapshot pointers) | `step_begin_resize_a`, `lemma_begin_resize_parity_a` |
| `page_aligned_fingerprint_store.rs:554-650` (`contains` reader path + retry loop) | `tab_lookup`, `lemma_reader_terminates`, `lemma_reader_consistent_snapshot` |
| `page_aligned_fingerprint_store.rs:656-839` (`contains_or_insert` resize-mode + normal-path) | `step_insert_during_resize_a`, `step_insert_stable_a`, `lemma_insert_stable_observable_a`, `lemma_cas_during_resize_observable_a` |
| `page_aligned_fingerprint_store.rs:723-741` (resize-mode CAS slot insert) | `cas_step`, `lemma_cas_soundness` |
| `page_aligned_fingerprint_store.rs:783-820` (normal-path CAS slot insert) | same `cas_step` model + `lemma_insert_then_lookup` |

### What is STILL abstracted in tier A

- Raw `slice::from_raw_parts` over `*mut HashTableEntry`. Tier A models the slice as a `Seq<u64>`; the pointer-to-slice conversion is axiomatic. Eliminating this would require `vstd::raw_ptr` annotations on `FingerprintShard::contains_or_insert` — open-ended rewrite work.
- Memory orderings (`Ordering::Acquire/Release/AcqRel`). Tier A treats atomics as sequentially consistent. The seqlock pattern is correct under SC, and the shipping code uses the standard release/acquire pairings — but a fully-relaxed-memory-model proof would require Verus support for the C++20 memory model (open-ended, not yet available).
- `resize_lock: Mutex<()>` (line 142). Tier A models "single resize at a time" via the seq parity bit. The mutex is what physically enforces this in shipping; tier A relies on the protocol-level parity invariant rather than the mutex directly.
- `mmap(MAP_ANONYMOUS)` zero-fill. Modeled as `Seq::new(cap, |_| 0)`. The kernel guarantee is axiomatic.

### Tier-A.5 (T13.4 partial): shipping-shape shadow methods

`shard_methods.rs` ships **17 verified items**, ~0.7s wall (`./run_proof.sh shadow`). It is the shipping-shape annotated shadow of `FingerprintShard`'s hot-path methods, using real Verus tracked- permission machinery (`PAtomicU64` + `Tracked<&PermissionU64>` + `Tracked<&mut PermissionU64>`) on real per-slot atomic cells.

| Shipping method | Shadow function (verified) | Lines covered |
|---|---|---|
| `contains` slot probe | `probe_slot_for_contains` | 634 (atomic load + 3-way fork) |
| `contains` 2-iteration unroll | `two_probe_contains` | 626-643 (loop body x2) |
| `contains` 3-iteration unroll | `three_probe_contains` | 626-643 (loop body x3) |
| `contains_or_insert` normal-path CAS | `cas_insert_or_observe` | 805-823 (CAS + 3-way result) |
| `contains_or_insert` per-iteration | `contains_or_insert_step` | 789-828 (load + CAS dispatch) |
| `rehash_batch_counted` per-slot step | `rehash_one_step` | 312-340 (CAS-or-skip) |

Bridge lemmas (shipping-shape -> tier-A spec):

| Bridge | Lemma |
|---|---|
| shadow CAS Inserted -> tier-A `cas_step` Some(...) | `lemma_cas_inserted_matches_tier_a` |
| shadow CAS preserves other slots | `lemma_cas_preserves_other_slots` |
| shadow probe Hit -> tier-A `tab_contents.contains(fp)` | `lemma_probe_hit_matches_tier_a` |
| shadow probe Empty -> tier-A `EMPTY` precondition | `lemma_probe_empty_matches_tier_a` |
| CAS-then-probe sees fp (shipping-shape `lemma_insert_then_lookup`) | `lemma_cas_then_probe_observes_fp` |
| `(fp + i) % cap` is in [0, cap) | `lemma_probe_index_in_bounds` |

What this tier gives beyond tier A:
1. **Real Verus permissions on real atomic cells.** Tier A modeled the table as `Seq<u64>`; this file uses `Vec<PAtomicU64>` plus a `Tracked<Map<int, PermissionU64>>` permission map — the same machinery a full T13.4 in-place rewrite would use.
2. **`requires`/`ensures` clauses tied to tier-A predicates.** Each shadow method's post-condition references tier-A's `tab_lookup`, `cas_step`, and `tab_insert` semantics via the bridge lemmas.
3. **Drop-in template for the in-place rewrite.** When the full T13.4 work happens, the rewrite team has a working blueprint for permission threading and contract shapes.

What the shadow file does NOT do:
- Does not replace the `FingerprintShard` itself. The `cargo build` is unchanged; the shadow file lives only under `verification/verus/` and is verified by `verus`, not compiled by `rustc` into the binary.
- Does not lift the outer probe loop into a single bounded-iteration Verus exec function. The 2- and 3-step unrolls demonstrate the shape; a fully-bounded `for probes in 0..cap` form requires a Verus loop with an inductive invariant and a per-slot permission swap. Tier A's `lemma_probe_terminus_bounded` already discharges termination at the spec level.
- Does not model the seqlock retry loop in shipping-shape. Tier A's `lemma_reader_terminates` covers that at the spec level.

### Tier-A.6 (T13.5): unbounded-fairness reader liveness

`reader_liveness.rs` extends tier A's bounded `lemma_reader_terminates` with the unbounded-liveness statement: "a writer cannot starve a reader indefinitely" under a temporal weak-fairness assumption that writers do not perpetually block in a resize. Run via `./run_proof.sh liveness`.

This file is preserved as the bounded-form fallback. The headline T13.3 + T13.5 result lives in `reader_liveness_v2.rs` (see below), which discharges all three axioms with explicit finite-prefix witnesses.

| Property | Lemma | Status |
|---|---|---|
| **Parity discipline** | `lemma_begin_resize_parity_seq`, `lemma_finalize_resize_parity_seq`, `lemma_stutter_parity_seq` | Proved (no axioms). |
| **Step monotonicity** | `lemma_seq_monotonic_seq`, `lemma_step_relation_monotonic` | Proved. |
| **Reader-attempt soundness** | `lemma_reader_attempt_soundness` | Proved. |
| **Finite-prefix monotonicity** | `theorem_prefix_seq_monotonic`, `theorem_prefix_seq_growth_bounded` | Proved by induction over `j - i`. |
| **Equal-seq implies stutter** | `theorem_equal_seq_implies_stutter` | Proved. |
| **Reader success is stutter** | `theorem_reader_success_is_stutter` | Proved. |
| **Writer-step parity in prefix** | `theorem_writer_step_parity_in_prefix` | Proved. |
| **Writer eventually finalizes** | `axiom_writer_eventually_finalizes` | Axiom (discharge plan in file). |
| **Reader can observe stutter** | `axiom_reader_can_observe_stutter` | Axiom (discharge plan in file). |
| **Extension composes** | `axiom_extension_composes` | Axiom (sequence-arithmetic glue). |
| **MAIN THEOREM** | `theorem_reader_eventually_succeeds` | Proved (modulo the three axioms). |
| **No-starvation corollary** | `theorem_no_starvation` | Proved (modulo the three axioms). |

### Tier-A.6 (T13.3 + T13.5, constructive): axiom-free reader liveness

`reader_liveness_v2.rs` is the constructive sibling of `reader_liveness.rs`. It proves the same headline statements (`theorem_reader_eventually_succeeds`, `theorem_no_starvation`) with **zero axioms**. Run via `./run_proof.sh reader-liveness-v2` (also accepts `liveness-v2`, `live-v2`, `l2`).

Each of the original three axioms is replaced by a constructive lemma whose witness is an explicit short `seq!` literal:

| Original axiom | Replacement lemma | Witness |
|---|---|---|
| `axiom_writer_eventually_finalizes` | `lemma_writer_eventually_finalizes` | `seq![s, finalize(s)]` — the resizing tail finalised |
| `axiom_reader_can_observe_stutter` | `lemma_reader_can_observe_stutter` | `seq![s, s]` — the stable tail stuttered once |
| `axiom_extension_composes` | `lemma_extension_composes` | `seq![s, finalize(s), finalize(s)]` — finalize then stutter |

`wf_prefix` over a 2- or 3-element literal reduces to a small number of `step_relation` checks that the SMT solver discharges by case analysis on `step_relation`'s three disjuncts. The composition witness inlines the case-2-then-case-1 reasoning that was previously admitted as `axiom_extension_composes`, so all three discharge in a single file with no temporal-logic machinery beyond the finite-prefix `Seq<ShardSeqState>` model.

| Property | Lemma | Status |
|---|---|---|
| All safety properties from `reader_liveness.rs` | (same names) | Proved (no axioms). |
| **Writer-fairness witness** | `lemma_writer_eventually_finalizes` | Proved (constructive 2-step witness). |
| **Reader-progress witness** | `lemma_reader_can_observe_stutter` | Proved (constructive 2-step stutter witness). |
| **Composition** | `lemma_extension_composes` | Proved (constructive 3-step finalize+stutter witness). |
| **MAIN THEOREM** | `theorem_reader_eventually_succeeds` | Proved (no axioms). |
| **No-starvation corollary** | `theorem_no_starvation` | Proved (no axioms). |

`./run_proof.sh reader-liveness-v2` reports `verification results:: 17 verified, 0 errors` in well under one second.

The original `reader_liveness.rs` is intentionally kept as the bounded-form fallback; it documents the temporal-logic shape and its axiom-discharge plan, both of which remain useful reference material for the eventual `state_machines!` port.

### Tier-A.7 (T13.4 Phase 1 + 1.5): shipping-shape verified wrapper

`shard_wrapper.rs` ships **34 verified items**, ~1s wall (`./run_proof.sh shard-wrapper`). It is the shipping-shape **wrapper struct** `VerifiedFingerprintShard` that mirrors the `FingerprintShard` itself field layout (slot array + capacity) and ships:

  - A **fully-bounded probe loop** (`bounded_contains_loop`) with inductive invariant on the probe index — the headline new capability beyond `shard_methods.rs`. The shadow file explicitly deferred this ("a fully-bounded `for probes in 0..cap` form would require a Verus loop with an inductive invariant ... out of scope for the 6-hour T13.4 timebox"). This file lifts it.
  - A **fully-bounded contains-or-insert loop** (`bounded_contains_or_insert_loop`) lifting lines 789-828 with the same loop invariant pattern, plus per-iteration permission-map mutation via `tracked_remove`/`tracked_insert`.
  - A **resize-mode dispatch wrapper** (`bounded_contains_during_resize`) lifting lines 578-622 — the old-table + new-table probe-or-skip control flow.
  - A **verified constructor** (`make_empty_shard`) that mints an `(VerifiedFingerprintShard, Tracked<Map<int, PermissionU64>>)` pair with the post-condition that every slot is `empty_slot()`. Models lines 187-256 (`FingerprintShard::new`) without the `mmap` allocation primitive (Phase-2 gap).
  - **Bridge lemmas** (`lemma_inserted_matches_cas_step`, `lemma_found_implies_in_view`, `lemma_inserted_visible_to_contains`, `lemma_hit_observation_implies_contains`, `lemma_empty_observation_locks_in`, `lemma_probe_index_step`, `lemma_wrapper_probe_index_in_range`) connecting wrapper outputs to tier-A's `tab_lookup` / `tab_insert` / `cas_step` predicates.

| Shipping source | Wrapper function (verified) |
|---|---|
| line 626-643: `contains` body | `bounded_contains_loop` |
| line 789-828: `contains_or_insert` body | `bounded_contains_or_insert_loop` |
| line 578-622: `contains` resize-mode dispatch | `bounded_contains_during_resize` |
| line 187-256: `FingerprintShard::new` | `make_empty_shard` |
| line 805-823: CAS at one slot | `cas_insert_or_observe` (re-stated from `shard_methods.rs`) |
| line 634: probe at one slot | `probe_slot_for_contains` (re-stated from `shard_methods.rs`) |
| line 559-651: `contains` outer seqlock retry loop | `bounded_seqlock_retry_contains` (Phase 1.5 add) |

What this tier gives beyond tier-A.5 (`shard_methods.rs`):

  1. **Bounded outer probe loop, fully verified end-to-end.** The shadow file is restricted to 2-step and 3-step unrolls because the loop-invariant lift was deferred. This file lifts it via the `index as int == probe_index(fp, probes as nat, cap as nat)` loop invariant + `decreases cap as u64 - probes` termination clause. This is the explicit "Step 3" deliverable from `T13.2-T13.4-design.md` (smallest standalone Verus win, no new `vstd` capability needed).

  2. **Shipping-shape struct.** The shadow has only `ShardCells { slots: Vec<PAtomicU64> }`. This file adds `capacity: usize` and a `Tracked<Map<int, PermissionU64>>` permission map, mirroring the `FingerprintShard` itself skeleton at lines 124-176. The `VerifiedFingerprintShard` is the Phase-1 wrapper struct.

  3. **Wrapper-level method bodies, not just per-iteration helpers.** The shadow proves single-iteration helpers (`probe_slot_for_contains`, `cas_insert_or_observe`). This file composes them into the full `bounded_contains_loop` and `bounded_contains_or_insert_loop` with the shipping-side bounded-iteration shape.

What this file does NOT do (Phase 2 / Phase 3):

  - **Does not replace the `FingerprintShard` itself.** The `cargo build` and `cargo test` are unchanged — the wrapper is additive and lives only under `verification/verus/`. Replacing the shipping type would require resolving the three concrete `vstd` capability gaps documented in `T13.2-T13.4-design.md`: atomic-pointer-swap with overlapping permission lifetimes, mmap-allocated `PointsToArray` with no provenance axiom, and `&self` + linear ghost token incompatibility under `AtomicInvariant`. The realistic path is a `state_machines!` reformulation; open-ended.
  - **Does not switch call sites in `runtime.rs`.** Phase 3 of the brief; out of scope until Phase 2 lands.
  - **Does not lift the seqlock retry loop's UNBOUNDED-fairness termination** into a verified exec function — tier A's `lemma_reader_terminates` covers it at the spec level. The Phase 1.5 add `bounded_seqlock_retry_contains` ships the BOUNDED form (`max_retries: u64` parameter, `decreases max_retries - retries`), matching shipping code's `MAX_RETRIES_BEFORE_PANIC = 1_000_000` ceiling.
  - **Does not model the `count: AtomicU64` / `state: AtomicU8` bookkeeping fields** (lines 134, 107). Tier A and the shadow explicitly omit them; this wrapper inherits that.

#### Phase 1.5 add (2026-05): bounded seqlock retry loop

`bounded_seqlock_retry_contains` lifts the shipping OUTER seqlock retry loop (lines 559-651) into a verified exec function. Bounded form via `max_retries: u64` parameter and `decreases max_retries - retries` termination clause. Models shipping code's `MAX_RETRIES_BEFORE_ PANIC` ceiling literally; the unbounded-fairness form remains open-ended (would require a writer-resize-count ghost decrement threaded through `finalize_resize`). Outcome enum `SeqlockRetryOutcome::{Stable(BoundedLookup), Exhausted}` makes the retry-budget exhaustion case total. +3 verified items (was 31, now 34).

### Tier-A.8 (T13.5 polish): `state_machines!` port of reader liveness

`reader_liveness_state_machine.rs` ships **15 verified items**, ~1s wall (`./run_proof.sh sm` or `./run_proof.sh reader-liveness-sm`). LTL-native restatement of v2's headline `theorem_no_starvation` using Verus's `state_machine!` macro.

**Status: optional polish.** The headline `theorem_no_starvation` already ships axiom-free in `reader_liveness_v2.rs` over a hand-rolled `Seq<ShardSeqState>` model. This file is the LTL-native equivalent: defines a `state_machine!(ShardSeq)` with three transitions (`begin_resize`, `finalize_resize`, `stutter`), two `#[invariant]` predicates, four `#[inductive]` proofs, plus a **refinement bridge** to `reader_liveness_v2.rs` showing every SM transition refines a v2 `step_relation_v2` step. The headline `theorem_no_starvation_sm` is restated and proved (constructively via the same 2-step stutter / 3-step finalize-stutter witnesses as v2). Zero axioms. Run via `./run_proof.sh sm`.

| Property | Lemma | Status |
|---|---|---|
| **State-machine init refinement** | `theorem_init_refines` | Proved. |
| **Per-transition refinement** | `lemma_begin_resize_refines`, `lemma_finalize_resize_refines`, `lemma_stutter_refines` | Proved. |
| **Step refinement** | `theorem_step_refines` | Proved (every SM step refines a v2 step). |
| **Prefix refinement** | `theorem_prefix_refines` | Proved (well-formed SM prefix lifts to v2 prefix). |
| **Reader-success refinement** | `lemma_reader_success_refines` | Proved. |
| **Constructive witnesses** | `lemma_reader_can_observe_stutter_sm`, `lemma_extension_composes_sm` | Proved (constructive 2-step / 3-step). |
| **MAIN THEOREM (SM-level)** | `theorem_reader_eventually_succeeds_sm` | Proved (no axioms). |
| **No-starvation corollary (SM-level)** | `theorem_no_starvation_sm` | Proved (no axioms). |

What this file does NOT do:
- Does NOT replace `reader_liveness_v2.rs`. v2 is the canonical 0-axiom proof; this is an LTL-native reformulation that future state-machines!-native consumers can build on.
- Does NOT introduce new soundness — same headline, different framework.
- Does NOT integrate with the shipping fingerprint shard. Like all other Verus proof files in this directory, this is verified by `verus`, not compiled by `rustc`.

### Tier-A.9 (T13.4 gap-1 PoC): `atomic_ptr_with_epoch.rs`

`atomic_ptr_with_epoch.rs` ships **16 verified items**, 0 errors (`./run_proof.sh epoch`). A `tokenized_state_machine!`-based skeleton proving that the RCU/seqlock atomic-pointer-swap pattern — Gap 1 in `T13.2-T13.4-design.md` — is verifiable in Verus today, with no new `vstd` primitive. Direct follow-up to [verus-lang/verus#2437](https://github.com/verus-lang/verus/issues/2437), where core Verus collaborator @tjhance pointed at `verus/examples/state_machines/counting.rs` as the pattern to use.

The protocol exposes `publish` / `reclaim` / `acquire_read` / `release_read` / `read_guards` transitions over an `Option<T>` storage slot, a `(nat, T)` counter, and a `Multiset<T>` of `ReadRef` tokens. The `read_guards` property realises the `guard` instruction tjhance referenced — gives `&T` access through a live `ReadRef` without consuming the linear permission. Verus's stability of the protocol's `main_inv` invariant proves: outstanding `ReadRef` count equals the counter, and every token names the same `T` as is stored.

What this PoC proves and what remains open is documented in `T13.2-T13.4-design.md` "Gap 1 update (2026-05-13)". Headline: single-epoch swap-blocks-until-readers-zero verifies; the multi-epoch generalisation, the exec-atomic wiring, and the gap-2 / gap-3 questions all resolve in existing Verus features — see the "Cross-survey of Verus examples" section in `T13.2-T13.4-design.md` for the full inventory.

### Tier-A.10 (T13.4 Phase 2 slice 1): `shard_exec_wired.rs`

`shard_exec_wired.rs` ships **19 verified items**, 0 errors (`./run_proof.sh exec-wired`). The first exec-wired artifact: a `VerifiedShard` struct that ties the `EpochProtocol` (shaped exactly after arc.rs's `RefCounter<Perm>`) to a real `PPtr<InnerShard>` allocation holding two `PAtomicU64`s — a slot atomic (the FingerprintShard hash-slot stand-in) and a refcount cell.

Five methods covering the full lifecycle:

- `new(initial_fp) -> Self` — allocates the inner cell with refcount 1, deposits the linear `PointsTo` into the protocol's storage_option, returns the first reader.
- `read(&self) -> u64` — `slot.load(...)` through the protocol's `reader_guard` + `Shared<AtomicInvariant>`. Validates the read path (gap-1: linear permission parked in protocol, shared `&` access via guard).
- `cas_insert(&self, expected, new_fp) -> bool` — `slot.compare_exchange(...)` through the same invariant. Validates the write path; mirrors the FingerprintShard hot path's CAS-from-0 on an empty slot.
- `clone(&self) -> Self` — CAS-loop on `rc_cell` to bump the refcount + protocol `do_clone` to mint a new reader token. Validates the Arc-of-Arc primitive: multiple readers of the same underlying allocation, each with its own protocol-tracked reference.
- `dispose(self)` — `fetch_sub_wrapping` on `rc_cell`; if the count was 1, calls `dec_to_zero` to withdraw the storage permission + frees the PPtr via `ptr.take` + `ptr.free`. Otherwise calls `dec_basic`. Completes the reclaim half of the lifecycle.

Together these validate gaps 1 + 3 end-to-end against real exec atomics plus the Arc-of-Arc multi-reader + reclaim primitive. The protocol uses arc.rs's split-field shape (`counter: nat` + `storage: Option<T>` + `reader: Multiset<T>`) with `do_deposit` / `do_clone` / `dec_basic` / `dec_to_zero` as transitions and `reader_guard` as the borrow property.

The reclaim path here is *not* the QSBR pattern FingerprintShard actually uses (lazy free on next resize via the seqlock); arc.rs's refcount-and-free is used as the closest Verus blueprint. The FingerprintShard `cleanup_old_memory` path would be its own state-machine refinement on top.

Plus a `demonstrate_swap_pattern()` exec function (slice 3c) that exercises two coexisting `VerifiedShard` instances through a swap-like sequence: epoch A's writer publishes + a reader clones; epoch B's writer publishes a separate allocation; A's reader continues across the publication; B's reader attaches; A is drained independently via successive `dispose`s; B is then drained too. Validates the multi-epoch composition by construction — both epochs maintain independent protocols, independent reader counts, and independent allocations.

Slice 5 adds **`ShardRegistry`** + **`demonstrate_registry_swap()`**: a lock-based RCU registry that wraps a `VerifiedShard` inside a `vstd::rwlock::RwLock<VerifiedShard, WellFormedShardPred>`. Two `&self` methods:

- `clone_current(&self) -> VerifiedShard` — acquires the read lock, clones the inner shard (minting a fresh reader on its allocation), releases.
- `swap(&self, new: VerifiedShard) -> VerifiedShard` — acquires the write lock, replaces the protected shard, returns the old one for the caller to dispose at their leisure. Existing outstanding clones from before the swap continue to access the OLD allocation via their own ReadRef tokens.

The correctness story (no use-after-free, no data races on the linear permissions) is the same as for a lock-free RCU swap because `WellFormedShardPred` carries `v.wf()` and `release_write` requires `inv(new_val)`. The runtime perf cost is the lock contention vs. truly atomic publication; the lock-free version using `vstd::atomic_ghost::AtomicU64<...>` carrying the protocol's tokens is a follow-up slice.

### Tier-A.12 (T13.4 Phase 2 slice 6): `shard_multi_slot.rs`

`shard_multi_slot.rs` ships **19 verified items**, 0 errors (`./run_proof.sh multi-slot`). Extends the single-slot `VerifiedShard` in `shard_exec_wired.rs` to two slots, validating that the per-slot `AtomicInvariant` bookkeeping composes for `N` slots: each `PAtomicU64` slot gets its own `PermissionU64` carried in `GhostStuff`, and `read_at(idx)` / `cas_insert_at(idx, ...)` branch on an index parameter without losing soundness.

The 2-slot hardcoded version validates the per-slot composition pattern. The N-slot extension (e.g. `Vec<PAtomicU64>` for `slots` + `Map<int, PermissionU64>` ghost) is mechanically the same shape — see the comment at the bottom of the file for the worked-out generalisation. Lifecycle methods (`clone` / `dispose`) and the `ShardRegistry` from `shard_exec_wired.rs` are not re-implemented here since they're slot-count-independent.

### Tier-A.11 (T13.4 Phase 2 slice 4): `mmap_external_body.rs`

`mmap_external_body.rs` ships **1 verified item**, 0 errors (`./run_proof.sh mmap`). The verified item is a `demonstrate_mmap_lifecycle` exec function that allocates a 2 MiB huge-page region via `mmap_allocate_huge_pages`, then immediately frees it via `munmap_huge_pages`. The two wrapper functions are `#[verifier::external_body]` — trusted at the FFI boundary the same way `vstd::raw_ptr::allocate` itself is.

This closes gap 2 from the original T13.4 design doc. The original framing claimed "Verus refuses to admit a fresh ghost token from an external-body function" — factually wrong, since `vstd::raw_ptr::allocate` (line 907 of `source/vstd/raw_ptr.rs`) is itself `external_body` and mints fresh `Tracked<PointsToRaw>` / `Tracked<Dealloc>` via `Tracked::assume_new()`. This file ports the exact same shape to `libc::mmap` with `MAP_HUGETLB | MAP_POPULATE` (the flags `FingerprintShard::allocate_huge_pages` uses), introducing a parallel `MmapDealloc` tracked struct + `MmapDeallocData` ghost struct mirroring `Dealloc` / `DeallocData`. The trust assumption is identical to vstd's: the implementation actually does what the `ensures` clause says.

Still not covered: mmap allocation (gap-2 closure), the exec wiring of the outer `AtomicPtr<InnerShard>` that publishes "which allocation is current" (the `AtomicPtrWithEpoch` shape from tjhance's reply, with its own ghost state tracking address→permission), and multi-slot tables (single u64 slot; the array variant is mechanical via `Vec<PAtomicU64>` + index). The protocol composition validated by `demonstrate_swap_pattern` is the prerequisite for the outer AtomicPtr wiring.

### Genuinely deferred to v1.2.0+

- **Full Verus tracked-pointer integration of the `FingerprintShard` itself (Phase 2 / Phase 3 of T13.4).** Annotating the shipping code with `Tracked<PointsToArray<HashTableEntry>>` and threading the permissions through every call site. Requires rewriting `allocate_huge_pages`, `allocate_file_backed`, the resize swap, and every `unsafe { std::slice::from_raw_parts(...) }` to use `vstd::raw_ptr::PPtr` (plus an `external_body` mmap wrapper modeled on `vstd::raw_ptr::allocate`). Substantial design-and-engineering work, but no longer blocked on any vstd capability gap — all three originally-documented gaps (atomic-pointer swap, mmap-allocated `PointsToArray`, `&self` + linear ghost) resolve in current Verus features. Working templates: `examples/state_machines/arc.rs` for the RCU exec wiring, `vstd::raw_ptr::allocate` for the mmap-shape external_body, `vstd::logatom` for the logically-atomic specification pattern that avoids per-step invariant opens. The shadow methods in `shard_methods.rs` and the shipping-shape wrapper in `shard_wrapper.rs` are the working blueprints for the shipping-code side; `atomic_ptr_with_epoch.rs` is the validated protocol skeleton.
- **Discharge of the three reader-liveness axioms.** Discharged constructively in `reader_liveness_v2.rs` (17 verified, 0 axioms, `./run_proof.sh reader-liveness-v2`). The `state_machines!` port ships in `reader_liveness_state_machine.rs` (15 verified, 0 axioms, `./run_proof.sh sm`). Both ship the headline `theorem_no_starvation`. The original `reader_liveness.rs` is preserved as the bounded-form fallback.

## Honest verdict on Verus for tlaplusplus

**Tier B is real value with bounded cost.** A 600-line proof file with 19 verified lemmas gives us a machine-checked artifact that the *protocol* is sound. If a future refactor of `page_aligned_fingerprint_store.rs` changes the protocol shape (e.g. adds a "cancel resize mid-flight" path), the proof will catch it during code review when the abstraction is updated to match.

**Tier A is open-ended effort.** Direct verification of the unsafe shipping code would require a substantial rewrite. For 1.0.0 the risk/value tradeoff favors keeping the shipping code idiomatic and the Verus proof at the protocol level.

**Recommendation for v1.1+**: extend this proof along two axes that *don't* require rewriting shipping code:

1. Model linear-probe collision behavior as a concrete `Seq<Option<u64>>` instead of `Set<u64>`, and prove the abstract `contains`/`insert` spec. This catches probe-sequence bugs without touching shipping.
2. Model worker-thread interleavings explicitly via Verus's `state_machines!` macro, so reader retries can be proved live (not just safe).

## How to reproduce

### Prerequisites

- Linux x86_64 (recommended; aarch64 works with the workaround below)
- 8 GB RAM, ~5 GB disk
- Rust toolchain (rustup) — Verus pins it to a specific version

### Install Verus from source

Verus does not yet ship a prebuilt aarch64 Linux binary (https://github.com/verus-lang/verus/releases/latest as of 2026-04-25 ships only x86_64 Linux + arm64 macOS + x86_64 macOS + x86_64 Windows). On x86_64 Linux:

```
wget https://github.com/verus-lang/verus/releases/latest/download/<asset>.zip
unzip ...
```

On aarch64 Linux, build from source:

```
git clone --depth 1 https://github.com/verus-lang/verus.git
cd verus/source

# Z3 4.12.5 — upstream's get-z3.sh hits a broken aarch64 zip
# (see verus issue tracker, the upstream zip contains an x86 binary
# under an arm64-named filename). Workaround:
sudo apt-get install -y z3   # 4.13.3 from Ubuntu 25.10 main
cp /usr/bin/z3 ./z3

. ../tools/activate
vargo --no-solver-version-check build --release --vstd-no-verify
```

`--vstd-no-verify` is required when using a non-pinned Z3, because `vstd` itself contains proof obligations that go through cleanly only on Z3 4.12.5. This does not affect the soundness of *our* proof — Verus still runs Z3 against `seqlock_resize.rs`'s VC, just not against the standard library's.

### Run the proof

```
cd verification/verus
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh                  # tier B (default)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh tier-a           # tier A
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh shard-methods    # tier-A.5 (shipping-shape shadow)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh shard-wrapper    # tier-A.7 (T13.4 Phase 1 + 1.5 wrapper)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh reader-liveness-v2  # T13.5 axiom-free reader liveness
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh sm               # T13.5 state_machines! port
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh epoch            # T13.4 gap-1 PoC (verus-lang/verus#2437 follow-up)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh exec-wired       # T13.4 Phase 2 slices 1-3c (exec wiring)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh mmap             # T13.4 Phase 2 slice 4 (gap-2 mmap external_body)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh multi-slot       # T13.4 Phase 2 slice 6 (per-slot AtomicInvariant)
```

Or run all under CI; see `.github/workflows/verus.yml` for the complete gate (T13.6). The CI gate currently runs tier-b, tier-a, shard-methods, and reader-liveness-v2; `shard-wrapper` is added in this commit and should be wired into the CI workflow alongside the others.

Successful output (shard-wrapper, T13.4 Phase 1):

```
Using verus: /home/ubuntu/verus/source/target-verus/release/verus
Verifying: shard_wrapper.rs

verification results:: 31 verified, 0 errors
```

Successful output (tier B):

```
Using verus: /home/ubuntu/verus/source/target-verus/release/verus
Verifying: seqlock_resize.rs

verification results:: 19 verified, 0 errors
```

Successful output (tier A):

```
Using verus: /home/ubuntu/verus/source/target-verus/release/verus
Verifying: seqlock_resize_tier_a.rs

verification results:: 31 verified, 0 errors
```

Both files verify in well under one second (after the 10-15 minute one-time Verus build).

### Toolchain versions used (2026-04-25)

- Verus: HEAD of `main` branch at clone time, commit `release/0.2026.04.19.6f7d4de` or newer.
- Rust: pinned to 1.95.0 by `verus/rust-toolchain.toml`.
- Z3: 4.13.3 (Ubuntu 25.10 apt) with `-V no-solver-version-check`. Verus's "preferred" Z3 is 4.12.5; behavior on 4.13.3 has not been formally validated by the Verus team but produces identical results for our small first-order proof obligations.
