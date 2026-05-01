# Verus proofs for tlaplusplus

This directory contains formal verification artifacts for tlaplusplus's
own correctness, separate from the model checker's TLA+ frontend.

The first artifact (T13, release 1.0.0) is `seqlock_resize.rs` — a
machine-checked proof of the seqlock-based dynamic-resize protocol used
by the lock-free fingerprint store
(`src/storage/page_aligned_fingerprint_store.rs`).

## What is proved

The proof file declares 19 lemmas that Verus discharges via the Z3 SMT
solver. Together they establish, on an abstract model of the protocol:

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

The main theorem is the precise machine-checked statement of the soundness
property the user cares about: **"Once a state's fingerprint is admitted to
the fingerprint store, no resize event can cause the model checker to
silently re-explore it or to silently miss a future violation by losing
track of it."**

## What is *assumed* (axioms / unproven preconditions)

The proof is at the **protocol abstraction layer**. It deliberately
abstracts:

1. **Pointer arithmetic and open-addressed probing.** The production
   code stores fingerprints in a flat array indexed by `fp % capacity`
   with linear probing on collisions. The proof models the table as a
   `Set<u64>`. Implication: the proof shows the *protocol* is correct,
   but does not (yet) prove that the probe sequence implementation
   actually realises that abstract set faithfully. A bug like "linear
   probe wraps incorrectly at the table boundary" would not be caught
   by this proof.

2. **Memory orderings of atomic operations.** The proof treats each
   protocol step as atomic. The production code uses `AcqRel`/`Acquire`
   pairings. The proof assumes these orderings are sufficient — i.e.
   that the abstract sequential semantics is the right model for the
   relaxed-atomic concurrent execution. (For the seqlock pattern this
   is the standard assumption, justified in the literature; see
   "C++ Concurrency in Action" Ch. 5.)

3. **Single ongoing resize.** The protocol model assumes at most one
   resize is in progress at a time. The production code enforces this
   via `resize_lock: Mutex<()>` (line 142). We do not model the mutex
   explicitly; we instead model "resize is in progress" as the
   `seq % 2 == 1` parity bit being set.

4. **Mmap allocation produces a zero-filled region.** Production uses
   `MAP_ANONYMOUS` (line 466 comment) which the kernel guarantees to
   zero-fill. The proof models `step_begin_resize` as setting
   `new_table` to the empty set, equivalent to all-zero entries.

5. **Rehash completion as an atomic ghost step.** The production code
   migrates entries one bucket at a time (`rehash_batch_counted` lines
   280–344), and concurrent inserts can land in `new_table` during the
   migration. Our `step_rehash_complete` step asserts the union as a
   single primitive; the inductive `lemma_step_preserves_contents` over
   `step_rehash_one` covers the per-entry case faithfully.

## Tier A (post-1.0.0): partial coverage shipped

`seqlock_resize_tier_a.rs` extends tier B along the three axes flagged
in the v1.1.0 backlog. The file is independently verified
(`./run_proof.sh tier-a` => `verification results:: 31 verified, 0
errors`, ~0.7s wall), in addition to tier B.

| Property | Lemma | Status |
|---|---|---|
| **T13.1 — Linear-probe table model.** Replace `Set<u64>` with `Seq<u64>` (open-addressed, 0 = empty sentinel). | `Table = Seq<u64>`, `tab_lookup`, `tab_insert`, `tab_contents`, `probe_index`, `probe_terminus_at`, `probe_terminus` | Shipped. Captures the production layout exactly (`HashTableEntry { fp: AtomicU64, ... }` at lines 101-121). |
| **T13.1 — Probe correctness.** Linear-probe insert is faithful to the abstract set. | `lemma_probe_terminus_bounded`, `lemma_probe_terminus_slot`, `lemma_probe_indices_distinct`, `lemma_probe_index_in_range`, `lemma_probe_walk_matches`, `lemma_insert_then_lookup`, `lemma_insert_preserves_contents`, `lemma_insert_adds_fp`, `lemma_lookup_implies_in_contents` | Shipped. `lemma_insert_then_lookup` is the headline: a successful linear-probe insert is observable to a subsequent lookup. The probe-distinctness lemma (`lemma_probe_indices_distinct`) is the modular-arithmetic core; proved via `vstd::arithmetic::div_mod::lemma_fundamental_div_mod` + nonlinear-arith assertions. |
| **T13.2 — CAS soundness (spec-level).** A successful CAS at slot S transitions table contents from C to C ∪ {fp}; failure leaves the table unchanged. | `cas_step`, `lemma_cas_soundness`, `lemma_cas_failure_no_clobber`, `lemma_cas_during_resize_observable_a` | Shipped. The CAS is modeled as a pure function on `Seq<u64>`, matching the abstract semantics of `entry.fp.compare_exchange(0, fp, AcqRel, Acquire)` at production lines 723-741. **What is NOT shipped:** Verus tracked pointers on the actual `AtomicU64` field (would require rewriting `FingerprintShard` to thread `Tracked<PointsTo<...>>` through every call). What IS shipped is the spec-level proof that the CAS algorithm preserves table soundness — the prerequisite for a future production-code annotation pass. |
| **T13.3 — Bounded-resize termination.** The reader retry loop terminates after at most `R + 1` iterations, where `R` is the number of resizes the writer performs during the reader's lifetime. | `reader_iters_bound`, `reader_step_consistent`, `lemma_reader_terminates`, `lemma_reader_progress`, `lemma_reader_consistent_snapshot` | Shipped (bounded form). **What is NOT shipped:** the unbounded-fairness case ("writers can't starve readers indefinitely") requires temporal-logic reasoning (LTL liveness with a fairness assumption on the writer's resize rate) that Verus does not yet have first-class support for. Documented as a tracked follow-up. |
| **End-to-end conservation.** Every fp present pre-resize is observable post-resize. | `rehash_complete_a`, `lemma_resize_preserves_contents_a`, `lemma_rehash_complete_preserves_old_a`, `lemma_finalize_promotes_new_table_a`, `theorem_no_fingerprint_lost_a`, `theorem_concurrent_insert_survives_a` | Shipped. The `theorem_no_fingerprint_lost_a` lemma is the table-level analog of tier-B's `theorem_no_fingerprint_lost`. |

### Production-code coverage now machine-checked at tier A

| Production source location | Tier-A spec |
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

- Raw `slice::from_raw_parts` over `*mut HashTableEntry`. Tier A models
  the slice as a `Seq<u64>`; the pointer-to-slice conversion is
  axiomatic. Eliminating this would require `vstd::raw_ptr` annotations
  on `FingerprintShard::contains_or_insert` — still an estimated 1-2
  agent-weeks of code-rewrite work.
- Memory orderings (`Ordering::Acquire/Release/AcqRel`). Tier A treats
  atomics as sequentially consistent. The seqlock pattern is correct
  under SC, and the production code uses the standard release/acquire
  pairings — but a fully-relaxed-memory-model proof would require Verus
  support for the C++20 memory model (research-grade, not yet
  available).
- `resize_lock: Mutex<()>` (line 142). Tier A models "single resize at
  a time" via the seq parity bit. The mutex is what physically
  enforces this in production; tier A relies on the protocol-level
  parity invariant rather than the mutex directly.
- `mmap(MAP_ANONYMOUS)` zero-fill. Modeled as `Seq::new(cap, |_| 0)`.
  The kernel guarantee is axiomatic.

### Tier-A.5 (T13.4 partial): production-shape shadow methods

`shard_methods.rs` ships **17 verified items**, ~0.7s wall
(`./run_proof.sh shadow`). It is the production-shape annotated shadow
of `FingerprintShard`'s hot-path methods, using real Verus tracked-
permission machinery (`PAtomicU64` + `Tracked<&PermissionU64>` +
`Tracked<&mut PermissionU64>`) on real per-slot atomic cells.

| Production method | Shadow function (verified) | Production lines covered |
|---|---|---|
| `contains` slot probe | `probe_slot_for_contains` | 634 (atomic load + 3-way fork) |
| `contains` 2-iteration unroll | `two_probe_contains` | 626-643 (loop body x2) |
| `contains` 3-iteration unroll | `three_probe_contains` | 626-643 (loop body x3) |
| `contains_or_insert` normal-path CAS | `cas_insert_or_observe` | 805-823 (CAS + 3-way result) |
| `contains_or_insert` per-iteration | `contains_or_insert_step` | 789-828 (load + CAS dispatch) |
| `rehash_batch_counted` per-slot step | `rehash_one_step` | 312-340 (CAS-or-skip) |

Bridge lemmas (production-shape -> tier-A spec):

| Bridge | Lemma |
|---|---|
| shadow CAS Inserted -> tier-A `cas_step` Some(...) | `lemma_cas_inserted_matches_tier_a` |
| shadow CAS preserves other slots | `lemma_cas_preserves_other_slots` |
| shadow probe Hit -> tier-A `tab_contents.contains(fp)` | `lemma_probe_hit_matches_tier_a` |
| shadow probe Empty -> tier-A `EMPTY` precondition | `lemma_probe_empty_matches_tier_a` |
| CAS-then-probe sees fp (production-shape `lemma_insert_then_lookup`) | `lemma_cas_then_probe_observes_fp` |
| `(fp + i) % cap` is in [0, cap) | `lemma_probe_index_in_bounds` |

What this tier gives beyond tier A:
1. **Real Verus permissions on real atomic cells.** Tier A modeled the
   table as `Seq<u64>`; this file uses `Vec<PAtomicU64>` plus a
   `Tracked<Map<int, PermissionU64>>` permission map — the same
   machinery a full T13.4 production rewrite would use.
2. **`requires`/`ensures` clauses tied to tier-A predicates.** Each
   shadow method's post-condition references tier-A's `tab_lookup`,
   `cas_step`, and `tab_insert` semantics via the bridge lemmas.
3. **Drop-in template for the production rewrite.** When the full
   multi-week T13.4 work happens, the rewrite team has a working
   blueprint for permission threading and contract shapes.

What the shadow file does NOT do:
- Does not replace the production `FingerprintShard`. Production
  `cargo build` is unchanged; the shadow file lives only under
  `verification/verus/` and is verified by `verus`, not compiled by
  `rustc` into the binary.
- Does not lift the outer probe loop into a single bounded-iteration
  Verus exec function. The 2- and 3-step unrolls demonstrate the
  shape; a fully-bounded `for probes in 0..cap` form requires a Verus
  loop with an inductive invariant and a per-slot permission swap.
  Tier A's `lemma_probe_terminus_bounded` already discharges
  termination at the spec level.
- Does not model the seqlock retry loop in production-shape. Tier A's
  `lemma_reader_terminates` covers that at the spec level.

### Genuinely deferred to v1.2.0+

- **Full Verus tracked-pointer integration of the production
  `FingerprintShard` itself.** Annotating the production code with
  `Tracked<PointsToArray<HashTableEntry>>` and threading the
  permissions through every call site (vs the shadow approach which
  ships annotated mirror methods). Requires rewriting
  `allocate_huge_pages`, `allocate_file_backed`, the resize swap, and
  every `unsafe { std::slice::from_raw_parts(...) }` to use
  `vstd::raw_ptr::PPtr` — multi-week work. The shadow methods in
  `shard_methods.rs` are the working blueprint for this rewrite.
- **Unbounded-fairness liveness.** "A writer cannot starve a reader
  indefinitely" requires LTL liveness with a fairness assumption.
  Verus's state-machine framework supports this but requires a
  redesign of the proof structure. Bounded termination is shipped now;
  unbounded liveness is tracked.
- **CI gate.** Verus build is ~10 min on aarch64, requires the Z3 apt
  workaround, and is not yet `cargo`-driven. CI integration is tracked
  as a separate piece of tooling work.

## Honest verdict on Verus for tlaplusplus

**Tier B is real value with bounded cost.** A 600-line proof file with
19 verified lemmas, completed in one focused agent run, gives us a
machine-checked artifact that the *protocol* is sound. If a future
refactor of `page_aligned_fingerprint_store.rs` changes the protocol
shape (e.g. adds a "cancel resize mid-flight" path), the proof will
catch it during code review when the abstraction is updated to match.

**Tier A is research-grade effort.** Direct verification of the unsafe
production code would require a multi-week rewrite. For 1.0.0 the
risk/value tradeoff favors keeping the production code idiomatic and
the Verus proof at the protocol level.

**Recommendation for v1.1+**: extend this proof along two axes that
*don't* require rewriting production code:

1. Model linear-probe collision behavior as a concrete `Seq<Option<u64>>`
   instead of `Set<u64>`, and prove the abstract `contains`/`insert`
   spec. This catches probe-sequence bugs without touching production.
2. Model worker-thread interleavings explicitly via Verus's
   `state_machines!` macro, so reader retries can be proved live (not
   just safe).

## How to reproduce

### Prerequisites

- Linux x86_64 (recommended; aarch64 works with the workaround below)
- 8 GB RAM, ~5 GB disk
- Rust toolchain (rustup) — Verus pins it to a specific version

### Install Verus from source

Verus does not yet ship a prebuilt aarch64 Linux binary
(https://github.com/verus-lang/verus/releases/latest as of
2026-04-25 ships only x86_64 Linux + arm64 macOS + x86_64 macOS +
x86_64 Windows). On x86_64 Linux:

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

`--vstd-no-verify` is required when using a non-pinned Z3, because
`vstd` itself contains proof obligations that go through cleanly only
on Z3 4.12.5. This does not affect the soundness of *our* proof —
Verus still runs Z3 against `seqlock_resize.rs`'s VC, just not against
the standard library's.

### Run the proof

```
cd verification/verus
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh           # tier B (default)
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh tier-a    # tier A
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh shadow    # tier-A.5 (production-shape shadow)
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

Both files verify in well under one second (after the 10-15 minute
one-time Verus build).

### Toolchain versions used (2026-04-25)

- Verus: HEAD of `main` branch at clone time, commit
  `release/0.2026.04.19.6f7d4de` or newer.
- Rust: pinned to 1.95.0 by `verus/rust-toolchain.toml`.
- Z3: 4.13.3 (Ubuntu 25.10 apt) with `-V no-solver-version-check`.
  Verus's "preferred" Z3 is 4.12.5; behavior on 4.13.3 has not been
  formally validated by the Verus team but produces identical results
  for our small first-order proof obligations.
