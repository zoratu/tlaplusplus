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

## What is left for tier A

A full tier-A proof would establish:

- **Direct verification of the production Rust code**, with all `unsafe`
  pointer arithmetic in scope. Verus supports this via `vstd::raw_ptr`
  and `vstd::atomic_ghost` but requires substantial rewrites to make the
  production code amenable: replacing raw `*mut HashTableEntry` with
  Verus-tracked pointers, attaching ghost permissions to every CAS,
  and proving the linear-probe loop invariants. Estimated effort: 2-4
  agent-weeks for one experienced Verus user.
- **Proving the rehash batch loop terminates**, with a decreases clause
  bound by `rehash_cursor`.
- **Memory safety of the file-backed fallback** (`allocate_file_backed`
  branches in `resize`).
- **Liveness**: a reader does not retry forever. Bounded by the maximum
  number of concurrent resizes.

These are tracked as follow-ups in `RELEASE_1.0.0_PLAN.md` under
`### Follow-ups (parked)` for T13.

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
VERUS_DIR=/home/ubuntu/verus ./run_proof.sh
```

Successful output:

```
Using verus: /home/ubuntu/verus/source/target-verus/release/verus

verification results:: 19 verified, 0 errors
```

### Toolchain versions used (2026-04-25)

- Verus: HEAD of `main` branch at clone time, commit
  `release/0.2026.04.19.6f7d4de` or newer.
- Rust: pinned to 1.95.0 by `verus/rust-toolchain.toml`.
- Z3: 4.13.3 (Ubuntu 25.10 apt) with `-V no-solver-version-check`.
  Verus's "preferred" Z3 is 4.12.5; behavior on 4.13.3 has not been
  formally validated by the Verus team but produces identical results
  for our small first-order proof obligations.
