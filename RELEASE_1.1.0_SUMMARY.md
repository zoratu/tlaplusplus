# tlaplusplus v1.1.0

Feature release rolling up the post-1.0 sweep — the first wave of items
the v1.0.0 plan had marked **DEFER TO 1.1.0**, plus a mid-cycle
soundness fix and a Verus production-shape proof tier.

## Highlights

### Symbolic Init — joint encoding (T5.4 + T5.5)

- **T5.4 — Streaming Init enumeration via a producer thread.** The
  Init producer now overlaps with worker exploration, so large
  cross-product Inits no longer block the first worker.
- **T5.5 — Joint Init + invariant Z3 encoding.** Init constraints and
  invariants are encoded as a single Z3 formula and enumerated
  together rather than enumerate-then-filter. Einstein-class spec:
  **44 min → 14 ms** wall-time (including Z3 startup). Falls back
  cleanly when the invariant body is not Z3-reducible.

### Liveness scaling (T10.1 / T10.2 / T10.3 / T10.4)

- **T10.1 / T10.3 / T10.4** — parallel-flatten via dashmap raw shards,
  trivial-SCC pre-filter, per-action transition shard.
- **T10.2 partial** — nested-DFS streaming-SCC oracle (opt-in via
  `--liveness-streaming`), with the O(N) red-DFS fix. Phase-2
  (drive from live exploration frontier) tracked for v1.2.0.

### Trace minimization (T9.1 / T9.2 / T9.3)

- Transitive variable relevance through operator inlining, median BFS
  seed, suffix shortening to alternate violations.

### Partial-order reduction (T7.1 / T7.2 / T7.3)

- Batched per-disjunct evaluation (PorBenchProcessGrid 19.7x →
  **39.2x**), smarter stubborn-set seed, **POR for liveness via the
  Peled (1994) visible-action proviso** — lifts the v1.0.0 safety-only
  restriction.

### Robustness (T11.3 / T11.4 / T11.5 / T12.1)

- T11.3 per-PR chaos-smoke gate; T11.4 spill Err-branch inflight leak
  fix; T11.5 violation-exit hang fix at NUMA-auto worker counts;
  T12.1 explicit 8 MB stack for the recursive-depth test.

### Verification (T13.1 – T13.6 partial)

- T13.1+T13.2+T13.3 — Verus tier A (31 lemmas, linear-probe model +
  spec-level CAS + bounded reader retry).
- T13.4 partial — tier-A.5 production-shape shadow methods (+17
  lemmas) for the `FingerprintShard` hot path.
- T13.5 — reader-liveness proof via Verus `state_machines!`.
- T13.6 — CI workflow gate for the Verus tier-A run.

### Compiled-vs-interpreted soundness fix (T101.1)

- **Soundness fix.** Compiler arithmetic associativity differed from
  the interpreter on five distinct shapes, producing silent Ok-vs-Err
  and value divergences. Caught by the T2 proptest harness; compiler
  parser tightened to match interpreter validation.

## Validation gates

All green on the v1.1.0 tag:

| Gate | Result |
|---|---|
| `cargo test --release` | 786 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 808 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 813 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `cargo test --release --test compiled_vs_interpreted` (PROPTEST_CASES=2048) | 17 pass / 0 fail |
| `cargo test --release --features symbolic-init --test joint_init_invariant_t5_5` | 3 pass / 0 fail |

## Compatibility

Drop-in for v1.0.x. No public-API or CLI changes; new behaviour is
opt-in (`--features symbolic-init` for T5.4/T5.5,
`--liveness-streaming` for T10.2 oracle). Fingerprint format,
checkpoint format, and state-graph dump format are all unchanged.
