# tlaplusplus v1.0.0

A Rust TLA+ model checker with TLC feature parity and 10.7x faster state
exploration on many-core systems. v1.0.0 is the first stable release.

## Headline numbers

- **756 default tests, 776 with failpoints, 774 with symbolic-init, 0 failures**
- **13/13** specs match TLC exactly via the diff CI gate
  ([scripts/diff_tlc.sh](scripts/diff_tlc.sh))
- **174/182 (95.6%)** of [tlaplus/Examples](https://github.com/tlaplus/Examples)
  pass full model checking at 60s; 182/182 pass analysis
- **10.7x faster** than Java TLC on the synthetic counter-grid benchmark
  (128-core AMD EPYC); up to 22x on NUMA-optimized configs
- **Verus tier-B + tier-A proofs** of the seqlock resize protocol — 19 + 31
  lemmas verified, including `theorem_no_fingerprint_lost` and the
  linear-probe-table soundness extension

## What's new since v0.3.0

### Correctness
- **Differential CI gate vs TLC** (T1) — 13 curated specs run under both
  checkers on every push; uncovered and fixed seven soundness bugs (T1.1,
  T1.3, T1.4, T1.5, T1.6, T2.4, T11.5)
- **Compiled-vs-interpreted proptest equivalence** (T2) — well-typed
  expression generator, 17 proptest cases, clean across 9 seeds at
  `PROPTEST_CASES=2048`
- **State-graph snapshot tests** (T3) — 12 specs pinned to 128-bit XxHash3
  digests, validated against TLC v2.19
- **Mutation testing audit** (T4) — `cargo-mutants` against eval/action paths;
  17 inline kill-tests added
- **Cross-arch CI matrix** (T12) — `[ubuntu-latest, ubuntu-24.04-arm]`
- **Regehr-style swarm testing** (T16) — both the proptest harness (random
  17-bit feature mask) and the chaos soak (1-4 concurrent failpoints per iter)

### Performance
- **Symbolic Init enumeration via Z3** (T5, opt-in `--features symbolic-init`)
  — 10-41x on filtered record-set Init shapes
- **Cross-node distributed work stealing** (T6) — TCP steal protocol with
  peer-down cooldown and termination consensus extension. Fixed three
  pre-existing v0.3.0 distributed-mode termination bugs along the way. See
  [RELEASE_1.0.0_LOG.md](RELEASE_1.0.0_LOG.md) `### T6` and `### T6.1` for the
  honest cluster-vs-independent benchmark.
- **Partial-order reduction via stubborn sets** (T7, opt-in `--por`) — 36.8x
  state reduction, 17.9x wall-time speedup on PorBenchProcessGrid
- **State compression in queue** (T8, default-on, opt-out
  `--queue-compression false`) — zstd ring between hot deques and disk; 13.2x
  ratio, -68% peak RSS at 1M items, +2% wall time
- **Liveness checking 550x faster** (T10) — iterative Tarjan +
  fingerprint-keyed graph + fast SCC fairness check; 63.39 s → 115.28 ms on
  LivenessBench

### Polish & robustness
- **Trace minimization on violation** (T9, default-on `--minimize-trace`) —
  earliest-violation truncation + BFS shortcut to fixed point + variable
  highlighting
- **1-hour chaos soak with swarm-mode failpoints** (T11 + T16b) — every
  failpoint in `src/chaos.rs` fired ≥23 times; 0 divergences, 0 hangs

### Verification
- **Verus tier-B + tier-A proofs of the seqlock resize protocol** (T13 +
  T13.1-T13.3). 19 lemmas at the abstract `Set<u64>` layer (tier B,
  `seqlock_resize.rs`); 31 lemmas at the concrete `Seq<u64>` linear-probe
  layer (tier A, `seqlock_resize_tier_a.rs`). Together they
  machine-check that no inserted fingerprint is ever lost during a resize,
  the spec-level CAS soundness theorem, and bounded reader-retry termination.
  See [verification/verus/README.md](verification/verus/README.md).

## Deferred to v1.1.0

Eight quality follow-ups are parked as the v1.1.0 roadmap (detail in
[RELEASE_1.0.0_PLAN.md](RELEASE_1.0.0_PLAN.md), each entry begins with
`**DEFER TO 1.1.0.**`):

- **T5.4, T5.5** — Streaming Init enumeration / joint Init+Solution symbolic
  encoding (Einstein-class workloads).
- **T10.2** — Streaming SCC discovery for 100M+ liveness.
- **T11.3, T11.4** — CI-gate chaos variant; `route_spill_batch` inflight
  accounting on disk-overflow push errors.
- **T13.4, T13.5, T13.6** — Verus production-code annotations,
  unbounded-fairness reader liveness, CI gate.

## Try it

```bash
git clone https://github.com/zoratu/tlaplusplus.git
cd tlaplusplus
cargo build --release

# Model-check a TLA+ spec
./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla --config /path/to/Spec.cfg

# Compare with TLC on the curated diff list
scripts/diff_tlc.sh
```

## Links

- [Full changelog](CHANGELOG.md)
- [Architecture & developer notes](CLAUDE.md)
- [Release plan & v1.1.0 backlog](RELEASE_1.0.0_PLAN.md)
- [Detailed work log](RELEASE_1.0.0_LOG.md)
- [Verus proof](verification/verus/seqlock_resize.rs)

## License

GNU GPLv3.
