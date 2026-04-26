# Changelog

## v1.0.0 (2026-04-26)

First stable release. The 1.0 cycle focused on correctness foundations,
high-leverage performance wins, polish, and a Verus-checked proof of the
fingerprint store's resize protocol.

### Correctness foundations

- **Differential testing vs TLC as a CI gate.** `scripts/diff_tlc.sh` plus
  `.github/workflows/diff-tlc.yml` runs 13 curated specs under both TLC and
  tlaplusplus on every push. State counts agree exactly; the harness uncovered
  five real divergences (T1.1, T1.3, T1.4, T1.5, T2.4) that were all fixed
  before the release. Two divergences (T1.6, T11.1) are documented and deferred
  to v1.1.0 with workarounds.
- **Compiled-vs-interpreted proptest equivalence.**
  `tests/compiled_vs_interpreted.rs` generates well-typed
  Int/Bool/Set/Seq/Record/Str expressions and asserts the text evaluator and
  the compiled-IR evaluator agree on every case. Wired into CI at
  `PROPTEST_CASES=128`; runs at 2048 across 9 seeds locally.
- **State-graph snapshot tests.** `tests/state_graph_snapshots.rs` pins the
  reachable set for 7 small TLA+ specs as 128-bit XxHash3 digests, validated
  against TLC v2.19.
- **Mutation-testing audit.** `cargo-mutants` was run against `src/tla/eval.rs`
  and `src/tla/action_exec.rs`; 17 inline kill-tests added to cover the
  consequential survivor categories.

### Soundness fixes (caught by the new gates)

- **T1.1.** Compiled `\E x \in S : Action(x)` inside a wrapper definition now
  routes through the interpreted action evaluator, so existential branches no
  longer silently produce zero successors. Surfaced under `QueueSegmentSync`
  (TLC: 1531 distinct, pre-fix: 5).
- **T1.3.** Wrapper-Next fairness constraint accepts any in-SCC transition
  when the constraint's action name matches the spec's Next definition,
  eliminating the false-positive SCC fairness violation seen on `WorkQueue`.
- **T1.4.** Compiled expression evaluator no longer mis-splits LET bodies on
  indented `\/`, fixing the `\A x : \E m \in {} : ...` always-TRUE shape that
  affected Paxos-style specs.
- **T1.5.** Next-splitter no longer slices `/\ guard /\ \/ A \/ B /\ shared`
  on the inner `\/`, so shared post-conditions are no longer dropped and
  spurious successors are no longer fabricated. Closes the VIEW projection
  mismatch on `ViewTest`.
- **T2.1, T2.2.** Compiled-expr parser now scans for top-level `EXCEPT` only
  at bracket depth 0, so nested record-EXCEPT and EXCEPT-inside-record-literal
  shapes compile correctly.
- **T2.3.** `..` precedence vs set ops is fixed in both interpreter and
  compiler — `n..m \subseteq S` and `S \union n..m` now type-check and
  evaluate correctly in both code paths.
- **T2.4.** Unary minus inside binary subtraction (`<<(-1 - r.a), 0>>` and
  similar) no longer compiles to `Unparsed`; a new `find_binary_minus_split`
  helper walks past unary-minus positions before splitting.

### Performance — high-leverage wins

- **T5. Symbolic Init enumeration via Z3** (opt-in, `--features symbolic-init`).
  Filtered record-set Init shapes (e.g. `{c \in [f1: 0..N, f2: 0..N] :
  pred(c)}`) are translated to SMT and enumerated symbolically. Benchmark on
  `TightCan`: N=15 brute-force 14.71 s → symbolic 1.40 s (10.5x); N=20
  58.42 s → 1.41 s (41x); N=40 brute-force exhausts the eval budget,
  symbolic finishes in 1.6 s.
- **T6. Cross-node distributed work stealing.** TCP-based steal protocol with
  steal-victim threshold, peer-down cooldown, and termination consensus
  extension. Three pre-existing v0.3.0 distributed-mode termination bugs
  surfaced and were fixed (so any multi-node cluster can now converge at all).
  Cluster mode remains opt-in (`--cluster-listen`); see T6.1 entry in
  `RELEASE_1.0.0_LOG.md` for the honest cluster-vs-independent benchmark.
- **T7. Partial-order reduction (POR) via stubborn sets** (opt-in, `--por`).
  Static read/write dependency analysis at module load; per-state stubborn-set
  computation under enabled-disjunct restriction. Safety-only — automatically
  rejected when fairness/liveness constraints are present. Benchmark on
  `PorBenchProcessGrid` (4 independent processes, MAX=4): full=625 states,
  POR=17 — **36.8x state reduction, 17.9x wall-time speedup**.
- **T8. State compression in the spillover queue** (default on, opt-out
  `--queue-compression false`). zstd-compressed in-memory ring sits between
  the hot work-stealing deques and the disk-backed overflow. Triggers only
  when the spill path is already engaged. Benchmark: 13.2x compression ratio,
  -68% peak RSS at 1M items, +2% wall time.
- **T10. Liveness checking scaling.** Iterative Tarjan + fingerprint-keyed
  graph + fast `check_fairness_on_scc_fp` cut SCC-based fairness
  post-processing from O(scc_size × tx × constraints) to O(tx × constraints).
  Benchmark on `LivenessBench` (32k states, 143k transitions, one giant SCC,
  6 WF constraints): liveness phase wall-time **63.39 s → 115.28 ms (~550x)**,
  total wall-time 64.92 s → 1.60 s, peak RSS 2.59 GB → 1.99 GB (-23%).

### Polish

- **T9. Trace minimization on violation** (default on, `--minimize-trace`).
  Two-phase: (A) earliest-violation truncation + BFS shortcut search to fixed
  point; (B) syntactic variable-relevance scan on the invariant body, used to
  mark unrelated state variables as "(noise)" in the printed trace. 30 s
  default budget. Diamond fixture: 9 → 6 states in 375 µs.
- **T11. 1-hour chaos soak harness.** `scripts/chaos_soak.sh` plus a
  `FailScenario::setup()` wiring under `cfg(feature = "failpoints")`. 1-hour
  c8g.xlarge soak against `CheckpointDrain`: **387 iterations, 0 divergences,
  0 hangs**. Every failpoint in `src/chaos.rs` (12 names) fired ≥23 times;
  the runtime tolerates persistent permanent failures in every checkpoint
  sub-step, FP-store-pressure path, and queue spill/load path.
- **T16. Regehr-style swarm testing** at two layers:
  - **T16a.** The T2 proptest harness picks a random subset of 17 shape
    categories per case (mean ~8.5 enabled), so each case explores a
    *minimal-feature-interaction* slice. The original uniform proptest is
    kept as a regression gate.
  - **T16b.** The chaos soak supports `--swarm-mode N|auto` (default 1; auto
    picks 1-4 concurrent failpoints per iter). 30-minute soak: 204 iters,
    0 divergences, 66/66 distinct concurrent failpoint pairs observed
    (exhaustive pair coverage).
- **T12. Cross-arch CI matrix.** diff-TLC workflow now runs as a
  `[ubuntu-latest, ubuntu-24.04-arm]` matrix (both archs run lib + bin tests,
  the diff harness, and the T2 proptest at `PROPTEST_CASES=128`).

### Verification

- **T13. Verus on the fingerprint store** — tier B, protocol-level proof.
  `verification/verus/seqlock_resize.rs` (600 lines, 19 verified lemmas)
  proves the headline soundness theorem `theorem_no_fingerprint_lost`: in
  any well-formed execution of the protocol, every inserted fingerprint
  remains observable from then on. Proof is at the protocol abstraction
  layer — table = `Set<u64>`, atomic step semantics; it does NOT verify the
  production code's pointer arithmetic, memory orderings, or linear-probe
  collision behavior. See `verification/verus/README.md` for the assumptions
  and the tier-A roadmap (T13.1–T13.3, deferred to v1.1.0).

### Distributed

- **T6.1. Cross-node re-benchmark on a real corpus spec.** Confirmed the
  v0.3.0 design's tradeoff: cluster mode (each node maintains its own FP
  store, redundant exploration but no global FP synchronization) is slower
  than independent-explorer mode on canonical workloads. Cluster mode stays
  opt-in (`--cluster-listen`). Global FP partitioning is multi-quarter work
  out of scope for v1.0.0; tracked in the v1.1.0 backlog.

### Test suite

- **727 default tests** (release, no extra features), 0 failures,
  3 ignored (one disk-checkpoint test pending serializable queue, two
  doc-tests for chaos/S3).
- **746 tests with `--features failpoints`**, 0 failures, 3 ignored.
- **Symbolic-init feature** adds 8 tests when `--features symbolic-init` is
  enabled (Z3-backed equivalence + brute-force-on-small).
- 13/13 specs pass `scripts/diff_tlc.sh` (state counts agree exactly with
  TLC v2.19 on every spec).
- 12 active state-graph snapshot tests, all match.
- T2 proptest equivalence harness clean across 9 seeds at
  `PROPTEST_CASES=2048`.
- Single ignored test: disk-checkpoint round-trip (needs serializable queue
  metadata, deferred to v1.1.0).

### Deferred to v1.1.0

The following items were classified during the T17 closeout sweep. Detail in
`RELEASE_1.0.0_PLAN.md` (each entry begins with `**DEFER TO 1.1.0.**`).

- **T1.6** — `FingerprintStoreResize` invariant evaluator returns `Bool(false)`
  instead of computing. Pre-existing, single-spec, low-impact.
- **T11.1** — `--queue-max-inmem-items` below natural state count causes the
  spill path to drop states. Documented workaround: keep cap above natural
  state count (default 50M is safe for ~all users).
- **T5.1, T5.2, T5.3** — symbolic Init enhancements (sequence-set Init,
  permutation symmetry, projection-based all-SAT).
- **T7.1, T7.2, T7.3** — POR enhancements (batched per-disjunct evaluation,
  smarter stubborn-set seed, POR for liveness).
- **T9.1, T9.2, T9.3** — trace minimization enhancements (transitive variable
  relevance, smarter BFS seed, suffix shortening).
- **T10.1–T10.4** — liveness scaling enhancements (parallel Tarjan, streaming
  SCC discovery, single-state SCC pre-filter, per-action transition shard).
- **T11.2, T11.3** — chaos soak follow-ups (re-soak with queue cap once T11.1
  fixed, CI-gate variant).
- **T12.1** — CI surfaced a stack-overflow on the deliberate
  unbounded-recursion test on GitHub Actions' default 2 MB thread stack;
  spot instances pass cleanly.
- **T13.1, T13.2, T13.3** — Verus tier A roadmap (verify production Rust
  directly, liveness, CI gate).

## v0.3.0 (2026-03-25)

### TLA+ Language Compatibility

**External corpus: 0 errors, 90% pass at 60s, 94% at 15 min** (up from 63% at start of release cycle). Every spec that defines a state machine (Init + Next) runs correctly.

- **SPECIFICATION definition chasing**: Follow definition reference chains to extract Init/Next from temporal formulas like `Spec == LiveSpec` where `LiveSpec == Init /\ [][Next]_vars /\ WF(Next)`
- **Disjunctive Init branches**: Handle `\/ Guard1 /\ var = expr1 \/ Guard2 /\ var = expr2` in Init by evaluating branch guards with known constants
- **Existential quantifier Init**: Expand `\E x \in S : body` in Init bodies by iterating domain values and extracting variable assignments
- **Init!N sub-expression references**: Support TLC-style `Init!1`, `Init!2` to reference specific conjuncts of a definition
- **Disjunctive Init bodies**: Handle Init definitions that are top-level disjunctions (`\/ branch1 \/ branch2`)
- **Parameterized operator calls in Init**: Inline-expand `XInit(x)` where `XInit(v) == v = 0` to extract variable assignments
- **Late-binding equality/membership**: After cross-product expansion, re-classify guards as equality assignments when dependencies are now satisfied
- **Deferred membership evaluation**: Membership sets that depend on other membership variables are deferred to cross-product phase
- **Outer parenthesis stripping**: `(h_turn = 1)` in Init correctly classified as equality assignment
- **INSTANCE variable shadowing**: Skip instance module variables that shadow definitions (e.g., Stuttering's `vars` vs Lock's `vars == <<pc, lock>>`)
- **Definition override preservation**: Save original definitions before `Init <- MCInit` override for missing variable recovery
- **Evaluation-only modules**: Specs without Init/Next/SPECIFICATION complete instantly with 0 states instead of erroring
- **Local .tla file loading for built-in modules**: When a spec ships its own `Functions.tla` alongside a built-in module, load definitions from the local file as fallback

### Community Modules

16 new built-in operators across 6 modules:

- **Bags**: EmptyBag, SetToBag, BagToSet, IsABag, BagIn, BagOfAll, BagUnion, CopiesIn
- **BagsExt**: BagAdd, BagRemove
- **IOUtils**: IOEnv, ndJsonDeserialize, JsonDeserialize, JsonSerialize, ToString
- **Bitwise**: BitsAnd, BitsOr, BitsXor, BitNot, LeftShift, RightShift, IsABitVector, IsANatural
- **Combinatorics**: Factorial, nCk, nPk
- **CSV**: CSVRead, CSVWrite
- **VectorClocks**: VCLess, VCLessOrEqual, VCMerge
- **Randomization**: RandomSubset

### Performance Optimizations

- **Trivial Next detection**: When `Next == UNCHANGED vars`, skip BFS exploration entirely — just enumerate Init states and check invariants
- **Constraint propagation**: For `{c \in [f1: 0..N, f2: 0..N] : c.f1 + c.f2 \in lo..hi}`, compute valid ranges directly instead of iterating all N² pairs (~3000x speedup for CoffeeCan)
- **Compiled predicate evaluation**: Use `compile_expr` + `eval_compiled` for record set filtering instead of re-parsing expression text per record
- **Range membership fast path**: `x \in a..b` evaluates as `a <= x && x <= b` instead of constructing the full integer set
- **Inline record set generation**: Generate records inline with predicate filtering instead of materializing the full record set
- **Vec-based constraint output**: Return constraint-propagated record sets as Seq (O(n)) instead of BTreeSet (O(n log n))
- **Lazy Init enumeration**: For cross-products exceeding 10M states, use odometer-style enumeration instead of materialization
- **Early exit for 0 initial states**: Evaluation-only modules complete instantly without spawning worker threads
- **FunAsSeq fix**: Corrected key indexing from `n..n+m-1` to `1..m` matching TLC semantics

### Distributed Model Checking

- **`--fetch-module` / `--fetch-config`**: Fetch spec files from S3 URIs for distributed runs where nodes don't share a filesystem
- **FLURM integration**: Plugin support for job scheduling on spot instances with automatic S3 file distribution
- **S3 checkpoint resume validated**: Round-trip checkpoint → clear → resume verified on spot instances

### Bug Fixes

- Fixed `FunAsSeq(f, n, m)` key indexing (was `n..n+m-1`, now `1..m`)
- Fixed empty expression handling in `Init!N` references with comment-stripped lines
- Resolve `IOEnv` and `EmptyBag` as zero-arg built-in operators (bare identifier usage)
- TLCGet ASSUME failures downgraded to warnings (TLC-specific runtime introspection)
- MAX_INIT_STATES raised from 1M to 10M
- Record set size limit raised to 10M

### Scaling Benchmark

Corpus run on c6gd.2xlarge (8 vCPU) instances, 60s timeout per spec:

| Nodes | vCPUs | Wall Time | Pass Rate | Speedup |
|-------|-------|-----------|-----------|---------|
| 1 | 8 | 29 min | 162/182 (89%) | 1.0x |
| 2 | 16 | 18 min | 161/182 (88%) | 1.6x |
| 4 | 32 | 8.6 min | 166/182 (91%)* | 3.4x |
| 8 | 64 | 7.5 min | 170/182 (93%)* | 3.9x |

*Adjusted for connectivity issues on some nodes

With longer timeouts on larger machines (192 cores):
- 300s: 169/182 (93%)
- 900s: 171/182 (94%)

### Test Suite

- 600 unit tests, all passing
- 620 failpoint tests (with `--features failpoints`), all passing
- 25/32 internal corpus specs passing
- 163/182 external corpus specs passing at 60s (0 errors, 19 timeouts)
- 161/182 analysis probes: FULL_PASS (88%), 0 FAIL

## v0.2.0

Initial release with parallel runtime, NUMA-aware work-stealing, lock-free fingerprint store, and native TLA+ frontend.
