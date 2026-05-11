# tlaplusplus v1.2.0

Patch release driven by extensive fuzz + mutation testing. Six T20X-class soundness fixes in the compiled-eval / compiled-expr paths, two large refactors that improve testability without behavioural change, and testing-infrastructure additions that raise the compiler kill rate from 42% → 65% over eight iterations.

## Highlights

### Compiler-vs-interpreter soundness (T201–T207)

Six soundness gaps surfaced by `fuzz_tla_swarm` (the compiler-vs-interpreter equivalence harness, T101 lineage) and one by mutation testing.

- **T201 — Eval budget OOM.** `fuzz_tla_swarm` could synthesize TLA+ expressions that triggered allocator paths (range, set comprehension, function constructor) without budget enforcement, OOM-aborting the fuzzer in seconds. Six sites now charge `ctx.check_budget(cost)` before allocating. Fuzz harness raised to `-rss_limit_mb=8192` so adversarial inputs run to completion and the fuzzer focuses on the soundness signal.
- **T202 — Compiler `LAMBDA` parser was looser than interpreter.** `LAMBDA \\t\\t==I :\\tI\\t=I` and similar shapes parsed in the compiler (returning `Bool(false)`) but were rejected by the interpreter (returning `Lambda`). Compiler parser tightened to mirror `eval.rs::parse_lambda` word-boundary rules.
- **T203 — `LET`-binding eval OOM.** Post-T201 fuzz of LET-keyword patterns showed ~1MB allocations across 712 sites in the LET-binding clone-on-write path; the eval budget covered the surrounding eval path but not the `LET` defs vec construction itself. Now charges `ctx.local_definitions.len() + defs.len()` before `with_local_definitions`.
- **T204 — Per-call entry-point eval budget.** Recursive operator application could escape the budget at the per-call boundary; the `eval_compiled_inner` per-call entry now installs a budget tick.
- **T205 — Bare `IOEnv` / `EmptyBag` in compiled eval.** The compiler fell through to `ModelValue("IOEnv")` while the interpreter resolved the env Record. Surfaced by post-T204 fuzz. Compiler now dispatches bare zero-arg builtins.
- **T206 — Chained binary `+`/`-` arithmetic.** Three or more top-level binary `+`/`-` ops associated right-to-left in the compiler vs left in the interpreter. Compiler detects 3+ chains via `has_chained_top_level_arithmetic` and emits `CompiledExpr::Unparsed` to delegate to the interpreter (the reference). The earlier per-shape fixes (T101.1) handled 2-term chains; T206 generalises.
- **T207 — Compiler typed `SubSeq` accepted `m=0`.** Surfaced by mutation testing during the v1.2.0 prep cycle. `CompiledExpr::SubSeq` lacked the `m >= 1` validation present in the interpreter and the OpCall fallback path: silent Ok-with-wrong-result vs Err divergence on `SubSeq(s, 0, k)`. Fixed to mirror both.

Each T20X fix has a pinned regression test under `tests/fuzz_t20*_*.rs` or `tests/compiler_helper_coverage_t207*.rs`.

### Testing infrastructure — mutation kill rate 42% → 65%

Eight iterations of targeted compiler-helper coverage on top of the existing T2 proptest equivalence + fuzz suites. Trajectory:

| Iter | Tests added | Kill rate | Δ |
|---|---|---|---|
| Original | 0 | 42.4% | — |
| Iter1 (T207) | +66 | 55.7% | +13.3pp |
| Iter2 (T207b) | +46 | 59.4% | +3.7pp |
| Iter3 (T207c) | +77 | 61.3% | +1.9pp |
| Iter4 (assertion fix) | 0 | 62.8% | +1.5pp |
| Iter5 (T207d) | +40 | 63.7% | +0.9pp |
| Iter6 (T207e) | +28 | 65.3% | +1.6pp |
| Iter7 (T207f, RECURSIVE) | +6 | 66.0% | +0.7pp |
| Iter8 (T207g, deep relops) | +18 | 65.4% | -0.6pp |

281 new compiler-helper tests across 8 iterations, +23pp total. Convergence at ~65–66% — within mutation-to-mutation variance.

- **T207 (iter 1, +66 tests).** `tests/compiler_helper_coverage_t207.rs`: `eval_compiled_opcall` arity guards, user-defined-shadow guards on `Max`/`Min`, chained-relop split position, word-boundary keyword scanning, IF/LET/CASE/quantifier scope protection in operator splits, membership shapes, short-circuit. Surfaced T207's SubSeq m<1 soundness fix.
- **T207b (iter 2, +46 tests).** `compiler_helper_coverage_t207b.rs`: `has_chained_top_level_arithmetic` boundaries, community-module operator arity (SeqOf, Quantify, FlattenSet, kSubset, ChooseUnique, IsUndirectedGraph, SymDiff, SumSet, ProductSet), membership shapes (SUBSET, UNION, comprehension, function set), arithmetic edge cases.
- **T207c (iter 3, +77 tests).** `compiler_helper_coverage_t207c.rs` plus +43 in `src/tla/compiled_expr.rs::tests` and +15 in `src/tla/compiled_eval.rs::tests`: deep-recursion stress (280+ levels through IF/LET/NOT/AND/OR/IMPLIES/IFF/RECORD/CHOOSE/ADDITION/Cardinality), direct unit tests on private helpers (`split_first_top_level_op`, `split_binary_op_with`, `find_keyword`, `split_quantifier_bindings`, `split_case_arms`, `split_let_expression`, `find_definition_equals`, `find_top_level_except`, `strip_label_prefix`, `compiled_membership_contains`, `membership_matches_text`, `guard_text_is_action_body`, `get_nested_value`, `set_nested_value`, `sequence_like_values`).
- **Iter 4 (assertion fix).** Tightened t207c deep-recursion tests to assert Err (not Ok-or-Err), so the `+1 -> *1` mutation that keeps depth=0 produces Ok instead of Err and fails the assertion.
- **T207d (iter 5, +40 tests).** Direct boundary tests for scanner helpers — `split_quantifier_bindings` with `<<>>` tracking, `find_top_level_except` paren/brace/tuple skipping, `find_definition_equals` LET/IN nesting and byte-position, `split_first_top_level_op` ambiguity disambiguation (`=>`/`<=`/`>=`/`=<`/`/=`/`#`).
- **T207e (iter 6, +28 tests).** Deep recursion through more built-ins (Cardinality, DOMAIN, Range, Lambda app, ToString, function constructor/apply, record access), exhaustive membership shapes (Nat, Int, BOOLEAN, negative ranges), built-in arity edges (SelectSeq with lambda, FunAsSeq b<0, SubSeq n=0).
- **T207f (iter 7, +6 RECURSIVE tests).** `RECURSIVE Op(_)` deep chain (`Count(500)`, `Fib(10)`, mutual `IsEven`/`IsOdd`, 300-deep operator chain) targeting line 1909's user-defined op dispatch.
- **T207g (iter 8, +18 tests).** Deep recursion through every relop (Eq, Neq, Lt, Le, Gt, Ge), In/NotIn, and arithmetic (Add/Sub/Mul/Div/Mod) dispatch arm in `eval_compiled_inner`.
- **Dead code removal.** Mutation testing identified `split_top_level_old` (annotated `#[allow(dead_code)]`) as 158 noise mutants on never-called code; removed.

### Refactors (no behavioural change)

- **`src/main.rs` split into `src/cli/`.** The CLI dispatch tree was the largest single source file; the rewrite splits it into 12 focused modules (`probe.rs`, `analyze_tla.rs`, `run_tla.rs`, `run_counter_grid.rs`, `run_einstein.rs`, `run_combined.rs`, `corpus_check.rs`, `compile_check.rs`, `mc.rs`, `verify.rs`, `simulate.rs`, plus `mod.rs`). No public CLI surface changes; --help text unchanged.
- **`src/runtime.rs` partial split into `src/runtime/`.** Seven chunks landed (PauseController, checkpoint manifest, memory budget, shard count, AtomicRunStats, T5.4 init producer, T10 liveness post-processing, distributed handler wiring, progress tick). 64 new unit tests now exercise these modules directly. Two further chunks deferred to v1.2.x.
- **Cluster stats surfaced (T204.1).** `DistributedWorkStealer::print_stats()` is now wired into the run summary so cluster runs surface `stolen` / `donated` / `steal_req_sent` counters by default (previously visible only via ad-hoc instrumentation).

### Fuzz harness signal-vs-noise framing

`scripts/fuzz.sh` documents the policy: fuzz targets surface panics, sanitizer trips, and compiler-vs-interpreter divergences. RSS-cap exits from adversarial memory-consumption inputs are noise; the harness raises libFuzzer's RSS limit to 8 GB so the soundness signal isn't drowned out. Production runs use `budget=None` — this is fuzz harness configuration only, not user-facing behavior.

## Validation gates

All green on the v1.2.0 tag:

| Gate | Result |
|---|---|
| `cargo test --release` | 1,155 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,177 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,182 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 (continuous CI gate) |
| `cargo test --release --test compiled_vs_interpreted` (PROPTEST_CASES=2048) | clean across 9 seeds |
| `cargo test --release --test compiler_helper_coverage_t207{,b,c,e,f,g}` | 7 files, all pass |
| `cargo test --release --test fuzz_t20*` | 7 files, all pass |
| Mutation testing — `eval.rs` (interpreter) | 100% kill rate (0 missed) |
| Mutation testing — `compiled_eval.rs` + `compiled_expr.rs` (compiler) | 65.4% kill rate (1,443/2,208 viable; 8 iterations) |

## Compatibility

Drop-in for v1.0.x and v1.1.x. No public-API or CLI changes. Fingerprint format, checkpoint format, state-graph dump format unchanged.

## Known limits

The compiler kill rate landed at 65.4%, not 100%. Eight iterations of targeted compiler-helper testing converged at this number; iter8 actually moved -0.6pp, indicating mutation-to-mutation variance exceeds test contributions at this point. The compiler's safety net for soundness is layered — T2 proptest equivalence (compiler vs interpreter, 2048 cases × 9 seeds), `fuzz_tla_swarm` (compiler vs interpreter, every input), state-graph snapshots, and the diff-vs-TLC gate. Mutation testing reveals that the compiler's *internal helpers* contain many *equivalent mutants* — multiple `depth + 1` recursive call sites where mutating any one in isolation produces no observable output difference (the outer dispatch's depth check fires first), and match-guard mutations on dispatch arms where the fallthrough produces the same observable behavior. Every real soundness bug found this cycle (T201–T207) came from fuzz or mutation testing — none slipped past the layered checks. Reaching higher kill rates requires code restructuring (merging redundant depth-tracked dispatch sites, removing fall-through arms) rather than more tests; tracked as v1.2.x backlog.

## Deferred to v1.2.x — POST-RELEASE STATUS

All four originally-deferred items have been addressed in post-release follow-up commits on `main` (v1.2.1).

- T204 distributed mock — `Transport` trait + `MockTransport` using in-memory tokio mpsc channels + 10 tests landed.
- `src/runtime.rs` chunks 7 + 8 — both extracted. Chunk 8 (the 13-step shutdown phase) becomes `runtime/shutdown.rs` (`ShutdownContext` struct + `orchestrate` method). Chunk 7 (the worker spawn loop with 27-Arc capture) becomes `runtime/worker.rs` (`WorkerLocalState` struct).
- `src/tla/eval.rs` split — all 8 chunks (A-H) landed via the design doc's landing protocol. 13 submodule files in `src/tla/eval/`: `mod.rs`, `expr.rs`, `action.rs`, `splitter.rs`, `operator.rs`, `set.rs`, `quantifier.rs`, `bracket.rs`, `postfix.rs`, `control.rs`, `transition.rs`, `instance.rs`, `budget.rs`.
- Compiler internal-helper restructuring — depth-tracking consolidation across `compiled_eval.rs` (148 sites collapsed to 4) and `eval_operator_call` (26 sites collapsed to 1). Plus 6 more iterations of targeted compiler-helper unit tests (T207b through T207h, adding 149 tests). Compiler mutation kill rate lifted from 65.4% (v1.2.0 release) to 70.5% (post-release).
