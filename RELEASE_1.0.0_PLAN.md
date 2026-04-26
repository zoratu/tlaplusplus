# Release 1.0.0 — Task Plan

Working through these in order. Each task is delegated to an agent. Builds/tests run on EC2 spot instances (small instances unless scale testing). Log entries in `RELEASE_1.0.0_LOG.md`.

## Phase 1 — Test infrastructure (correctness foundations)

These come first because every later change needs a regression gate.

- [x] **T1. Differential testing against TLC as CI gate.** Done — `scripts/diff_tlc.sh` + `.github/workflows/diff-tlc.yml`. 9 curated specs, 100% parity on active list. Surfaced 3 real divergences (T1.1–T1.3 below).

### Bugs uncovered by T1 (block 1.0.0; fix before T2/T3)

- [x] **T1.1. SOUNDNESS: compiled `\E x \in S : ActionCall(x)` inside a wrapper definition silently produced zero successors.** Fixed: compiled-IR `Guard` arm now detects action-call shape at runtime and dispatches to the interpreted action evaluator. QueueSegmentSync_Buggy now reports the correct violation; QueueSegmentSync_Fixed now reports 1531 distinct states (matches TLC). Both back in active diff list. Regression test in `src/tla/action_exec.rs::tests::exists_wrapping_action_call_in_compiled_ir_enumerates_all_bindings`. See `RELEASE_1.0.0_LOG.md` for root cause / commit.
- [x] **T1.2. VIEW projection ignored in fingerprinting** — investigated, NOT a VIEW bug. `evaluate_view` is wired correctly; the inflated state count comes from the Next-splitter (T1.5). VIEW will start collapsing automatically once T1.5 lands. See log entry for evidence.
- [x] **T1.5. SOUNDNESS: `split_action_body_disjuncts` sliced `/\ guard /\ \/ A \/ B /\ shared` on the inner `\/`.** Fixed: (a) `split_action_body_disjuncts` returns the body unsplit when `split_action_body_clauses` reports more than one top-level conjunct; (b) `normalize_branch_expr` no longer strips a leading `/\` when subsequent `/\` conjuncts remain (otherwise the first conjunct loses its delimiter and `split_top_level("/\\")` silently corrupts); (c) the compiled-IR `Guard` arm dispatches a guard text that is itself an action body (top-level `\/` containing primes or `UNCHANGED`) to the interpreted action evaluator. ViewTest now reports 106 distinct (matches TLC) and is promoted to the active diff list (12/12 pass). Regression tests: `src/tla/action_ir.rs::split_action_body_disjuncts_does_not_split_nested_disjunction_inside_outer_conjunction`, `src/tla/action_exec.rs::evaluates_conjunctive_next_with_inner_disjunction_and_shared_post`, `tests/next_splitter_t1_5.rs`. Closes T1.2.
- [ ] **T1.3. Fairness SCC false positive.** `corpus/internals/WorkQueue.tla`: state counts agree (15003) but tlaplusplus reports an SCC fairness violation TLC accepts. `src/fairness.rs` SCC handling when SCC has only the implicit Next action.
- [ ] **T1.4. Compiled `LET ... IN /\ \A a \in Q : \E m \in {} : ... /\ ...` incorrectly evaluates to TRUE for non-empty Q.** Surfaced by T1.1 fix when the compiled `Guard` previously masked it via a downstream Send-call drop. Reproduces in the Paxos-style probe test (`src/tla/action_exec.rs::tests::parsed_paxos_style_probe_keeps_let_locals_in_scope_with_operator_overrides` — currently asserts ≥3 with a TODO comment). Likely a quantifier-scope or `\A`-over-empty-`\E` bug in `src/tla/compiled_eval.rs::CompiledExpr::Forall` when the body itself contains an `\E` over a LET-bound set.

- [ ] **T2. Compiled-vs-interpreted proptest equivalence.** proptest generator for TLA+ expressions. For each: evaluate via `eval_expr` and `compile_expr` + `eval_compiled`, assert equality. Catches silent drift in the compiled fast path.
- [ ] **T3. Snapshot tests for state graphs.** Small specs whose full reachable fingerprint set we pin. Catches off-by-one errors in successor generation that pure count-checks miss.
- [ ] **T4. Mutation testing audit.** Run `cargo-mutants` on `src/tla/eval.rs` and `src/tla/action_exec.rs`. Identify surviving mutants → file gaps → write tests to kill them.

## Phase 2 — High-leverage performance

- [ ] **T5. Symbolic Init enumeration.** Z3 or BDD-backed enumeration for filtered record sets that currently brute-force. Targets: Einstein, CoffeeCan-large, MCBinarySearch. Solves the entire Init-bound class.
- [ ] **T6. Cross-node distributed work stealing.** Current FLURM mode is independent exploration with no cross-node steal. Add a TCP/gRPC steal protocol so an idle node pulls work from a busy one. Should close the 8-node gap toward linear speedup.
- [ ] **T7. Partial-order reduction (POR).** Stubborn sets or ample sets. For specs with independent actions, can cut state space 10–100x. Big win for distributed-protocol corpus entries.

## Phase 3 — Polish

- [ ] **T8. State compression in queue.** zstd-compress cold queue segments (we already depend on zstd) or structural sharing for record-heavy states. Reduces memory footprint, enables larger models.
- [ ] **T9. Trace minimization on violation.** When an invariant violation is found, shrink the counter-example (proptest-style) before reporting. Quality-of-life for users.
- [ ] **T10. Liveness checking scaling.** Iterative tableau or on-the-fly fairness for large state graphs. Currently correct but slow at scale.
- [ ] **T11. Long-run soak / chaos.** Run failpoint chaos for hours under random fault injection. Catches accumulation bugs that single-fault tests miss.
- [ ] **T12. Cross-arch CI matrix.** Corpus runs only on aarch64 today. Add x86_64 to CI to catch endianness/atomic-ordering issues.

## Phase 4 — Verification

- [ ] **T13. Verus on fingerprint store.** Pick the seqlock resize protocol — the most consequential lock-free invariant. Prove "no fingerprint lost across resize" and "every fingerprint inserted is observable to subsequent readers." Scope: one module, not the whole codebase.

## Phase 5 — Release

- [ ] **T14. Update CHANGELOG.md, CLAUDE.md, README.** Bump version to 1.0.0.
- [ ] **T15. Tag, push, and prepare gh release.** User triggers the actual `gh release create`.

## Working rules

- All builds and `cargo test` runs happen on EC2 spot instances (`REDACTED i` for small instances). Local machine is read-only for code.
- Each agent updates `RELEASE_1.0.0_LOG.md` when it finishes (or hits a blocker).
- Agents are spawned serially within a phase to share spot infrastructure.
- Commit work-in-progress per task so an interruption doesn't lose state.
