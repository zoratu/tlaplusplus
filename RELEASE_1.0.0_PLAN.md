# Release 1.0.0 — Task Plan

Working through these in order. Each task is delegated to an agent. Builds/tests run on EC2 spot instances (small instances unless scale testing). Log entries in `RELEASE_1.0.0_LOG.md`.

## Phase 1 — Test infrastructure (correctness foundations)

These come first because every later change needs a regression gate.

- [ ] **T1. Differential testing against TLC as CI gate.** Pick a curated subset of corpus specs (small, fast), run both tlaplusplus and TLC, diff state counts and violation reports. Make it a CI step that fails the build on divergence.
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
