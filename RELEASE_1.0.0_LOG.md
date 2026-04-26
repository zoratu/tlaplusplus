# Release 1.0.0 — Work Log

Chronological log of work toward 1.0.0. Each entry: date, task ID, agent, outcome, commit (if any), follow-ups.

---

## 2026-04-25

### Setup
- Wrote `RELEASE_1.0.0_PLAN.md` with 15 tasks across 5 phases.
- Working rule confirmed: all builds run on EC2 spot instances; local machine is code-edit-only.
- Starting with Phase 1 (test infrastructure) so later perf work has a regression gate.

### T1 — Differential testing against TLC as CI gate (Phase 1)

**Status:** harness landed, CI workflow wired, end-to-end validated. **4 real divergences uncovered** (parked for follow-up bugs, not patched).

**Artifacts:**
- `scripts/diff_tlc.sh` — runs each curated spec under both checkers, parses
  `<N> states generated, <M> distinct states found` lines plus violation/deadlock
  markers, exits non-zero on divergence. Treats expected violations correctly
  (skips state-count compare when either side stopped early). Has an allowlist
  escape hatch via `corpus/diff_test/allowlist.tsv`.
- `corpus/diff_test/list.tsv` — 9 curated specs, all <10s under TLC, covering:
  set ops, record sets, signed arithmetic, nested EXCEPT, RECURSIVE operators,
  EXTENDS chains, expected-violation case, plus a moderately large internal
  model (CheckpointDrain, ~26K distinct states, ~3s) so we exercise the parallel
  runtime, not just trivial state spaces.
- `corpus/diff_test/allowlist.tsv` — empty placeholder with usage comment.
- `.github/workflows/diff-tlc.yml` — runs on push and PR to main, builds debug
  binary, caches cargo, uploads `.diff-tlc-out/` logs on failure. Uses
  `actions/setup-java@v4` (temurin 17) and `dtolnay/rust-toolchain@stable`.

**Commit:** `c91de88` — `test: differential testing harness vs TLC as CI gate (T1)`

**End-to-end validation:**
- Provisioned `c8g.xlarge` spot instance via `REDACTED l 72` (4 vCPU, 8GB,
  us-west-2). Tailscale hostname `REDACTED-INSTANCE`. Installed Rust stable
  (1.95) + JDK 17. Synced repo, ran `cargo build --bin tlaplusplus` (~5 min),
  ran `scripts/diff_tlc.sh`. Result: **9/9 passing, exit 0**.
- An earlier 13-spec list also ran cleanly through the harness and surfaced the
  divergences below before they were trimmed.

**State-count parity (current curated list, all 9 specs):**
- TLC and tlaplusplus agree on distinct state counts for every passing spec
  (tested values range from 3 to 26,344). For the one expected-violation spec
  (SimpleCounterViolation), violation status agrees and we deliberately skip
  state-count compare since both checkers stop early.

**Real divergences uncovered (NOT patched per agent rules; documented in
`corpus/diff_test/list.tsv` and listed below for follow-up):**

1. `corpus/language/ViewTest.tla` — VIEW projection appears to not be applied to
   fingerprinting. tlaplusplus reports 121 distinct, TLC 106 (the spec uses
   `VIEW <<x, y>>` to elide a monotonically-increasing `timestamp` field; TLC
   correctly collapses states that differ only in `timestamp`, tlaplusplus does
   not). Look at `src/models/tla_native.rs::resolve_view` and how the runtime
   feeds the projection into the fingerprint store.

2. `corpus/internals/QueueSegmentSync.tla` (both `_Fixed.cfg` and `_Buggy.cfg`)
   — INIT/NEXT cfg style produces only 5 distinct states under tlaplusplus vs
   1531 under TLC. Each generated state has exactly 1 successor (debug log:
   `1 successors generated`), so the `Next == \/ A \/ B \/ ...` disjunction with
   `\E segId \in localSegments : ConsumeSegment(segId)` etc. is not enumerating
   the disjuncts properly. The Buggy cfg, which TLC reports a violation on, is
   silently passing under tlaplusplus — i.e. **soundness bug, not just a count
   mismatch**. Suspect `src/tla/action_exec.rs` or `src/tla/action_ir.rs`.
   Reproduces with both `INIT/NEXT` and `SPECIFICATION Spec` cfg forms.

3. `corpus/internals/WorkQueue.tla` — distinct state counts agree (15003), but
   tlaplusplus's fairness checker reports an SCC fairness violation that TLC
   accepts. Message: "Fairness violation in SCC 0 (1 states): Fairness
   constraint may be violated: action 'Next' does not occur in SCC". TLC says
   "No error has been found." Investigate `src/fairness.rs` SCC handling when
   the SCC has only the implicit Next action.

These 4 divergences are tracked inline in `corpus/diff_test/list.tsv` (commented
KNOWN DIVERGENT SPECS section). The CI gate is intentionally narrowed to
specs that already agree, so any *future* regression on the working surface
fails CI immediately. Re-add divergent specs to the active list when fixed.

**Caveats / follow-ups:**
- Fairness divergence (#3) suggests the broader corpus may have similar latent
  divergences; T2 (proptest equivalence) and T3 (snapshot tests) should help
  surface them. Worth a deliberate sweep of the full corpus comparing
  tlaplusplus against TLC once we tackle T2/T3.
- Spec coverage skew toward language features; we should add ViewTest /
  QueueSegmentSync back as soon as the underlying bugs are fixed, since they
  exercise VIEW and INIT/NEXT cfg respectively — both production paths.
- Per-spec timeout set to 60s by default, 120s in CI to absorb cold-cache
  variance on Actions runners.

### Decision: insert bug-fix tasks before T2

Plan amended to add T1.1 (soundness, QueueSegmentSync), T1.2 (VIEW), T1.3
(fairness SCC) as blockers for 1.0.0. Doing them before T2 because (a) T1.1 is
a soundness bug — silent false-pass — which blocks any release, and (b) T2's
proptest equivalence would re-surface the same eval/exec issues, so fixing
first avoids wasted parallel investigation. Spawning T1.1 next.

### T1.1 — SOUNDNESS fix: compiled `Exists`-over-action-call silently dropped successors

**Status:** fixed, validated against TLC, regression test added, both
QueueSegmentSync configs back in the active diff harness (11/11 specs pass).

**Symptom:** `corpus/internals/QueueSegmentSync.tla` with both `_Fixed.cfg`
and `_Buggy.cfg` produced only 5 distinct states under tlaplusplus vs 1531
under TLC. The Buggy cfg, which TLC reports an invariant violation on,
silently passed under tlaplusplus — the dangerous shape: false confidence in
correctness.

**Root cause:** in `src/tla/compiled_expr.rs::compile_action_clause` the
`ActionClause::Exists { binders, body }` variant was lowered to
`CompiledActionClause::Exists` with `body_clauses = compile_action_body_clauses(body)`.
When `body` is a bare action call like `ConsumeSegment(segId)`, it doesn't
parse as `\E`/IF/primed-anything, so `compile_action_clause_text` falls
through to `CompiledActionClause::Guard { expr: compile_expr(call) }`. The
runtime `Guard` handler then evaluated the call as a *boolean expression* —
it inlined the action body (which contains primed assignments) and tried to
treat `consumedSegments' = ...` as an equality predicate. The variable
`consumedSegments'` was undefined in scope, so the guard returned `false`
and the existential branch silently produced zero successors.

This pattern is widespread: any `Action == \E x \in S : OtherAction(x)`
wrapper definition (or any `Next` body that contains one) was affected.
Standalone `\E x \in S : Action(x)` written directly inside a top-level
disjunction was *not* affected because the outer execution path reaches
`execute_branch` → `execute_exists_branch` → recursive `execute_branch` on
the body, which correctly hits `parse_action_call` and dispatches via
`compile_action_ir`. The bug only triggered when the existential was
wrapped inside a definition whose body was itself compiled via
`compile_action_ir` → `CompiledActionIr::from_ir` → `Exists` clause.

**Fix:** `src/tla/compiled_eval.rs:2039` (`eval_compiled_clause_to_branch`,
`Guard` arm). Added a runtime check via the new helper
`try_eval_compiled_guard_as_action`. If the guard's text parses as an
identifier-style call to a definition that `looks_like_action`, we route
through `eval_action_body_multi` (the interpreted action evaluator), which
already handles action calls correctly via `expand_action_call_multi`. The
new `text` field on `CompiledActionClause::Guard` carries the original
source so the runtime can do this discrimination — compile-time can't
because operator definitions aren't in scope yet. Non-action expressions
(plain guards like `Cardinality(S) > 0`) parse-fail or lookup-fail and
fall through to ordinary boolean evaluation, so the perf hot path is
unchanged for normal guards.

**Validation:**
- `cargo run --release -- run-tla --module corpus/internals/QueueSegmentSync.tla --config corpus/internals/QueueSegmentSync_Buggy.cfg --allow-deadlock --skip-system-checks` →
  `violation=true (1 violations found)`, invariant `SegmentsRecoverable`
  violated, 4-step trace printed (matches TLC's discovery shape).
- Same on `_Fixed.cfg` → `1,531 distinct states found, violation=false`.
  Distinct count matches TLC exactly.
- `cargo test --release` → all 601 tests pass (474 lib + 90 bin + 37 integration).
- `scripts/diff_tlc.sh` → 11/11 specs pass (the 9 prior + both new
  QueueSegmentSync entries).

**Regression test:** `src/tla/action_exec.rs::tests::exists_wrapping_action_call_in_compiled_ir_enumerates_all_bindings`
constructs the minimal repro (Inc(i), IncAny == \E i \in S : Inc(i),
Next == IncAny over S = {1,2,3}) and asserts 3 successors with x' values
{1, 2, 3}. Pre-fix this returned 0; post-fix it returns 3.

**Diff list update:** moved both `queue_segment_sync_*` entries out of the
KNOWN DIVERGENT SPECS comment into the active list in
`corpus/diff_test/list.tsv`. The buggy variant gates on the violation
being detected; the fixed variant gates on the 1531-state count.

**Discovered (not patched):** the Paxos-style probe test now produces 21
successors instead of 0/3. Tracing showed that with my fix dispatching
`Send(...)` correctly, a *separate* pre-existing bug in `compile_expr`
becomes visible: the LET-wrapping-Forall expression
`LET Q1b == {} ; Q1bv == {} IN /\ \A a \in Q : \E m \in Q1b : m.acc = a /\ ...`
incorrectly evaluates to `true` for non-empty Q (where the `\A` body should
make it false). Filed as **T1.4** (in plan) — symptom only; not the same
bug as T1.1 and not in scope here. The Paxos test's assertion was already
permissive (accepted 0 OR 3) to mask the original bug; updated to assert a
floor of ≥3 (catches regression that re-drops Send) with a comment
pointing at T1.4.

**Commit:** `4865611` — `fix(soundness): expand action calls inside compiled
\E ... : Action(x) (T1.1)`

### T1.2 — VIEW projection investigation: misdiagnosis, real bug is in Next splitter (NOT patched)

**Status:** investigated, **the original diagnosis was wrong**. VIEW projection
IS being applied to fingerprinting. The 121-vs-106 discrepancy in `ViewTest.tla`
is caused by a **wider parser bug**: top-level `\/` inside a `/\`-conjunction
Next body is incorrectly treated as a top-level disjunction, producing spurious
successor states. Per the working rule "if the bug is wider, document and stop",
no patch was applied. Filed as a follow-up (recommend **T1.5** — Next splitter).

**Investigation evidence (run on spot `REDACTED-INSTANCE`, c8g.xlarge):**

1. Confirmed `TlaModel::fingerprint` (`src/models/tla_native.rs:346`) does call
   `evaluate_view` and the global FP store dedupes correctly. With
   `TLAPP_VIEW_DEBUG=1` instrumented, the view eval returns
   `Seq([Int(x), Int(y)])` for every state and there are exactly 121 unique
   `<<x, y>>` results — meaning the FP store has 121 *distinct view-projected
   values*. So VIEW is wired in.

2. Removed VIEW from the cfg (`SPECIFICATION Spec\nINVARIANTS TypeOK`, no
   `VIEW StateView`). Result: still **321 generated, 121 distinct**. The full
   state count under no-view is also 121 because the buggy Next happens to
   produce a one-to-one (x,y)↔timestamp relationship (timestamp ends up equal
   to the y-increment count along any path). Confirmed VIEW has nothing to do
   with the discrepancy.

3. Instrumented `TlaModel::next_states` to dump successors. From init
   `(t=0, x=0, y=0)` tlaplusplus generates **3 successors**:
     - `(t=0, x=0, y=0)` — pure stutter, identical to source
     - `(t=0, x=1, y=0)` — `x` incremented but **timestamp NOT incremented**
     - `(t=1, x=0, y=1)` — correct `y`-disjunct outcome
   Expected (TLC, 2 successors): `(t=1, x=1, y=0)` and `(t=1, x=0, y=1)`.

4. Instrumented `evaluate_next_states_with_instances` to print
   `split_action_disjuncts(next_body)`. Output:
   ```
   -> 3 disjuncts:
     [0] "/\\ x + y < 15\n    /\\"
     [1] "/\\ x < 10\n          /\\ x' = x + 1\n          /\\ y' = y"
     [2] "/\\ y < 10\n          /\\ y' = y + 1\n          /\\ x' = x\n    /\\ timestamp' = timestamp + 1"
   ```
   The Next body
   ```
   /\ x + y < 15
   /\ \/ /\ x < 10 /\ x' = x + 1 /\ y' = y
      \/ /\ y < 10 /\ y' = y + 1 /\ x' = x
   /\ timestamp' = timestamp + 1
   ```
   should be *one* action body with an inner 2-way disjunction, but the
   splitter slices on the inner `\/` characters at top level, yielding three
   bogus "branches" — one with no primed assignments (→ stutter), one missing
   the timestamp clause, and one with the timestamp clause attached only to
   the `y`-disjunct.

**Root cause (file:line):** `src/tla/action_exec.rs:799 split_action_disjuncts`
delegates to `src/tla/action_ir.rs:141 split_action_body_disjuncts`. When
`split_indented_action_disjuncts` returns `None` (because no top-level line
*starts* with `\/` after trimming — here every meaningful line starts with
`/\`), the code falls through to `split_top_level(trimmed, "\\/")`, which
greedily splits on every top-level `\/` regardless of whether it sits inside
a top-level conjunction. The interpreter path
(`evaluate_next_states_with_instances`) is therefore inconsistent with the
analyze-tla path (`compile_action_ir_branches`), which correctly cross-products
the disjunction inside each conjunct.

**Why this is wider than VIEW:** the broken splitting affects *any* spec whose
Next has the shape `/\ guard /\ \/ branch1 \/ branch2 /\ shared_postcondition`
— a **very common** TLA+ idiom for "guarded branches with a shared post-step
update". Specs that happen not to use this layout (e.g. those that lift
disjunction to the very top: `Next == \/ Action1 \/ Action2`) are unaffected,
which is why the corpus pass rate (174/182) doesn't reflect the breadth of
this. Beyond VIEW, this is a **soundness-relevant** bug: it can cause
tlaplusplus to either miss states (if the spurious branches drop required
clauses, as here) or invent states that violate guards. Either direction can
hide invariant violations.

**Recommended fix sketch (NOT applied; for follow-up task):**
`split_action_body_disjuncts` should return `vec![trimmed]` when the input is
a top-level `/\`-conjunction (i.e. starts with `/\` and `split_top_level` on
`/\` yields ≥2 parts). The downstream interpreter/IR-compile path already
handles inner disjunctions correctly per conjunct.

**Validation that no patch was applied:**
- `./target/release/tlaplusplus run-tla --module corpus/language/ViewTest.tla --config corpus/language/ViewTest.cfg --allow-deadlock --skip-system-checks`
  still reports `321 states generated, 121 distinct` — same as before
  investigation. All instrumentation reverted.
- `corpus/diff_test/list.tsv` left unchanged: `ViewTest` remains in the
  KNOWN DIVERGENT SPECS comment block. The 11-spec curated harness is
  unaffected (no `cargo test` rerun needed since no source change was
  retained).

**Commits:** none — investigation only; no source changes shipped.

**Follow-up filed:** add a new task (T1.5 in plan) to fix
`split_action_body_disjuncts` so that top-level `/\` bodies are not split on
inner `\/`. Once that lands, ViewTest's 106-distinct count should fall out
automatically and the diff harness can re-include it.

### T1.5 — SOUNDNESS fix: Next splitter sliced `/\ guard /\ \/A \/B /\ shared` on the inner `\/`

**Date:** 2026-04-26.

**Symptom:** `corpus/language/ViewTest.tla` reported 121 distinct states under
tlaplusplus vs TLC's 106. The Next body has the canonical "guarded branches
with a shared post-step update" shape:

```
Next ==
    /\ x + y < 15
    /\ \/ /\ x < 10 /\ x' = x + 1 /\ y' = y
       \/ /\ y < 10 /\ y' = y + 1 /\ x' = x
    /\ timestamp' = timestamp + 1
```

The outer connective is `/\` and the inner disjunction has two branches; the
correct semantics is "exactly two successors per state, each carrying both
the outer guard and the shared `timestamp' = timestamp + 1` post-condition."
Pre-fix, tlaplusplus produced THREE bogus "branches" (a stutter, a branch
missing `timestamp' = timestamp + 1`, and a branch with it), inflating the
state graph and masking VIEW projection collapsing.

**Root cause (3 collaborating bugs):**

1. `src/tla/action_ir.rs:141 split_action_body_disjuncts` — when
   `split_indented_action_disjuncts` returned `None` (no line *starts* with
   `\/` after dedent because every meaningful line starts with `/\`), the code
   fell through to `split_top_level(trimmed, "\\/")`, which greedily slices
   on every top-level `\/` regardless of whether it sits inside a top-level
   conjunction. **Fix:** before falling through, call
   `split_action_body_clauses(trimmed)`; if it returns ≥ 2 conjuncts, the
   outer connective is `/\` (not `\/`) and we return `vec![trimmed]` — the
   whole body is one action with a nested disjunction. Downstream IR
   compilation expands the nested disjunction with the shared clauses
   correctly.

2. `src/tla/action_exec.rs::normalize_branch_expr` — was stripping ALL
   leading `/\` in a loop. For a list-style conjunction
   `/\ guard /\ \/ A \/ B /\ shared`, this made the first conjunct lose its
   delimiter while later conjuncts kept theirs, so `split_top_level("/\\")`
   in the inline-action body path silently produced a corrupt clause split.
   **Fix:** only strip the leading `/\` if the body has no further top-level
   `/\` conjunct (i.e. the body is shaped `/\ X` with a single conjunct).
   Added `has_top_level_conjunct` helper that scans byte-by-byte respecting
   bracket / paren / brace / angle / string nesting.

3. `src/tla/compiled_eval.rs::eval_compiled_clause_to_branch` Guard arm —
   when `compile_action_ir` produces a flat IR for `/\ guard /\ DISJ /\ shared`,
   the disjunctive sub-clause `\/ /\ x' = ... \/ /\ y' = ...` is wrapped as
   a single `Guard { expr, text }` clause. `eval_compiled_guard` evaluated
   `text` as a *boolean* (true because at least one disjunct was satisfied
   in the current state) and then returned the existing branch unchanged —
   silently discarding the inner primed assignments. **Fix:** added
   `guard_text_is_action_body` heuristic — when the guard text is a top-level
   `\/` containing primed identifiers (`X'`) or `UNCHANGED`, dispatch through
   `crate::tla::eval::eval_action_body_multi` (the interpreted action
   evaluator) instead of the boolean guard, then merge the resulting branches
   back into the compiled-IR branch state.

**Why all 3 changes were needed:** fix (1) alone caused ViewTest to spin
forever at 1 distinct state because the inline action body path then
compiled the body via the broken `normalize_branch_expr`, which yielded a
mangled clause split that the compiled IR evaluated as a stutter. Fix (2)
plus (1) made the interpreted IR see the correct conjunctive structure but
the compiled-IR fast path still took the boolean-guard route on the
disjunctive sub-clause and produced a stutter successor. Fix (3) closes the
loop so the compiled-IR fast path produces the same successors as the
interpreted IR.

**Files changed:**
- `src/tla/action_ir.rs:141` `split_action_body_disjuncts` (early-return
  when body is a top-level conjunction).
- `src/tla/action_exec.rs:845` `normalize_branch_expr` + new
  `has_top_level_conjunct` helper.
- `src/tla/compiled_eval.rs:2039` Guard arm + new
  `guard_text_is_action_body` helper at line ~2168.

**Regression tests added:**
- Unit: `src/tla/action_ir.rs::split_action_body_disjuncts_does_not_split_nested_disjunction_inside_outer_conjunction` — asserts the splitter returns 1 disjunct for the canonical body and that `compile_action_ir_branches` expands it into 2 branches each carrying outer guard + shared post.
- Unit (end-to-end): `src/tla/action_exec.rs::evaluates_conjunctive_next_with_inner_disjunction_and_shared_post` — calls `evaluate_next_states` and asserts exactly 2 successors with `timestamp' = 1` in both.
- Integration: `tests/next_splitter_t1_5.rs` — minimal 15-line spec exercising the pattern, plus a multi-step test confirming `timestamp' = timestamp + 1` persists across 5 successor steps.

**Validation:**
- `./target/release/tlaplusplus run-tla --module corpus/language/ViewTest.tla --config corpus/language/ViewTest.cfg --workers 2` → **191 generated, 106 distinct, 0 violation** (matches TLC exactly, was 121 before).
- `cargo test --release` → **603 tests passing, 0 failing** (476 lib + 90 bin + 15 + 11 + 5 + 6 across tests/, 2 ignored). New unit + integration tests both green.
- `scripts/diff_tlc.sh` → **12/12 PASS, 0 fail, 0 allowlisted** (ViewTest now in active list, 106 vs 106).
- Sample corpus parity (29 of 32 internal corpus specs run to completion under 30s timeout each, the other 3 are expected-large `Combined`, `BloomAutoSwitch`, `FingerprintResize`): **all 29 produce identical distinct counts to baseline, 0 panics, 0 errors**. Notably WorkQueue=15003, CheckpointDrain=26344, NegativeIntTest=10360, EnabledTest=121, LivenessTest=157 all match prior runs.

**Wider scope discovered:** None — the fix is narrowly contained. T1.2 (VIEW projection) was a misdiagnosis whose symptom was actually T1.5; this fix closes both. The `compile_action_ir_branches` path was already correct (it's used by analyze-tla); only the runtime `evaluate_next_states` path needed the three-part fix above.

**Commit:** `2881d8b`.

### T1.4 — SOUNDNESS fix: compiled expr mis-split LET body on indented `\/`, silently TRUE for `\A x : \E m \in {} : ...`

**Date:** 2026-04-26.

**Symptom:** the parked test
`src/tla/action_exec.rs::tests::parsed_paxos_style_probe_keeps_let_locals_in_scope_with_operator_overrides`
asserted only `>= 3` Phase2a successors instead of the correct exact `3`,
because the compiled-expression evaluator returned TRUE for the guard

```tla
\A a \in Q : \E m \in Q1b : m.acc = a
```

at an initial Paxos state where `Q1b = {}` (no `1b` messages yet). Since
`\E m \in {} : ...` is FALSE for any body, the universal must be FALSE
for any non-empty `Q` — but the compiled fast path silently returned
TRUE, over-counting Phase2a successors by `|Quorum| × |Value| = 6` per
ballot (21 instead of 3). Any Paxos-style or distributed-protocol spec
using the universally-quantified-existential pattern would silently pass
invariants it should fail.

**Root cause:** in `src/tla/compiled_expr.rs::compile_expr` the multi-line
LET body inside `Phase2a`,

```text
/\ \A a \in Q : \E m \in Q1b : m.acc = a
/\ \/ Q1bv = {}
   \/ \E m \in Q1bv : ...
```

is dispatched to `formula::split_top_level(body, "\\/")` BEFORE the
`/\` split. `formula::split_top_level` is body-delimiter aware (it
correctly treats `\\E m \\in S : body` as one unit) but is **not**
indentation-aware: once `in_quantifier_body` becomes true after `\\E m
\\in Q1b :` the splitter's only short-circuit for `\\/` is "did the
quantifier body itself start with `\\/`?" — for our shape it didn't, so
the splitter happily sliced on the indented inner `\\/` lines and
returned **three** disjuncts, producing `Or([Forall, Eq(Q1bv, {}),
Exists])` instead of the correct nested `And([Forall, Or([Eq,
Exists])])`. The flat `Or` includes `Q1bv = {}` which is TRUE at the
initial state, so the entire compiled expression evaluated TRUE and
silently masked the Paxos guard.

The interpreted evaluator was unaffected — it has a separate
`split_indented_top_level_boolean` pre-pass (`src/tla/eval.rs:5478`).
Only the compiled fast path lacked it.

**Fix location:** `src/tla/compiled_expr.rs:557-578` plus a new helper
`split_indented_top_level_boolean` (~line 1000) and its companion
`normalize_multiline_boolean_indentation`. Before falling through to the
symbol-based `\\/` and `/\\` splits, `compile_expr` now tries
indentation-based splitting on `/\\` first, then on `\\/`. The helper
finds the smallest indent at which a line begins with the delimiter and
splits only at that level — inner indented `\\/` and `/\\` are kept as
part of the current clause for recursive `compile_expr` calls. After the
fix the same Phase2a body compiles to `And([Forall { ... }, Or([Eq,
Exists])])`, the universal correctly evaluates to FALSE at the initial
state, and Phase2a is disabled (the only enabled action is Phase1a).

**Regression tests added** (all in `src/tla/compiled_eval.rs::forall_tests`):
- `t1_4_exists_over_empty_set_is_false` — minimal `\\E m \\in {} : TRUE`
  must be FALSE.
- `t1_4_forall_over_inner_empty_exists_is_false` — outer `\\A a \\in {1,
  2, 3}` over inner empty `\\E` must be FALSE.
- `t1_4_forall_over_inner_exists_with_let_bound_empty_set_is_false` —
  same shape but with the inner-`\\E` domain bound by a LET.
- `t1_4_paxos_phase2a_guard_with_empty_msgs_is_false` — exact Phase2a
  guard shape with `msgs = {}` and a non-empty `Q`.

Plus updated two existing tests to assert the (now correct) exact counts
instead of the buggy over-counts:
- `tests::parsed_paxos_style_probe_keeps_let_locals_in_scope_with_operator_overrides`
  — strengthened from `>= 3` to exactly `3` (Phase1a per ballot in
  `Ballot = 0..2`); removed the TODO comment.
- `tests::parsed_module_probe_matches_model_state_for_phase2a_let_guards`
  — corrected from `2` to `1` (the buggy `2` came from the same `Or`
  mis-shape; the correct semantics for a single Quorum × single Value ×
  single Ballot Phase2a is one new `2a` message).

**Validation:**
- `cargo test --release` → **609 tests passing, 0 failing** (480 lib +
  90 bin + 15 + 11 + 5 + 6 + 2 across `tests/`, 1 ignored). The four new
  T1.4 unit tests are part of the 480.
- `scripts/diff_tlc.sh` → **12/12 PASS, 0 fail, 0 allowlisted**. No
  regressions; CheckpointDrain (26344 distinct), QueueSegmentSync
  (1531/32 distinct), ViewTest (106 distinct) all still match TLC.
- Sample corpus parity (5 internal specs with quantifiers, run on the
  spot under `--workers 4`): **WorkStealingTermination → 64,805 distinct
  (matches TLC 64,805)**, CheckpointDrain matched in the diff harness,
  FingerprintStoreResize produces the same pre-existing
  `expected Int, got Bool(false)` invariant evaluation issue **before
  and after the fix** (so it is unrelated to T1.4 and does not
  regress). BloomAutoSwitch and FingerprintResize did not finish under
  the 2-min timeout but reached the same per-progress-line state counts
  as baseline runs; no panics, no `\E`/`\A` divergences observed.
- Phase2a probe count: was `21` (over-counted, `|Quorum| × |Value| × 3
  ballots`), now exactly `3` (Phase1a per ballot, Phase2a correctly
  disabled).

**Wider scope discovered:** None new. The fix is contained to
`compile_expr` and is symmetric with the long-existing
`split_indented_top_level_boolean` in the interpreted evaluator. The
old indentation-aware splitter `split_top_level_old` in
`src/tla/compiled_expr.rs` (still marked `#[allow(dead_code)]`) attempted
this logic via column tracking but was disabled because trimming the
expression broke its first-line indent calculation; the new helper sits
strictly above the body-delimiter splitter and complements rather than
replaces it.

**Commit:** `a7dfa1a`.

### T1.3 — fairness SCC false positive on wrapper-Next stutter SCCs

**Date:** 2026-04-26.

**Symptom:** `corpus/internals/WorkQueue.tla` was on the KNOWN
DIVERGENT list because tlaplusplus reported

> "Fairness violation in SCC 0 (1 states): Fairness constraint may
> be violated: action 'Next' does not occur in SCC"

while TLC said "No error has been found." Distinct state counts
agreed (15003 under both checkers) — purely a fairness-checker false
positive on the single-state SCC formed by the spec's explicit
`Terminated /\ UNCHANGED vars` stutter disjunct.

**Root cause:** `check_fairness_on_scc` matched transitions strictly
by `t.action.name == constraint.action_name`. The action labeller
(`evaluate_next_states_labeled_with_instances` →
`extract_action_name`) takes each top-level disjunct of `Next` and
extracts the head identifier — `WorkerTakeItem`, `LoaderLoad`,
`Terminated`, etc. — falling back to `next_name` only when the
disjunct has no parseable head. So the labels in WorkQueue's state
graph are all subaction names; *none* of them carry the literal
string `"Next"`. When the spec declared `WF_vars(Next)` (fairness on
the entire next-state relation), the SCC walker correctly identified
the terminated state as a self-loop SCC, then asked "does any
transition labelled `Next` exist in this SCC?" — the answer was no
for every SCC, so every SCC tripped the violation. The first
non-trivial SCC the walker hit was the terminated stutter (a
1-state self-loop), which surfaced as the reported violation.

This is conceptually wrong: when `WF_vars(Next)` is declared on the
wrapper Next action, every transition in the state graph IS a Next
step by definition (because every successor was generated by
evaluating Next). So any in-SCC edge satisfies the constraint.

**Fix location:** `src/fairness.rs:163-237` — added
`check_fairness_on_scc_with_next(..., next_action_name:
Option<&str>)` which treats `t.action.name == action_name` OR
`constraint.action_name == next_action_name` as a satisfying
transition. The original `check_fairness_on_scc` is preserved as a
thin wrapper that passes `None`. `src/model.rs:60-71` adds
`Model::next_action_name() -> Option<&str>` (default `None`) so the
runtime can ask the model for its wrapper Next name.
`src/models/tla_native.rs:427-429` returns `Some(&self.next_name)`
for TLA+ models. `src/runtime.rs:2852-2858` passes
`model.next_action_name()` into the new entry point.

The narrow scope is deliberate: the fix only relaxes the wrapper-Next
case (constraint action name == spec's Next definition name).
Fairness on a *named subaction* (e.g., `WF_vars(WorkerTakeItem)`)
still requires that named subaction to actually fire in every SCC
where it might be enabled — true-positive detection is preserved.

**Regression tests added** (in `tests/wrapper_next_fairness_t1_3.rs`):
- `wf_vars_next_with_explicit_stutter_disjunct_does_not_falsely_violate_fairness`
  — minimal 4-state spec (`x = 0..3`) with `Done == /\ x = 3 /\
  UNCHANGED vars` and `Spec == Init /\ [][Next]_vars /\
  WF_vars(Next)`. Asserts no violation. Pre-fix this would report a
  fairness violation on the `x = 3` self-loop SCC.
- `wf_vars_named_subaction_still_detects_real_violation` — spec
  with a 2-state `Toggle` cycle and `WF_vars(NeverFires)` declared
  on a permanently-disabled subaction. Asserts the violation IS
  reported with `PropertyType::Liveness` and message mentioning
  `NeverFires`. This is the true-positive sanity check.
- `corpus/internals/WrapperNextFairness.tla` + `.cfg` — minimal
  external spec that mirrors the failing pattern, parked under
  internals for ad-hoc reproduction.

**Validation:**
- `cargo test --release` on the spot instance:
  **611 tests pass** (480 lib + 90 bin + 15+11+2+5+6+2 across
  `tests/`, 1 ignored, 0 failed). The two new T1.3 tests are
  among the 2 in `tests/wrapper_next_fairness_t1_3.rs`.
- `bash scripts/diff_tlc.sh` on the spot instance: **13/13 PASS,
  0 fail, 0 allowlisted.** WorkQueue is now in the active list and
  reports `states=37621 distinct=15003 violation=no` matching TLC
  byte-for-byte. The previously-active 12 specs (CheckpointDrain,
  ViewTest, QueueSegmentSync_Fixed/_Buggy, etc.) still pass.
- `corpus/diff_test/list.tsv` updated: WorkQueue promoted from the
  KNOWN DIVERGENT comment block to the active diff list.
- True-positive detection sanity-checked end-to-end via the second
  unit test, plus by running CheckpointDrain.tla
  (26344 distinct, no violation) and WorkStealingTermination.tla
  (64805 distinct, no violation), both with subaction-level WF/SF
  constraints — no regressions.

**Wider scope discovered:** None. The fix is a precise edge-case
patch contained to the fairness checker's matching predicate; it
does not touch the SCC walker, the action labeller, or the
constraint extractor. The pre-existing `BuchiChecker` API in
`src/liveness.rs` (only used by unit tests) was left on the original
`check_fairness_on_scc` entry point — no changes needed there.

**Commit:** `08c7ec8`.

