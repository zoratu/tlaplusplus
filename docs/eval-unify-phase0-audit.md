# Dual-Evaluator Unification — Phase 0 Audit

Status: Phase 0 (safety net + audit). **No model behavior change.** This
document maps the surface area for migrating the model's hot-path evaluation
from the interpreted path (`eval_expr`) onto the compiled path
(`eval_compiled`).

## Background

`tlaplusplus` has **two** expression evaluators:

- **Interpreted** — `src/tla/eval/` : `eval_expr(text, ctx)` →
  `classify_op` → `eval_expr_inner`. Re-parses the expression *text* on every
  call. This is the reference semantics.
- **Compiled** — `src/tla/compiled_expr.rs::compile_expr(text)` →
  `CompiledExpr` AST, evaluated by
  `src/tla/compiled_eval.rs::eval_compiled(&CompiledExpr, ctx)`.

The compiled path is a **superset wrapper** over the interpreted one: any
construct `compile_expr` cannot model is emitted as
`CompiledExpr::Unparsed(text)`, and `eval_compiled` on `Unparsed(s)` calls
`eval_expr(s, ctx)` (`compiled_eval.rs:1146`). So the interpreted path is the
compiled path's leaf fallback.

**The divergence class this project fixes:** the model *hot path*
(`src/models/tla_native.rs`, `src/tla/action_exec.rs`) evaluates Init / Next /
guards / domains by calling `eval_expr` **directly**, whereas
`check_invariants` evaluates through `eval_compiled`. Same predicate text, two
mechanisms — so they can disagree (behind MCBakery + PRs #157–#159).

**Invariant we want to guarantee (and the Phase-0 proptest checks):**

```
eval_expr(P, s) == eval_compiled(compile_expr(P), s)   for all predicates P, states s
```

Both-`Ok`-equal, both-error, and both-panic all count as "equal".

---

## Table 1 — Direct `eval_expr` callers outside `src/tla/eval/`

Categories:
- **(a) MODEL hot path** — Init/Next/guard/domain eval driving state
  exploration. **These are the Phase 1 migration targets.**
- **(b) CLI / analysis** — `analyze-tla` probe machinery, run/assume drivers.
  Not on the exploration hot path; migrate opportunistically, lower priority.
- **(c) symbolic-init** — Z3 symbolic Init enumeration (feature-gated).
- **(d) internal** — inside the compiled evaluator itself (the `Unparsed`
  leaf fallback + compiled sub-evaluations that delegate a fragment to the
  interpreter). Migrating (a) does not remove these; they are the seam.

Test-only call sites (`#[cfg(test)]`) are excluded — notably
`src/cli/probe.rs` ≥ L3033, `src/tla/compiled_eval.rs` `eval_both_paths`
(L5435), and `src/tla/compiled_expr.rs` L4866.

### (a) MODEL hot path — Phase 1 targets

| File:line | Site / role |
|---|---|
| `src/models/tla_native.rs:952` | View expression eval (symmetry `VIEW`) |
| `src/models/tla_native.rs:1015` | State-constraint predicate eval |
| `src/models/tla_native.rs:2652` | Init: domain-set eval for a binder |
| `src/models/tla_native.rs:2745` | Init: constant/definition reference resolution |
| `src/models/tla_native.rs:2754` | Init: definition-body eval |
| `src/models/tla_native.rs:2885` | Init: clause is-Bool classification |
| `src/models/tla_native.rs:2919` | Init: state-constraint short-circuit |
| `src/models/tla_native.rs:2991` | Init: clause value eval |
| `src/models/tla_native.rs:3012` | Init: definition-body eval |
| `src/models/tla_native.rs:3026` | Init: clause value eval (alt path) |
| `src/models/tla_native.rs:3075` | Init: membership set eval |
| `src/models/tla_native.rs:3117` | Init: definition-body set eval |
| `src/models/tla_native.rs:3130` | Init: membership set eval (alt path) |
| `src/models/tla_native.rs:3290` | Init: binder domain-set eval |
| `src/models/tla_native.rs:3383` | Init: assignment RHS value eval |
| `src/models/tla_native.rs:3400` | Init: membership set eval |
| `src/models/tla_native.rs:3442` | Init: `[x \in D \|-> ...]` domain eval |
| `src/models/tla_native.rs:3475` | Init: function-body eval per domain point |
| `src/models/tla_native.rs:3489` | Init: boolean-clause eval |
| `src/models/tla_native.rs:3555` | Next/guard eval |
| `src/models/tla_native.rs:3606` | guard/value eval (nested) |
| `src/models/tla_native.rs:3638` | guard/value eval (nested) |
| `src/models/tla_native.rs:3694` | Init predicate `= TRUE` check |
| `src/models/tla_native.rs:3742` | Init assignment RHS eval |
| `src/models/tla_native.rs:3756` | Init membership set eval |
| `src/tla/action_exec.rs:559` | Successor: operator-arg eval (outer ctx) |
| `src/tla/action_exec.rs:611` | Successor: operator-arg eval |
| `src/tla/action_exec.rs:808` | Successor: `\in`-binder domain eval |
| `src/tla/action_exec.rs:1163` | Successor: constant reference eval |
| `src/tla/action_exec.rs:1176` | Successor: RHS value eval |
| `src/tla/action_exec.rs:1193` | Successor: primed-assignment value eval |

**Count: 25 in `tla_native.rs`, 6 in `action_exec.rs` — the Phase 1 set.**
(The guard site at L3555 now also carries the opt-in Phase-0 consistency check
call — see Task 3 — but still evaluates via `eval_expr`; behavior is
unchanged.)

### (b) CLI / analysis — lower priority

| File:line(s) | Role |
|---|---|
| `src/cli/probe.rs` L304, 425, 547, 574, 603, 711, 808, 1347, 1892, 1931, 2085, 2189, 2363, 2473, 2486, 2503, 2701, 2731, 2807, 2838, 2855, 2858, 3021 | `analyze-tla` probe machinery (23 prod sites): constant/definition probing, action-clause value/guard/membership eval, primed-assignment eval. Mirrors the hot path but for the analyzer, not exploration. |
| `src/cli/run_tla.rs:659` | Liveness predicate eval in the run driver |
| `src/cli/run_model.rs:318` | `ASSUME` body eval at startup |
| `src/cli/analyze_tla.rs:177, 237, 255` | `analyze-tla`: reference / value / set eval |

### (c) symbolic-init (feature `symbolic-init`)

| File | Sites | Role |
|---|---|---|
| `src/tla/symbolic_init.rs` | ~32 call sites across L507–L2561 (both `eval_expr(...)` and `crate::tla::eval_expr(...)`) | Z3-backed symbolic Init enumeration: evaluates predicate fragments, domain sets, element/range bounds while building the symbolic encoding. Feature-gated; not in the default hot path. |

### (d) internal — the compiled evaluator's own delegations (the seam)

| File:line | Role |
|---|---|
| `src/tla/compiled_eval.rs:1146` | **The `Unparsed(s)` leaf fallback** — `eval_compiled` delegates any un-modeled construct to `eval_expr`. This is what makes the compiled path a superset. |
| `src/tla/compiled_eval.rs:305` | Primed-name resolution sub-eval |
| `src/tla/compiled_eval.rs:1407` | Set-comprehension domain sub-eval |
| `src/tla/compiled_eval.rs:1461` | Membership RHS sub-eval |
| `src/tla/compiled_eval.rs:1821` | LAMBDA-body sub-eval |
| `src/tla/compiled_eval.rs:2846` | Staged operator-arg sub-eval |
| `src/tla/compiled_eval.rs:3099` | Quantifier/binder domain sub-eval |

> Migrating (a) onto `eval_compiled` does **not** delete (d) — these remain
> the fallback seam. Phase 2 (see Table 2) is about shrinking the `Unparsed`
> surface so the fallback fires less often.

---

## Table 2 — The `Unparsed` surface (Phase 2 map)

Every `compile_expr` (and helper) site that emits `CompiledExpr::Unparsed`,
i.e. the construct classes the compiler currently punts to the interpreter.
"Portable" = could be modeled as a real `CompiledExpr` node with bounded
effort; "hard" = genuinely needs interpreter machinery (live definition
context, inline instances, associativity re-derivation).

| Site (`compiled_expr.rs`) | Trigger construct | Portable to `CompiledExpr`? |
|---|---|---|
| L992 | Empty expression after dedent | Trivial (both paths error / no-op) — portable |
| L1158 | `LAMBDA` with non-identifier / unsupported param shape | Hard-ish — needs richer Lambda node; simple lambdas already compiled |
| L1288 | `x \in <bracket-type-expr>` (e.g. `\in [D -> R]`, `\in SUBSET S`) — membership against a *type set* | Portable — model function-set / SUBSET membership as a node |
| L1303 | `split_comparison` matched an operator with no compiled node (catch-all `_`) | Portable — enumerate remaining comparison operators |
| L1405 | Top-level **user-defined infix operator** (`a \preceq b`) — unknown precedence | Hard — precedence is defined at runtime in the live def context |
| L1430 | **Chained top-level arithmetic** (`a + b + c`) — compiler/interpreter splitter associativity differs | Portable-with-care — unify the additive splitter, then compile directly |
| L1630 | Final fallback: anything the cascade didn't recognize | Residual — shrinks as other rows are handled |
| L2559 | **Multi-binder map comprehension** `{ e : x \in S, y \in T }` | Portable — add a multi-binder SetComprehension node |
| L2577 | **Multi-binder filter** `{ x \in S, y \in T : P }` | Portable — same multi-binder node |
| L3287 | `LET alias == INSTANCE M [WITH ...] IN body` — **inline instance** | Hard — needs instance registration in the compiled Let (see MEMORY: LET-INSTANCE bug) |
| L3307 | `LET` with definitions the compiler couldn't bind (e.g. **recursive function defs** `f[S \in SUBSET opts]`) | Hard — recursive/function-valued LET bindings |
| L3428 | `\E` predicate binder that is a **tuple binder** (`\E <<x,y>> \in S : ...`) | Portable — add tuple-binder destructuring to Exists |
| L3505 | `\A` predicate binder that is a **tuple binder** | Portable — same tuple-binder support for Forall |
| L3565 | `CHOOSE` with a **tuple binder** | Portable — same tuple-binder support for Choose |

Plus the generic terminal fallback in `compile_expr` (the last
`CompiledExpr::Unparsed(expr.to_string())`) which catches any shape none of
the structured handlers claimed.

### Phase-2 grouping

- **Quick wins (portable, low risk):** tuple-binder support for
  `\E`/`\A`/`CHOOSE` (L3428/L3505/L3565); multi-binder set comprehensions
  (L2559/L2577); type-set membership (L1288); remaining comparison ops
  (L1303).
- **Portable-with-care:** chained arithmetic (L1430) — unify the additive
  splitter first, guarded by the equivalence proptest.
- **Genuinely hard (leave on the interpreter fallback):** user-defined infix
  operators (L1405), inline `INSTANCE` in `LET` (L3287), recursive /
  function-valued `LET` bindings (L3307), exotic `LAMBDA` shapes (L1158).
  These depend on the live definition context and are the long tail; the
  `Unparsed` fallback should remain for them indefinitely.

---

## Task 1 — the strengthened equivalence proptest

`tests/compiled_vs_interpreted.rs` already fuzzes compiled-vs-interpreted
equivalence with a wide typed grammar (arith / comparisons / set ops /
quantifiers / IF / LET / CASE / records / EXCEPT / sequences / function
application / opcalls), swarm-mode category masking (T16a), and three-layout
rendering (aligned / shallow-head / deep-head, from #159).

Phase-0 strengthening (this branch): the equivalence is now checked against a
**panel of 5 distinct states** (`build_states`) over the same schema, instead
of one fixed state. The states are chosen to flip common sub-predicate
outcomes:

- `x` vs `y` ordering (`<`, `>`, `==`) and `b` true/false,
- `S`/`T` disjoint, overlapping, subset/superset, empty, and equal (so `\in`,
  `\subseteq`, `\union`, `\intersect`, `\` change verdict),
- empty / singleton / longer sequences,
- records with positive / zero / negative fields,
- functions with differing and **empty** domains (so `f[1]`/`f[2]` sometimes
  error on **both** paths — exercising the both-error branch).

`check_equivalence(expr)` now iterates every state via
`check_equivalence_in_state`, tagging any divergence with the state index. A
mismatch in any single state fails the case, so each generated predicate — and
its sub-predicates — is exercised where it is both true and false. Both-Ok-equal
/ both-error / both-panic remain "equal". Case count stays CI-bounded
(`PROPTEST_CASES`, default 256).
