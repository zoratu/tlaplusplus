# Chaff testing — measuring the sensitivity of our own test suite

Our differential and corpus gates answer one question well: *does this build
agree with TLC today?* They do **not**, by themselves, answer a second,
equally important one: *would a gate actually fail if a known bug class were
reintroduced?*

That gap is not hypothetical. Several real semantic bugs have reached `main`
because no gate exercised the relevant path — a bare `<>[]P` liveness property
left unchecked, a guard-disjunction split that dropped shared primed conjuncts,
a `--features symbolic-init` regression that CI never compiled. Each is an
*uncaught chaff*: a defect the suite was, in principle, blind to.

Chaff testing closes that gap. It is the method behind Brown PLT's
[Examplar](https://blog.brownplt.org/2024/01/01/examplar.html): a good test
suite is a **classifier** that separates correct implementations ("wheat") from
deliberately-buggy ones ("chaff"), and you measure the suite by **how many
chaffs it catches**.

## How it works

1. **Chaffs** (`chaffs.toml`, `src/chaff.rs`) are intentional, feature-gated
   semantic regressions, each sourced from a bug we have actually shipped. A
   chaff lives at the exact source site of a real fix and reintroduces the old,
   wrong behavior when activated.

2. Chaffs are gated behind `--features chaff` and selected at runtime via the
   `TLAPP_CHAFF` env var. **Without the feature there is no way to activate a
   chaff**: `chaff::active` is unconditionally `false`, so no environment can turn
   one on and the production behavior is unchanged. The call sites are left
   *unconditional* (not `#[cfg]`-removed) on purpose, so the default `tests` build
   compiles and type-checks every chaff branch and the feature cannot bit-rot.
   `chaff::active` is `#[inline(always)]`, so in a release (optimized) build each
   `if chaff::active(..)` folds to the constant `false` and the guarded branch is
   dropped — the zero-cost claim holds for optimized builds, which is what ships;
   an unoptimized debug build keeps a trivially-predictable dead branch. The
   `chaff` feature must never be part of `default` and must never ship in a
   release artifact.

3. `scripts/chaff_matrix.sh` activates each chaff in turn and runs the gate that
   is *supposed* to catch it, asserting the gate **fails**. A chaff the gate does
   not catch is a proven hole — the script exits non-zero and names it.

This is Examplar's "conceptual mutation testing": the chaffs are *semantic*
mistakes (wrong junction splitting, wrong prime staging, unchecked temporal
shapes), **not** line-level mutants from a tool like `cargo-mutants`. Our bug
history is conceptual, so the high-signal chaff set is hand-curated from it.
Automated line-level mutation would drown the signal in trivially-caught or
semantically-irrelevant noise — the same warning Examplar gives.

## Running it

```bash
# Build with the chaff feature (REQUIRED — otherwise every chaff is a no-op).
cargo build --release --features chaff

# Run the full matrix (needs TLC for diff_tlc-gated chaffs: java + tools/tla2tools.jar).
scripts/chaff_matrix.sh

# One chaff:
CHAFF_ONLY=let_helper_prime_as_guard scripts/chaff_matrix.sh

# Include not-yet-wired (status=todo) chaffs — they should report UNCAUGHT:
CHAFF_INCLUDE_TODO=1 scripts/chaff_matrix.sh
```

Per project convention, build and run on a spot instance, not locally.

## Chaff status vocabulary

Each `[[chaff]]` in `chaffs.toml` has a `status`:

- **`wired`** — toggle is in the source *and* the matrix has demonstrated the
  gate catches it. These are what the matrix asserts on; a `wired` chaff that
  goes UNCAUGHT fails CI as a real suite hole.
- **`open`** — toggle is in the source but the gate does *not* (yet) catch it: a
  tracked finding, with a `note` explaining why. The matrix skips these so CI
  stays green while the gap is documented.
- **`todo`** — located but the toggle is not yet inserted.

## Three findings the tool surfaced — all resolved (8 of 9 caught, 1 open)

The first CI run (2026-11-07) reported: control OK, **5 caught**, **3 open**.
The three were not false alarms — each was a case where a first-draft chaff did
not reproduce the bug via the gated spec's *actual* evaluation path, which is
precisely what the matrix exists to reveal. All three are now `wired` and
confirmed CAUGHT (final run 2026-11-08: control clean, **8 of 9 wired chaffs
caught, 0 holes**; the 9th, `eventually_always_unchecked`, is an `open` finding —
its toggle is wired but no spec exercises it yet):

- `assert_short_circuit_disabled` — `run-tla` uses the **compiled** evaluator by
  default (the dual-evaluator unification), so the first draft on `CompiledExpr::And`
  and a second draft on the *interpreted* clause loop were both unreached.
  **Fix:** sited the chaff on the compiled clause loop
  `eval_compiled_clause_sequence`, where the false-guard short-circuit actually
  runs for this spec.
- `let_helper_prime_as_guard` — two lessons in one. The existing `LetHiddenPrime`
  spec nests the `LET` under `\E` (interpreted/quantifier path), so it never
  reaches the compiled LET arm; **and** the #175 fix is *two* defensive layers —
  the compiled LET gate *and* an interpreted cross-check that re-runs on an empty
  compiled result for LET-bodied actions. **Fix:** added a top-level spec
  `LetHiddenPrimeTop` that reaches the compiled LET arm, and a second toggle in
  `body_has_let_with_primes` so the chaff defeats both layers.
- `constants_in_fingerprint` — a genuine *oracle* limitation: carrying constants
  in every state changes neither the distinct-state count nor the invariant
  verdict, so a count-based differential gate structurally cannot see it.
  **Fix:** added a schema-level gate (`tests/chaff_const_schema.rs`) asserting no
  state carries a constant key.

Each carries a `note` in `chaffs.toml` with the full diagnosis. This is the
Examplar discipline made concrete: a chaff is only useful if it reproduces a
failure the gate's oracle can actually observe *on the path the checker takes* —
when it can't, strengthen the oracle (the constants case) or re-site the chaff
onto the real evaluator (the other two).

## Adding a chaff

1. Find the source site of a fix for a bug class worth guarding (start from a
   `corpus/diff_test/list.tsv` regression entry).
2. Wrap the fix so the **old, buggy** behavior runs under the chaff:

   ```rust
   if crate::chaff::active("my_bug_id") {
       old_buggy_behavior()   // reintroduce the historical defect
   } else {
       the_fix()              // current, correct behavior
   }
   ```

3. Add a `[[chaff]]` entry to `chaffs.toml` (id, bug, origin, spec, gate, site),
   set `status = "wired"`, and confirm `CHAFF_ONLY=my_bug_id scripts/chaff_matrix.sh`
   reports it **CAUGHT**.
4. If the matrix reports the chaff **UNCAUGHT**, the suite has a hole: add or
   strengthen a spec in `corpus/diff_test/list.tsv` (or the relevant gate) until
   the chaff is caught, then commit both.

## Relationship to the broader testing plan

Chaff testing is Enhancement 1 of a four-part plan inspired by the Examplar /
PICK / Hegel line of work:

1. **Chaff matrix** (this doc) — measure suite sensitivity to reintroduced bugs.
2. **PICK-style distinguishing traces** for the liveness/temporal layer.
3. **Metamorphic property tests** over whole model runs (no TLC oracle needed).
4. **Deterministic, minimizing chaos** — shrink a diverging failpoint schedule
   to a committable regression.

## CI

A dedicated job builds `--features chaff` and runs the matrix. This also closes
the "feature CI never compiled" class directly: the chaff feature itself cannot
bit-rot unnoticed, because CI compiles and exercises it every run.
