# `src/main.rs` refactor plan

`src/main.rs` is 11,711 lines. This document is the authoritative split
plan. Path B (doc only) was chosen for this branch — see
[Why Path B for this attempt](#why-path-b-for-this-attempt) for the
specific blockers — but the proposed tree below is intended to be picked
up verbatim by a follow-up landing branch with cargo access on a Linux
runner.

## Goal

Split `src/main.rs` into a `src/cli/` module tree with one file per
subcommand handler, plus shared infra. **Behavior must be identical**:
same flags, same exit codes, same output format, same error messages.
The 90 in-file unit tests, the 13/13 differential-vs-TLC gate, and the
chaos smoke gate must all stay green.

## Topology survey

Top-level item counts in current `src/main.rs`:

| Section | Lines | Items | Notes |
|---|---|---|---|
| `use` block | 1–25 | — | 17 `use` lines, mostly into `tlaplusplus::tla::*` and `tlaplusplus::models::*` |
| `parse_byte_size` | 28–67 | 1 fn | Trivial, pure helper |
| `Cli` / `RuntimeArgs` / `StorageArgs` / `S3Args` / `ClusterArgs` | 69–263 | 5 structs | clap derive; ~200 lines of `#[arg]` annotations |
| `Command` enum | 264–421 | 1 enum, 7 variants | One variant per subcommand |
| `build_engine_config` | 423–504 | 1 fn | Maps `RuntimeArgs`+`StorageArgs`+S3 hint → `EngineConfig` |
| `print_stats` | 505–575 | 1 fn | TLC-shaped output |
| `run_model_with_s3` | 576–861 | 1 fn (generic over `M: Model`) | Wires checkpoint + S3 resume around `run_model` |
| `run_system_checks` | 862–870 | 1 fn | THP / cgroup warnings |
| `evaluate_assumes` | 871–918 | 1 fn | Runs ASSUME blocks at startup |
| `collect_coverage` / `dump_state_graph` | 919–1136 | 2 fns | Coverage and state-graph output |
| `model_fingerprint` | 1137–1150 | 1 fn | TlaState → u64 |
| `print_difftrace` / `print_difftrace_with_relevance` | 1151–1204 | 2 fns | Trace pretty-printer |
| `fetch_s3_file` | 1205–1259 | 1 fn | S3 URI → local temp path |
| `maybe_setup_cluster` | 1260–1353 | 1 fn | Distributed transport bootstrap |
| `fn main()` | 1354–2629 | 1 fn | 1,275-line dispatcher with 7 match arms |
| `list_checkpoints` / `list_s3_checkpoints` | 2632–2972 | 2 fns | `ListCheckpoints` subcommand impl |
| `format_num` | 2973–2989 | 1 fn | Pretty-print integers with separators |
| `inject_constants_into_definitions` (+helpers) | 2990–3083 | 4 fns | Used by `analyze-tla` and `run-tla` |
| Probe machinery | 3084–6094 | ~70 fns + 1 enum | Used **only** by `analyze-tla`; large internal subgraph |
| `mod tests` | 6095–11711 | 90 `#[test]` fns | All probe-machinery tests; `use super::*` |

`fn main()` match-arm sizes:

| Variant | Lines | Approx LOC |
|---|---|---|
| `RunCounterGrid` | 1365–1389 | 25 |
| `RunFlurmLifecycle` | 1390–1409 | 20 |
| `RunHighBranching` | 1410–1429 | 20 |
| `RunAdaptiveBranching` | 1430–1552 | 123 |
| `AnalyzeTla` | 1553–2036 | 484 |
| `RunTla` | 2037–2616 | 580 |
| `ListCheckpoints` | 2617–2625 | 10 |

## Proposed module tree

```
src/cli/mod.rs                   — Cli struct, Command enum, fn run() entry, dispatch
src/cli/args.rs                  — RuntimeArgs, StorageArgs, S3Args, ClusterArgs, parse_byte_size
src/cli/shared.rs                — build_engine_config, print_stats, run_system_checks,
                                    fetch_s3_file, maybe_setup_cluster, format_num,
                                    model_fingerprint, print_difftrace[_with_relevance],
                                    inject_constants_*, config_value_to_*
src/cli/run_model.rs             — run_model_with_s3<M>, evaluate_assumes,
                                    collect_coverage, dump_state_graph
src/cli/run_counter_grid.rs      — handler for RunCounterGrid
src/cli/run_flurm_lifecycle.rs   — handler for RunFlurmLifecycle
src/cli/run_high_branching.rs    — handler for RunHighBranching
src/cli/run_adaptive_branching.rs — handler for RunAdaptiveBranching (incl. monitor-thread block)
src/cli/run_tla.rs               — handler for RunTla (~580 LOC)
src/cli/analyze_tla.rs           — handler for AnalyzeTla (~484 LOC); calls into cli::probe
src/cli/list_checkpoints.rs      — handler for ListCheckpoints; both list_checkpoints + list_s3_checkpoints
src/cli/probe/mod.rs             — re-exports for the probe subgraph
src/cli/probe/state_seed.rs      — seed_probe_state_*, type-invariant constraint kinds, refine_*
src/cli/probe/representative.rs  — representative_value_*, try_create_representative_*,
                                    pick_representative_from_set, sample_param_value*
src/cli/probe/action_eval.rs     — probe_action_*, probe_let_*, probe_exists_*,
                                    probe_stuttering_*, expand_probe_action_call,
                                    body_contains_*_action_call, definition_is_*
src/cli/probe/scan.rs            — collect_*, infer_*, find_*, parse_probe_*, split_probe_*,
                                    sample_exists_quantifier_binders, expr_probe_is_ready,
                                    is_probe_sampling_limitation_error, build_probe_eval_context,
                                    build_action_expr_probe_context, normalize_probe_clause_expr
src/cli/probe/temporal.rs        — starts_with_temporal_*, contains_temporal_*,
                                    should_skip_action_expr_probe, should_short_circuit_probe_guard
src/cli/probe/util.rs            — strip_probe_*, take_probe_group, find_probe_*,
                                    next_probe_word, matches_probe_keyword_at,
                                    is_probe_word_char, trim_probe_ascii_end,
                                    apply_instance_substitutions_to_text
src/cli/probe/tests.rs           — the 90 `#[test]` fns currently in `mod tests`
src/main.rs                      — `fn main() -> anyhow::Result<()> { tlaplusplus::cli::run() }`
                                    plus the `#[cfg(feature = "failpoints")]`
                                    `let _failpoint_scenario = fail::FailScenario::setup();`
                                    line that must stay in `fn main` so the scenario lives
                                    for the whole process
```

Estimated post-split LOC per file (probe subtree split is approximate; see
[risk #2](#risks-and-coupling-the-blocking-issues)):

| File | Approx LOC |
|---|---|
| `src/cli/mod.rs` | ~120 (Cli, Command, dispatch) |
| `src/cli/args.rs` | ~230 |
| `src/cli/shared.rs` | ~600 |
| `src/cli/run_model.rs` | ~520 |
| `src/cli/run_counter_grid.rs` | ~40 |
| `src/cli/run_flurm_lifecycle.rs` | ~35 |
| `src/cli/run_high_branching.rs` | ~35 |
| `src/cli/run_adaptive_branching.rs` | ~150 |
| `src/cli/run_tla.rs` | ~620 |
| `src/cli/analyze_tla.rs` | ~510 |
| `src/cli/list_checkpoints.rs` | ~350 |
| `src/cli/probe/{state_seed,representative,action_eval,scan,temporal,util}.rs` | ~3,000 spread |
| `src/cli/probe/tests.rs` | ~5,600 |
| `src/main.rs` | ~10 |

## Library exposure

`fn run()` is the only new public entry. Everything else stays
crate-private:

```rust
// src/lib.rs
pub mod cli;             // ← new
// (existing pub mod ... unchanged)

// src/cli/mod.rs
pub fn run() -> anyhow::Result<()> { ... }   // ← only public symbol
mod args;
mod shared;
mod run_model;
mod run_counter_grid;
mod run_flurm_lifecycle;
mod run_high_branching;
mod run_adaptive_branching;
mod run_tla;
mod analyze_tla;
mod list_checkpoints;
mod probe;
```

Within `cli::`, prefer `pub(crate)` (or `pub(super)` for probe internals
called only by sibling probe files). Do **not** widen anything to plain
`pub`.

## Dependency graph (the constraint that drives the order)

```
                               ┌──── args.rs ────┐
                               │                 │
                  ┌─ shared.rs ┤                 ├─ all subcommand handlers
                  │            │                 │
                  └─ run_model.rs                │
                                                 │
              ┌── probe/util.rs ─┐               │
              │                  ├── probe/scan.rs ──┐
              │                  │                   ├── probe/state_seed.rs ─┐
              │                  │                   │                        ├─ analyze_tla.rs
              │                  │                   ├── probe/representative.rs ┘
              │                  │                   │
              │                  │                   └── probe/action_eval.rs ─┘
              └── probe/temporal.rs ─┘

run_counter_grid, run_flurm_lifecycle, run_high_branching,
run_adaptive_branching ──► shared.rs + run_model.rs
run_tla ──► shared.rs + run_model.rs (no probe)
analyze_tla ──► shared.rs + probe::*
list_checkpoints ──► (mostly self-contained, uses S3 SDK)
mod.rs ──► all handlers
main.rs ──► cli::run()
```

`probe/*.rs` files have intra-module cycles in the current code (e.g.,
`probe_action_clause_expr` calls `probe_action_body_into_ctx` which
calls `probe_disjunctive_action_body` which calls back into clause
handling). The split must keep those mutually-recursive groups in the
**same file** or expose them via a shared `pub(super)` surface in
`probe/mod.rs`.

## Recommended landing order (for the follow-up branch)

Each step is independently compilable. **After each step, on a Linux
runner**: `cargo check --release && cargo check --release --features failpoints`.

1. **Move parse_byte_size + the four `Args` structs + `Command` enum**
   into `src/cli/args.rs`. Re-export from `src/cli/mod.rs`. Keep `Cli`
   in `src/cli/mod.rs`. Update `src/main.rs` to `use tlaplusplus::cli::*;`
   so `Cli::parse()` still works in the existing `fn main`. **Compile.**
2. **Lift `fn main`'s body** into `cli::run()` exactly as-is (one giant
   match). `src/main.rs` becomes
   `fn main() -> anyhow::Result<()> { tlaplusplus::cli::run() }` plus
   the `#[cfg(feature = "failpoints")]` setup line moved into `cli::run`.
   At this point everything still lives in `cli/mod.rs` but it compiles.
   **Compile + run `cargo test --release` (full suite).**
3. **Move `shared.rs` symbols** (`build_engine_config`, `print_stats`,
   `run_system_checks`, `fetch_s3_file`, `maybe_setup_cluster`,
   `format_num`, `model_fingerprint`, `print_difftrace*`,
   `inject_constants_*`, `config_value_to_*`). Update call sites in
   `cli::run`. **Compile.**
4. **Move `run_model.rs`** (`run_model_with_s3`, `evaluate_assumes`,
   `collect_coverage`, `dump_state_graph`). **Compile.**
5. **Move the four trivial subcommand handlers** (`run_counter_grid`,
   `run_flurm_lifecycle`, `run_high_branching`, `run_adaptive_branching`).
   Each handler becomes
   `pub(super) fn handle(args: ...) -> anyhow::Result<()> { ... }`
   and `cli::run` dispatches with one line per arm. **Compile.**
6. **Move `list_checkpoints`** with both its sync and async helpers.
   The async helper is invoked via `tokio::runtime::Builder` *inside*
   `list_checkpoints`; preserve that pattern verbatim. **Compile.**
7. **Move `run_tla` handler.** No probe dependency, so it's a clean
   move once `shared.rs` and `run_model.rs` are in place. **Compile +
   run all tests + `scripts/diff_tlc.sh`.** Both must be green before
   moving probe code.
8. **Probe subtree.** This is the largest and riskiest step. Recommended
   internal order:
   1. Move `probe/util.rs` (leaf utilities: `strip_probe_*`,
      `find_probe_*`, character predicates). **Compile.**
   2. Move `probe/temporal.rs` (leaf). **Compile.**
   3. Move `probe/scan.rs` (depends on util + temporal). **Compile.**
   4. Move `probe/representative.rs` (depends on scan/util). **Compile.**
   5. Move `probe/state_seed.rs`. **Compile.**
   6. Move `probe/action_eval.rs`. This is the biggest cluster
      (~25 mutually recursive probe_* fns) — keep them together.
      **Compile.**
   7. Move `probe/mod.rs` re-exports.
9. **Move `analyze_tla` handler** — now that probe is a clean
   sub-tree, the handler simply calls `cli::probe::*`. **Compile +
   run all tests.**
10. **Move `mod tests`** to `src/cli/probe/tests.rs`. The tests use
    `use super::*` today; they'll need
    `use super::{state_seed::*, representative::*, action_eval::*,
    scan::*, temporal::*, util::*};` plus the existing
    `use tlaplusplus::tla::*` lines. Run `cargo test --release` and
    confirm all 90 tests in this module still discover and pass.
11. **Final gates** (all on the Linux runner):
    - `cargo build --release`
    - `cargo build --release --features failpoints`
    - `cargo build --release --features symbolic-init`
    - `cargo test --release` (756 tests)
    - `cargo test --release --features failpoints -- --test-threads=2` (776)
    - `cargo test --release --features symbolic-init` (774)
    - `scripts/diff_tlc.sh` (13/13)
    - `scripts/chaos_smoke.sh`
12. Commit `refactor: split src/main.rs (11,711 lines) into src/cli/ modules`.

Each numbered step is one commit. The branch should have ~12 commits;
revert is per-step.

## Risks and coupling (the blocking issues)

### 1. Test-module reach

The 90 unit tests at `src/main.rs:6095-11711` (~5,616 LOC, 48% of the
file) all do `use super::*` and reach into private symbols. A grep of
the test module names suggests they cover: probe-machinery
(`representative_value_*`, `sample_param_value_*`, `probe_action_*`,
`infer_action_param_samples_*`), constraint-collection
(`collect_type_invariant_constraints`, `representative_value_from_*`),
and parser helpers (`parse_probe_action_call`, `parse_probe_action_if`,
`split_probe_quantifier`).

After the split each tested symbol must be `pub(super)` (visible from
`probe/tests.rs`). This is a wide surface. It is *correct* for them to
stay `pub(super)` — they are tests of probe internals — but accidental
widening to `pub(crate)` or `pub` would leak internals to the rest of
`tlaplusplus`. **Mitigation**: after step 10, run
`cargo doc --no-deps --document-private-items` and grep for
`probe::`-prefixed items in the public surface; assert nothing escapes
`pub(super)`.

### 2. Probe-fn intra-cycles

The `probe_*` family has mutual recursion that crosses my proposed
file boundaries. Examples found in the survey:

- `probe_action_body_into_ctx` ↔ `probe_action_body_via_runtime_eval`
  ↔ `probe_disjunctive_action_body` ↔ `probe_action_clause_expr`
- `body_contains_probeable_action_call` ↔
  `body_contains_contextual_probeable_action_call` ↔
  `definition_is_contextually_probeable_action`
- `representative_value_from_set_expr` calls
  `representative_value_from_definition_body` which calls back into
  representative-from-set logic via `try_create_representative_record`.

The proposed `probe/{state_seed,representative,action_eval}.rs` split
keeps each cluster intact, but `probe/scan.rs` reaches into all three.
A safer alternative: collapse `probe/*.rs` into one `probe.rs`
(~3,000 LOC) and only split tests off. This still gets `src/main.rs`
down to ~3,400 lines (handlers + probe), which is the bulk of the
benefit. **The follow-up branch should default to the
single-`probe.rs` shape unless step 8.6 reveals a clean cut.**

### 3. `RunAdaptiveBranching` monitor-thread closure

The `RunAdaptiveBranching` arm spawns a background monitor thread that
captures `model.clone()` and the `Arc<AtomicBool>` done flag. Lifting
this into `run_adaptive_branching::handle()` is mechanical, but the
thread must be joined inside the handler (current code joins before
printing stats). Preserve that order; `done.store(true)` then `join()`
then `print_stats(...)`.

### 4. `#[cfg(feature = "failpoints")]` scenario lifetime

Today:

```rust
fn main() -> anyhow::Result<()> {
    #[cfg(feature = "failpoints")]
    let _failpoint_scenario = fail::FailScenario::setup();
    ...
}
```

After the refactor `_failpoint_scenario` must outlive the dispatcher.
Move the binding into `cli::run()`, NOT into `main.rs` and NOT into a
helper that drops it on return. The `_` prefix is required to suppress
unused-binding warnings in non-failpoints builds (without it the
`#[cfg]` gate would fire an `unused_variables` lint).

### 5. `tlaplusplus::tla::tla_state` test-only import

`src/main.rs:15-16` carries
`#[cfg(test)] use tlaplusplus::tla::tla_state;` — this `use` lives in
the file scope, not in `mod tests`. When tests move to
`src/cli/probe/tests.rs`, the `#[cfg(test)]` import must move with
them, or the new file scope will still need it for compile in test
builds.

### 6. The two-phase parse pattern

`Command::AnalyzeTla` calls `parse_tla_module_file(&module)?` then
`scan_module_closure(&module)?` then `TlaModel::from_files(...).ok()`,
and uses the *original* parsed module if `from_files` fails. This
fall-back is load-bearing for the analysis output (analyze must still
emit something useful when the model can't be built). When moving to
`analyze_tla::handle`, the early-return-on-error pattern that's
idiomatic in handlers must NOT be applied here — keep the
`.ok()` + conditional re-assignment exactly as written.

### 7. Differential-vs-TLC gate

`scripts/diff_tlc.sh` greps stdout for exact strings emitted by
`print_stats` and the violation block (`violation=true|false`,
`violation_message=...`, `violation_state=...`). Any drift in print
formatting fails the gate. After step 3, before any handler moves, do
a no-op golden run (one `run-counter-grid` invocation, capture
stdout) and diff against the same invocation on `main`. Zero diff
required.

### 8. `--features symbolic-init` build

This feature flag wires Z3 into `src/tla/`. It does not touch
`src/main.rs` directly today, but adding `pub mod cli;` to `lib.rs`
means the cli code is compiled under all feature combinations. Step 11
must include the `--features symbolic-init` build and test variants.

## Why Path B for this attempt

Three blockers, in order of severity:

1. **No cargo on this machine.** Project rules: "All cargo on a
   c8g.metal-24xl spot — NEVER on this Mac." Without per-step
   `cargo check`, the recommended landing order
   ([above](#recommended-landing-order-for-the-follow-up-branch))
   collapses — every step has to be moved blind, and a single
   `pub(super)` vs `pub(crate)` mistake in step 8 (probe subtree)
   would silently break the test-module visibility added in step 10.
   The 8-hour budget cannot absorb a debug cycle that requires
   re-uploading the worktree to a spot.

2. **Differential gate cannot run here.** `scripts/diff_tlc.sh`
   needs the JVM TLC (`tools/tla2tools.jar`) and the 13 curated
   specs to produce reference outputs. Behavior-identical claims
   without this gate are unverifiable.

3. **The probe sub-tree is one decision, not a mechanical move.**
   Choosing between `probe.rs` (single file, ~3,000 LOC) and
   `probe/{...}.rs` (six files) requires either (a) a half-day
   mapping the call graph or (b) iterating with the compiler. Without
   the compiler, (a) is the only option, and (a) needs the test gate
   to validate. The follow-up branch with a Linux runner can do this
   in 2-3 hours.

The refactor itself is **straightforward and safe** once a Linux
runner is in the loop. The blocker is purely environmental, not
architectural. This document is intended to make the follow-up
branch a one-sitting job.

## Follow-up TASKs spotted during survey

(Per the working rules, real bugs noticed during refactor are filed
as TASKs, not fixed inline.)

- **TASK-MAIN-1**: `Command::RunAdaptiveBranching` discards `s3`
  args with a `_` binding (line 1438) and a comment "S3 not yet
  integrated with adaptive branching". Either wire S3 in or remove
  the `s3: S3Args` field from this variant. (Refactor-neutral.)
- **TASK-MAIN-2**: `fn main()` is 1,275 lines — well past the point
  where rust-analyzer goto-definition and IDE rename become slow.
  This refactor fixes that.
- **TASK-MAIN-3**: `#[cfg(test)] use tlaplusplus::tla::tla_state;`
  at file scope (line 15-16) is dead in the current `main.rs` — no
  reference outside the test module. After step 10 it lives only in
  `probe/tests.rs`. Confirm and remove if unused.
- **TASK-MAIN-4**: Several `print_stats`/violation-print blocks are
  duplicated verbatim across the four synthetic-model arms
  (RunCounterGrid, RunFlurmLifecycle, RunHighBranching,
  RunAdaptiveBranching). After the split, each handler can call a
  shared `shared::print_outcome(name, &outcome)` helper. Keep this
  out of the refactor PR; file as a follow-up.
