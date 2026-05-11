# tlaplusplus v1.2.3

Patch release landing T10.2 phase 2 stage 3 — the hot-loop DFS lift
that was the headline parked item across v1.2.1 / v1.2.2.

## Highlights

### T10.2 phase 2 stage 3 — DFS exploration via separate worker (SHIPPED)

Two prior agents had stopped at the same surface for the same reason:
lifting the BFS exploration in `src/runtime/worker.rs::run_worker` (733
LOC, 27-Arc capture, T5.4/T6/T11.5 ordering invariants) is a 5+ day
risky rewrite. This release uses a third approach that sidesteps the
universal lift entirely:

- **New file `src/runtime/dfs_worker.rs`** (941 LOC) — a SEPARATE
  single-worker single-node DFS exploration function with its own
  ctx struct. Drops features irrelevant to streaming-SCC mode: no
  checkpoint pause, no cluster steal, no init producer, no
  distributed donate, no auto-tune throttle, no backpressure.
- **`src/runtime.rs` dispatch branch** (+199 LOC) — when
  `--liveness-streaming-exploration` is on AND
  `model.has_fairness_constraints()` AND no distributed stealer is
  attached, the run path spawns ONE `dfs_worker::run_dfs_worker`
  instead of the normal worker fleet. Init producer conditionally
  skipped to avoid double-counting `states_distinct`.
- **`src/runtime/worker.rs` is byte-unchanged.** Default behaviour is
  identical to v1.2.2 — the new code path activates only when the
  flag is on.
- **In-band Tarjan-style coloring** via `PageAlignedColorMap` (the
  Stage 1 deliverable shipped in v1.2.1). Per-DFS-frame stack with
  `BlueFrame { state, successor_iter, color_owned }`. Nested-DFS red
  probe fires when an accepting state's frame pops (Gray → Black).

#### Liveness-verdict parity (BFS vs DFS)

| Spec | states_distinct (BFS=DFS) | violation (BFS=DFS) | Notes |
|---|---|---|---|
| WrapperNextFairness (passing) | 4 | None | |
| NamedSubaction (NeverFires) | 2 | Liveness | "NeverFires" present |
| ThreeCycle (Idle WF) | 3 | Liveness | "Idle" present |
| SafetyOnly (no fairness) | identical | identical | DFS dispatch correctly skipped via `has_fairness_constraints` gate |

Plus the v1.2.2 `streaming_scc_exploration_parity` suite (3 tests)
re-validated under the new code path with no DIVERGENCE.

#### Memory benchmark

Honest finding: the design doc's predicted >50% RSS reduction at
100M+ scale is **NOT** realized in this stage. On a 9,261-state
counter spec at `--max=20`:

| Config | RSS | Wall |
|---|---|---|
| BFS, --fp-expected-items 100K | 189 MB | 1.11s |
| DFS (flag on), 100K | 193 MB | 1.10s |
| BFS, 100M (default) | 1729 MB | 1.67s |
| DFS, 100M | 1751 MB | 1.67s |

The DFS path adds ~3-22 MB for the page-aligned color map
(proportional to `--fp-expected-items`). The actual memory win
requires Stage 4 (drop the labeled-transitions adjacency-map
materialization), which depends on this stage's architecture but is a
separate piece of work.

**What this stage delivers**: the architecture (separate function,
color-map coloring in-band, single-worker dispatch, parity-verified
correctness) on which Stage 4 can drop labeled_transitions without
touching the BFS path.

## Validation gates

All green on the v1.2.3 tag:

| Gate | Result |
|---|---|
| `cargo test --release` | 1,212 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,234 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,239 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `cargo test --release --test compiled_vs_interpreted` (PROPTEST_CASES=2048) | 17 pass × 3 seeds |
| `cargo test --release --test streaming_scc_oracle` | 2 / 2 (no DIVERGENCE) |
| `cargo test --release --test streaming_scc_exploration_parity` | 3 / 3 |
| `cargo test --release --test dfs_worker_parity` (NEW) | 4 / 4 |
| `cargo test --release --test wrapper_next_fairness_t1_3` | 2 / 2 |
| Verus across 6 proof files | 133 verified items, 0 errors, 0 axioms |

## Compatibility

Drop-in for v1.2.0 / v1.2.1 / v1.2.2. No public-API changes; the
`--liveness-streaming-exploration` CLI flag exists since v1.2.2 and
defaults OFF.

## Code-organization deltas vs v1.2.2

- New `src/runtime/dfs_worker.rs` (941 LOC, 3 unit tests)
- New `tests/dfs_worker_parity.rs` (4 integration tests)
- `src/runtime.rs`: +199 LOC dispatch branch + producer-skip
- `src/cli/args.rs`: +25 LOC docstring update for the flag
- Tests: 1,205 → 1,212 (+7)
- Verus proofs unchanged at 133 verified items

## Still parked

- **T10.2 phase 2 stages 4 + 5** — Stage 4 (drop labeled_transitions
  + memory benchmark proving > 50% RSS reduction) and Stage 5
  (cluster-mode wiring of `RedDfsProbe`/`RedDfsResponse` + corpus
  revalidation) build on the architecture this release lands. Both
  remain multi-day per the design doc.
- **T13.4 Phases 2+3** — production-code Verus rewrite of
  `FingerprintShard`. Blocked by three documented `vstd` capability
  gaps; needs upstream work, not application-side iteration.
