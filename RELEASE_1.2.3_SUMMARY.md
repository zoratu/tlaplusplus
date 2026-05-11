# tlaplusplus v1.2.3

T10.2 phase 2 stage 3. The hot-loop DFS exploration that was the headline parked item across v1.2.1 and v1.2.2 lands via a scoped third approach.

## T10.2 phase 2 stage 3 — DFS exploration via separate worker

Two prior agents stopped at the same surface for the same reason: lifting the BFS exploration in `src/runtime/worker.rs::run_worker` is a multi-day risky rewrite. The function captures 27 distinct `Arc<...>` clones and threads T5.4/T6/T11.5 ordering invariants throughout. This release uses a different approach that sidesteps the universal lift.

A new file `src/runtime/dfs_worker.rs` implements a separate single-worker single-node DFS exploration function with its own ctx struct. It drops features irrelevant to streaming-SCC mode: no checkpoint pause, no cluster steal, no init producer streaming, no distributed donate, no auto-tune throttle, no backpressure.

A dispatch branch in `src/runtime.rs` checks `--liveness-streaming-exploration`, `model.has_fairness_constraints()`, and the absence of a distributed stealer. When all three hold, the run path spawns one `dfs_worker::run_dfs_worker` instead of the normal worker fleet. The init producer is conditionally skipped to avoid double-counting `states_distinct`.

`src/runtime/worker.rs` is byte-unchanged. The default code path is identical to v1.2.2; the new code path activates only when the flag is on.

The DFS worker uses in-band Tarjan-style coloring via the `PageAlignedColorMap` shipped in v1.2.1. Per-DFS-frame stack with `BlueFrame { state, successor_iter, color_owned }`. The nested-DFS red probe fires when an accepting state's frame pops (Gray → Black).

## Liveness-verdict parity (BFS vs DFS)

| Spec | states_distinct (BFS=DFS) | violation (BFS=DFS) | Notes |
|---|---|---|---|
| WrapperNextFairness (passing) | 4 | None | |
| NamedSubaction (NeverFires) | 2 | Liveness | "NeverFires" present |
| ThreeCycle (Idle WF) | 3 | Liveness | "Idle" present |
| SafetyOnly (no fairness) | identical | identical | DFS dispatch correctly skipped via `has_fairness_constraints` gate |

The v1.2.2 `streaming_scc_exploration_parity` suite (3 tests) re-validates under the new code path with no DIVERGENCE.

## Memory benchmark — architecture-only in this stage

The design doc's predicted >50% RSS reduction at 100M+ scale is not realized in this stage. On a 9,261-state counter spec at `--max=20`:

| Config | RSS | Wall |
|---|---|---|
| BFS, --fp-expected-items 100K | 189 MB | 1.11 s |
| DFS (flag on), 100K | 193 MB | 1.10 s |
| BFS, 100M (default) | 1729 MB | 1.67 s |
| DFS, 100M | 1751 MB | 1.67 s |

The DFS path adds ~3-22 MB for the page-aligned color map (proportional to `--fp-expected-items`). The actual memory win requires Stage 4 (drop the labeled-transitions adjacency-map materialization), which depends on this stage's architecture but is a separate piece of work.

What this stage delivers is the architecture: a separate function, color-map coloring in-band, single-worker dispatch, and parity-verified correctness. Stage 4 (v1.2.4) drops `labeled_transitions` and realizes the memory win without touching the BFS path.

## Validation gates

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

Drop-in for v1.2.0/v1.2.1/v1.2.2. No public-API changes; the `--liveness-streaming-exploration` CLI flag exists since v1.2.2 and defaults OFF.

## Code-organization deltas vs v1.2.2

New `src/runtime/dfs_worker.rs` with 3 unit tests. New `tests/dfs_worker_parity.rs` with 4 integration tests. `src/runtime.rs` adds the dispatch branch and producer-skip. `src/cli/args.rs` adds a docstring update for the flag. Tests at 1,212 (was 1,205). Verus proofs unchanged at 133 verified items.

## Still parked

- T10.2 phase 2 stages 4 + 5 — Stage 4 drops `labeled_transitions` and proves the memory win; Stage 5 wires multi-worker DFS via cross-partition routing.
- T13.4 Phases 2+3 — production-code Verus rewrite of `FingerprintShard`. Blocked by three documented `vstd` capability gaps.
