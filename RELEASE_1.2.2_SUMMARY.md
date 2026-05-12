# tlaplusplus v1.2.2

Patch release continuing the long-parked items work.

## T13.5 `state_machines!` port — shipped axiom-free

LTL-native restatement of the headline `theorem_no_starvation` via Verus's `state_machine!` macro. Proves the theorem from a refinement bridge to the v1.1.0 constructive proof in `reader_liveness_v2.rs`.

A new file `verification/verus/reader_liveness_state_machine.rs` (15 verified items, ~1s wall, run via `./run_proof.sh sm`) defines `state_machine!(ShardSeq)` with three transitions (`begin_resize`, `finalize_resize`, `stutter`), two `#[invariant]` predicates, and four `#[inductive]` proofs. A refinement bridge (`theorem_step_refines`, `theorem_prefix_refines`) shows every state-machine transition refines a v2 `step_relation_v2` step. The headline `theorem_no_starvation_sm` is proved constructively with the same 2-step stutter and 3-step finalize-stutter witnesses as v2. Zero axioms.

## T13.4 — wrapper-level seqlock retry verified

A new `bounded_seqlock_retry_contains` in `verification/verus/shard_wrapper.rs` (+3 verified items, total now 34). Wrapper-level seqlock retry loop with `decreases max_retries - retries` termination — closes another piece of the gap between the shadow methods and the shipping retry loop.

T13.4 Phases 2+3 (shipping-code annotation of `FingerprintShard`) remain parked. The Phase 2 probe reconfirmed all three documented `vstd` capability gaps:

1. `unsafe { std::slice::from_raw_parts(table_ptr, capacity) }` in the contains probe loop — `table_ptr` comes from a swappable `AtomicPtr`; modeling without `external_body` isn't possible today.
2. `mmap(MAP_HUGETLB | MAP_POPULATE)` in `FingerprintShard::new` — `Tracked<PointsToArray<...>>` requires a memory-provenance axiom or an `external_body` permission mint.
3. `unsafe impl Send` with `&self` everywhere — `vstd::AtomicInvariant` workaround would add open/close per probe step, regressing the 220K-states/sec hot path.

## T13.4 + T13.5 verified item totals

| Proof file | Verified items | Wall time |
|---|---|---|
| `seqlock_resize.rs` (tier-b) | 19 | 0.96s |
| `seqlock_resize_tier_a.rs` (tier-a) | 31 | 1.29s |
| `shard_methods.rs` | 17 | 1.06s |
| `reader_liveness_v2.rs` | 17 | 1.02s |
| `shard_wrapper.rs` | 34 | 1.82s (was 31 in v1.2.1) |
| `reader_liveness_state_machine.rs` (NEW) | 15 | ~1s |
| Total | 133 (was 115) | |

## T10.2 phase 2 — stage 3 partial (data-structure integration)

Wires the `PageAlignedColorMap` data structure (Stage 1 deliverable) into the post-BFS oracle path behind a new `--liveness-streaming-exploration` flag. Validates the data structure end-to-end on real fairness specs.

`src/streaming_scc.rs` gains `nested_dfs_color_map` plus helpers and 5 unit tests. `src/runtime/liveness.rs` gains the page-aligned oracle integration. `src/runtime.rs` gains the `liveness_streaming_exploration` config field. `src/cli/args.rs` and `src/cli/shared.rs` add the new `--liveness-streaming-exploration` flag. `tests/streaming_scc_exploration_parity.rs` adds 3 parity tests (Tarjan vs color-map; observed `color_map_cycle=true tarjan_cycle=true` and `color_map_cycle=false tarjan_cycle=false` on the existing fairness fixtures).

Stages 3 (hot-loop DFS lift), 4 (cross-partition routing + memory benchmark), and 5 (cluster + corpus revalidation) remain parked. Two agents in succession have stopped at the same surface for the same reason: lifting the BFS exploration into per-worker DFS requires touching `src/runtime/worker.rs::run_worker` preserving T5.4 init-producer ordering, T6 cluster idle-flag handshake, T11.5 violation-finish ordering, plus auto-tune throttle, checkpoint pause points, stolen-state drain, donate-tx, distributed bloom skip, backpressure, and parent-map tracking. The realistic path forward is dedicated focused-session work that can sit on the spot host with a hot build cache.

## Validation gates

| Gate | Result |
|---|---|
| `cargo test --release` | 1,205 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,222 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,227 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `cargo test --release --test compiled_vs_interpreted` (PROPTEST_CASES=2048) | 17 pass × 3 seeds |
| `cargo test --release --test streaming_scc_oracle` | 2 / 2 (no DIVERGENCE) |
| `cargo test --release --test streaming_scc_exploration_parity` | 3 / 3 (NEW; no DIVERGENCE) |
| `cargo test --release --test wrapper_next_fairness_t1_3` | 2 / 2 |
| `cargo test --release --test cross_node_steal_handshake` | 3 / 3 |
| Verus across 6 proof files | 133 verified items, 0 errors, 0 axioms |

## Compatibility

Drop-in for v1.2.0 / v1.2.1. No public-API or CLI changes (the new `--liveness-streaming-exploration` flag is additive and defaults OFF).

## Code-organization deltas vs v1.2.1

1 new Verus proof file (`reader_liveness_state_machine.rs`, 15 verified items). `shard_wrapper.rs` extended with `bounded_seqlock_retry_contains` (+3 verified). 1 new src test file (`tests/streaming_scc_exploration_parity.rs`). Tests at 1,205 (was 1,197). Verus proofs at 133 (was 115).

## Still parked

- T10.2 phase 2 stage 3 (the actual hot-loop DFS lift), stages 4+5.
- T13.4 Phases 2+3.
