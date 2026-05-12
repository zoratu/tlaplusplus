# tlaplusplus v1.2.4

T10.2 phase 2 stage 4. The streaming-SCC memory win is realized: the DFS exploration path no longer materializes the labeled-transitions adjacency map.

## T10.2 phase 2 stage 4 — drop labeled_transitions, in-band fairness verdict

`DfsWorkerCtx` no longer carries `Option<Arc<DashMap<u64, Vec<LabeledTransition<S>>>>>`. The DFS worker now builds a thin local `LocalAdjacency = Vec<(u64, u64, String)>` (source fingerprint, destination fingerprint, action name) — no state clones.

A new `run_inband_fairness_check` runs Tarjan + the existing `check_fairness_on_scc_fp_sharded` predicate on the local triple list when DFS exploration completes. The verdict equivalence with the post-processing path holds by construction since both paths use the same predicate. Constraint-failing edges are still recorded (matches the BFS ordering at `worker.rs:495-506`: append-then-filter); descent skips them.

A new `dfs_inband_verdict_done: Arc<AtomicBool>` is shared between the DFS worker and `runtime/shutdown.rs`. When set, `liveness::run_post_processing` is skipped entirely; the run summary prints `Liveness post-processing: skipped — DFS worker already produced in-band fairness verdict (T10.2 stage 4)`.

The dispatch in `runtime.rs` drops the `labeled_transitions` `DashMap` immediately when DFS path fires; the run summary prints `T10.2 stage 4 DFS dispatch ... labeled-transitions=DROPPED (in-band verdict)`.

## Memory benchmark

A new `tests/dfs_memory_benchmark.rs` (gate-8, `#[ignore]` by default) runs a synthetic `HighFanoutGrid` spec with `WF_vars(Move)` constraint: 600×600 grid, 360,000 reachable states, 1,797,600 transitions, ~5 successors per state plus self-loop on every state forming non-trivial SCCs. Result on aarch64 spot:

| Run | Peak RSS | Wall | Δ vs baseline |
|---|---|---|---|
| BFS | 877 MiB | 3 s | +876.8 MiB |
| DFS | 513 MiB | 24 s | +513.2 MiB |

DFS / BFS = 0.59, a 41.5% reduction. DFS is single-worker by Stage 4 design and is slower than the BFS fleet on this spec; the deliverable is the memory ratio, not throughput. Stage 5 lifts single-worker DFS to multi-worker.

## Edge cases handled

BFS records labeled transitions before the constraint filter (`worker.rs:495-506`). The DFS path's `LocalAdjacency` does the same so verdicts stay identical: `compute_labeled_successors` returns `(state, action_name, passes_constraints)` tuples; descent skips `!passes_constraints` entries but the edge is still recorded.

Stage 3's in-band red probe is exercise-only in Stage 4 — it touches the page-aligned color-map CAS path on every accepting-state pop but the verdict comes from the post-exploration Tarjan pass. The probe is bounded to 4096 iterations to keep its cost amortized on large graphs.

The benchmark tunes down `fp_expected_items`, `queue_compression`, and `fp_cache_capacity_bytes` so the measured delta reflects the actual `labeled_transitions` vs `LocalAdjacency` difference rather than upfront allocator overhead.

## Cross-partition routing — deferred to v1.2.5

Stage 4 also called for cross-partition routing infrastructure for future multi-worker DFS. The protocol variants (`PartitionEdge`/`PartitionEdgeAck`/`RedDfsProbe`/`RedDfsResponse`) already exist in `src/distributed/protocol.rs` with log-and-drop handler stubs (Stage 2 deliverable, v1.2.1). Wiring up multi-shard dispatch is independent of the labeled_transitions drop; single-worker DFS already delivers the memory win documented above.

## Validation gates

| Gate | Result |
|---|---|
| `cargo test --release` | 1,212 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,234 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,239 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `streaming_scc_oracle` | 2 / 2 (no DIVERGENCE) |
| `streaming_scc_exploration_parity` | 3 / 3 |
| `dfs_worker_parity` (verdict equivalence after lt drop) | 4 / 4 |
| `wrapper_next_fairness_t1_3` | 2 / 2 |
| `dfs_memory_benchmark` (NEW) | DFS / BFS = 0.59 ✓ (≤ 0.7 threshold) |
| Verus across 6 proof files | 133 verified items, 0 errors, 0 axioms |

## Compatibility

Drop-in for v1.2.0–v1.2.3. No public-API changes. The `--liveness-streaming-exploration` flag exists since v1.2.2 and defaults OFF.

## Code-organization deltas vs v1.2.3

`src/runtime/dfs_worker.rs` rewritten internally. `src/runtime.rs` gets the dfs_inband_verdict_done wiring. `src/runtime/shutdown.rs` gets the post-processing skip. New `tests/dfs_memory_benchmark.rs`. Tests at 1,212 (memory benchmark is `#[ignore]` by default; run via `cargo test --release --test dfs_memory_benchmark -- --ignored`). Verus proofs unchanged at 133 verified items.

## Still parked

- T10.2 phase 2 stage 5 — multi-worker DFS via cross-partition routing.
- T13.4 Phases 2+3 — shipping-code Verus rewrite of `FingerprintShard`. Blocked by three documented `vstd` capability gaps.
