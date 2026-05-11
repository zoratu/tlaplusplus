# tlaplusplus v1.2.4

Patch release landing T10.2 phase 2 stage 4 — the streaming-SCC
memory win is finally realized.

## Highlights

### T10.2 phase 2 stage 4 — drop labeled_transitions, in-band fairness verdict

v1.2.3 landed the architecture (separate `dfs_worker.rs`, in-band
coloring) but honestly noted the predicted memory win wasn't
realized: the DFS path still populated `labeled_transitions` so the
post-processing Tarjan oracle could run on it. This release lifts
that.

- **`labeled_transitions` dropped on the DFS path.** `DfsWorkerCtx`
  no longer carries the `Option<Arc<DashMap<u64, Vec<LabeledTransition<S>>>>>`
  field. The DFS worker builds a thin local
  `LocalAdjacency = Vec<(u64, u64, String)>` (fingerprints + action
  name only — **no state clones**).
- **In-band fairness verdict.** New `run_inband_fairness_check`
  function runs Tarjan + the existing
  `check_fairness_on_scc_fp_sharded` predicate on the local triple
  list directly when DFS exploration completes. Verdict equivalence
  with the post-processing path holds by construction (same
  predicate). Constraint-failing edges still recorded (matches BFS
  `worker.rs:495-506` ordering: append-then-filter), only descent
  skips them.
- **Post-processing skip.** New `dfs_inband_verdict_done: Arc<AtomicBool>`
  is shared between the DFS worker and `runtime/shutdown.rs`. When
  set, `liveness::run_post_processing` is skipped entirely. Banner:
  `Liveness post-processing: skipped — DFS worker already produced
  in-band fairness verdict (T10.2 stage 4)`.
- **`runtime.rs` dispatch**: when DFS path fires, the
  `labeled_transitions` DashMap is dropped immediately (never
  populated by anyone). Banner: `T10.2 stage 4 DFS dispatch ...
  labeled-transitions=DROPPED (in-band verdict)`.

### Memory benchmark

New `tests/dfs_memory_benchmark.rs` (gate-8). Synthetic
`HighFanoutGrid` spec with `WF_vars(Move)` constraint:
- 600×600 grid → **360,000 reachable states**, **1,797,600 transitions**
- ~5 successors per state + self-loop (non-trivial SCCs)

Result on aarch64 spot:

| Run | Peak RSS | Wall | Δ vs baseline |
|---|---|---|---|
| BFS | **877 MiB** | 3 s | +876.8 MiB |
| DFS | **513 MiB** | 24 s | +513.2 MiB |
| **Ratio** | **DFS / BFS = 0.59** | | **41.5% reduction** |

DFS is slower (single-worker vs the BFS fleet) — by Stage 4 design.
The deliverable is the memory win, not throughput.

### Cross-partition routing — deferred

Stage 4 also calls for cross-partition routing infrastructure for
future multi-worker DFS. The protocol variants
(`PartitionEdge`/`PartitionEdgeAck`/`RedDfsProbe`/`RedDfsResponse`)
already exist in `src/distributed/protocol.rs` with log-and-drop
handler stubs (Stage 2 deliverable shipped in v1.2.1). Wiring up
multi-shard dispatch is independent of the labeled_transitions drop;
single-worker DFS already delivers the memory win documented above.

### Edge cases handled

- BFS records labeled transitions **before** the constraint filter
  (`worker.rs:495-506`); the DFS path's `LocalAdjacency` does the
  same to keep verdicts identical. `compute_labeled_successors`
  returns `(state, action_name, passes_constraints)` tuples; descent
  skips `!passes_constraints` entries but the edge is still recorded.
- Stage 3's in-band red probe is exercise-only in Stage 4 (touches
  the page-aligned color-map CAS path on every accepting-state pop
  but the verdict comes from the post-exploration Tarjan pass);
  bounded to 4096 iterations to keep its cost amortized on large
  graphs.
- Benchmark tunes down `fp_expected_items`, `queue_compression`, and
  `fp_cache_capacity_bytes` so the measured delta reflects the
  actual `labeled_transitions` vs `LocalAdjacency` difference rather
  than upfront allocator overhead.

## Validation gates

All green on the v1.2.4 tag:

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
| `dfs_memory_benchmark` (NEW) | DFS RSS 513 / BFS RSS 877 = **0.59** ✓ (≤ 0.7 threshold) |
| Verus across 6 proof files | 133 verified items, 0 errors, 0 axioms |

## Compatibility

Drop-in for v1.2.0–v1.2.3. No public-API changes. The
`--liveness-streaming-exploration` flag exists since v1.2.2 and
defaults OFF.

## Code-organization deltas vs v1.2.3

- `src/runtime/dfs_worker.rs`: 941 → 1,450 LOC (rewrite of internal
  successor / coloring path; +509 / -381 net)
- `src/runtime.rs`: +60 LOC (DfsInbandVerdictDone wiring)
- `src/runtime/shutdown.rs`: +24 LOC (skip post-processing when DFS
  verdict already in hand)
- New `tests/dfs_memory_benchmark.rs` (394 LOC)
- Tests: 1,212 → 1,212 (memory benchmark is `#[ignore]` by default;
  run via `cargo test --release --test dfs_memory_benchmark -- --ignored`)
- Verus proofs unchanged at 133 verified items

## Still parked

- **T10.2 phase 2 stage 5** — multi-worker DFS via cross-partition
  routing (single-node, then multi-node). Independent of this
  release; stage 4 already delivers the memory win for single-worker
  fairness checking.
- **T13.4 Phases 2+3** — production-code Verus rewrite of
  `FingerprintShard`. Blocked by three documented `vstd` capability
  gaps; needs upstream work, not application-side iteration.
