# tlaplusplus v1.2.5

T10.2 phase 2 stage 5, Layer A. The single-worker DFS exploration shipped in v1.2.3 / v1.2.4 is lifted to a fingerprint-partitioned worker pool with cross-partition routing.

## T10.2 phase 2 stage 5 Layer A — single-node multi-worker DFS pool

A new `src/runtime/dfs_pool.rs` implements an N-worker DFS pool that shares the v1.2.4 in-band fairness verdict path. Each worker owns the partition `partition_for_fp(fp, N) == self.id` (same bit-mixing as `home_numa`). Per-worker DFS stack, per-worker `LocalAdjacency` triples, per-worker `state_by_fp` cache, per-worker crossbeam mpsc `Receiver`. Cross-partition successors are shipped over `outbox_tx[owner]`; local successors stay local.

Termination uses an `AtomicI64` Mattern in-flight counter incremented on each cross-partition send and decremented as the first action in `handle_explore_msg` on receive. Termination requires `inflight == 0 && all_idle && my_inbox_empty` confirmed twice (Mattern two-round). After the worker join, per-worker triples are merged and the same in-band fairness check from stage 4 (`dfs_worker::run_inband_fairness_check`) runs once over the union, so verdict equivalence with the stage-4 single-worker path holds by construction.

A new `--dfs-workers` flag controls pool size (1 selects the stage-4 single-worker path).

## Throughput

`tests/dfs_pool_throughput_benchmark.rs` runs the existing `HighFanoutGrid` synthetic spec at dim=250 (62,500 reachable states, ~5 successors per state, 311,500 edges) on the EC2 spot:

| Workers | Wall | Peak RSS Δ |
|---|---|---|
| 1 | 6.11 s | 326.2 MiB |
| 4 | 2.10 s | 346.6 MiB |

Speedup 2.91×, memory ratio 1.06× (pool's per-worker scratch + crossbeam channel buffers add only ~6% to peak RSS at this scale). Stage 4's BFS-vs-DFS memory win carries through: `dfs_memory_benchmark` gate-7 still shows DFS 511.9 MiB vs BFS 897.6 MiB = 43.0% reduction.

## Verdict equivalence

`tests/dfs_pool_parity.rs` runs the same fairness specs at pool sizes 1 / 2 / 4 and asserts identical liveness verdicts and identical `states_distinct` counts.

## Layer B — multi-node cluster DFS — deferred to v1.2.6

The protocol variants (`PartitionEdge`, `PartitionEdgeAck`, `RedDfsProbe`, `RedDfsResponse`) and log-and-drop handler arms shipped in stage 2 (v1.2.1) and remain in `src/distributed/handler.rs` as no-ops. Wiring them into the pool requires node-aware partitioning (`partition_for_fp(fp, num_nodes * dfs_workers_per_node)` returning `(node_id, worker_id)`), a tokio-async-to-sync-thread bridge between `Transport::recv()` and the worker's `inbox_rx`, extending `TerminationToken { inflight_partition_edges: Option<u64> }` into a two-round consensus that accumulates per-node in-flight counts, and a 2-node integration test rig. Layer A delivers the headline single-node speedup on its own.

## Validation gates

| Gate | Result |
|---|---|
| `cargo test --release` | 1,220 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,242 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,247 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `streaming_scc_oracle` | 2 / 2 (no DIVERGENCE) |
| `streaming_scc_exploration_parity` | 3 / 3 |
| `dfs_worker_parity` (verdict equivalence) | 4 / 4 |
| `dfs_memory_benchmark` (NEW) | DFS 511.9 MiB / BFS 897.6 MiB = 0.57 ✓ |
| `dfs_pool_parity` (NEW) | 3 / 3 across pool sizes 1 / 2 / 4 |
| Verus across 6 proof files | 133 verified items, 0 errors, 0 axioms |

## Compatibility

Drop-in for v1.2.0–v1.2.4. No public-API changes. The new `--dfs-workers` flag is additive and defaults to 1 (stage-4 single-worker behavior).

## Code-organization deltas vs v1.2.4

New `src/runtime/dfs_pool.rs`. New `tests/dfs_pool_parity.rs` with 3 tests. New `tests/dfs_pool_throughput_benchmark.rs` (ignored by default). `src/runtime/dfs_worker.rs` exposes a few helpers as `pub(super)` for the pool. `src/runtime.rs`, `src/cli/args.rs`, `src/cli/shared.rs` add the `--dfs-workers` flag plumbing. Tests at 1,220 (was 1,212). Verus proofs unchanged at 133 verified items.

## Still parked

- T10.2 phase 2 stage 5 Layer B — multi-node cluster DFS via the existing `PartitionEdge`/`RedDfsProbe` protocol variants.
- T13.4 Phases 2+3 — production-code Verus rewrite of `FingerprintShard`, blocked by three documented `vstd` capability gaps.
