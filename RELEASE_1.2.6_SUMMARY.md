# tlaplusplus v1.2.6

T10.2 phase 2 stage 5, Layer B. The single-node multi-worker DFS pool shipped in v1.2.5 is extended to span multiple cluster nodes via cross-node routing.

## T10.2 phase 2 stage 5 Layer B — multi-node cluster DFS

A new `src/runtime/dfs_cluster_bridge.rs` provides the async-to-sync bridge between `Transport::recv()` (async, boxed-future return) and the per-worker crossbeam mpsc inboxes (sync). The bridge thread drains incoming `PartitionEdge` messages and forwards them into the appropriate worker's `inbox_tx`. Outbound cross-node sends use fire-and-forget `Handle::spawn`; the inflight counter guarantees correctness regardless of when the network catches up.

`src/runtime/dfs_pool.rs` gains a `DfsPoolClusterCtx` and cluster-aware partition routing. Each worker's partition is `(node_id, worker_id)` under the existing `partition_for_fp` bit-mixing — local partitions stay on the local crossbeam mpsc, remote partitions ship over the `Transport` as `PartitionEdge` messages.

Termination uses a two-round consensus over the existing `TerminationToken.inflight_partition_edges: Option<u64>` field (shipped in stage 2, v1.2.1, with `#[serde(default)]` so existing tokens still decode). A node declares idle only when its local Mattern condition holds (`inflight == 0 && all_idle && my_inbox_empty`) AND it has received a confirmed token from each peer showing the same. The verdict comes from per-node in-band Tarjan over the merged `LocalAdjacency` triples — same predicate as the v1.2.4 single-worker path and the v1.2.5 single-node pool.

## Validation

`tests/dfs_cluster_layer_b.rs` (4 default tests + 1 ignored memory benchmark) uses the existing `MockTransport` (T204, v1.2.1) to wire two transport instances through a shared `MockNetwork`. No real TCP. The tests assert identical liveness verdicts and identical `states_distinct` counts when the same fairness spec is run under the single-node pool vs the 2-node cluster.

Per-node cluster memory at dim=8 (64 states): single-node DFS RSS delta 0.6 MiB, 2-node cluster RSS delta 0.8 MiB total → 0.4 MiB per node = 0.64× single-node baseline.

Throughput at dim=6 (36 states) is dominated by tokio runtime startup and bridge spin-up; meaningful wall-time deltas would need a bigger benchmark. The deliverable for Layer B is verdict equivalence under cluster topology, not throughput.

## Deliberate carve-outs

The following are documented as non-blocking for Layer B's verdict-equivalence deliverable:

- **Cross-partition red-DFS send path** stays log-only. The receive arms for `RedDfsProbe` reply `NotFound` to keep the wire shape stable. The DFS pool's verdict comes from the post-exploration in-band Tarjan over the merged per-node `LocalAdjacency` triples, not the in-loop red probe. Cross-partition red-DFS would only matter if Layer B switched to streaming SCC discovery, which is a separate roadmap item.
- **`RequestStateBlob` send-side** stays a no-op (receive arms reply with empty blobs). Trace reconstruction across partition boundaries is deferred. The single-coordinator pattern keeps trace reconstruction local to whichever node owns the violating SCC.
- **`--dfs-cluster-listen` CLI flag wiring** is deferred. Shipping multi-node DFS pool runs go through `dfs_cluster_test_api`; threading the bridge through `EngineConfig` + `cli/args.rs` + `cli/shared.rs` is a separate flag-day.
- **Per-node fingerprint stores share no state with peers.** Layer B cluster-DFS mode and the existing T6 independent-exploration cluster mode are mutually exclusive runtimes that share the protocol but not the bridge.

## Validation gates

| Gate | Result |
|---|---|
| `cargo test --release` | 1,227 pass / 0 fail / 11 ignored |
| `cargo test --release --features failpoints` | 1,249 pass / 0 fail / 11 ignored |
| `cargo test --release --features symbolic-init` | 1,254 pass / 0 fail / 11 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `streaming_scc_oracle` | 2 / 2 (no DIVERGENCE) |
| `streaming_scc_exploration_parity` | 3 / 3 |
| `dfs_worker_parity` | 4 / 4 |
| `dfs_pool_parity` | 3 / 3 |
| `cross_node_steal_handshake` (T6 cluster mode unaffected) | 3 / 3 |
| `distributed_handler_with_mock` | 5 / 5 |
| `dfs_cluster_layer_b` (NEW) | 4 / 4 (+ 1 ignored memory benchmark, 0.64× per-node ratio ✓) |
| Verus across 6 proof files | 133 verified items, 0 errors, 0 axioms |

## Compatibility

Drop-in for v1.2.0–v1.2.5. No public-API or CLI changes. Shipping multi-node DFS pool runs go through `dfs_cluster_test_api`; the `--dfs-cluster-listen` user-facing flag is queued for a later release.

## Code-organization deltas vs v1.2.5

New `src/runtime/dfs_cluster_bridge.rs` with 3 unit tests. New `tests/dfs_cluster_layer_b.rs` with 5 tests. `src/runtime.rs` adds a `pub mod dfs_cluster_test_api` and the bridge module declaration. `src/runtime/dfs_pool.rs` gains the cluster ctx, cluster-aware partition routing, and bridge-aware termination predicate. Tests at 1,227 (was 1,220). Verus proofs unchanged at 133 verified items.

## Still parked

- T13.4 Phases 2+3 — shipping-code Verus rewrite of `FingerprintShard`, blocked by three documented `vstd` capability gaps.
