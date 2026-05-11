# tlaplusplus v1.2.1

Patch release closing out the v1.2.0 deferred-items list. 25 commits
on top of v1.2.0 — no new user-facing features, no behavioural change.
The release is dominated by structural refactoring and additional
verification work.

## Highlights

### v1.2.0 deferred items — all closed

The v1.2.0 release notes flagged four items as "Deferred to v1.2.x".
All four are now landed:

- **T204 distributed mock for the cluster handler** — `Transport`
  trait + `MockTransport` (in-memory tokio mpsc channels) +
  `DistributedWorkStealer` switched to `Arc<dyn Transport>`. 5 unit
  tests + 5 integration tests in `tests/distributed_handler_with_mock.rs`
  exercise the inbound handler / bloom-and-termination / steal-trigger
  paths without real TCP.
- **`src/runtime.rs` extraction chunks 7 + 8** — both extracted.
  Chunk 8 (13-step shutdown phase) → `src/runtime/shutdown.rs`
  (`ShutdownContext` struct + `orchestrate` method). Chunk 7 (worker
  spawn loop with 27-Arc capture) → `src/runtime/worker.rs`
  (`WorkerLocalState` struct). `runtime.rs` final size: **1,644 LOC**
  (down from 4,323 at v1.0.0, -62%).
- **`src/tla/eval.rs` split** — all 8 chunks (A–H) of the design doc's
  landing protocol landed. Original 11,533-line monolith → 13
  submodule files in `src/tla/eval/`: `mod.rs`, `expr.rs`, `action.rs`,
  `splitter.rs`, `operator.rs`, `set.rs`, `quantifier.rs`,
  `bracket.rs`, `postfix.rs`, `control.rs`, `transition.rs`,
  `instance.rs`, `budget.rs`. External API surface unchanged
  (`crate::tla::eval::<name>` paths still resolve via `pub(crate) use`
  re-exports).
- **Compiler internal-helper restructuring** — depth-tracking
  consolidation across `compiled_eval.rs` (148 → 4 sites) and
  `eval_operator_call` (26 → 1 site). Plus 6 more iterations of
  targeted compiler-helper unit tests (T207b through T207h, +149
  tests on top of v1.2.0's T207). Compiler mutation kill rate:
  **65.4% (v1.2.0) → 70.5% (v1.2.1)** across 11 cargo-mutants
  iterations on the v1.2.1 surface. Interpreter (`eval/`) remains at
  100%.

### Older parked items partially landed

- **T10.2 phase 2 — stages 1+2 of 5** — strictly-additive foundation
  for in-exploration streaming SCC (the actual memory win for 100M+
  liveness):
  - **Stage 1**: `PageAlignedColorMap` (524 LOC, 7 tests) — 2-bit
    Color enum, 32 slots per `AtomicU64`, 2 MB hugepage mmap with
    anonymous fallback, `set_preferred_node` per shard, lock-free CAS
    via `compare_exchange_weak`.
  - **Stage 2**: protocol variants — `PartitionEdge`/`PartitionEdgeAck`,
    `RedDfsProbe`/`RedDfsResponse`, `RequestStateBlob`/`StateBlobResponse`,
    plus `Option<u64>` extensions on `TerminationToken` (existing
    tokens still decode via `#[serde(default)]`).
  Stages 3–5 (worker DFS body, partition routing, cluster mode) remain
  parked per the design doc's 5-week effort estimate. Default
  `--liveness-streaming` flag stays OFF — phase 2 is opt-in just like
  phase 1.

- **T13.4 Verus tracked-pointer integration — Phase 1** — production-
  shape verified wrapper at `verification/verus/shard_wrapper.rs`
  (1,083 LOC, **31 verified items**). Lifts the bounded outer probe
  loop that `shard_methods.rs` deferred ("a fully-bounded
  `for probes in 0..cap` form would require a Verus loop with an
  inductive invariant"). Five proof files now verifying with **115
  total verified items** (was 84). Phases 2 (production-code
  annotation) and 3 (call site swap) remain parked behind the three
  documented `vstd` capability gaps.

### Stale checkbox closed

- **T204.1** (cluster stats surfacing) was actually shipped in commit
  `5caabf9` for v1.2.0; the unchecked box in `RELEASE_1.0.1_PLAN.md`
  was stale and is now marked done.

## Validation gates

All green on the v1.2.1 tag:

| Gate | Result |
|---|---|
| `cargo test --release` | 1,197 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,219 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,224 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 (continuous CI gate) |
| `cargo test --release --test compiled_vs_interpreted` (PROPTEST_CASES=2048) | 17 pass × 3 seeds |
| `cargo test --release --test streaming_scc_oracle` | 2 / 2 (no DIVERGENCE) |
| `cargo test --release --test wrapper_next_fairness_t1_3` | 2 / 2 |
| `cargo test --release --test cross_node_steal_handshake` | 3 / 3 |
| Verus tier-B + tier-A + shard-methods + reader-liveness-v2 + shard-wrapper | 19 + 31 + 17 + 17 + 31 = 115 verified items, 0 errors |
| Mutation testing — `eval.rs` (interpreter) | 100% kill rate |
| Mutation testing — compiled_eval.rs + compiled_expr.rs (compiler) | 70.5% kill rate (1,364 / 1,934 viable, 11 iterations) |

## Compatibility

Drop-in for v1.2.0. No public-API or CLI changes. Fingerprint format,
checkpoint format, state-graph dump format unchanged.

## Code-organization deltas vs v1.2.0

- `src/runtime.rs`: 2,451 → 1,644 LOC (-33%)
- `src/tla/eval.rs` (monolith) → 13-file submodule tree under `src/tla/eval/`
- 14 new files in `src/{runtime,storage,distributed,tla/eval}/`
- 1 new Verus proof file (`shard_wrapper.rs`, 31 verified items)
- 8 new test files (`compiler_helper_coverage_t207{b,c,d,e,f,g,h}.rs` + `distributed_handler_with_mock.rs`)
- Tests: 1,155 → 1,197 (+42)
- Verus proofs: 84 → 115 verified items (+31)

## Still parked (multi-release backlog)

- **T10.2 phase 2 stages 3+4+5** — runtime hot-loop rewrite + partition
  routing + cluster mode. Multi-week per the refined design doc.
- **T13.4 Phases 2+3** — production-code Verus rewrite of `FingerprintShard`.
  Blocked by three documented `vstd` capability gaps (atomic-pointer-
  swap with overlapping permission lifetimes; mmap-allocated
  `PointsToArray` provenance; `&self` + linear ghost token under
  `AtomicInvariant`). Realistic path is a `state_machines!`
  reformulation jointly with T13.5's discharge — research-grade.
- **T13.5 `state_machines!` port** — optional polish; the headline
  `theorem_no_starvation` already shipped axiom-free in v1.1.0 via
  `reader_liveness_v2.rs`.
