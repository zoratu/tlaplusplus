# tlaplusplus v1.2.1

Patch release closing out the v1.2.0 deferred-items list. No new user-facing features and no behavioural change. The release is dominated by structural refactoring and additional verification work.

## v1.2.0 deferred items — all closed

### T204 distributed mock for the cluster handler

A new `Transport` trait (dyn-compatible via boxed futures, no `async-trait` dep) plus an in-memory `MockTransport` using tokio mpsc channels. `DistributedWorkStealer::transport` switched to `Arc<dyn Transport>`. 5 unit tests + 5 integration tests in `tests/distributed_handler_with_mock.rs` exercise the inbound handler, bloom-and-termination, and steal-trigger paths without real TCP.

### `src/runtime.rs` extraction chunks 7 + 8

Both extracted. Chunk 8 (the 13-step shutdown phase) becomes `src/runtime/shutdown.rs` with a `ShutdownContext` struct grouped per-step, and an `orchestrate` method. Chunk 7 (the worker spawn loop with 27-Arc capture) becomes `src/runtime/worker.rs` with a `WorkerLocalState` struct.

### `src/tla/eval.rs` split

All 8 chunks (A–H) of the design doc's landing protocol landed. The monolith becomes 13 submodule files in `src/tla/eval/`: `mod.rs`, `expr.rs`, `action.rs`, `splitter.rs`, `operator.rs`, `set.rs`, `quantifier.rs`, `bracket.rs`, `postfix.rs`, `control.rs`, `transition.rs`, `instance.rs`, `budget.rs`. The external API surface is unchanged — `crate::tla::eval::<name>` paths still resolve via `pub(crate) use` re-exports.

### Compiler internal-helper restructuring

Depth-tracking consolidation across `compiled_eval.rs` and `eval_operator_call`. Plus 6 more iterations of targeted compiler-helper unit tests (T207b through T207h, adding 149 tests on top of v1.2.0's T207). Compiler mutation kill rate: 65.4% (v1.2.0) → 70.5% (v1.2.1) across 11 cargo-mutants iterations on the v1.2.1 surface. Interpreter (`eval/`) remains at 100%.

## Older parked items partially landed

### T10.2 phase 2 — stages 1+2 of 5

Strictly-additive foundation for in-exploration streaming SCC (the eventual memory win for 100M+ liveness checking). Stage 1 ships `PageAlignedColorMap` (7 unit tests): 2-bit Color enum, 32 slots per `AtomicU64`, 2 MB hugepage mmap with anonymous fallback, `set_preferred_node` per shard, lock-free CAS via `compare_exchange_weak`. Stage 2 ships protocol variants: `PartitionEdge`/`PartitionEdgeAck`, `RedDfsProbe`/`RedDfsResponse`, `RequestStateBlob`/`StateBlobResponse`, plus `Option<u64>` extensions on `TerminationToken` (existing tokens still decode via `#[serde(default)]`).

Stages 3–5 (worker DFS body, partition routing, cluster mode) remain parked per the design doc's effort estimate. Default `--liveness-streaming` flag stays OFF — phase 2 is opt-in just like phase 1.

### T13.4 Verus tracked-pointer integration — Phase 1

Shipping-shape verified wrapper at `verification/verus/shard_wrapper.rs` (31 verified items). Lifts the bounded outer probe loop that `shard_methods.rs` deferred. Five proof files now verify with 115 total verified items (was 84). Phases 2 (shipping-code annotation) and 3 (call site swap) remain parked behind the three documented `vstd` capability gaps.

## Stale checkbox closed

T204.1 (cluster stats surfacing) was actually shipped in commit `5caabf9` for v1.2.0; the unchecked box in `RELEASE_1.0.1_PLAN.md` was stale and is now marked done.

## Validation gates

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
| Mutation testing — `compiled_eval.rs` + `compiled_expr.rs` (compiler) | 70.5% kill rate (1,364 / 1,934 viable, 11 iterations) |

## Compatibility

Drop-in for v1.2.0. No public-API or CLI changes. Fingerprint format, checkpoint format, state-graph dump format unchanged.

## Code-organization deltas vs v1.2.0

`src/tla/eval.rs` monolith → 13-file submodule tree under `src/tla/eval/`. 14 new files in `src/{runtime,storage,distributed,tla/eval}/`. 1 new Verus proof file (`shard_wrapper.rs`, 31 verified items). 8 new test files (`compiler_helper_coverage_t207{b,c,d,e,f,g,h}.rs` + `distributed_handler_with_mock.rs`). Tests at 1,197 (was 1,155). Verus proofs at 115 (was 84).

## Still parked

- T10.2 phase 2 stages 3+4+5 — runtime hot-loop rewrite + partition routing + cluster mode.
- T13.4 Phases 2+3 — shipping-code Verus rewrite of `FingerprintShard`, blocked by three documented `vstd` capability gaps.
- T13.5 `state_machines!` port — optional polish; the headline `theorem_no_starvation` already shipped axiom-free in v1.1.0 via `reader_liveness_v2.rs`.
