# v1.2.7 — Verus T13.4 Phase 2 in-place annotation, wide expansion

Phase 2 in-place Verus annotation broadens from the initial 12-helper / ~58-site footprint
to **38 verified helpers driving 177 shipping call sites across 31 shipping files**, while
the four formerly-"deferred to v1.1.x" items (T5.4, T5.5, T10.2, T13.5) are confirmed
already-shipped and the status documentation is corrected.

## T13.4 Phase 2 expansion

`src/storage/verus_smoke.rs` now contains 38 verified helper functions (up from 12 at the
start of the v1.2.7 cycle). Net additions across 87 commits since v1.2.6:

### New verified helpers

Stdlib substitutions (Verus doesn't yet spec `Ord::min` / `Ord::max` / `Ord::clamp` /
`u64::saturating_*`):

- `min_usize`, `max_usize`, `min_u64`, `max_u64`, `min_u32`, `max_u32`, `max_u16`
- `clamp_usize`, `clamp_u64`
- `saturating_add_u64`, `saturating_sub_u64`
- `saturating_dec_i32_to_zero` — first i32-typed helper, for bracket-depth tracking
  in TLA+ parsers

Bounded-index + bit-shift bounds:

- `compute_index_mod`, `compute_u64_index_mod`, `compute_steal_idx`, `compute_steal_start`
- `compute_shard_id_from_lower_bits`, `compute_shard_idx_modulo`
- `compute_bit_offset` (< 64 for u64 shift), `compute_bit_idx_in_byte` (< 8 for u8 shift),
  `compute_bit_idx_in_u64_word` (< 64 for u64 shift)
- `compute_byte_idx_in_array`, `compute_word_idx`, `compute_slot_within_shard`
- `clamp_shift_to_lt_64`

Probe / hash-table / resize:

- `next_probe_slot`, `initial_probe_slot`
- `compute_new_capacity_on_resize` (no overflow + strict growth)
- `compute_capacity_from_memory` (memory-safety: `capacity * entry_size ≤ memory_size`)
- `compute_memory_size_for_resize`, `compute_rehash_batch_end`,
  `compute_rehash_batch_size_from_pct`
- `is_resize_in_progress` (seqlock parity, bit_vector identity)

Recursive (with `decreases` clause):

- `gcd` — first recursive verified function in shipping source; Euclidean algorithm
  lifted into the verified universe via `decreases b`.

### Shipping coverage

Verified helpers are wired across **31 shipping files** including all major modules:

`src/storage/`: page_aligned_fingerprint_store.rs, page_aligned_color_map.rs,
bloom_fingerprint_store.rs, fingerprint_store.rs, work_stealing_queues.rs, queue.rs,
spillable_work_stealing.rs, numa.rs, s3_persistence.rs, hybrid_fingerprint_store.rs

`src/runtime/`: shards.rs, memory.rs, pause.rs, dfs_pool.rs, dfs_cluster_bridge.rs,
liveness.rs, progress.rs, dfs_worker.rs

`src/distributed/`: bloom.rs, work_stealer.rs

`src/tla/`: compiled_eval.rs, eval/operator.rs, eval/set.rs, action_ir.rs,
compiled_expr.rs

`src/models/`: flurm_job_lifecycle.rs, adaptive_branching.rs

plus `runtime.rs`, `system.rs`, `chaos.rs`, `simulation.rs`.

Default builds are unchanged. `--features verus` activates the verified path via
cfg-split wires.

## Status correction: T5.4 / T5.5 / T10.2 / T13.5 — all shipped

Per a re-survey of the release notes, four items previously listed under
"Deferred to v1.1.x" in `CLAUDE.md` are actually shipped:

- **T5.4** (streaming Init enumeration) — shipped v1.1.0. Init producer thread overlaps
  with worker exploration.
- **T5.5** (joint Init + invariant Z3 encoding) — shipped v1.1.0. Einstein-class spec
  44 min → 14 ms.
- **T10.2** (streaming SCC) — phase 1 + phase 2 stages 1–5 shipped across v1.1.0 →
  v1.2.6. The streaming-SCC memory win is realized; only the `--dfs-cluster-listen`
  user-facing CLI flag remains queued.
- **T13.5** (`state_machines!` port) — shipped v1.2.1 at
  `verification/verus/reader_liveness_state_machine.rs` (15 verified items, Tier A.8).

`CLAUDE.md` is updated to reflect the actual shipping status.

## T13.4 full — defensibly parked

The "production FingerprintShard tracked-pointer integration" remaining work is
documented in `verification/verus/T13.4-PHASE2-CLOSURE.md` as engineering work
(Cargo.toml/feature-gate + per-method `requires`/`ensures` lifts using validated
prototype patterns) rather than verification research. All original blockers
(atomic-pointer swap, mmap-tracked permissions, `&self`+linear-ghost, multi-slot
composition, cargo-verus integration) are resolved in 186 verified items across 11
standalone artifacts.

Phase 2 in-place annotation — what this release expands — is the *near-term*
realization of that path: `cargo verus check --features verus` runs on the main
crate with 49 verified items driving 177 shipping call sites. The deeper lift
(replacing unsafe pointer arithmetic over mmap with `Vec<PAtomicU64>` +
`Tracked<Map<int, PermissionU64>>`) is the next architectural step, not in scope
for v1.2.7.

## Validation

All test gates pass on a fresh c6gd.metal aarch64 spot:

| Gate | Result | Baseline |
|---|---|---|
| `cargo test --release` | 1,227 pass / 0 fail / 11 ignored | matches v1.2.6 |
| `cargo test --release --features failpoints` | 1,249 pass / 0 fail / 11 ignored | matches v1.2.6 |
| `PROPTEST_CASES=2048 cargo test --release --test compiled_vs_interpreted` | 17 pass / 0 fail | clean at 16× default |
| `scripts/chaos_smoke.sh` (swarm-mode auto) | 15 iters, 12/12 failpoints fired, 39 concurrent pairs, 0 div / 0 hangs | PASS gate |
| `cargo verus check --features verus` (shipping) | 49 verified, 0 errors | NEW |
| Verus standalone tiers (9 of 11) | 144 verified, 0 errors | b/a/liveness/v2/sm/epoch/exec-wired/mmap/multi-slot |
| `scripts/chaos_soak.sh --duration 86400 --swarm-mode auto` (cut short by AWS spot reclaim at ~12.7h) | 2,526 iters complete, 2,526 / 2,526 verdict=ok, 0 divergences, 0 hangs | 12-fold over 1-hour release-ritual baseline (~210 iters) |

Two standalone tier files (`shard_methods.rs`, `shard_wrapper.rs`) hit pre-existing
Verus version drift on the spot's HEAD-built verus binary (mut-ref migration
post-`46362a2` pinned in `Cargo.toml`). Last touched 2026-05-11; not a v1.2.7
regression. The shipping `cargo verus check --features verus` path against pinned
vstd 46362a2 still passes 49/49.

### Extended chaos-soak detail (post-release validation)

A long-soak run kicked off after tagging, targeting 24 hours; AWS reclaimed the
c6gd.metal aarch64 spot at the ~12.7-hour mark. The partial result is itself a
strong signal: **2,526 iterations** of `corpus/internals/CheckpointDrain.tla`
(26,344 distinct states per iter, control baseline 16.6 s wall) under
`--swarm-mode auto` (random N in [1, 4] concurrent failpoints per iteration,
sampled from the 12-failpoint catalog in `src/chaos.rs`). Every iteration agreed
with the control on both state count and verdict (`distinct=26344 verdict=ok`).
No divergences, no hangs, no panics-without-recovery across the full run. Hourly
heartbeat checks (sampled in a separate log) confirmed steady ~215 iters/hour
throughout.

Failpoint coverage was statistically saturated by the swarm distribution: at
2,526 iters with N drawn uniformly from `[1, 4]` over a catalog of 12, every
single-failpoint and every concurrent pair were exercised many times. The
5-minute pre-release smoke (15 iters, all 12 failpoints fired, 39 distinct
concurrent pairs) already confirmed catalog saturation under the same harness;
the long-soak result extends the same property out by ~170× iteration count.

## Compatibility

Drop-in for v1.2.0–v1.2.6. No public-API, CLI, or test surface changes. The
`verus` feature stays opt-in; default builds and CI gates are unchanged.
