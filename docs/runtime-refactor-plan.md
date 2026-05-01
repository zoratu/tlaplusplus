# `src/runtime.rs` Refactor Plan

Status: **PARTIALLY LANDED (Path A, chunks 1-7 of 8)** — see "Landed
chunks" appendix at the bottom of this file. Chunk 7 (worker spawn loop)
and chunk 8 (shutdown orchestration) remain inline in `runtime.rs`
pending a follow-up that introduces the `WorkerCtx<M>` struct from the
"Concurrency-coupling analysis" section. The worker loop in particular
requires bundling 27 captured Arcs into a single context type so the
extraction does not regress the T5.4 producer / T6 cluster steal /
T11.5 violation-finish ordering invariants documented below.

`src/runtime.rs` is 4,323 lines. It accreted T5.4 (init producer thread),
T6 (cluster steal trigger wiring), T10/T10.1-T10.4 (parallel liveness
post-processing), T10.2 (streaming-SCC oracle), and T11.5 (violation
handler / `queue.finish()` ordering) on top of the original BFS worker
loop, termination detection, and checkpoint coordinator.

This document captures the proposed module split, the dependency graph
between submodules, and — most importantly — the concurrency-coupling
analysis that forced the conclusion that the refactor cannot be landed
in the time budget under the working constraint that all `cargo` runs
must happen on a remote spot host.

## Why this is design-only

The split itself is mechanical, but every chunk has to be validated by
the full gate set on the remote host:

- `cargo test --release` (~790 tests)
- `cargo test --release --features failpoints` (~812 tests, single-threaded)
- `cargo test --release --features symbolic-init` (~813 tests)
- `scripts/diff_tlc.sh` (13/13 specs vs TLC v2.19)
- `cargo test --release --test wrapper_next_fairness_t1_3`
- `cargo test --release --test cross_node_steal_handshake`
- `cargo test --release --test streaming_init_t5_4`

Running each chunk's gates over the remote spot loop, with build cache
warmups, conservatively costs 25-40 min/chunk. With 8 chunks (see
**Move plan** below) that is 4-6 hours of pure validation, plus
diagnosis + re-roll on any chunk that breaks. The 8-hour budget does
not absorb a single broken chunk; landing partial moves violates the
binary quality bar. Path B keeps the design captured and defers the
move to a session that can sit on the spot host with a hot build cache.

## Top-level definitions today (line numbers)

Index of every top-level definition in `src/runtime.rs` as of
`f161249`:

| Lines | Name | Visibility | Notes |
|---|---|---|---|
| 36-188 | `EngineConfig` + `Default` | `pub` | Config struct with 50+ fields |
| 191-196 | `PropertyType` | `pub` | Safety/Liveness enum |
| 199-206 | `Violation<S>` | `pub` | Trace-bearing violation report |
| 209-214 | `RunOutcome<S>` | `pub` | Returned by `run_model` |
| 217-235 | `RunStats` | `pub` | Final stats snapshot |
| 238-277 | `AtomicRunStats` + impls | private | Shared across threads |
| 279-305 | `PauseController` + `Default` | private | Checkpoint pause coordinator |
| 307-310 | `QUIESCENCE_*` consts | private | Wait-budget constants |
| 312-326 | `next_quiescence_timeout_secs` | private | Pure helper, has tests |
| 328-365 | `PauseController::init_*/get_*` impls | private | Worker tracking |
| 367-594 | `PauseController` register/wait/resume impls | private | The lock-free park/unpark protocol |
| 596-622 | `CheckpointPauseQueue` trait + 2 impls | private | Bridges `WorkStealingQueues` and `SpillableWorkStealingQueues` |
| 624-647 | `request_checkpoint_pause`, `pause_worker_after_empty_pop_during_checkpoint` | private | Pause-flag ordering helpers (tested directly) |
| 649-670 | `CheckpointManifest` (serde) | private | On-disk schema |
| 672-684 | `compute_effective_memory_max` | private | Memory budget math |
| 686-699 | `should_enable_file_backed_fingerprint_store` / `should_start_fingerprint_memory_monitor` | private | Pure predicates, have tests |
| 701-730 | `apply_memory_budget` | private | Memory budget math |
| 732-758 | `write_checkpoint_manifest` | private | Atomic-rename writer with failpoints |
| 761-770 | `load_checkpoint_manifest` | private | Resume reader |
| 773 | `MAX_CHECKPOINT_FILES` | private | const |
| 784-861 | `write_validated_rolling_checkpoint` | private | Read-back validation + retention |
| 865-903 | `prune_old_checkpoints` | private | Rolling-window cleanup |
| 906-921 | `CheckpointContext<'a, T>` | private, `dead_code` | Used only by orphan `checkpoint_once` |
| 924-994 | `checkpoint_once` | private, `dead_code` | Old (pre-T11.5) checkpoint helper, now superseded by inline checkpoint thread body |
| 1004-1076 | `calculate_optimal_shard_count` | private | Pure heuristic |
| **1078-3631** | **`run_model<M>`** | `pub` | The 2,553-line entry point |
| 3641-3691 | `reconstruct_trace` | `pub` | Used by simulation; not internal to run_model |
| 3697-3740 | `reconstruct_trace_limited` | `pub` | Used inline in worker safety-violation path |
| 3743-4068 | `mod tests` | `cfg(test)` | 9 tests (see test list below) |
| 4072-end | `mod failpoint_tests` | `cfg(test, feature = "failpoints")` | 12+ chaos tests |

External API surface (from `lib.rs`):
```rust
pub use runtime::{EngineConfig, PropertyType, RunOutcome, RunStats, Violation, run_model};
```

External `pub` consumers:

- `src/lib.rs` re-exports the six items above.
- `src/simulation.rs` imports `PropertyType, Violation`.
- `reconstruct_trace` and `reconstruct_trace_limited` are `pub` but
  not re-exported. Both are reachable as `crate::runtime::*` paths.

## Proposed module tree

```
src/runtime/
├── mod.rs                  ~280  — pub re-exports, EngineConfig + Default + PropertyType,
│                                   Violation, RunOutcome, RunStats, AtomicRunStats,
│                                   reconstruct_trace, reconstruct_trace_limited.
│                                   The pub run_model<M> entry point lives here as a
│                                   thin orchestrator (see below).
├── config.rs              ~200  — EngineConfig fields + Default. Split out from mod.rs
│                                   only if mod.rs grows past ~300; otherwise inline.
├── stats.rs               ~120  — AtomicRunStats + RunStats + the per-worker
│                                   flush_local_stats closure (extracted as a free fn
│                                   that takes &mut counters + &AtomicRunStats).
├── pause.rs               ~440  — PauseController + QUIESCENCE_* constants +
│                                   next_quiescence_timeout_secs +
│                                   wait_for_quiescence{_attempt} + register/resume +
│                                   CheckpointPauseQueue trait + 2 impls +
│                                   request_checkpoint_pause +
│                                   pause_worker_after_empty_pop_during_checkpoint.
│                                   All 4 unit tests for these helpers move here.
├── checkpoint/
│   ├── mod.rs             ~30  — re-exports the public-to-runtime symbols
│   ├── manifest.rs       ~200  — CheckpointManifest + write_/load_/write_validated_/
│   │                              prune_old_/MAX_CHECKPOINT_FILES.
│   ├── context.rs        ~120  — CheckpointContext + checkpoint_once. These are
│   │                              currently dead_code; moving them keeps history
│   │                              traceable. Filing as follow-up: delete in a
│   │                              separate commit after refactor lands.
│   └── thread.rs         ~250  — spawn_checkpoint_thread() — extracted from the
│                                   inline closure body inside run_model
│                                   (lines 1681-1897). Takes Arc clones of the
│                                   shared atomics and the queue/fp_store/pause
│                                   handles, returns the JoinHandle. Captures
│                                   the lightweight-vs-emergency branch unchanged.
├── memory.rs              ~140  — compute_effective_memory_max +
│                                   should_enable_file_backed_fingerprint_store +
│                                   should_start_fingerprint_memory_monitor +
│                                   apply_memory_budget +
│                                   spawn_memory_monitor_thread (extracted from
│                                   lines 1350-1421) + spawn_memory_pressure_monitor
│                                   (the progress-thread branch lines 1916-2003 that
│                                   handles MemoryStatus::Critical/Warning/Ok and
│                                   talks to WorkerThrottle).
├── shards.rs              ~100  — calculate_optimal_shard_count (pure helper +
│                                   its sole call site moves with it).
├── init_producer.rs       ~160  — spawn_init_producer<M>(...) — extracts the T5.4
│                                   producer closure (lines 1554-1669). Owns the
│                                   `init_producing` AtomicBool's Drop guard, the
│                                   INIT_BATCH_SIZE batching, and the
│                                   contains_or_insert_batch ordering. Returns
│                                   (Option<JoinHandle<Result<u64>>>, Arc<AtomicBool>).
├── progress.rs            ~200  — spawn_progress_thread(...) extracts
│                                   lines 1903-2094: the format_with_commas helper,
│                                   ETA computation, and the emergency-checkpoint
│                                   poll. Memory-pressure handling stays inside
│                                   memory.rs::spawn_memory_pressure_monitor.
├── worker/
│   ├── mod.rs             ~30  — re-exports spawn_worker.
│   ├── batch.rs          ~240  — process_batch closure body extracted as a method
│   │                              on a WorkerCtx struct (see "Worker context"
│   │                              below). Lines 2651-2769.
│   └── loop.rs            ~480  — spawn_worker<M>(...) — body is lines 2246-2833,
│                                   minus process_batch (in batch.rs) and
│                                   flush_local_stats (in stats.rs).
├── distributed.rs         ~100  — spawn_distributed_handlers(...) — wraps the
│                                   `if let Some(ref stealer) = config.distributed_stealer`
│                                   block lines 2148-2211: inbound handler,
│                                   bloom/termination task, and the T6 steal-trigger
│                                   task. This module is a thin facade over the
│                                   already-existing `distributed::handler` API.
├── liveness.rs            ~520  — Post-BFS fairness pipeline: lines 3022-3589.
│                                   T10.1 parallel-flatten (parallel_flatten_shards
│                                   + serial_flatten), T10.3 trim, Tarjan call,
│                                   T10.4 per-action shard build, fairness check
│                                   loop, optional T10.2 streaming-SCC oracle.
│                                   Public entry: run_liveness_post_processing<M>.
└── shutdown.rs            ~180  — Final-phase orchestration: worker join + crash
                                   recovery (lines 2837-2867), init producer join
                                   (2868-2882), progress/auto-tuner/checkpoint
                                   thread shutdown (2884-2916), exit checkpoint
                                   write (2918-2975), fp_store flush + compression
                                   stats summary (2977-2999), error drain
                                   (3001-3008), violation collection (3010-3019).
```

Total: ~3,540 LOC of moved code + ~280 LOC orchestrator in `mod.rs`.

### `run_model` after refactor (sketch, ~280 lines in `mod.rs`)

```rust
pub fn run_model<M: Model>(model: M, config: EngineConfig) -> Result<RunOutcome<M::State>> {
    let model = Arc::new(model);
    setup_work_dir(&config)?;
    let effective_memory_max = memory::compute_effective_memory_max(&config);
    let worker_plan = system::build_worker_plan(/* ... */);
    let fp_store = build_fp_store(&config, &worker_plan, effective_memory_max)?;  // ~80 LOC
    let mem_monitor = memory::spawn_memory_monitor_thread(/* ... */);
    let (queue, worker_states) = build_queue(&config, &worker_plan)?;             // ~30 LOC

    let shared = SharedRuntimeState::new(&config, &worker_plan, /* atomics */);
    let init_producer = init_producer::spawn(&model, &queue, &fp_store, &shared);
    let checkpoint = checkpoint::thread::spawn(/* ... */);
    let progress = progress::spawn(/* ... */);
    let labeled_transitions = build_labeled_transitions(&model);
    let mut auto_tuner = autotune::start(&config, &shared);

    pause.init_worker_tracking(/* ... */);
    let numa_diagnostics = NumaDiagnostics::new(/* ... */);
    pause.set_numa_diagnostics(numa_diagnostics);

    distributed::spawn_handlers_if_any(&config, &queue, &shared);

    let workers: Vec<_> = (0..worker_plan.worker_count)
        .map(|id| worker::spawn(id, &model, &queue, &fp_store, &shared, &config, &labeled_transitions))
        .collect();

    shutdown::orchestrate(workers, init_producer, progress, auto_tuner, checkpoint, mem_monitor,
                          &queue, &fp_store, &shared, &config, &labeled_transitions, started_at)
}
```

`SharedRuntimeState` is the **only new abstraction** the refactor
introduces. Everything else is pure code motion.

## Concurrency-coupling analysis

The reason this is hard is that `run_model` is a single function whose
correctness depends on the **textual order** of statements that touch
several Arc-wrapped atomics. Each move chunk has to preserve:

### 1. T5.4 init-producer ordering (lines 1538-1669, 2244, 2416-2423, 2868-2882)

Invariants:

- `init_producing: Arc<AtomicBool>` is set to `true` **before** the
  producer thread is spawned (line 1538), then cleared by a Drop guard
  inside the producer thread closure (lines 1571-1577) so that a
  panic inside the producer doesn't strand workers.
- Workers (line 2420) check `init_producing.load(Acquire)` after
  `has_pending_work()` returns `false` and before terminating. Order
  matters: queue check first, producer check second. Reversing this
  introduces a race where a state pushed by the producer between the
  two checks would be missed by the terminating worker.
- The producer joins **after** all workers join (line 2872 vs line
  2840). This is load-bearing: the workers' termination check above
  reads `init_producing` and a producer that exits before workers
  must have cleared the flag (via Drop) so workers can terminate.

Refactor risk: extracting `spawn_init_producer` is mechanical, but the
worker-loop check at line 2420 must stay in the **same textual position
relative to the queue check and the cluster-idle handshake**. This
means `worker/loop.rs` must keep the empty-pop branch (lines 2353-2461)
in one continuous block — splitting it across helpers would obscure
the ordering and risk a future contributor reordering them.

### 2. T6 cluster steal-trigger and idle-flag handshake (lines 2148-2211, 2452-2458, 2826-2832)

Invariants:

- `stealer.set_locally_idle(true)` is called on the empty-pop path
  (line 2454) before the worker waits 10ms. T6 fix history (line 2448
  comment): previously this was only set on worker exit, which deadlocked
  cluster termination consensus.
- `stealer.set_locally_idle(false)` is called as soon as a worker pops
  a state (line 2474), and `note_local_work()` fires every
  STATS_FLUSH_INTERVAL (line 2495).
- `stealer.set_locally_idle(true)` also fires on the **last** worker
  exit (line 2828: `if remaining == 1`).
- `spawn_steal_trigger_task` is spawned unconditionally inside the
  `Some(stealer)` branch (line 2203), even though it short-circuits
  on singleton clusters internally.

Refactor risk: the three idle-flag transitions in the worker loop must
all stay in `worker/loop.rs`. The steal-trigger spawn moves cleanly
into `distributed.rs::spawn_handlers_if_any`. The `local_pending_fn`
closure (lines 2167-2173) — which bridges `pending_count()` over-reporting
via `has_pending_work()` — must remain as a unit; it cannot be split
across files.

### 3. T11.5 violation finish ordering (lines 2530-2557)

Invariants:

- On a safety violation that triggers `should_stop`, the order is
  fixed: (a) `worker_violation_tx.try_send`, (b) `violation_count.fetch_add`,
  (c) `worker_stop.store(true, Release)`, (d) `worker_queue.finish()`,
  (e) `worker_queue.worker_idle(id)`, (f) `break`.
- `queue.finish()` (step d) is **the** fix for T11.5: without it,
  orphan items in the violator's local deque keep `should_terminate`
  returning false for siblings, causing a hang. The final `queue.finish()`
  in the `run_model` shutdown path (line 2897) is a separate call that
  fires after all workers join.

Refactor risk: the violation-handler block (lines 2499-2559) cannot be
extracted to a free function unless the function takes `&worker_queue`,
`&worker_stop`, `&worker_violation_tx`, `&worker_violation_count`, and
the trace-reconstruction closure references (`worker_parent_map`,
`worker_state_map`). That signature is so wide that the helper would
not actually shrink the worker loop in any meaningful way — the helper
body is also shorter than its signature. Recommendation: keep it inline
in `worker/loop.rs`. Same for the T5.4/T6 idle handshake.

### 4. T10 liveness post-processing dependency on transitions DashMap

The fairness path (lines 3022-3590) reads `labeled_transitions: Arc<DashMap<...>>`
that workers populate during exploration (lines 2586-2606). Workers
must have all joined before the post-processing reads, because the
parallel-flatten path uses raw shard locks (`unsafe { let raw_iter = guard.iter() }`)
and would race with concurrent inserts.

The barrier is the worker-join loop (line 2840). The init-producer
join (line 2872) happens after worker join, so producer-side fingerprint-only
states are not in `labeled_transitions` (only workers populate it via
`worker_model.next_states_labeled`).

Refactor risk: `liveness.rs` must be invoked **after**
`shutdown::orchestrate`'s worker-join phase, before the `RunOutcome`
is built. The cleanest split is: `shutdown::orchestrate` returns the
joined workers and the run_stats snapshot, then the orchestrator in
`mod.rs::run_model` calls `liveness::run_post_processing` with the
DashMap. The current code structure already follows this order; the
refactor preserves it.

### 5. Per-worker state ownership

`worker_states: Vec<WorkerState>` (line 1445) is moved into the
worker spawn closures via `into_iter().enumerate()`. Each worker owns
its own `WorkerState`. Refactor risk: `worker::spawn` must take
`worker_state: WorkerState` by value, not by `&mut` — moving across
the thread boundary is the existing semantics.

### 6. Closures over local `flush_local_stats` and `process_batch`

`flush_local_stats` (lines 2297-2329) and `process_batch` (lines
2651-2769) are closures defined inside the worker loop. Both close over
`&mut` locals (`local_states_processed`, `pending_batch`, `local_fp_dedup`,
`unique_states`, `unique_fps`, `batch_seen`, `local_fp_cache`,
`local_fp_cache_hits`, `states_with_home_numa`, `fps_to_check`,
`states_to_check`).

Extracting them as free functions requires either:

(a) Passing **all** these `&mut` locals as parameters (function
    signature ~12 params, `&mut` for half of them, plus
    `&Arc<UnifiedFingerprintStore>`, `&Arc<DashMap<...>>`,
    `&Arc<AtomicU64>`, `&AtomicRunStats`).

(b) Bundling them into a `WorkerLocalState` struct that the worker
    loop owns, with `WorkerLocalState::process_batch(&mut self, ...)`
    methods. This is the recommended path but introduces a new
    struct and ~120 LOC of plumbing per method.

Either option is correctness-preserving but **non-trivial** to validate
without running the full test gate set on each iteration.

### 7. Shutdown ordering (lines 2837-2999)

The shutdown phase has 13 distinct steps in a specific order:

1. Worker join (with crash recovery) — line 2840
2. Init producer join — line 2872
3. `stop.store(true)` + progress thread join — lines 2886-2889
4. Auto-tuner join — lines 2892-2894
5. `queue.finish()` — line 2897
6. `mem_monitor_stop.store(true)` + unpark + monitor join — 2899-2909
7. `checkpoint_thread_stop.store(true)` + `pause.resume()` + checkpoint thread join — 2910-2916
8. Exit checkpoint write — 2919-2975
9. `fp_store.flush()` (best-effort) — 2981-2983
10. Compression stats print — 2985-2999
11. Worker error drain — 3001-3008
12. Violation collection — 3010-3019
13. Liveness post-processing — 3048-3590

Steps 1-12 are fixed; reordering breaks crash-recovery semantics or
risks deadlock. Step 13 must come after step 11 (errors) but before
the final `RunOutcome` build because `violation` may be reassigned by
liveness.

Refactor risk: `shutdown.rs::orchestrate` must encode these 13 steps
in this order. The function signature is wide (it needs Arc clones of
every shared atomic, every JoinHandle, every queue/fp_store handle,
plus `&config` and `started_at`).

## Move plan (sequence)

If a future session attempts the move, the chunks should be applied
**in this order**, with the full gate set run between each:

| # | Chunk | Touches | Why first/last |
|---|---|---|---|
| 1 | `pause.rs` | Lines 279-647 + 4 unit tests | Self-contained; tests inline; zero closures |
| 2 | `checkpoint/manifest.rs` + `checkpoint/context.rs` | Lines 649-994 | Pure-data + dead_code helpers; tests stay in mod.rs for chunk 1 |
| 3 | `memory.rs` + `shards.rs` | Lines 672-730, 1004-1076, 1350-1421, 1916-2003 | Memory monitor + budget calc; isolated threads |
| 4 | `stats.rs` | Lines 217-277 + flush_local_stats extraction | Touches worker loop; **first risky chunk** |
| 5 | `init_producer.rs` | Lines 1538-1669, 2868-2882 | T5.4; preserves Drop guard ordering |
| 6 | `progress.rs` + `distributed.rs` | Lines 1903-2094, 2148-2211 | Independent threads |
| 7 | `worker/loop.rs` + `worker/batch.rs` | Lines 2213-2834 | **Largest risk**; T6, T11.5, T5.4 worker checks |
| 8 | `liveness.rs` + `shutdown.rs` + final `mod.rs` cleanup | Lines 2837-3631 | Orchestrator becomes the new run_model |

Validation cost (estimated, on c8g.metal-24xl with hot cache):

| Gate | Time |
|---|---|
| `cargo test --release` | ~6 min |
| `cargo test --release --features failpoints --test-threads 2` | ~12 min |
| `cargo test --release --features symbolic-init` | ~7 min |
| `scripts/diff_tlc.sh` | ~5 min (13 specs sequential) |
| Targeted regressions (T1.3 fairness, T6 steal, T5.4 producer) | ~3 min |

Total per chunk: ~33 min. 8 chunks = ~4.4 hours of clean validation,
with zero re-rolls. Realistic with a single re-roll allowance per
chunk: ~6.5 hours. Not safely landable in 8 hours when the operator
can't watch the spot host.

## Tests that move with each chunk

| Chunk | Tests | Where they go |
|---|---|---|
| 1 | `quiescence_schedule_respects_total_budget`, `quiescence_schedule_clamps_to_remaining_budget`, `worker_pauses_when_checkpoint_is_requested_between_pause_point_and_pop`, `request_checkpoint_pause_sets_controller_and_queue_flags`, `pause_worker_after_empty_pop_is_noop_without_checkpoint` | `pause.rs::tests` |
| 2 | (none move — manifest tests live in `tests/checkpoint_chaos.rs`) | — |
| 3 | `file_backed_fingerprint_store_uses_effective_memory_cap`, `memory_monitor_requires_file_backed_fingerprint_store` | `memory.rs::tests` |
| 4-7 | (no unit tests in current file touch these) | — |
| 8 | `writes_checkpoint_manifest_on_exit`, `t102_concurrent_worker_errors_do_not_deadlock`, `resumes_from_disk_queue_checkpoint` (ignored), all 12 `failpoint_tests` | `runtime/mod.rs::tests` (or a sibling `tests.rs`) |

## Visibility audit

Every helper currently private stays private after refactor. The
private-to-`runtime/`-but-cross-submodule items become `pub(super)`
(or `pub(crate::runtime)` via `pub(crate)` if Rust ever stabilizes
nested `pub(in path)` ergonomics). Specifically:

- `PauseController`, `next_quiescence_timeout_secs`,
  `request_checkpoint_pause`, `pause_worker_after_empty_pop_during_checkpoint`,
  `CheckpointPauseQueue` — `pub(super)` from `pause.rs`.
- `CheckpointManifest`, `MAX_CHECKPOINT_FILES`, `write_checkpoint_manifest`,
  `load_checkpoint_manifest`, `write_validated_rolling_checkpoint`,
  `prune_old_checkpoints` — `pub(super)` from `checkpoint::manifest`.
- `AtomicRunStats` — `pub(super)` from `stats.rs`.
- `compute_effective_memory_max`, `should_enable_file_backed_fingerprint_store`,
  `should_start_fingerprint_memory_monitor`, `apply_memory_budget`,
  `calculate_optimal_shard_count` — `pub(super)` from their respective files.

External `pub` items (the six in `lib.rs::pub use`) keep `pub`
visibility from `runtime/mod.rs`.

## What gets fixed in follow-ups (not now)

Bugs and code-smell I noticed while reading; **none are fixed inline**:

1. `CheckpointContext` and `checkpoint_once` (lines 906-994) are
   `#[allow(dead_code)]`. They've been superseded by the inline
   checkpoint-thread body. File a follow-up to delete after refactor.
2. `RunStats::queue` is built in the final `RunOutcome` (lines 3613-3625)
   with **zeros** for `spilled_items`, `spill_batches`, `loaded_segments`,
   `loaded_items`, `max_inmem_len`, `spill_lost_permanently`. The actual
   `WorkStealingStats` doesn't carry these — they live on the spillable
   queue's own stats. The on-exit-checkpoint manifest (line 2952) calls
   `queue.stats()` which **does** return spill numbers — different shape.
   File follow-up: unify the stats schema.
3. `process_batch` line 2769 returns `Ok(())` but its `.is_ok()` check
   on lines 2776-2785 / 2793-2802 — only the error path matters. This
   is fine but the closure is awkward; converting to a method on
   `WorkerLocalState` would make the control flow clearer.
4. The compressed-ring summary print (lines 2985-2999) is in
   `run_model` shutdown but conditionally fires only if the spillable
   queue has compression stats. Belongs in `shutdown.rs`.

## Commit hash this plan was written against

`f161249` (`chore: remove extracted Java class files, update .gitignore`)
on branch `worktree-agent-a6e9544466376a95d`.

## Landed chunks (Path A, partial)

Six commits on top of `f161249` extract the easy + medium chunks. The
file shrinks from 4,323 LOC to 2,451 LOC (-43%); the bulk of what
remains is the worker spawn loop (587 LOC, chunk 7) and the 13-step
shutdown phase (chunk 8). All gates (790 default tests, 812 failpoint,
817 symbolic-init, 13/13 diff_tlc, chaos smoke 12/12) remain green.

| Chunk | File(s) | LOC moved | Notes |
|---|---|---:|---|
| 1 | `runtime/pause.rs` | 526 | PauseController + 5 inline tests |
| 2 | `runtime/checkpoint.rs` | 315 | Manifest schema + writer + dead `checkpoint_once` |
| 3 | `runtime/memory.rs` + `runtime/shards.rs` | 130 + 83 | Budget + 2 inline tests; shard heuristic |
| 4 | `runtime/stats.rs` | 50 | AtomicRunStats |
| 5 | `runtime/init_producer.rs` | 174 | T5.4 producer thread + Drop guard |
| 6 | `runtime/progress.rs` + `runtime/distributed.rs` | 226 + 100 | Progress thread + T6 handler wiring |
| 7 | `runtime/liveness.rs` | 598 | T10/T10.1-T10.4 + T10.2 oracle |

Total: ~2,202 LOC moved across 9 new files (one of which is `liveness.rs`
at 598 LOC, almost entirely a verbatim move of the post-BFS fairness
pipeline). `runtime.rs` retains the orchestration scaffold + worker
spawn loop + shutdown.

### Not landed (deferred to a follow-up session)

**Chunk 7 (worker spawn loop, 587 LOC).** The worker thread closure
captures 27 distinct `Arc<...>` clones plus the per-worker `WorkerState`
by value. Internal `flush_local_stats` and `process_batch` closures
themselves close over 12+ `&mut` locals. The "Closures over local
flush_local_stats and process_batch" section above describes the two
options: a 12-param free function or a `WorkerLocalState` struct with
~120 LOC of plumbing. Either is doable but each option needs a focused
session that can iterate quickly on the spot host (compile errors will
cascade through the worker body), and the T6 idle-flag handshake plus
T11.5 violation-finish ordering must be re-validated against
`wrapper_next_fairness_t1_3`, `streaming_init_t5_4`, and
`cross_node_steal_handshake` after each iteration.

**Chunk 8 (shutdown.rs, ~250 LOC).** The 13-step shutdown phase
(worker join → init join → progress stop → auto-tuner join → queue
finish → mem monitor stop+unpark+join → checkpoint stop+resume+join
→ exit checkpoint write → fp_store flush → compression stats →
worker error drain → violation collection) is currently 13 sequential
fragments interleaved with state collection. Extracting it requires
either a wide-signature `shutdown::orchestrate(...)` function or a
ShutdownContext struct. The signature is wide because the function
needs Arc clones of every shared atomic, every JoinHandle, every
queue/fp_store handle, plus `&config` and `started_at`.

### Validation snapshot at the chunk-7 head

- `cargo build --release` — clean.
- `cargo build --release --features failpoints` — clean.
- `cargo test --release` — 790 pass, 0 fail, 8 ignored.
- `cargo test --release --features failpoints --test-threads=2` — 812
  pass, 0 fail, 8 ignored.
- `cargo test --release --features symbolic-init` — 817 pass, 0 fail,
  8 ignored.
- `scripts/diff_tlc.sh` — 13/13 specs match TLC v2.19 state counts.
- `scripts/chaos_smoke.sh` — 12/12 failpoints exercised, 0 divergences.
- `cargo test --release --test wrapper_next_fairness_t1_3` — 2/2 pass
  (T1.3 fairness regression).
- `cargo test --release --test streaming_init_t5_4` — 1/1 pass (T5.4
  producer regression).
- `cargo test --release --test cross_node_steal_handshake` — 3/3 pass
  (T6 handshake).
