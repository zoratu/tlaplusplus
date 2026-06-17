# Changelog

## v1.2.13 (2026-06-17)

RSS-based queue spilling is now on by default, plus a small successor-construction allocation win.

### PR #104 — `--queue-memory-ceiling-pct` defaults to 75 (RSS spill on by default)

v1.2.12 shipped the RSS-based spill trigger as opt-in (default 0/off). This makes it default-on at 75% of total RAM, so the in-memory state queue is memory-bounded out of the box: a run whose pending set would outgrow RAM spills to disk and stays responsive instead of getting OOM-killed. `0` still means explicit-off; lower it (e.g. 60) for more headroom. This is the one default-behavior change in the release — specs whose peak RSS stays under the ceiling never spill, so runs that fit in memory are unaffected.

Broadly validated on a 126 GB box (94 GB ceiling): `cargo test` 1,236/0/10, `diff_tlc` 13/13 (the gate runs 13 real specs through the binary with the new default on), and MCBinarySearch / CoffeeCan-100 / MCKVSSafetyMedium all complete with identical distinct counts. MCKVSSafetyMedium — the largest spec that fits — finishes with an empty spill directory (peak RSS under the ceiling, trigger never fires), confirming no regression for memory-fitting runs. Big specs get the protection validated in v1.2.12 (MCKVSSafetyLarge: RSS bounded 81–95 GB, responsive, ~2.1M distinct/min sustained where it previously OOM-thrashed).

### PR #105 — reuse existing key Arc on successor update

Per-successor state construction did `next.insert(Arc::from(var), value)` for every changed variable, where `next` is a clone of the parent state. Since the parent already contains every (fixed) state-variable key, a changed var is an update of an existing key — so the `Arc::from` allocation and the BTreeMap rebalance were unnecessary. Now `*next.get_mut(var) = value` reuses the existing key Arc and overwrites the value in place. Byte-identical result (fingerprints/serialization unchanged); +1.7% throughput on MCKVSSafetyMedium in a same-box A/B. The larger per-successor cost — the parent-state BTreeMap clone itself — remains; reducing it needs a structurally-shared map and is gated on fingerprint/serialization equivalence, so it's deferred.

### Gauntlet

| Gate | Result |
|---|---|
| `cargo test --release` | 1,236 pass / 0 fail / 10 ignored |
| `scripts/diff_tlc.sh` | 13 / 13 specs match TLC v2.19 |

Drop-in for v1.2.12 except the one intended default change above (RSS spill ceiling, which is transparent for memory-fitting runs and protective for the rest).

## v1.2.12 (2026-06-17)

Memory-bounded queue spilling. Two opt-in spill triggers that let the in-memory state queue stay bounded on specs whose pending set would otherwise outgrow RAM. Both default off, so existing runs are unchanged.

### PR #101 — byte-based spill trigger (`--queue-max-inmem-bytes`)

The queue previously spilled to disk only when the in-memory item COUNT crossed `--queue-max-inmem-items` (default 50M). For specs with large states that over-commits memory — an item count well under the cap can still be tens of GB when each state carries large nested records or accumulating logs. This adds a byte budget (accepts unit suffixes: `80GB`, `512MB`) that spills when either the count cap or the byte budget is crossed. The byte footprint is estimated from a sampled `bincode::serialized_size` EWMA seeded by `--estimated-state-bytes`. Sampling (1-in-4096 via `Iterator::inspect`) runs only when the budget is active, so the default count-based config pays zero new cost. E2E-verified: CoffeeCan-100 with a 1MB budget forces heavy spilling and still finds the identical 5,150 distinct states as the baseline.

### PR #102 — RSS-based spill trigger (`--queue-memory-ceiling-pct`)

The byte estimate above under-counts actual heap — a `TlaState`'s nested `BTreeMap`/`Arc`/`HashedArc` plus allocator overhead occupy ~1.5–2× the serialized size — so on a big spec the budget can stay under its threshold while real memory drives the box into swap. This adds a ground-truth trigger that spills based on actual process RSS. When RSS crosses the configured percentage of total RAM, workers spill successors to disk, the loader caps the hot queue at ~1M items under pressure (with hysteresis, never starving workers), and the spill coordinator periodically `malloc_trim()`s so freed heap returns to the OS and RSS reflects the spill.

Validated on MCKVSSafetyLarge (c6g.metal, 64 workers, `--queue-memory-ceiling-pct 75`, 20 min): RSS bounded — oscillates 81–95 GB instead of growing unbounded — the box stays responsive, throughput holds at ~2.1M distinct/min, and the run reaches 39.2M distinct and keeps climbing. The serialized-estimate approach on the same spec collapsed to ~5K states/min in swap thrash. MCKVSSafetyMedium with the same flag completes at the identical 17,220,672 distinct and never spills (peak RSS stays under the ceiling), so the trigger is dormant for specs that fit — no regression.

### Gauntlet

| Gate | Result |
|---|---|
| `cargo test --release` | 1,236 pass / 0 fail / 10 ignored (+2 new spill-trigger tests) |
| `scripts/diff_tlc.sh` | 13 / 13 specs match TLC v2.19 |

Drop-in for v1.2.11. No default-behavior change — both triggers are opt-in. Auto-enabling a sane RSS ceiling by default is a candidate follow-up pending broad corpus no-regression validation.

## v1.2.11 (2026-06-16)

Two related runtime fixes that surfaced when investigating an apparent hang on MCKVSSafetyLarge.

### PR #97 — flush local_popped at threshold

`WorkStealingQueues` already flushed `local_pushed` to `global_pushed` every `flush_threshold` pushes, but the symmetric `local_popped` counter was only flushed when a worker exited. During a normal run `pending_count() = global_pushed - global_popped` over-reported queue depth — the progress meter showed "stabilizing" / no progress for specs that were actually still draining. Workers themselves were unaffected (termination uses `has_pending_work()` which reads real deque state). Fix: symmetric `bump_local_popped` helper applied at all 7 pop sites.

### PR #98 — auto-switch FP store self-deadlock + silent state drop

With #97 making the queue depth visible, MCKVSSafetyLarge still didn't terminate within a 30-minute budget, with the log filling up with `auto_switch: batch try_read exceeded 10 attempts, switch_pending=true`.

Root cause: `AutoSwitchingFingerprintStore::contains_or_insert{,_batch}` called `maybe_trigger_switch()` **inside** the `match &*state` block — while still holding a read guard on `self.state`. `maybe_trigger_switch` → `switch_to_hybrid` → `self.state.write()` then self-deadlocks on parking-lot RwLock (calling thread holds a read guard on the same lock). Other workers' `try_read` keeps failing (writer queued, no new readers admitted), they bail out at attempt 21, but the original switching thread stays permanently stuck and `switch_pending` stays `true`.

**Important secondary discovery**: the bail-out path marks fingerprints as "seen" (duplicates) **without inserting them**. During the livelock window — which lasted indefinitely once it started — newly-found states were silently dropped from the FP store. The previously-reported "MCKVSSafetyLarge converged at 20,571,284 distinct" was an undercount; with the deadlock fixed and bail-out no longer firing indefinitely, the same spec finds 24M+ distinct states (and is still climbing past the wall budget on this run — exposes a separate scaling issue worth its own fix where the bloom-store auto-switch doesn't fire before in-memory FP store OOMs).

Fix: defer `maybe_trigger_switch` until after explicitly dropping the state read guard in both paths.

### Gauntlet

| Gate | Result |
|---|---|
| `cargo test --release` | 1,234 pass / 0 fail / 10 ignored |
| `scripts/diff_tlc.sh` | 13 / 13 specs match TLC v2.19 |

### Follow-up open

`maybe_trigger_switch`'s memory-pressure detection currently fires only on a `check_interval`-based sampling cadence. On MCKVSSafetyLarge-scale specs with high state-generation rate the in-memory FP store outgrows 85% RAM faster than the next sampled check, so the bloom transition never fires and OOM-killer terminates the run. Tuning either `check_interval` or the memory-pressure trigger is the natural next step; not in this release.

Drop-in for v1.2.10. State-count reductions previously reported on MCKVSSafetyLarge under prior releases were likely undercounts due to the silent-drop bug above; re-running gives larger (correct) counts.

## v1.2.10 (2026-06-15)

Symmetry-reduction correctness fix that unlocks **MCKVSSafetyMedium
full-MC completion**.

### The bug

`canonicalize_tla_state` in `src/symmetry.rs` claimed to compute the
lexicographically smallest permutation of symmetric values, but the
implementation sorted the values that *appeared* in the state and
mapped them to the alphabetically-first labels of the symmetric group.
For any state that uses all symmetric values (the common case for
mid-to-late exploration), this degenerates to the identity permutation
— no reduction in practice, regardless of the `SYMMETRY` clause.

### The fix (PR #95)

Enumerate all permutations of each symmetric group (capped at 5! = 120
per group), apply every combination to the state, and return the
lex-min candidate. Two states in the same symmetry orbit therefore
collapse to the same canonical form, which is the contract the
fingerprint store relies on.

### Measured impact (c6g.metal ap-south-1, 64 workers)

| Spec | Before | After |
|---|---|---|
| MCKVSSafetyMedium 600s | 32,109,138 distinct, queue growing | **COMPLETES** at 17,210,068 distinct (≈ the 2× orbit size of TxIdSymmetric) |
| MCCheckpointCoord 5min | 7,053,401 distinct, queue growing | 3,391,543 distinct, queue growing (~2.08×) |

MCKVSSafetyMedium full-MC was previously the most prominent
"exploration-bound at 600s" spec on the corpus; with correct symmetry
it completes within the budget. **9 of the original 11 timeout
specs now resolved.**

MCCheckpointCoord's reduction is ~2× rather than the nominal 6×
(3-element NodeSymmetry) because not every state variable carries
symmetric values — expected behavior, not a bug.

### Gauntlet

| Gate | Result |
|---|---|
| `cargo test --release` | 1,234 pass / 0 fail / 10 ignored (+2 new symmetry tests) |
| `cargo test --release --features failpoints` | 1,256 pass / 0 fail / 10 ignored |
| `cargo test --release --features symbolic-init` | 1,261 pass / 0 fail / 10 ignored |
| `scripts/diff_tlc.sh` | 13 / 13 specs match TLC v2.19 |

Two new unit tests pin the orbit-collapse contract for 2- and
3-element symmetric groups.

### Cost

Per-state canonicalize cost goes from O(N) to O(N! × N) where N is the
symmetry-group size. With the 5-element cap, worst case is 120
candidates per state. For typical TLA+ specs (2–4 element symmetric
groups) it's 2–24 candidates. The state-count reduction recovers
this overhead for any spec where symmetry actually reduces the orbit.

Drop-in for v1.2.9. The semantics change for specs with SYMMETRY:
fingerprints for symmetric-state pairs now collapse correctly, which
means state counts on those specs will be lower (correct) than
before. Spec checkpoints from v1.2.9 and earlier should not be
resumed under v1.2.10 because the fingerprint shape for symmetric
states differs.

## v1.2.9 (2026-06-14)

Tail-end perf cleanups (PRs #91, #92) and a corpus re-sweep that closes
**8 of the original 11 timeout specs** on `tlaplus/Examples`.

- **PR #91**: skip a redundant `unchanged_vars` re-insert loop in the
  successor-state constructor. Every Unchanged-clause handler already
  populates `branch.staged` with the unchanged value via
  `entry(var).or_insert_with(|| current_value.clone())`, so the
  subsequent staged-iteration loop already covers the same key with the
  same value. +2% distinct@600s on MCKVSSafetyMedium; mostly a cleanup.
- **PR #92**: drop `format!` from the operator/action compile-cache
  lookup hot path. Both `THREAD_LOCAL_OPERATOR_CACHE` (compiled_eval)
  and `THREAD_LOCAL_ACTION_CACHE` (action_exec) keyed each lookup on
  `format!("{}:{}", name, body)` — a per-hit heap allocation. The cache
  value is determined by body alone (name was a debug label), so we now
  look up by `&str` and skip the allocation. Both caches also switch
  `std::HashMap` → `ahash::AHashMap` (SIP's HashDoS resistance is
  unnecessary overhead for thread-local non-adversarial keys; the
  pre-warmed `DashMap` already defaults to ahash). Perf-neutral on
  this measurement, contained cleanup.

### Corpus unlocks on current main (cumulative over PRs #84 — #92)

Re-sweep on c6g.metal ap-south-1, 64 workers, against the original
"11 timeout specs" list:

| Spec                      | Was             | Now |
|---------------------------|-----------------|------|
| MCBinarySearch            | ∞ stuck         | **1 s**  (27,953 distinct, fairness verified) |
| CoffeeCan-100             | timeout         | **16 s** (5,150 distinct) |
| CoffeeCan-1000            | record-set cap crash | **9 s** (501,500 distinct) |
| CoffeeCan-3000            | crash → timeout | **1 min 23 s** (4,504,500 distinct) |
| btree                     | timeout @ 331K  | **33 s** — finds fairness violation in an 11,340-state SCC |
| Elevator-SafetyMedium     | timeout         | **1 s** — finds 2 invariant violations |
| Elevator-SafetyLarge      | timeout         | **6 s** (390,625 distinct) — finds 2 violations |
| Einstein                  | timeout (199M cross-product) | **1 s** under `--features symbolic-init` |
| MCKVSSafety Medium/Large | bounded         | **31.5M / 22.6M distinct @ 600s** (+6.5× / +2.5× since pre-#84) |

Of the remaining three:
- **c1cs** is genuinely exploration-bound — throughput is ~30× higher
  than pre-#84 (62K distinct in 15 min vs ~2K previously), but the
  state space is large enough that a 15-minute budget still doesn't
  finish.
- **MCCheckpointCoordination** runs correctly when the MC wrapper
  module is used (`MCCheckpointCoordination.tla`, not the base
  `CheckpointCoordination.tla`). At 5 minutes / 64 workers it has
  explored 7M distinct states with the queue still growing —
  exploration-bound rather than algorithmic-stall. The cfg comment
  ("Finishes in around one minute") suggests TLC's symmetry reduction
  is more effective on this spec than ours; worth a separate look.
- **MCKVSSafety Medium/Large** remain "bounded": they still don't
  fully enumerate at 600s, but throughput on them more than tripled
  (1.4M distinct/min Medium, 2.4M distinct/min Large).

The four newly-found violations on Elevator + btree are model checker
output (the runtime does its job), not runtime bugs; whether they
indicate real spec issues is a separate question for the spec authors.

### Cargo features

`--features symbolic-init` builds need
`BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include/z3"` on Amazon Linux 2023
where the `z3-devel` package installs headers under `/usr/include/z3/`
instead of `/usr/include/`. This is an env-var-only workaround; no
Cargo/build changes required.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,232 pass / 0 fail / 10 ignored |
| `scripts/diff_tlc.sh` | 13 / 13 specs match TLC v2.19 |

Drop-in for v1.2.8. No public-API or CLI changes.

## v1.2.8 (2026-06-13)

Eval-path performance batch. Six merged PRs (#84 — #89) compound through
the interpreted/compiled hot paths and the fingerprint store. Net result on
MCKVSSafetyMedium full-MC (c7g.16xlarge eu-north-1, 64 workers, 600s budget):
distinct@600s **4,826,272 → 31,482,090 (+6.5×)**; states/min generated
**4.86M → 56.66M (+11.7×)**.

| PR  | What                                                       | Win on MCKVSSafetyMedium |
|----:|------------------------------------------------------------|--------------------------|
| #84 | `HashedArc<T>` — cache structural hash on TlaValue arcs     | ~neutral on full-MC (-14% on `analyze-tla` MCBinarySearch) |
| #85 | Cache `TLAPP_TRACE_*` env probes (compiled-side) in `OnceLock` | distinct@600s 4.78M → 10.53M (+2.5×) |
| #86 | Bump default `shard_count` 128 → 256                        | distinct@600s 10.45M → 11.23M (+7.5%) |
| #87 | Short-circuit `x ∈ {y ∈ S : P(y)}` filter-set membership    | MCBinarySearch full-MC ∞ → 1s (~17,000×); CoffeeCan-100 unlocked |
| #88 | Structural short-circuit for record-set `x ∈ [field: D, …]` | CoffeeCan-1000/3000 unlocked from `record set too large` cap |
| #89 | Cache `TLAPP_TRACE_EVAL` / `TLAPLUSPLUS_DEBUG_SYMBOLIC_INIT` env probes | distinct@600s 11.85M → 31.48M (+2.66×) |

Three previously-timing-out specs are now solved or progressing under 60s
budgets (MCBinarySearch, CoffeeCan-100/-1000/-3000, plus Einstein under
`--features symbolic-init`). c1cs and btree remain genuinely
exploration-bound at minutes-scale.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,232 pass / 0 fail / 10 ignored |
| `cargo test --release --features failpoints` | 1,254 pass / 0 fail / 10 ignored |
| `cargo test --release --features symbolic-init` | 1,259 pass / 0 fail / 10 ignored |
| `scripts/diff_tlc.sh` | 13 / 13 specs match TLC v2.19 |
| `PROPTEST_CASES=2048 cargo test --release --test compiled_vs_interpreted` | 17 / 0 (17 strategies × 2048 cases ≈ 34K) |
| `scripts/chaos_smoke.sh` (swarm-mode auto) | 12/12 failpoints, 0 divergences, 0 hangs |

Drop-in for v1.2.7. No public-API or CLI changes — all wins are runtime
defaults and eval-path micro-optimizations. Caching debug-trace env probes
in `OnceLock` means the trace-eprintln paths still fire if the env var is
set at process start; setting them mid-run is unsupported (was racy before;
deterministically off now).

## v1.2.7 (2026-05-20)

Verus T13.4 Phase 2 in-place annotation expansion. `src/storage/verus_smoke.rs` grows
from 12 to 38 verified helper functions driving **177 shipping call sites across 31
shipping files** (was ~58 sites / ~9 files at the v1.2.7 cycle start). New helpers:
`min_usize` / `max_usize` / `min_u64` / `max_u64` / `min_u32` / `max_u32` / `max_u16`
(stdlib `Ord::min` / `Ord::max` substitutes since Verus doesn't yet spec them),
`clamp_usize` / `clamp_u64`, `saturating_add_u64` / `saturating_sub_u64` /
`saturating_dec_i32_to_zero`, plus a recursive `gcd` with `decreases b` termination
proof. The cfg-split pattern keeps default builds unchanged; `--features verus`
activates verified paths.

Status correction: T5.4 (streaming Init enumeration), T5.5 (joint Init+invariant Z3
encoding / Einstein 44 min → 14 ms), T10.2 (streaming SCC, phase 1 + phase 2 stages
1–5), and T13.5 (`state_machines!` port) were already shipped — `CLAUDE.md`'s
"Deferred to v1.1.x" section is updated to reflect actual status. The remaining
T13.4 full lift (`Vec<PAtomicU64>` + `Tracked<Map<int, PermissionU64>>` rewrite of
`FingerprintShard`) is defensibly parked per `verification/verus/T13.4-PHASE2-CLOSURE.md`
— all original verification-research blockers resolved in 186 verified items across 11
standalone artifacts; what remains is mechanical engineering work.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,227 pass / 0 fail / 11 ignored |
| `cargo test --release --features failpoints` | 1,249 pass / 0 fail / 11 ignored |
| `PROPTEST_CASES=2048 cargo test --release --test compiled_vs_interpreted` | 17 / 0 (16× default) |
| `scripts/chaos_smoke.sh` (swarm-mode auto) | 12/12 failpoints, 39 concurrent pairs, 0 div, 0 hangs |
| `cargo verus check --features verus` (shipping) | 49 verified, 0 errors |
| Verus standalone (9 of 11 tiers) | 144 verified, 0 errors |
| `scripts/chaos_soak.sh --duration 86400 --swarm-mode auto` (post-release; cut short by spot reclaim at ~12.7h) | 2,526 iters, 100% verdict=ok, 0 div, 0 hangs |

See `RELEASE_1.2.7_SUMMARY.md` for the full helper inventory, file-coverage list,
the T13.4 closure-status update, and the extended chaos-soak detail.

Drop-in for v1.2.0–v1.2.6. No public-API or CLI changes.

## v1.2.6 (2026-05-10)

T10.2 phase 2 stage 5, Layer B. The single-node multi-worker DFS pool shipped in v1.2.5 is extended to span multiple cluster nodes via cross-node routing.

A new `src/runtime/dfs_cluster_bridge.rs` provides the async-to-sync bridge between `Transport::recv()` and the per-worker crossbeam mpsc inboxes. `src/runtime/dfs_pool.rs` gains a `DfsPoolClusterCtx` and cluster-aware partition routing: each worker's partition is `(node_id, worker_id)`; local partitions stay on the local crossbeam mpsc, remote partitions ship over the `Transport` as `PartitionEdge` messages. Termination uses a two-round consensus over the existing `TerminationToken.inflight_partition_edges` field.

`tests/dfs_cluster_layer_b.rs` (4 default + 1 ignored memory benchmark) uses the existing `MockTransport` (T204, v1.2.1) to wire two transport instances through a shared `MockNetwork`. No real TCP. Identical liveness verdicts and identical state-distinct counts confirmed across single-node pool vs 2-node cluster runs of the same fairness spec.

Per-node cluster memory at dim=8 (64 states): single-node DFS 0.6 MiB delta, 2-node cluster 0.8 MiB total → 0.4 MiB per node = 0.64× single-node baseline.

Deliberate carve-outs: cross-partition red-DFS send path stays log-only (verdict comes from per-node in-band Tarjan); `RequestStateBlob` send-side stays a no-op (trace reconstruction across nodes deferred); `--dfs-cluster-listen` CLI flag wiring deferred (shipping multi-node DFS pool runs go through `dfs_cluster_test_api`).

| Gate | Result |
|---|---|
| `cargo test --release` | 1,227 pass / 0 fail / 11 ignored |
| `cargo test --release --features failpoints` | 1,249 pass / 0 fail / 11 ignored |
| `cargo test --release --features symbolic-init` | 1,254 pass / 0 fail / 11 ignored |
| `dfs_cluster_layer_b` (NEW) | 4 / 4 (+ 1 ignored, 0.64× per-node memory ratio ✓) |
| `cross_node_steal_handshake` (T6 cluster mode unaffected) | 3 / 3 |
| Verus proofs (6 files) | 133 verified items, 0 errors, 0 axioms |

Drop-in for v1.2.0–v1.2.5. No public-API or CLI changes.

## v1.2.5 (2026-05-10)

T10.2 phase 2 stage 5, Layer A. The single-worker DFS exploration shipped in v1.2.3 / v1.2.4 is lifted to a fingerprint-partitioned worker pool with cross-partition routing.

A new `src/runtime/dfs_pool.rs` implements an N-worker DFS pool. Each worker owns the partition `partition_for_fp(fp, N) == self.id`. Per-worker DFS stack, per-worker `LocalAdjacency`, per-worker crossbeam mpsc inbox; cross-partition successors get shipped over the owner's outbox channel. Termination uses an `AtomicI64` Mattern in-flight counter with two-round confirmation (`inflight == 0 && all_idle && my_inbox_empty`). After join the per-worker triples are merged and the v1.2.4 in-band fairness check runs once over the union, so verdict equivalence holds by construction.

`tests/dfs_pool_throughput_benchmark.rs` on `HighFanoutGrid` dim=250 (62,500 states, 311,500 edges): 1-worker DFS 6.11 s / 326.2 MiB; 4-worker pool 2.10 s / 346.6 MiB. Speedup 2.91×, memory ratio 1.06×. Stage 4's BFS-vs-DFS memory win carries through (DFS / BFS = 0.57 on the 360K-state benchmark).

`tests/dfs_pool_parity.rs` (3 tests) confirms identical verdicts and identical state-distinct counts across pool sizes 1 / 2 / 4.

A new `--dfs-workers` flag controls pool size (1 = stage-4 single worker).

Layer B (multi-node cluster DFS via the existing `PartitionEdge`/`RedDfsProbe` protocol variants) deferred to v1.2.6.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,220 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,242 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,247 pass / 0 fail / 8 ignored |
| `dfs_pool_parity` (NEW) | 3 / 3 |
| `dfs_pool_throughput_benchmark` (NEW) | 4-worker speedup 2.91× ✓ |
| Verus proofs (6 files) | 133 verified items, 0 errors, 0 axioms |

Drop-in for v1.2.0–v1.2.4. No public-API changes; `--dfs-workers` is additive and defaults to 1.

## v1.2.4 (2026-05-10)

T10.2 phase 2 stage 4. The streaming-SCC memory win is realized: the DFS exploration path no longer materializes the labeled-transitions adjacency map.

`DfsWorkerCtx` no longer carries the `labeled_transitions` field. The DFS worker builds a thin local `LocalAdjacency = Vec<(u64, u64, String)>` (source fingerprint, destination fingerprint, action name) — no state clones. A new `run_inband_fairness_check` runs Tarjan + the existing `check_fairness_on_scc_fp_sharded` predicate on the local triple list when DFS exploration completes. A new `dfs_inband_verdict_done: Arc<AtomicBool>` is shared with `runtime/shutdown.rs`; when set, `liveness::run_post_processing` is skipped entirely.

A new `tests/dfs_memory_benchmark.rs` (`#[ignore]` by default) runs a synthetic `HighFanoutGrid` spec at 600×600 (360,000 reachable states, 1,797,600 transitions): BFS peaks at 877 MiB / 3 s, DFS at 513 MiB / 24 s — DFS / BFS = 0.59, a 41.5% reduction. DFS is single-worker by Stage 4 design; the deliverable is the memory ratio. Stage 5 lifts single-worker DFS to multi-worker.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,212 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,234 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,239 pass / 0 fail / 8 ignored |
| `dfs_worker_parity` (verdict equivalence) | 4 / 4 |
| `dfs_memory_benchmark` (NEW) | DFS / BFS = 0.59 ✓ |
| Verus proofs | 133 verified items, 0 errors, 0 axioms |

Drop-in for v1.2.0–v1.2.3. No public-API changes.

## v1.2.3 (2026-05-10)

T10.2 phase 2 stage 3. The hot-loop DFS exploration that was the headline parked item across v1.2.1 and v1.2.2 lands via a scoped third approach.

A new file `src/runtime/dfs_worker.rs` implements a separate single-worker single-node DFS exploration function with its own ctx struct. It drops features irrelevant to streaming-SCC mode: no checkpoint pause, no cluster steal, no init producer, no auto-tune throttle, no backpressure. `src/runtime/worker.rs` is byte-unchanged; default behaviour is identical to v1.2.2.

A dispatch branch in `src/runtime.rs` checks `--liveness-streaming-exploration`, `model.has_fairness_constraints()`, and the absence of a distributed stealer. When all three hold, the run path spawns one DFS worker instead of the normal worker fleet.

The DFS worker uses in-band Tarjan-style coloring via the `PageAlignedColorMap` shipped in v1.2.1. Per-DFS-frame stack with `BlueFrame { state, successor_iter, color_owned }`. The nested-DFS red probe fires when an accepting state's frame pops (Gray → Black).

`tests/dfs_worker_parity.rs` adds 4 BFS-vs-DFS parity tests (WrapperNextFairness, NamedSubaction, ThreeCycle, SafetyOnly) — identical state counts and liveness verdicts.

The predicted >50% RSS reduction at 100M+ scale is not realized in this stage — DFS still populates `labeled_transitions`. That lift is Stage 4 (v1.2.4). What this stage delivers is the architecture on which Stage 4 can drop labeled_transitions without touching BFS.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,212 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,234 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,239 pass / 0 fail / 8 ignored |
| Verus proofs (6 files) | 133 verified items, 0 errors, 0 axioms |

Drop-in for v1.2.0 / v1.2.1 / v1.2.2. No public-API changes.

## v1.2.2 (2026-05-10)

Patch release continuing the long-parked items work.

T13.5 `state_machines!` port shipped axiom-free. LTL-native restatement of `theorem_no_starvation` via Verus's `state_machine!` macro plus refinement bridge to `reader_liveness_v2.rs`. New file `verification/verus/reader_liveness_state_machine.rs` (15 verified items). Zero axioms.

T13.4 wrapper extension: a new `bounded_seqlock_retry_contains` in `shard_wrapper.rs` (+3 verified items, was 31 now 34). T13.4 Phases 2+3 (shipping-code annotation) remain parked behind 3 documented `vstd` capability gaps.

Verus totals: 115 → 133 verified items (+18) across 6 proof files.

T10.2 phase 2 stage 3 partial: wires `PageAlignedColorMap` into the post-BFS oracle path behind the new `--liveness-streaming-exploration` flag (defaults OFF). Validates the data structure end-to-end; 3 new parity tests (Tarjan vs color-map). The actual hot-loop DFS lift (the headline Stage 3 deliverable) remains parked — same risk surface as the chunk-7 worker.rs refactor.

| Gate | Result |
|---|---|
| `cargo test --release` | 1,205 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,222 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,227 pass / 0 fail / 8 ignored |
| Verus proofs (6 files) | 133 verified items, 0 errors, 0 axioms |

Drop-in for v1.2.0 / v1.2.1. No public-API or CLI changes.

## v1.2.1 (2026-05-10)

Patch release closing out the v1.2.0 deferred-items list. 25 commits on top of v1.2.0; no new user-facing features and no behavioural change. Drop-in for v1.2.0.

### v1.2.0 deferred items — all closed

T204 distributed mock for the cluster handler — `Transport` trait + `MockTransport` (in-memory tokio mpsc channels); `DistributedWorkStealer` switched to `Arc<dyn Transport>`. 10 new tests exercise the inbound handler / bloom-and-termination / steal-trigger paths without real TCP.

`src/runtime.rs` extraction chunks 7 + 8 — both extracted. Chunk 8 (13-step shutdown phase) → `src/runtime/shutdown.rs` (`ShutdownContext` + `orchestrate`). Chunk 7 (worker spawn loop, 27-Arc capture) → `src/runtime/worker.rs` (`WorkerLocalState`).

`src/tla/eval.rs` split — all 8 chunks (A–H) of the design doc landed. The monolith becomes 13 submodule files under `src/tla/eval/`. External API surface unchanged.

Compiler internal-helper restructuring — depth-tracking consolidation (148 + 26 → 4 + 1 sites). Plus 6 more iterations of targeted tests (T207b–T207h, +149 tests). Compiler mutation kill rate: 65.4% → 70.5% across 11 iterations.

### Older parked items partially landed

T10.2 phase 2 stages 1+2 of 5 — strictly-additive foundation. Stage 1: `PageAlignedColorMap` (7 unit tests) — 2-bit Color enum, hugepage mmap, NUMA-shard placement, lock-free CAS. Stage 2: protocol variants (`PartitionEdge`, `RedDfsProbe`, `RequestStateBlob`, etc.) + `Option<u64>` extensions on `TerminationToken`. Stages 3–5 still parked.

T13.4 Phase 1 — shipping-shape verified wrapper at `verification/verus/shard_wrapper.rs` (31 verified items). Bounded outer probe loop now verified — closes the gap `shard_methods.rs` deferred. 115 total Verus items verified (was 84). Phases 2+3 parked behind documented `vstd` capability gaps.

### Validation

| Gate | Result |
|---|---|
| `cargo test --release` | 1,197 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,219 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,224 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| Verus proofs (5 files) | 115 verified items, 0 errors |
| Mutation testing — `eval.rs` (interpreter) | 100% kill rate |
| Mutation testing — compiled_eval.rs + compiled_expr.rs | 70.5% kill rate (8 iters) |

## v1.2.0 (2026-05-07)

Patch release driven by extensive fuzz + mutation testing. Six T20X soundness fixes in the compiled-eval / compiled-expr paths, two large refactors that improve testability without behavioural change, and +281 targeted compiler-helper tests across 8 iterations that lift the compiler mutation kill rate from 42% → 65%.

### Compiler-vs-interpreter soundness (T201–T207)

- **T201** — Fuzz harness OOM. Six allocator sites (range, set comprehension, function constructor) now charge `ctx.check_budget` before allocating. Fuzz harness raised to `-rss_limit_mb=8192`.
- **T202** — Compiler `LAMBDA` parser was looser than interpreter. Tightened to mirror `eval.rs::parse_lambda` word-boundary rules.
- **T203** — `LET`-binding eval OOM. Compiler now charges `local_definitions.len() + defs.len()` before `with_local_definitions`.
- **T204** — Per-call entry-point eval budget tick.
- **T205** — Bare `IOEnv` / `EmptyBag` in compiled eval. Compiler now dispatches bare zero-arg builtins instead of falling through to `ModelValue`.
- **T206** — Chained binary `+`/`-` arithmetic. Compiler detects 3+ chains via `has_chained_top_level_arithmetic` and emits `CompiledExpr::Unparsed` to delegate to the interpreter (the reference). Generalises the per-shape T101.1 fixes.
- **T207** — Compiler typed `SubSeq` accepted `m=0`. Surfaced by mutation testing. `CompiledExpr::SubSeq` now mirrors the interpreter's `m >= 1` validation.

### Testing infrastructure — mutation kill rate 42% → 65%

Eight iterations of compiler-helper coverage:

- **T207 (iter 1, +66 tests).** Arity guards, scope protection, membership shapes, short-circuit. **42.4% → 55.7%**. Surfaced T207 SubSeq m<1 soundness fix.
- **T207b (iter 2, +46 tests).** Chain-detection boundaries, community-module ops, arithmetic edges. **55.7% → 59.4%**.
- **T207c (iter 3, +77 tests).** Deep-recursion stress (280+ levels), direct unit tests on private scanner helpers. **59.4% → 61.3%**.
- **Iter 4 (assertion fix).** Tightened deep-recursion assertions from Ok-or-Err to Err-only. **61.3% → 62.8%**.
- **T207d (iter 5, +40 tests).** Direct boundary tests for scanner helpers — quantifier/EXCEPT/definition-equals/relop-disambiguation. **62.8% → 63.7%**.
- **T207e (iter 6, +28 tests).** More built-in deep recursion, exhaustive membership shapes. **63.7% → 65.3%**.
- **T207f (iter 7, +6 RECURSIVE tests).** `RECURSIVE Op(_)` chains for user-defined op dispatch. **65.3% → 66.0%**.
- **T207g (iter 8, +18 tests).** Deep recursion through every relop and arithmetic dispatch arm. **66.0% → 65.4%** (mutation-to-mutation variance dominates at this point).
- **Dead code.** Removed `split_top_level_old` (annotated `#[allow(dead_code)]`, never called) — surfaced as 158 noise mutants in iter 1.

Convergence at ~65–66% reflects equivalent mutants in compiler internal helpers (multiple `depth + 1` sites where the outer dispatch's depth check fires first, masking inner mutations). The compiler's real soundness guarantee is the layered safety net — T2 proptest equivalence
+ `fuzz_tla_swarm` + diff-vs-TLC + state-graph snapshots. Every real
soundness bug found this cycle (T201–T207) came from those layers.

### Refactors (no behavioural change)

- **`src/main.rs` split.** CLI dispatch tree split into 12 modules under `src/cli/`.
- **`src/runtime.rs` partial split.** Seven extraction chunks landed (PauseController, checkpoint manifest, memory budget, shard count, AtomicRunStats, T5.4 init producer, T10 liveness post- processing, distributed handler wiring, progress tick); +64 unit tests.
- **T204.1.** `DistributedWorkStealer::print_stats()` wired into the cluster run summary so steal counters surface by default.

### Validation

| Gate | Result |
|---|---|
| `cargo test --release` | 1,155 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 1,177 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 1,182 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| Mutation testing — `eval.rs` (interpreter) | 100% kill rate |
| Mutation testing — compiled_eval.rs + compiled_expr.rs | 65.4% kill rate (8 iters) |

Drop-in for v1.0.x and v1.1.x. No public-API or CLI changes.

## v1.1.0 (2026-04-25)

Feature release rolling up the post-1.0 sweep — the first wave of items the v1.0.0 plan had marked **DEFER TO 1.1.0**, plus a mid-cycle soundness fix surfaced by the differential gates and a Verus shipping-shape proof tier.

### Symbolic Init — joint encoding (T5.4 + T5.5)

- **T5.4 — Streaming Init enumeration via a producer thread.** The `Model::initial_states()` contract gained a streaming variant; large cross-product Inits no longer materialize before the first worker can start, and the Init producer overlaps with state exploration. This is the architectural prerequisite that made T5.5 representable.
- **T5.5 — Joint Init + invariant Z3 encoding.** When `--features symbolic-init` is enabled and the spec's invariants are reducible to the Init shape, Init constraints and invariants are encoded as a single Z3 formula and enumerated together rather than enumerate-then- filter. End-to-end on the Einstein-class spec the v1.0.0 plan called out as the canonical hard case: **44 min → 14 ms** wall-time, including Z3 startup. Reverts cleanly to the v1.0.0 enumerate-then- filter path when the invariant body is not Z3-reducible.

### Liveness scaling (T10.1 / T10.2 / T10.3 / T10.4)

- **T10.1, T10.3, T10.4 — Liveness post-processing follow-ups.** Parallel-flatten of the SCC transition table via dashmap raw shards (~20% on the LivenessBench shape at N=10), trivial-SCC pre-filter for sparse graphs, per-action transition shard so per-constraint checks hit only the relevant shard (~6x per-constraint check on multi-WF shapes).
- **T10.2 — Streaming-SCC oracle (nested DFS), opt-in via `--liveness-streaming`.** Phase-1 oracle: a structural self-test that validates the nested-DFS ordering against the existing iterative Tarjan output on every supported shape. Includes the O(N) red-DFS fix (Cyan color check rather than rebuilding the `blue_path` HashSet). Phase-2 — driving the oracle from the live exploration frontier — has its design pinned and is tracked for v1.2.0.

### Trace minimization (T9.1 / T9.2 / T9.3)

- **T9.1 — Transitive variable relevance.** The relevance scan now follows operator-inlining edges, so variables that only reach the invariant through a chain of helper definitions are correctly marked as relevant rather than collapsed to "(noise)".
- **T9.2 — Median BFS seed.** The BFS shortcut search now seeds from the median of the violation-trace prefix rather than the head, which cuts shortcut-search work on long traces.
- **T9.3 — Suffix shortening.** A second pass attempts to shorten the post-violation suffix by replaying alternate transitions; useful when the original trace took a long detour to reach the witness state.

### Partial-order reduction follow-ups (T7.1 / T7.2 / T7.3)

- **T7.1 — Batched per-disjunct evaluation.** The stubborn-set computation now batches per-disjunct enabledness checks rather than re-evaluating each disjunct's guard from scratch. PorBenchProcessGrid: 19.7x → **39.2x** state reduction.
- **T7.2 — Smarter stubborn-set seed.** Seed selection now prefers actions with the smallest dependency closure rather than picking the first enabled action in source order, which produces consistently smaller stubborn sets on irregular dependency graphs.
- **T7.3 — POR for liveness via the Peled (1994) visible-action proviso.** Lifts the v1.0.0 safety-only restriction. POR now composes with WF/SF fairness when the visible-action proviso is satisfiable; the runtime still rejects POR cleanly on shapes where it is not.

### Robustness (T11.3 / T11.4 / T11.5 / T12.1)

- **T11.3 — Per-PR chaos-smoke gate.** A 5-minute CI variant of the 1-hour soak that runs every PR; covers the same swarm-mode failpoint matrix at lower iteration depth so regressions in chaos-tolerance surface before merge.
- **T11.4 — `route_spill_batch` Err-branch inflight leak fix.** The spill router's error path was leaking the `inflight_spilled` counter, which caused `has_pending_work()` to return false even when items were stuck in the spill pipeline; observed under fault injection. Now decremented in the Err arm so termination detection stays consistent with the queue's actual contents.
- **T11.5 — Violation-exit hang fix.** Workers spinning in `pop_slow_path` did not observe `queue.finish()` when a violation set `worker_stop=true` at NUMA-auto worker counts, so they never exited. `pop_slow_path` now checks `self.finished` alongside `pause_requested` and the violation handler calls `worker_queue.finish()`.
- **T12.1 — Recursive-depth test stack.** The deliberate-unbounded- recursion test was crashing on x86_64 because the default thread stack was below the test's 8 MB depth; fixed by allocating an 8 MB thread stack for that test only.

### Verification (T13.1 – T13.6 partial)

- **T13.1 + T13.2 + T13.3 — Verus tier A.** A 31-lemma proof at `verification/verus/seqlock_resize_tier_a.rs` covering a `Seq<u64>` linear-probe model, spec-level CAS soundness, and bounded reader-retry termination. Sits one rung below the shipping code: the model is the shipping protocol, not the shipping pointers.
- **T13.4 partial — Tier-A.5 shipping-shape shadow methods.** A new shadow-method tier in the Verus crate adds 17 lemmas that mirror the hot-path `FingerprintShard` methods one-to-one, which closes the shape gap between the tier-A model and the shipping code without yet threading `Tracked<PointsTo<HashTableEntry>>` through the real type. Phase-2 (shipping-code annotations) tracked for v1.2.0.
- **T13.3 + T13.5 — Constructive reader-liveness proof, axiom-free.** A new proof at `verification/verus/reader_liveness_v2.rs` discharges the unbounded-fairness reader-liveness theorem `theorem_no_starvation` with **0 axioms** (17 verified, 0 errors via `./run_proof.sh reader-liveness-v2`, ~0.8s wall). The three previous protocol-shape axioms (`axiom_writer_eventually_finalizes`, `axiom_reader_can_observe_stutter`, `axiom_extension_composes`) are replaced by constructive lemmas with explicit short-`seq!` witnesses (2- and 3-element extensions); `wf_prefix` over each short literal reduces to a small case-split on `step_relation`'s three disjuncts that the SMT solver discharges automatically. The original `verification/verus/reader_liveness.rs` (14 verified plus 3 documented `external_body` axioms) is preserved as the bounded-form temporal-trace fallback and reference for the eventual `state_machines!` port.
- **T13.6 — CI gate for the Verus tier-A run.** A workflow gate that runs the full Verus tier-A proof on every push so a regression in the model-level guarantees fails the build rather than silently accumulating.

### Compiled-vs-interpreted soundness fix (T101.1)

- **T101.1 — Compiler arithmetic associativity.** The compiled-IR parser was associating chained `+`/`-` differently than the interpreter on five distinct shapes, which produced silent Ok-vs-Err and value divergences that the T2 proptest harness caught. Tightened the compiler parser to match interpreter validation exactly. New regression cases pinned in `tests/compiled_vs_interpreted.rs`. **Soundness fix** — these are not panic-resistance fixes, they were genuine wrong answers from the compiled path.

### Validation gates

| Gate | Result |
|---|---|
| `cargo test --release` | 786 pass / 0 fail / 8 ignored |
| `cargo test --release --features failpoints` | 808 pass / 0 fail / 8 ignored |
| `cargo test --release --features symbolic-init` | 813 pass / 0 fail / 8 ignored |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `cargo test --release --test compiled_vs_interpreted` (PROPTEST_CASES=2048) | 17 pass / 0 fail |
| `cargo test --release --features symbolic-init --test joint_init_invariant_t5_5` | 3 pass / 0 fail |

### Compatibility

Drop-in for v1.0.x. No public-API or CLI changes; new behaviour is opt-in (`--features symbolic-init` for T5.4/T5.5, `--liveness-streaming` for T10.2 oracle). Fingerprint format, checkpoint format, and state-graph dump format are all unchanged.

## v1.0.1 (2026-04-25)

Patch release covering the **"Bugs Rust Won't Catch"** audit — three sub-audits (T101 / T102 / T103) targeting the classes of defects that the Rust type system cannot catch on its own: parser/compiler panics on adversarial input, silently-discarded `Result` values, and lossy UTF-8 conversions.

### T101 — Parser/evaluator panic-resistance (fuzz audit)

- Stood up `cargo-fuzz` targets across the TLA+ parser and the compiled-IR evaluator and ran them long enough to drive crash counts to zero.
- Fixed **four panic classes** in the parser/compiler, all stemming from `&str[..]` slicing on non-character boundaries when the input contains non-ASCII bytes — affected sites included the indexed-op-call parser, recursive-decl parser, INSTANCE substitution, CFG comment stripping, and two LET-binding range computations in `compiled_expr` and `eval`.
- Added **7 regression tests** in `tests/fuzz_panic_regressions_t101.rs` pinning each crash so a future regression is caught immediately.
- Wired the swarm-equivalence fuzz target so symmetry-reduced and un-reduced runs are diff-checked on every fuzz iteration.

### T102 — `Result`-discard audit (silent-error audit)

- Audited every `let _ = ...`, `.ok()`, and `#[must_use]`-bypass in the codebase and classified each as intentional (logged channel-closed, best-effort cleanup) or a real bug.
- **Headline fix:** the runtime's per-worker `error_tx` could deadlock under concurrent send when the receiver had already drained — converted to a non-blocking try-send with an explicit drop-on-full path so workers never block on error reporting.
- Fixed **9 additional propagation sites** where I/O errors, checkpoint-write failures, and parser warnings were being swallowed silently; each now either propagates upstream or logs a structured warning.

### T103 — Lossy UTF-8 conversion audit

- Audited every `String::from_utf8_lossy` and `OsStr::to_string_lossy` call site to confirm none were on a soundness-critical path (state hashing, fingerprint identity, action equality).
- **Two fixes,** both on defensive observability paths:
  - S3 checkpoint key construction was using `_lossy` on a borrowed `OsStr` derived from a checkpoint path, which could corrupt the key on filesystems with non-UTF-8 path components. Switched to a strict UTF-8 conversion that returns an error rather than silently uploading to a mangled key.
  - The disk-stats code path that reports spilled-segment sizes used `_lossy` on a path it then logged; switched to strict UTF-8 so a non-UTF-8 path triggers a warning rather than producing a corrupt log line.
- **No state-path soundness issue was found** — the audit confirmed the hot path (state serialization, fingerprinting) never goes through a lossy conversion.

## v1.0.0 (2026-04-27)

First stable release. The 1.0 cycle focused on correctness foundations, high-leverage performance wins, polish, and a Verus-checked proof of the fingerprint store's resize protocol.

### Correctness foundations

- **Differential testing vs TLC as a CI gate.** `scripts/diff_tlc.sh` plus `.github/workflows/diff-tlc.yml` runs 13 curated specs under both TLC and tlaplusplus on every push. State counts agree exactly; the harness uncovered seven real divergences (T1.1, T1.3, T1.4, T1.5, T1.6, T2.4, T11.5) that were all fixed before the release.
- **Compiled-vs-interpreted proptest equivalence.** `tests/compiled_vs_interpreted.rs` generates well-typed Int/Bool/Set/Seq/Record/Str expressions and asserts the text evaluator and the compiled-IR evaluator agree on every case. Wired into CI at `PROPTEST_CASES=128`; runs at 2048 across 9 seeds locally.
- **State-graph snapshot tests.** `tests/state_graph_snapshots.rs` pins the reachable set for 7 small TLA+ specs as 128-bit XxHash3 digests, validated against TLC v2.19.
- **Mutation-testing audit.** `cargo-mutants` was run against `src/tla/eval.rs` and `src/tla/action_exec.rs`; 17 inline kill-tests added to cover the consequential survivor categories.

### Soundness fixes (caught by the new gates)

- **T1.1.** Compiled `\E x \in S : Action(x)` inside a wrapper definition now routes through the interpreted action evaluator, so existential branches no longer silently produce zero successors. Surfaced under `QueueSegmentSync` (TLC: 1531 distinct, pre-fix: 5).
- **T1.3.** Wrapper-Next fairness constraint accepts any in-SCC transition when the constraint's action name matches the spec's Next definition, eliminating the false-positive SCC fairness violation seen on `WorkQueue`.
- **T1.4.** Compiled expression evaluator no longer mis-splits LET bodies on indented `\/`, fixing the `\A x : \E m \in {} : ...` always-TRUE shape that affected Paxos-style specs.
- **T1.5.** Next-splitter no longer slices `/\ guard /\ \/ A \/ B /\ shared` on the inner `\/`, so shared post-conditions are no longer dropped and spurious successors are no longer fabricated. Closes the VIEW projection mismatch on `ViewTest`.
- **T1.6.** `<=>` (logical equivalence) was silently mis-parsed as `<= >` because the `=>` splitter consumed the `=>` *inside* `<=>`. Fixed in both interpreter (`eval_expr_inner` now splits `<=>` before `=>`) and compiled evaluator (new `CompiledExpr::Iff` variant placed before `Implies`). `FingerprintStoreResize.tla` now matches TLC exactly.
- **T2.1, T2.2.** Compiled-expr parser now scans for top-level `EXCEPT` only at bracket depth 0, so nested record-EXCEPT and EXCEPT-inside-record-literal shapes compile correctly.
- **T2.3.** `..` precedence vs set ops is fixed in both interpreter and compiler — `n..m \subseteq S` and `S \union n..m` now type-check and evaluate correctly in both code paths.
- **T2.4.** Unary minus inside binary subtraction (`<<(-1 - r.a), 0>>` and similar) no longer compiles to `Unparsed`; a new `find_binary_minus_split` helper walks past unary-minus positions before splitting.

### Performance — high-leverage wins

- **T5. Symbolic Init enumeration via Z3** (opt-in, `--features symbolic-init`). Filtered record-set Init shapes (e.g. `{c \in [f1: 0..N, f2: 0..N] : pred(c)}`) are translated to SMT and enumerated symbolically. Benchmark on `TightCan`: N=15 brute-force 14.71 s → symbolic 1.40 s (10.5x); N=20 58.42 s → 1.41 s (41x); N=40 brute-force exhausts the eval budget, symbolic finishes in 1.6 s.
- **T6. Cross-node distributed work stealing.** TCP-based steal protocol with steal-victim threshold, peer-down cooldown, and termination consensus extension. Three pre-existing v0.3.0 distributed-mode termination bugs surfaced and were fixed (so any multi-node cluster can now converge at all). Cluster mode remains opt-in (`--cluster-listen`); see T6.1 entry in `RELEASE_1.0.0_LOG.md` for the honest cluster-vs-independent benchmark.
- **T7. Partial-order reduction (POR) via stubborn sets** (opt-in, `--por`). Static read/write dependency analysis at module load; per-state stubborn-set computation under enabled-disjunct restriction. Safety-only — automatically rejected when fairness/liveness constraints are present. Benchmark on `PorBenchProcessGrid` (4 independent processes, MAX=4): full=625 states, POR=17 — **36.8x state reduction, 17.9x wall-time speedup**.
- **T8. State compression in the spillover queue** (default on, opt-out `--queue-compression false`). zstd-compressed in-memory ring sits between the hot work-stealing deques and the disk-backed overflow. Triggers only when the spill path is already engaged. Benchmark: 13.2x compression ratio, -68% peak RSS at 1M items, +2% wall time.
- **T10. Liveness checking scaling.** Iterative Tarjan + fingerprint-keyed graph + fast `check_fairness_on_scc_fp` cut SCC-based fairness post-processing from O(scc_size × tx × constraints) to O(tx × constraints). Benchmark on `LivenessBench` (32k states, 143k transitions, one giant SCC, 6 WF constraints): liveness phase wall-time **63.39 s → 115.28 ms (~550x)**, total wall-time 64.92 s → 1.60 s, peak RSS 2.59 GB → 1.99 GB (-23%).

### Polish

- **T9. Trace minimization on violation** (default on, `--minimize-trace`). Two-phase: (A) earliest-violation truncation + BFS shortcut search to fixed point; (B) syntactic variable-relevance scan on the invariant body, used to mark unrelated state variables as "(noise)" in the printed trace. 30 s default budget. Diamond fixture: 9 → 6 states in 375 µs.
- **T11. 1-hour chaos soak harness.** `scripts/chaos_soak.sh` plus a `FailScenario::setup()` wiring under `cfg(feature = "failpoints")`. 1-hour c8g.xlarge soak against `CheckpointDrain`: **387 iterations, 0 divergences, 0 hangs**. Every failpoint in `src/chaos.rs` (12 names) fired ≥23 times; the runtime tolerates persistent permanent failures in every checkpoint sub-step, FP-store-pressure path, and queue spill/load path.
- **T16. Regehr-style swarm testing** at two layers:
  - **T16a.** The T2 proptest harness picks a random subset of 17 shape categories per case (mean ~8.5 enabled), so each case explores a *minimal-feature-interaction* slice. The original uniform proptest is kept as a regression gate.
  - **T16b.** The chaos soak supports `--swarm-mode N|auto` (default 1; auto picks 1-4 concurrent failpoints per iter). 30-minute soak: 204 iters, 0 divergences, 66/66 distinct concurrent failpoint pairs observed (exhaustive pair coverage).
- **T12. Cross-arch CI matrix.** diff-TLC workflow now runs as a `[ubuntu-latest, ubuntu-24.04-arm]` matrix (both archs run lib + bin tests, the diff harness, and the T2 proptest at `PROPTEST_CASES=128`).

### Verification

- **T13. Verus on the fingerprint store** — tier B, protocol-level proof. `verification/verus/seqlock_resize.rs` (600 lines, 19 verified lemmas) proves the headline soundness theorem `theorem_no_fingerprint_lost`: in any well-formed execution of the protocol, every inserted fingerprint remains observable from then on. Proof is at the protocol abstraction layer — table = `Set<u64>`, atomic step semantics; it does NOT verify the shipping code's pointer arithmetic, memory orderings, or linear-probe collision behavior. See `verification/verus/README.md` for the assumptions and the tier-A roadmap (T13.1–T13.3, deferred to v1.1.0).

### Distributed

- **T6.1. Cross-node re-benchmark on a real corpus spec.** Confirmed the v0.3.0 design's tradeoff: cluster mode (each node maintains its own FP store, redundant exploration but no global FP synchronization) is slower than independent-explorer mode on canonical workloads. Cluster mode stays opt-in (`--cluster-listen`). Global FP partitioning is multi-quarter work out of scope for v1.0.0; tracked in the v1.1.0 backlog.

### Soundness fixes shipped in the final integration validation

- **T1.6.** `<=>` (logical equivalence) was silently mis-parsed because the `=>` splitter consumed the `=>` *inside* `<=>`. Fixed in both interpreter and compiled-IR paths; `FingerprintStoreResize.tla` now matches TLC exactly (52,376 generated, 15,970 distinct, 0 violations).
- **T11.1.** `--queue-max-inmem-items` below natural state count caused the spill path to drop states. Root cause: items in the spill pipeline were invisible to `has_pending_work()`/`should_terminate()`. Fixed via `inflight_spilled` AtomicU64 counter; 5 deterministic runs at cap=2000 now return exactly 26,344 distinct each.
- **T11.5.** Violation-exit hang under timeout-wrapper at NUMA-auto worker counts. Workers spinning in `pop_slow_path` did not observe `queue.finish()` so they never exited after a violation set `worker_stop=true`. Two-line fix: pop_slow_path now checks `self.finished` alongside `pause_requested`, and the violation handler calls `worker_queue.finish()`.

### Quality follow-ups shipped

- **T5.1+T5.2+T5.3** — Symbolic Init handles sequence-set comprehensions and Distinct-shortcut permutation symmetry; near-tautology detection covered by the existing v0.3.0 sum-range constraint propagation.
- **T5.6** — Tightened the symbolic-init `Distinct` shortcut to require per-position chain evidence (proptest divergence fix).
- **T7.1+T7.2+T7.3** — POR enhancements: batched per-disjunct evaluation (PorBenchProcessGrid 19.7x → 39.2x), smarter stubborn-set seed, POR for liveness via Peled (1994) visible-action proviso.
- **T9.1+T9.2+T9.3** — Trace minimization: transitive variable relevance through operator inlining, multi-source BFS seed, suffix shortening.
- **T10.1+T10.3+T10.4** — Liveness scaling: parallel-flatten via dashmap raw shards (~20% on N=10), trivial-SCC pre-filter for sparse graphs, per-action transition shard (~6x per-constraint check).
- **T11.2** — Re-soak validated T11.1 fix at cap=2000 driving the spill path under fault injection (166 iters, 0 divergences).
- **T12.1** — Cross-arch CI stack-overflow on the deliberate unbounded recursion test fixed by allocating an 8 MB thread stack for that test only.
- **T13.1+T13.2+T13.3** — Verus tier A: 31 lemmas verified including `theorem_no_fingerprint_lost_a` over a `Seq<u64>` linear-probe model, spec-level CAS soundness, and bounded reader-retry termination. Lives at `verification/verus/seqlock_resize_tier_a.rs`.

### Test suite

- **756 default tests** (release, no extra features), 0 failures, 8 ignored (disk-checkpoint round-trip pending serializable queue + per-test ignores for chaos/S3 doctests + a few env-dependent integration ignores).
- **776 tests with `--features failpoints`**, 0 failures.
- **774 tests with `--features symbolic-init`**, 0 failures.
- 13/13 specs pass `scripts/diff_tlc.sh` (state counts agree exactly with TLC v2.19 on every spec).
- 12 active state-graph snapshot tests, all match.
- T2 proptest equivalence harness clean across 9 seeds at `PROPTEST_CASES=2048` (validated on 3 fresh seeds: 1, 7, 42).
- 10-minute swarm-mode chaos soak: 63 iterations, 0 divergences, 0 hangs; 61 distinct concurrent failpoint pairs observed.
- Verus tier-A: 31 lemmas verified, 0 errors.

### Deferred to v1.1.0

The following items remain on the post-1.0.0 roadmap. Detail in `RELEASE_1.0.0_PLAN.md` (each entry begins with `**DEFER TO 1.1.0.**`).

- **T5.4** — Streaming Init enumeration / eager invariant filtering during cross-product (Einstein-class workloads).
- **T5.5** — Joint Init+Solution symbolic encoding (single-shot Z3 query for full Einstein-style specs).
- **T10.2** — Streaming SCC discovery during exploration (on-the-fly liveness for 100M+ state spaces).
- **T11.3** — CI-gate variant of the chaos soak (~5 min nightly form).
- **T11.4** — `route_spill_batch` inflight-counter accounting on disk-overflow push errors.
- **T13.4** — Shipping-code Verus annotations (`Tracked<PointsTo<HashTableEntry>>` threaded through `FingerprintShard`).
- **T13.5** — Unbounded-fairness reader liveness via Verus `state_machines!` macro.
- **T13.6** — CI gate for Verus tier-A run.

## v0.3.0 (2026-03-25)

### TLA+ Language Compatibility

**External corpus: 0 errors, 90% pass at 60s, 94% at 15 min** (up from 63% at start of release cycle). Every spec that defines a state machine (Init + Next) runs correctly.

- **SPECIFICATION definition chasing**: Follow definition reference chains to extract Init/Next from temporal formulas like `Spec == LiveSpec` where `LiveSpec == Init /\ [][Next]_vars /\ WF(Next)`
- **Disjunctive Init branches**: Handle `\/ Guard1 /\ var = expr1 \/ Guard2 /\ var = expr2` in Init by evaluating branch guards with known constants
- **Existential quantifier Init**: Expand `\E x \in S : body` in Init bodies by iterating domain values and extracting variable assignments
- **Init!N sub-expression references**: Support TLC-style `Init!1`, `Init!2` to reference specific conjuncts of a definition
- **Disjunctive Init bodies**: Handle Init definitions that are top-level disjunctions (`\/ branch1 \/ branch2`)
- **Parameterized operator calls in Init**: Inline-expand `XInit(x)` where `XInit(v) == v = 0` to extract variable assignments
- **Late-binding equality/membership**: After cross-product expansion, re-classify guards as equality assignments when dependencies are now satisfied
- **Deferred membership evaluation**: Membership sets that depend on other membership variables are deferred to cross-product phase
- **Outer parenthesis stripping**: `(h_turn = 1)` in Init correctly classified as equality assignment
- **INSTANCE variable shadowing**: Skip instance module variables that shadow definitions (e.g., Stuttering's `vars` vs Lock's `vars == <<pc, lock>>`)
- **Definition override preservation**: Save original definitions before `Init <- MCInit` override for missing variable recovery
- **Evaluation-only modules**: Specs without Init/Next/SPECIFICATION complete instantly with 0 states instead of erroring
- **Local .tla file loading for built-in modules**: When a spec ships its own `Functions.tla` alongside a built-in module, load definitions from the local file as fallback

### Community Modules

16 new built-in operators across 6 modules:

- **Bags**: EmptyBag, SetToBag, BagToSet, IsABag, BagIn, BagOfAll, BagUnion, CopiesIn
- **BagsExt**: BagAdd, BagRemove
- **IOUtils**: IOEnv, ndJsonDeserialize, JsonDeserialize, JsonSerialize, ToString
- **Bitwise**: BitsAnd, BitsOr, BitsXor, BitNot, LeftShift, RightShift, IsABitVector, IsANatural
- **Combinatorics**: Factorial, nCk, nPk
- **CSV**: CSVRead, CSVWrite
- **VectorClocks**: VCLess, VCLessOrEqual, VCMerge
- **Randomization**: RandomSubset

### Performance Optimizations

- **Trivial Next detection**: When `Next == UNCHANGED vars`, skip BFS exploration entirely — just enumerate Init states and check invariants
- **Constraint propagation**: For `{c \in [f1: 0..N, f2: 0..N] : c.f1 + c.f2 \in lo..hi}`, compute valid ranges directly instead of iterating all N² pairs (~3000x speedup for CoffeeCan)
- **Compiled predicate evaluation**: Use `compile_expr` + `eval_compiled` for record set filtering instead of re-parsing expression text per record
- **Range membership fast path**: `x \in a..b` evaluates as `a <= x && x <= b` instead of constructing the full integer set
- **Inline record set generation**: Generate records inline with predicate filtering instead of materializing the full record set
- **Vec-based constraint output**: Return constraint-propagated record sets as Seq (O(n)) instead of BTreeSet (O(n log n))
- **Lazy Init enumeration**: For cross-products exceeding 10M states, use odometer-style enumeration instead of materialization
- **Early exit for 0 initial states**: Evaluation-only modules complete instantly without spawning worker threads
- **FunAsSeq fix**: Corrected key indexing from `n..n+m-1` to `1..m` matching TLC semantics

### Distributed Model Checking

- **`--fetch-module` / `--fetch-config`**: Fetch spec files from S3 URIs for distributed runs where nodes don't share a filesystem
- **FLURM integration**: Plugin support for job scheduling on spot instances with automatic S3 file distribution
- **S3 checkpoint resume validated**: Round-trip checkpoint → clear → resume verified on spot instances

### Bug Fixes

- Fixed `FunAsSeq(f, n, m)` key indexing (was `n..n+m-1`, now `1..m`)
- Fixed empty expression handling in `Init!N` references with comment-stripped lines
- Resolve `IOEnv` and `EmptyBag` as zero-arg built-in operators (bare identifier usage)
- TLCGet ASSUME failures downgraded to warnings (TLC-specific runtime introspection)
- MAX_INIT_STATES raised from 1M to 10M
- Record set size limit raised to 10M

### Scaling Benchmark

Corpus run on c6gd.2xlarge (8 vCPU) instances, 60s timeout per spec:

| Nodes | vCPUs | Wall Time | Pass Rate | Speedup |
|-------|-------|-----------|-----------|---------|
| 1 | 8 | 29 min | 162/182 (89%) | 1.0x |
| 2 | 16 | 18 min | 161/182 (88%) | 1.6x |
| 4 | 32 | 8.6 min | 166/182 (91%)* | 3.4x |
| 8 | 64 | 7.5 min | 170/182 (93%)* | 3.9x |

*Adjusted for connectivity issues on some nodes

With longer timeouts on larger machines (192 cores):
- 300s: 169/182 (93%)
- 900s: 171/182 (94%)

### Test Suite

- 600 unit tests, all passing
- 620 failpoint tests (with `--features failpoints`), all passing
- 25/32 internal corpus specs passing
- 163/182 external corpus specs passing at 60s (0 errors, 19 timeouts)
- 161/182 analysis probes: FULL_PASS (88%), 0 FAIL

## v0.2.0

Initial release with parallel runtime, NUMA-aware work-stealing, lock-free fingerprint store, and native TLA+ frontend.
