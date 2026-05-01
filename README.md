# TLA++

A Rust implementation of TLA+ model checking with TLC feature parity, achieving **10.7x faster** state exploration than Java TLC on many-core systems. 182/182 (100%) of the [tlaplus/Examples](https://github.com/tlaplus/Examples) corpus passes analysis; **174/182 (95.6%)** also pass full model checking at 60s.

**v1.0.0 (2026-04-27)** ships with:
- 756 default tests + 776 with failpoints + 774 with symbolic-init, 0 failures
- Differential CI gate vs TLC (13/13 specs match exactly)
- Compiled-vs-interpreted proptest equivalence (clean across 9 seeds at 2048 cases)
- State-graph snapshot tests (12 specs pinned to 128-bit XxHash3 digests)
- Symbolic Init enumeration (Z3-backed, 10-41x; opt-in via `--features symbolic-init`)
- Partial-order reduction (stubborn-set, opt-in `--por`; 36.8x state reduction on protocol-style specs)
- State compression in queue (zstd, default-on; 13x ratio, -68% peak RSS)
- Trace minimization on violation (default-on `--minimize-trace`)
- Liveness checking 550x faster (iterative Tarjan + fingerprint-keyed fairness)
- 1-hour chaos soak with swarm-mode failpoint injection (0 divergences across 12 failpoints)
- Verus tier-B proof of the seqlock resize protocol (19 verified lemmas)
- Cross-arch CI matrix (ubuntu-latest x86_64 + ubuntu-24.04-arm)

See [CHANGELOG.md](CHANGELOG.md) for the full v1.0.0 changelist and
[RELEASE_1.0.0_PLAN.md](RELEASE_1.0.0_PLAN.md) for the v1.1.0 backlog.

## Performance

### Synthetic Model (Counter-Grid)

Benchmarked on 128-core AMD EPYC (c6a.metal, 256GB RAM):

| Metric | tlaplusplus | Java TLC | Speedup |
|--------|-------------|----------|---------|
| States/minute | 10.5M | 980K | **10.7x** |
| CPU utilization | 95%+ | ~60% | - |
| Memory efficiency | Lock-free | GC pauses | - |

### Sustained Throughput (192-core ARM, c8g.metal-48xl)

Tested on real-world Raft/consensus specs with 30-minute runs:
- **1.0–1.75M states/min** sustained with zero checkpoint stalls
- **31–53M states** explored per 30-minute run
- Lightweight periodic checkpoints (0ms pause, no queue drain)

### TLC Comparison (8 workers, tlaplus/Examples corpus)

| Spec | States | tlaplusplus | TLC | Speedup |
|------|--------|-------------|-----|---------|
| MCReachable | 8 | 2.7s | 67.4s | **25.0x** |
| CoffeeCan100Beans | 5,150 | 2.7s | 1.1s | 0.4x |
| DieHard | 16 | 2.3s | 0.7s | 0.3x |

**Note:** For small models (< 100 states), TLC's JVM starts faster than our NUMA-aware runtime initialization (~2s overhead). The speedup manifests on large state spaces where lock-free fingerprinting, NUMA-local memory, and zero GC pauses dominate.

### Distributed Scaling (3-node cluster, 8 workers/node)

Independent exploration with work stealing — zero network on the hot path:

| Spec | 1 Node (30s) | 3 Nodes (30s) | Speedup |
|------|-------------|--------------|---------|
| Consensus spec (large) | 236K states | 1,069K states | **4.5x** |
| Checkpoint coordination | 262K states | 1,403K states | **5.4x** |

Super-linear scaling: 3x workers → 4.5–5.4x throughput. Each node runs a fully independent model checker with its own fingerprint store. Network is only used for work stealing (when a node's queue empties) and periodic Bloom filter exchange for dedup.

```bash
# Launch 3-node cluster
node0$ tlaplusplus run-tla --module Spec.tla --cluster-listen 0.0.0.0:7878 \
  --cluster-peers node1:7878,node2:7878 --node-id 0
node1$ tlaplusplus run-tla --module Spec.tla --cluster-listen 0.0.0.0:7878 \
  --cluster-peers node0:7878,node2:7878 --node-id 1
node2$ tlaplusplus run-tla --module Spec.tla --cluster-listen 0.0.0.0:7878 \
  --cluster-peers node0:7878,node1:7878 --node-id 2
```

### NUMA Scaling (384-core, 6 NUMA nodes, 760GB RAM)

| Configuration | %usr | %sys | States/min |
|--------------|------|------|------------|
| 380 workers (all NUMA) | 60-70% | 20-38% | 5-9M |
| 192 workers (auto, 3 NUMA) | **99%+** | **<1%** | **10-22M** |

## TLC Feature Parity

| Feature | Status |
|---------|--------|
| Safety invariant checking | Working |
| Liveness/temporal properties | Working |
| Fairness constraints (WF/SF) | Working |
| Deadlock detection | Working |
| ENABLED operator | Working (including parameterized actions) |
| Symmetry reduction | Working |
| Simulation mode | Working (`--simulate`) |
| Error traces | Working (BFS parent tracking with `--trace-parents`) |
| Diff traces | Working (`--difftrace`) |
| Coverage/profiling | Working (`--coverage`) |
| Continue after violation | Working (`--continue --max-violations N`) |
| State graph dump | Working (`--dump FILE`) |
| ASSUME evaluation | Working |
| CHECK_DEADLOCK | Working (`--allow-deadlock`) |
| State/action constraints | Working |
| S3 checkpoint/resume | Working |
| Spot instance support | Working (SIGTERM handler) |

### Community Modules

| Module | Operators |
|--------|-----------|
| DyadicRationals | Zero, One, Add, Half, IsDyadicRational, PrettyPrint |
| SequencesExt | RemoveAt, SeqOf |
| Functions | FoldFunction, FoldFunctionOnSet |
| Folds | MapThenFoldSet |
| FiniteSetsExt | FoldSet, Quantify, SymDiff, FlattenSet, kSubset, ChooseUnique, SumSet, ProductSet, IsInjective |
| UndirectedGraphs | IsUndirectedGraph, IsLoopFreeUndirectedGraph, ConnectedComponents, AreConnectedIn, IsStronglyConnected |
| Graphs | IsDirectedGraph, Successors, Predecessors, InDegree, OutDegree, Roots, Leaves, Transpose, IsDag |
| Relation | TransitiveClosure |
| Proof modules | FiniteSetTheorems, NaturalsInduction, SequenceTheorems (theorems evaluate to TRUE) |

## Build

```bash
cargo build --release
```

## Quick Start

```bash
# Model check a TLA+ spec
./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla \
  --config /path/to/Spec.cfg

# Analyze a spec (parse + expression probing, no model checking)
./target/release/tlaplusplus analyze-tla \
  --module /path/to/Spec.tla \
  --config /path/to/Spec.cfg

# Simulation mode (random sampling)
./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla \
  --simulate --simulate-depth 100 --simulate-traces 10000

# With full error traces and diff output
./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla \
  --trace-parents --difftrace
```

## Testing

```bash
# Run all tests (756 tests)
cargo test --release

# Run with chaos/failpoint testing (776 tests)
cargo test --release --features failpoints

# Run with Z3-backed symbolic Init enumeration (T5)
# Requires: apt-get install libz3-dev clang libclang-dev
cargo test --release --features symbolic-init

# Differential gate vs TLC (13 curated specs)
scripts/diff_tlc.sh

# Compiled-vs-interpreted proptest equivalence (T2)
PROPTEST_CASES=2048 cargo test --release --test compiled_vs_interpreted

# State-graph snapshot tests (T3)
cargo test --release --test state_graph_snapshots

# Run fuzzing (requires nightly)
cargo +nightly fuzz run fuzz_tla_module
```

The test suite includes:
- **756 unit/integration tests** covering runtime, storage, TLA+ evaluation, and model checking correctness
- **776 tests with `--features failpoints`** (adds chaos/fault-injection coverage)
- **774 tests with `--features symbolic-init`** (adds Z3-backed enumeration coverage)
- **17 proptest cases** in the compiled-vs-interpreted equivalence harness (uniform + Regehr-style swarm mask), plus 19 set-algebra/serialization property tests
- **23 chaos/failpoint tests** for single-fault-path correctness
- **8 fuzz targets** for parser robustness
- **13/13 differential-vs-TLC specs** (`scripts/diff_tlc.sh`)
- **12 active state-graph snapshot tests** validated against TLC v2.19
- **174/182 external corpus specs** (tlaplus/Examples) pass full model checking; 182/182 pass analysis
- **32/32 internal corpus specs** pass analysis
- **17 mutation-kill tests** added during the v1.0.0 cargo-mutants audit (`src/tla/eval.rs`, `src/tla/action_exec.rs`)

### Pre-release chaos soak

`scripts/chaos_soak.sh` runs the binary (built with `--features failpoints`)
in a tight loop with random failpoint injection — covers every failpoint in
`src/chaos.rs` and verifies that final state count + invariant verdict still
match a control run. This is expensive (~1 hour wall time on a small spot
instance) and is run manually as a release ritual, not per PR.

### Per-PR chaos smoke (T11.3, CI gate)

`scripts/chaos_smoke.sh` is the per-PR variant of the soak — same harness,
shorter duration. It runs in ~5 minutes, gates on `>= 6` of the 12
failpoints actually firing, and is wired into
`scripts/REDACTED` (run on a fresh EC2 spot per push) so every push to `main` and every PR
exercises the chaos path. Total CI budget ~10 min including the failpoint
build. Locally:

```bash
cargo build --release --features failpoints
scripts/chaos_smoke.sh                # 5 min, swarm-mode auto, gate >= 6 failpoints
scripts/chaos_smoke.sh --duration 600 # longer smoke, same gate
```

Validated on a 2-vCPU spot: 5m17s wall, 21 iterations, 12/12 failpoints
exercised, 0 divergences, 0 hangs.

```bash
# Build with failpoints support
cargo build --release --features failpoints

# Run a 1-hour soak against CheckpointDrain (default), single failpoint per iter
scripts/chaos_soak.sh --duration 3600

# Swarm mode (T16b — Regehr et al. "Swarm Testing", ICST 2012):
# random 1-4 concurrent failpoints per iter, surfaces fault-cascade bugs
scripts/chaos_soak.sh --duration 3600 --swarm-mode auto

# Or target a different spec / pin a fixed concurrent-failpoint count
scripts/chaos_soak.sh --duration 3600 \
  --spec WorkStealingTermination \
  --swarm-mode 3 \
  --per-iter-timeout 90
```

`--swarm-mode` selects how many failpoints fire concurrently per iteration:

- `1` (default) — single failpoint, T11-baseline behaviour, backward-compatible.
- `N` (positive integer) — exactly N concurrent failpoints (clamped to catalog size).
- `auto` — random N in `[1, --swarm-max]`, default `--swarm-max 4`.

Output: per-iteration TSV log under `.chaos-soak/iterations.tsv` (with
`swarm_n` column for concurrency count and comma-joined `failpoint`/`action`
columns), retained logs for any divergent / hanging iteration under
`.chaos-soak/logs/`, and a `summary.txt` with both the per-failpoint
coverage matrix and a top concurrent-pair matrix (for swarm-mode runs).

## Configuration

### Runtime Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--workers` | auto | Worker count (0 = auto from NUMA topology) |
| `--core-ids` | all | CPU list (e.g., "2-127") |
| `--numa-pinning` | true | Enable NUMA-aware CPU binding |
| `--auto-tune` | true | Dynamically adjust workers based on CPU sys% |
| `--fp-shards` | auto | Fingerprint store shard count |
| `--fp-expected-items` | 100M | Initial capacity hint |
| `--fp-batch-size` | 512 | States per fingerprint batch |
| `--checkpoint-interval-secs` | 0 | Checkpoint frequency (0 = disabled) |
| `--queue-max-inmem-items` | 50M | Max items before disk spilling |

### Model Checking Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--allow-deadlock` | false | Allow deadlocked states (like TLC's -deadlock) |
| `--trace-parents` | false | Enable BFS parent tracking for full error traces |
| `--max-trace-states` | 10M | Memory limit for parent tracking |
| `--simulate` | false | Random simulation instead of exhaustive BFS |
| `--simulate-depth` | 100 | Max steps per simulation trace |
| `--simulate-traces` | 1000 | Number of random traces |
| `--coverage` | false | Print action coverage summary |
| `--continue` | false | Continue exploring after violations |
| `--max-violations` | 1 | Stop after N violations |
| `--dump FILE` | none | Write state graph to file |
| `--difftrace` | false | Show only changed variables in traces |

### S3 Persistence (Spot Instances)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--s3-bucket` | none | S3 bucket (enables persistence when set) |
| `--s3-region` | auto | AWS region |
| `--s3-prefix` | auto | Path prefix in bucket |
| `--s3-upload-interval-secs` | 10 | Background upload frequency |
| `--fresh` | false | Start fresh, ignore existing checkpoints |

## Architecture

### Core Components

- **Runtime Engine** (`src/runtime.rs`) — Parallel state exploration with NUMA-aware work-stealing, checkpoint coordination, liveness checking
- **TLA+ Frontend** (`src/tla/`) — Native parser, expression evaluator (text + compiled paths), action IR compiler
- **Fingerprint Store** (`src/storage/page_aligned_fingerprint_store.rs`) — Lock-free CAS, page-aligned, dynamic resize at 85% load
- **Work-Stealing Queues** (`src/storage/work_stealing_queues.rs`) — Per-worker deques, NUMA-aware stealing, disk spilling
- **Simulation** (`src/simulation.rs`) — Random behavior sampling with xoshiro256** PRNG
- **Coverage** (`src/coverage.rs`) — Action fire counting and profiling

### Chaos Testing

Available failpoints (`--features failpoints`):
- `checkpoint_write_fail` — Fail checkpoint writes
- `fp_store_shard_full` — Simulate fingerprint store pressure
- `worker_panic` — Crash individual workers
- `queue_spill_fail` — Fail queue disk operations

Recovery: worker crashes continue with N-1 workers, I/O failures retry with backoff, memory pressure triggers emergency checkpoints.

## License

GNU GPLv3
