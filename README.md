# TLA++

A Rust implementation of TLA+ model checking with TLC feature parity, achieving **10.7x faster** state exploration than Java TLC on many-core systems. 182/182 (100%) of the [tlaplus/Examples](https://github.com/tlaplus/Examples) corpus passes analysis.

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
# Run all tests (577 tests)
cargo test

# Run with chaos/failpoint testing (597 tests)
cargo test --features failpoints

# Run property-based tests
cargo test proptests

# Run fuzzing (requires nightly)
cargo +nightly fuzz run fuzz_tla_module
```

The test suite includes:
- **577 unit/integration tests** covering runtime, storage, TLA+ evaluation, and model checking correctness
- **19 property-based tests** (proptest) verifying set algebra laws and serialization roundtrips
- **23 chaos/failpoint tests** for fault injection
- **8 fuzz targets** for parser robustness
- **182/182 external corpus specs** (tlaplus/Examples) pass analysis
- **29/29 internal corpus specs** pass analysis
- **14 mutation-validated correctness tests** for the compiled action evaluator

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
