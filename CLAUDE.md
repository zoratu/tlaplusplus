# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tlaplusplus` is a Rust native implementation of the TLA+ model checker (TLC) that accepts `.tla` and `.cfg` files. The goal is to address TLC performance pain points by providing:

- No managed GC pauses (native Rust memory/runtime)
- Reduced worker I/O blocking (hot cache + bloom precheck + disk spill queue)
- Cgroup-aware worker sizing and memory budgeting
- Robust checkpoint/resume with disk-backed queue + manifest
- NUMA-aware CPU pinning for many-core systems

## Common Commands

### Build and Test

```bash
# Build release binary
cargo build --release

# Run tests
cargo test

# Run specific test
cargo test <test_name>

# Build and run in debug mode
cargo run -- <subcommand>
```

### Running Model Checks

```bash
# Run synthetic counter-grid model (for stress testing)
./target/release/tlaplusplus run-counter-grid \
  --max-x 10000 --max-y 10000 --max-sum 20000 \
  --workers 0 --core-ids 2-127 \
  --memory-max-bytes 206158430208 \
  --numa-pinning=true \
  --work-dir ./.tlapp-local

# Resume from checkpoint
./target/release/tlaplusplus run-counter-grid \
  --max-x 10000 --max-y 10000 --max-sum 20000 \
  --resume=true --clean-work-dir=false \
  --work-dir ./.tlapp-local

# Analyze a TLA+ spec
cargo run -- analyze-tla \
  --module /absolute/path/to/Spec.tla \
  --config /absolute/path/to/Spec.cfg
```

### TLC Corpus Validation

```bash
# Run language coverage corpus model
scripts/tlc_check.sh

# Run full indexed corpus and emit summary
scripts/tlc_corpus.sh

# Run public corpus entries
scripts/tlc_public_corpus.sh

# Refresh public corpus SHAs
scripts/refresh_public_corpus_shas.sh
```

## Architecture

### Core Components

The system is organized into several key layers:

**1. Model Trait (`src/model.rs`)**
- Defines the interface that all models must implement
- `Model::State` must be: Clone, Debug, Eq, Hash, Send, Sync, Serialize, DeserializeOwned
- Key methods: `initial_states()`, `next_states()`, `check_invariants()`

**2. Runtime Engine (`src/runtime.rs`)**
- Core parallel state exploration engine with N worker threads
- Uses `run_model()` function as main entry point
- Coordinates: workers, checkpoint thread, fingerprint store, state queue
- Worker lifecycle: pop state → check invariants → compute successors → batch fingerprint check → enqueue unique states
- Pause/resume mechanism for safe checkpointing

**3. Storage Layer (`src/storage/`)**

**Fingerprint Store** (`fingerprint_store.rs`):
- Sharded (configurable shard count) for concurrent access
- Three-tier architecture per shard:
  1. Bloom filter (fast false-positive pre-check)
  2. In-memory hot set (LRU-style bounded cache)
  3. Disk-backed persistent store (sled)
- Batch API: `contains_or_insert_batch()` for amortizing synchronization cost
- Periodic background flush to disk (configurable interval)

**Disk-Backed Queue** (`queue.rs`):
- Bounded in-memory frontier (configurable size)
- Overflow spills to disk segments in batches
- On-demand reload when in-memory queue drains
- Supports checkpoint flush and resume from persisted segments

**4. System Layer (`src/system.rs`)**
- Cgroup-aware worker planning: reads `/sys/fs/cgroup` to respect cpuset and CPU quota
- NUMA node discovery and CPU pinning via `sched_setaffinity`
- CPU list parsing (supports ranges like "2-127" or "2-63,96-127")
- Memory budget calculation from cgroup limits

**5. TLA+ Frontend (`src/tla/`)**

The native TLA+ frontend is **work in progress**. Current implementation status:

**Parsing & Analysis**:
- `module.rs`: Parse TLA+ module structure (constants, variables, definitions, EXTENDS)
- `cfg.rs`: Parse `.cfg` files (CONSTANTS, INIT, NEXT, SPECIFICATION, invariants, properties)
- `scan.rs`: Scan module closure for operator usage and language features

**Evaluation**:
- `value.rs`: TlaValue enum (Int, Bool, String, Set, Seq, Record, ModelValue, Function)
- `eval.rs`: Expression evaluator with support for basic operators, set operations, quantifiers
- `action_ir.rs`: Compile action definitions into intermediate representation (IR)
- `action_exec.rs`: Execute action IR to compute successor states
  - `evaluate_next_states()`: Core function for branch execution
  - `probe_next_disjuncts()`: Diagnostic probe for coverage analysis

**Formula Analysis**:
- `formula.rs`: Clause classification (primed assignments, UNCHANGED, guards)
- `split_top_level()`: Split disjunctions/conjunctions while respecting nesting

### Data Flow

```
.tla/.cfg files
    ↓
[TLA Frontend] → parse module + config
    ↓
[analyze-tla] → feature probe + Init seeding + Next branch coverage
    ↓
[Model trait impl] → initial_states() + next_states()
    ↓
[Runtime] → run_model()
    ↓
Workers → compute successors → batch FP check → enqueue
    ↓
[Fingerprint Store] → bloom + hot cache + sled disk
[State Queue] → in-memory frontier + disk spill segments
    ↓
[Checkpoint] → periodic pause + flush + manifest write
```

### Worker Architecture

Each worker runs in a tight loop:
1. Check pause signal (for checkpoint coordination)
2. Pop state from queue (with backoff sleep if empty)
3. Increment active worker counter
4. Check invariants (stop on violation if configured)
5. Compute successor states via `model.next_states()`
6. Accumulate successors into batch
7. When batch full or state complete:
   - Hash all states in batch
   - Dedup within batch using local HashSet
   - Call `fp_store.contains_or_insert_batch()`
   - Enqueue unique states to queue
8. Decrement active worker counter
9. Exit when queue drained AND no active workers

### Checkpointing

**Pause mechanism**:
- Checkpoint thread sets `pause.requested` flag
- Workers check flag at loop start, increment paused counter, wait on condvar
- Checkpoint thread waits for `paused_workers >= live_workers && active_workers == 0`

**Checkpoint content**:
- Queue: flush in-memory states to disk segments
- Fingerprints: flush hot cache to sled
- Manifest: JSON file with stats, config, timestamps

**Resume**:
- If `--resume=true`, queue loads existing disk segments on startup
- Fingerprint store automatically loads from sled
- Initial states skipped if queue has pending work

## Current Development Status

As of the last codex status:
> I'm now wiring runtime-facing APIs on top of the branch executor (evaluate_next_states) and then building TlaModel on top of it, so the checker can run .tla directly instead of just analyzing.

**What works**:
- Full runtime/storage/system layer
- Synthetic models (counter-grid, flurm-job-lifecycle)
- TLA+ parsing and analysis (`analyze-tla` command)
- Expression evaluation for common operators
- Action IR compilation and execution framework

**In progress**:
- TlaModel implementation connecting TLA frontend to runtime
- Full Next branch execution coverage
- Handling advanced TLA+ operators (DOMAIN, EXCEPT nested updates, etc.)

**Not yet implemented**:
- Temporal/liveness checking (safety/invariant focus for now)
- Symmetry reduction
- Full TLA+ language coverage (see `analyze-tla` output for gaps)

## Key Implementation Notes

### Model Implementation Pattern

Synthetic models live in `src/models/`. To add a new model:

1. Implement the `Model` trait
2. Define `State` as a concrete type (usually a struct with `#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]`)
3. Add subcommand to `src/main.rs` with model-specific parameters
4. Use `build_engine_config()` + `run_model()` pattern

### TLA+ Integration Pattern

The bridge from `.tla` to `Model` trait:

1. Parse module with `parse_tla_module_file()`
2. Parse config with `parse_tla_config()`
3. Seed initial state from `Init` using `classify_clause()` + `eval_expr()`
4. Implement `next_states()` using `evaluate_next_states()` on `Next` body
5. Implement invariant checks using `eval_expr()` on invariant formulas

### Performance Tuning

Key parameters for many-core systems:

- `--workers`: Set to 0 for auto (respects cgroup limits)
- `--core-ids`: Explicit CPU list to intersect with cgroup cpuset
- `--numa-pinning`: Enable for NUMA-aware round-robin worker→CPU mapping
- `--fp-shards`: Should be ≥ worker count for concurrency (default 64)
- `--fp-batch-size`: Larger batches amortize sync cost (default 512)
- `--queue-inmem-limit`: Balance memory vs disk I/O (default 5M states)
- `--memory-max-bytes`: Hard ceiling for automatic budget tuning

## Code Organization Principles

- **Separation of concerns**: Runtime (worker scheduling, checkpointing) is independent of storage (FP store, queue) and model semantics
- **Trait abstraction**: `Model` trait allows both synthetic models and TLA+-derived models
- **Lock-free where possible**: Workers use atomic counters + sharded stores to minimize contention
- **Graceful degradation**: Falls back to reasonable defaults if cgroup limits unavailable
- **Type safety**: Heavy use of newtypes and enums (TlaValue) to catch errors at compile time

## Corpus and Validation

The repository includes TLC corpus validation:
- `corpus/index.tsv`: Corpus run index
- `corpus/public/`: Public corpus source list and lockfile
- `tools/tla2tools.jar`: Vendored official TLC jar (v1.7.4)
- Scripts produce summary TSV and per-run logs under `.tlc-out/`

Use corpus validation to ensure compatibility as the native frontend evolves.
