# tlaplusplus (prototype)

`tlaplusplus` is a Rust prototype runtime for explicit-state model checking aimed at the TLC pain points you described:

- no managed GC pauses (native memory/runtime)
- reduced worker I/O blocking (hot cache + bloom precheck + disk spill queue)
- cgroup-aware worker sizing and memory budgeting
- robust checkpoint/resume with disk-backed queue + manifest

## What is implemented now

- Parallel state exploration engine (`N` worker threads)
- Cgroup-aware worker planner:
  - respects cpuset and CPU quota limits
  - optional explicit core list (`--core-ids`)
  - optional NUMA-aware CPU pinning
- Sharded fingerprint store:
  - in-memory hot set
  - bloom filter pre-check
  - batch contains/insert path for successor batches
  - persistent exact set in `sled` (disk-backed)
- Disk-backed state queue:
  - bounded in-memory frontier
  - overflow spill segments to disk
  - on-demand reload when in-memory queue drains
- Checkpointing:
  - periodic checkpoints (`--checkpoint-interval-secs`)
  - final checkpoint on exit
  - resume from persisted queue/fingerprint store (`--resume`)
- Synthetic model (`counter-grid`) for stress testing queue/fingerprint throughput

## Build

```bash
cargo build --release
```

## TLC corpus validation

Latest official TLC jar is vendored at:

- `tools/tla2tools.jar` (symlink)
- `tools/tla2tools-v1.7.4.jar`
- `corpus/index.tsv` (corpus run index)
- `corpus/public/public_corpus.tsv` (public corpus source list)
- `corpus/public/public_corpus.lock.tsv` (pinned SHA lockfile)

Run the language-coverage corpus model:

```bash
scripts/tlc_check.sh
```

Run fairness/liveness flavor:

```bash
scripts/tlc_check.sh corpus/language_coverage/LanguageFeatureMatrix.tla \
  corpus/language_coverage/LanguageFeatureMatrixFair.cfg \
  .tlc-out/language_coverage_fair
```

Run explicit `INIT`/`NEXT` config override corpus model:

```bash
scripts/tlc_check.sh corpus/language_coverage/InitNextTemporalQuant.tla \
  corpus/language_coverage/InitNextTemporalQuant.cfg \
  .tlc-out/init_next_temporal_quant
```

Note: `InitNextTemporalQuant.tla` includes `\AA`/`\EE` temporal formulas as language artifacts. TLC currently parses these definitions but does not support checking them directly as `PROPERTY` formulas.

Run the full indexed corpus and emit a summary table:

```bash
scripts/tlc_corpus.sh
```

Summary output:

- `.tlc-out/corpus/summary.tsv`
- Per-run logs under `.tlc-out/corpus/<id>/tlc.log`

Resolve/pin public corpus SHAs:

```bash
scripts/refresh_public_corpus_shas.sh
```

Run pinned public corpus entries:

```bash
scripts/tlc_public_corpus.sh
```

Public corpus outputs:

- `.tlc-out/public-corpus/summary.tsv`
- Per-run logs under `.tlc-out/public-corpus/<id>/tlc.log`

## Native TLA frontend progress

Analyze a real TLA spec and config to extract required language/features:

```bash
cargo run -- analyze-tla \
  --module /absolute/path/to/Spec.tla \
  --config /absolute/path/to/Spec.cfg
```

The analyzer currently provides:

- Structured module parse (constants/variables/definitions).
- Config parse (`CONSTANTS`, `SPECIFICATION`, `INIT`/`NEXT`, invariants/properties).
- Feature surface counts to drive implementation ordering.
- `Init` seeding probe (`probe_init_*`) to materialize realistic probe state from `Init`.
- Action-clause probe (`expr_probe_*`) showing expression-evaluator coverage on action bodies.
- `Next` branch probe (`next_branch_probe_*`) that attempts native branch execution and reports supported vs unsupported disjuncts.

Gap report for OpenPort `Combined.tla`:

- `notes/combined-gap-report.md`
- Raw output: `notes/combined-analysis.txt`

## Run locally

```bash
./target/release/tlaplusplus run-counter-grid \
  --max-x 10000 \
  --max-y 10000 \
  --max-sum 20000 \
  --workers 0 \
  --core-ids 2-127 \
  --memory-max-bytes 206158430208 \
  --numa-pinning=true \
  --fp-shards 128 \
  --fp-expected-items 500000000 \
  --fp-batch-size 2048 \
  --queue-inmem-limit 1000000 \
  --queue-spill-batch 100000 \
  --checkpoint-interval-secs 30 \
  --work-dir ./.tlapp-local
```

Resume from an existing checkpoint/work dir:

```bash
./target/release/tlaplusplus run-counter-grid \
  --max-x 10000 \
  --max-y 10000 \
  --max-sum 20000 \
  --resume=true \
  --clean-work-dir=false \
  --work-dir ./.tlapp-local
```

To force overflow testing:

```bash
./target/release/tlaplusplus run-counter-grid \
  --max-x 500 \
  --max-y 500 \
  --max-sum 2000 \
  --workers 16 \
  --queue-inmem-limit 100 \
  --queue-spill-batch 50 \
  --work-dir ./.tlapp-spill
```

## Run on your many-core host

Use `scripts/remote_bench.sh` (defaults to your host/key and `/mnt/weka`):

```bash
scripts/remote_bench.sh --max-x 20000 --max-y 20000 --max-sum 40000 --workers 96
```

The remote script defaults to `--core-ids 2-127` for many-core runs.

Environment overrides:

- `REMOTE_HOST` (default: `ubuntu@98.83.110.220`)
- `REMOTE_KEY` (default: `~/.ssh/weka-isaiah-us-east-1_ed25519`)
- `REMOTE_DIR` (default: `/mnt/weka/tlaplusplus`)

## TLA+ compatibility target

Target behavior is direct support for existing TLA+ language/tool semantics without per-model adapters. The current runtime work in this repo is the backend/runtime layer (storage, scheduling, checkpointing, CPU/memory placement).

Immediate next integration targets:

1. Parse TLA+ modules through existing TLA+ frontend tooling (SANY/TLC) and execute the resulting semantics in this runtime.
2. Differential-check behavior against TLC on sampled traces/configs.
3. Preserve operator coverage for the full language subset used by real specs.

## Caveats (current prototype)

- No temporal/liveness checking yet (safety/invariant focus).
- No symmetry reduction yet.
- Queue/fingerprint formats are internal and not stabilized.
- The runtime backend is implemented; full direct `.tla` frontend execution is not finished yet.
