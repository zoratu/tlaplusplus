# TLA++ Usage Guide

This guide covers common usage patterns for tlaplusplus, from basic model checking to advanced S3-backed spot instance runs.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Running TLA+ Specs](#running-tla-specs)
3. [Checkpoint and Resume](#checkpoint-and-resume)
4. [S3 Persistence for Spot Instances](#s3-persistence-for-spot-instances)
5. [NUMA Optimization](#numa-optimization)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Advanced Configuration](#advanced-configuration)

## Basic Usage

### Running the Built-in Test Model

The `run-counter-grid` command runs a synthetic stress test model useful for benchmarking:

```bash
# Quick test (small state space)
./target/release/tlaplusplus run-counter-grid \
  --max-x 100 --max-y 100 --max-sum 200

# Full stress test (large state space)
./target/release/tlaplusplus run-counter-grid \
  --max-x 10000 --max-y 10000 --max-sum 20000
```

### Analyzing a TLA+ Spec

Use `analyze-tla` to parse and analyze a TLA+ specification without running the model checker:

```bash
cargo run -- analyze-tla \
  --module /path/to/Spec.tla \
  --config /path/to/Spec.cfg
```

This shows:
- Parsed operators and definitions
- Language features used
- Configuration (constants, invariants, properties)

## Running TLA+ Specs

### Basic Model Check

```bash
./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla \
  --config /path/to/Spec.cfg
```

### With Worker Configuration

```bash
# Auto-detect optimal workers (recommended)
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --workers 0

# Fixed worker count
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --workers 64

# Pin to specific cores
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --core-ids "2-127"
```

### With Memory Configuration

```bash
# Limit memory usage
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --memory-max-bytes 128000000000  # 128GB

# Adjust fingerprint store capacity
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --fp-expected-items 500000000  # Expect 500M distinct states
```

## Checkpoint and Resume

### Enable Periodic Checkpoints

```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --checkpoint-interval-secs 300  # Every 5 minutes
```

### Resume from Checkpoint

```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --resume true
```

Checkpoints are stored in the work directory (default: `.tlapp/`):
- `checkpoints/latest.json` - Current checkpoint metadata
- `fingerprints/` - Seen state hashes
- `queue-spill/` - Pending states (zstd compressed)

## S3 Persistence for Spot Instances

### Initial Run with S3

```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --s3-bucket my-tla-checkpoints \
  --s3-region us-west-2 \
  --s3-prefix runs/spec-v1
```

When S3 is enabled:
- Checkpoints default to every 10 minutes
- Background upload every 10 seconds
- Graceful SIGTERM handling for spot preemption

### Resume After Spot Termination

```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --s3-bucket my-tla-checkpoints \
  --s3-region us-west-2 \
  --s3-prefix runs/spec-v1 \
  --resume true
```

### S3 Authentication

```bash
# Option 1: Instance profile (recommended for EC2)
# No configuration needed - uses instance metadata

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-west-2

# Option 3: Credentials file (~/.aws/credentials)
[default]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
```

### Spot Instance Launch Script Example

```bash
#!/bin/bash
# spot-run.sh - Script for spot instance model checking

BUCKET="my-tla-checkpoints"
REGION="us-west-2"
PREFIX="runs/myspec-$(date +%Y%m%d-%H%M%S)"

# Check for existing run to resume
if aws s3 ls "s3://$BUCKET/$PREFIX/manifest.json" 2>/dev/null; then
  RESUME="--resume true"
  echo "Resuming from existing checkpoint"
else
  RESUME=""
  echo "Starting fresh run"
fi

./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla \
  --config /path/to/Spec.cfg \
  --s3-bucket "$BUCKET" \
  --s3-region "$REGION" \
  --s3-prefix "$PREFIX" \
  $RESUME
```

## NUMA Optimization

### Auto-Detection (Recommended)

```bash
# Let tlaplusplus detect NUMA topology
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --workers 0 \
  --numa-pinning true
```

### Check NUMA Configuration

```bash
# On Linux, check NUMA topology
numactl --hardware

# Example output:
# node distances:
# node   0   1   2   3   4   5
#   0:  10  15  17  21  28  26
#   1:  15  10  15  21  28  28
#   ...
# Nodes 0,1,2 have distance ≤17 - these are "close"
```

### Manual NUMA Control

```bash
# Pin to specific NUMA nodes (advanced)
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --core-ids "0-63,128-191"  # First 2 NUMA nodes
```

## Monitoring and Debugging

### Progress Output

During execution, progress is logged every 10 seconds:
```
Progress(123) at 2024-01-15 10:30:45: 15,234,567 states generated (4,523,456 s/min),
3,456,789 distinct states found (1,234,567 ds/min), 2,345,678 states left on queue.
ETA: queue growing
```

Key metrics:
- **states generated**: Total state transitions explored
- **s/min**: States per minute (throughput)
- **distinct states**: Unique states in fingerprint store
- **ds/min**: Distinct states per minute
- **queue**: Pending states to explore

### Debug Build

For detailed debugging, build without optimizations:
```bash
cargo build
./target/debug/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg
```

Debug builds include:
- Loader debug output
- Symmetry reduction statistics
- Constraint filter debugging

### Check Invariant Violations

By default, the model checker stops on the first violation:
```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --stop-on-violation true  # default
```

## Advanced Configuration

### Full Parameter Reference

```bash
./target/release/tlaplusplus run-tla --help
```

### Performance Tuning

For very large models (>1B states):

```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --workers 0 \                          # Auto-detect
  --fp-expected-items 2000000000 \       # 2B expected states
  --fp-batch-size 1024 \                 # Larger batches
  --queue-max-inmem-items 100000000 \    # 100M in memory
  --checkpoint-interval-secs 300 \       # 5 min checkpoints
  --s3-bucket my-bucket                  # S3 backup
```

### Resource-Constrained Systems

For systems with limited memory:

```bash
./target/release/tlaplusplus run-tla \
  --module Spec.tla --config Spec.cfg \
  --workers 4 \                          # Fewer workers
  --memory-max-bytes 16000000000 \       # 16GB limit
  --queue-max-inmem-items 10000000 \     # 10M in memory
  --fp-expected-items 50000000           # 50M expected states
```
