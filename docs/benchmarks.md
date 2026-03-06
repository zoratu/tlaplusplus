# TLA++ Benchmarks

Benchmark results for tlaplusplus on various hardware configurations.

## Hardware Configurations

| Instance | CPU | Cores | RAM | NUMA Nodes |
|----------|-----|-------|-----|------------|
| c6a.metal | AMD EPYC 7R13 | 128 | 256GB | 2 |
| c7gn.metal | AWS Graviton3 (Neoverse-V1) | 64 | 128GB | 1 |
| r8g.metal-24xl | AWS Graviton4 (Neoverse-V2) | 96 | 768GB | 1 |
| r8g.metal-48xl | AWS Graviton4 (Neoverse-V2) | 192 | 1.5TB | 2 |
| u-6tb1.metal | Intel Xeon 8375C | 384 | 6TB | 6 |

## Throughput Comparison: tlaplusplus vs TLC

Tested on c6a.metal (128-core AMD EPYC) with Combined.tla model:

| Metric | tlaplusplus | Java TLC | Speedup |
|--------|-------------|----------|---------|
| States/minute | 10.5M | 980K | **10.7x** |
| CPU utilization | 95%+ | ~60% | - |
| Memory efficiency | Lock-free | GC pauses | - |
| Steady-state throughput | Maintained | Degrades over time | - |

## NUMA Impact Analysis

Tested on u-6tb1.metal (384-core, 6 NUMA nodes) with Combined.tla:

| Configuration | Workers | %usr | %sys | States/min | Notes |
|--------------|---------|------|------|------------|-------|
| All NUMA nodes | 384 | 48% | 51% | ~120K | Heavy cross-NUMA traffic |
| All workers, optimized | 384 | 97.5% | 2.1% | ~100K | NUMA-local allocation |
| **NUMA-optimized (auto)** | 192 | **99%** | **<1%** | **~220K** | 3 close NUMA nodes only |

**Key finding**: Using only workers on close NUMA nodes (distance ≤20) achieves **2x higher throughput** by eliminating cross-NUMA memory access.

## Model-Specific Benchmarks

### Combined.tla (Complex Multi-Module TLA+ Spec)

Realistic production TLA+ specification with:
- Multiple interacting modules
- Complex state transitions
- Large state space (~billions of states)

| Hardware | Workers | Throughput | Distinct States | Notes |
|----------|---------|------------|-----------------|-------|
| r8g.metal-24xl (96-core) | 96 | 4.0-4.5M s/min | 170M+ | Sustained |
| r8g.metal-48xl (192-core) | 96 | 4.0-4.5M s/min | 900M+ | Similar per-worker |
| c6a.metal (128-core) | 124 | 10.5M s/min | 50M+ | Best per-core |

### SimpleCounter.tla (Small Model)

Minimal model for quick verification:

| Hardware | Workers | Time to Complete | States |
|----------|---------|------------------|--------|
| MacBook Pro M1 | 8 | <1s | ~100 |
| c6a.metal | 128 | <1s | ~100 |

### CounterGrid (Synthetic Stress Test)

Synthetic model for benchmarking raw throughput:

```bash
./target/release/tlaplusplus run-counter-grid \
  --max-x 10000 --max-y 10000 --max-sum 20000
```

| Hardware | Workers | Peak Throughput | Notes |
|----------|---------|-----------------|-------|
| c6a.metal (128-core) | 124 | 15M+ s/min | Synthetic, no TLA+ parsing |
| r8g.metal-48xl (192-core) | 192 | 12M+ s/min | ARM, slightly lower per-core |

## Checkpoint Performance

Checkpoint overhead measured during Combined.tla runs:

| Queue Size | Checkpoint Time | Impact |
|------------|-----------------|--------|
| 10M items | 30-60s | Minimal |
| 30M items | 5-10 min | Noticeable pause |
| 70M items | 15-25 min | Significant pause |

**Recommendation**: Use smaller checkpoint intervals (5 min) to keep queue sizes manageable.

## Memory Usage

Memory scaling for fingerprint store and queue:

| Distinct States | FP Store Size | Queue (peak) | Total RAM |
|-----------------|---------------|--------------|-----------|
| 50M | ~4GB | ~8GB | ~15GB |
| 100M | ~8GB | ~15GB | ~30GB |
| 500M | ~40GB | ~50GB | ~100GB |
| 1B | ~80GB | ~100GB | ~200GB |

## S3 Persistence Overhead

Background S3 upload impact:

| Operation | Latency | Throughput Impact |
|-----------|---------|-------------------|
| Checkpoint upload | <5s for 100MB | <1% |
| Background sync | 10s interval | Negligible |
| Resume download | 30-120s for 1GB | One-time |

## Running Your Own Benchmarks

```bash
# Quick benchmark on local machine
./target/release/tlaplusplus run-counter-grid \
  --max-x 1000 --max-y 1000 --max-sum 2000

# Full benchmark with checkpointing
./target/release/tlaplusplus run-tla \
  --module /path/to/Spec.tla \
  --config /path/to/Spec.cfg \
  --workers 0 \
  --checkpoint-interval-secs 300

# Observe output for:
# - States/minute throughput
# - CPU utilization (top command)
# - Memory usage (free -h)
```

## Methodology Notes

- All benchmarks run with `--release` optimizations
- Throughput measured after 5+ minutes of steady-state operation
- CPU utilization from `top` or `/proc/stat`
- Memory usage excludes OS buffers/cache
- TLC benchmarks use `-workers auto` flag
