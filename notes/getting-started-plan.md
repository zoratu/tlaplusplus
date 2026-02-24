# Getting Started Plan (from scalable-model-checker.md)

This repository now has a runtime core that targets the two bottlenecks called out in the OpenPort notes:

- GC pressure: solved by moving core exploration/storage to Rust
- synchronous I/O stalls: reduced via hot-set + bloom pre-check + disk spill queue
- checkpoint fragility: addressed with periodic/final checkpoints and resume-from-disk queue state
- many-core scheduling: cgroup-aware worker planning + optional NUMA pinning + explicit core ID control

## Near-term work to connect to real `.tla` specs

1. Build a TLA+ frontend bridge:
   - Use existing TLA+ parser/tooling (SANY/TLC frontend).
   - Execute semantics directly in this runtime without per-model adapters.

2. Preserve TLC semantics coverage:
   - Keep full operator fidelity for constructs used in your models:
     - sets, sequences, records, functions, EXCEPT updates, quantifiers.
   - Add regression corpus for tricky constructs from OpenPort/ELP/FLURM specs.

3. Differential testing against TLC:
   - Run small bounded configs for:
     - `/Volumes/OWC 1M2/Users/isaiah/src/flurm/tla/FlurmJobLifecycle.tla`
     - `/Volumes/OWC 1M2/Users/isaiah/src/openport/specs/Combined.tla`
     - `/Volumes/OWC 1M2/Users/isaiah/src/elp/tla/TenantIsolation.tla`
   - Compare reachable-state counts and first-counterexample traces.

4. Scale/perf work:
   - Batched fingerprint inserts/lookups (implemented, tune batch/shard settings).
   - NUMA affinity pinning for worker/fingerprint shard locality (implemented, tune policy).
   - Background prefetch for queue segments.
   - Optional distributed shard mode.
