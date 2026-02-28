# NUMA-Local Fingerprint Routing Design

## Problem

On 6-NUMA systems, 83% of fingerprint checks hit remote memory (140ns vs 60ns).
This causes 20-38% kernel time and limits throughput to 5-9M states/min.

## Solution: Route States to Their Fingerprint's Home NUMA

### Key Insight

Each fingerprint deterministically maps to one NUMA node. If we route states to their
fingerprint's home NUMA, all fingerprint checks become local.

### Architecture

```
State Flow:

  Worker generates      Route by         Workers on home NUMA
  successor state  →  fingerprint  →   check local shards only
       ↓                  ↓                    ↓
  Compute F = hash(S)    NUMA = F % N     Push to local queue
                              ↓                    ↓
                    Push to NUMA's injector   Process locally
```

### Changes Required

#### 1. Fingerprint Store: Deterministic NUMA Sharding

```rust
// Before: fp & shard_mask (random NUMA)
// After: deterministic NUMA assignment

fn shard_for_fp(&self, fp: u64) -> (usize, usize) {  // (numa_node, shard_id)
    let numa = self.home_numa(fp);
    let shard_within_numa = (fp as usize) % self.shards_per_numa;
    let shard_id = numa * self.shards_per_numa + shard_within_numa;
    (numa, shard_id)
}

fn home_numa(&self, fp: u64) -> usize {
    // Use upper bits for NUMA routing (less collision with shard selection)
    ((fp >> 32) ^ (fp >> 16) ^ fp) as usize % self.num_numa_nodes
}
```

#### 2. Work-Stealing Queues: Per-NUMA Injectors

```rust
struct WorkStealingQueues<T> {
    // Before: global: Injector<T>
    // After: per-NUMA injectors
    numa_injectors: Vec<Injector<T>>,
    num_numa_nodes: usize,
    // ... rest unchanged
}
```

#### 3. State Pushing: Route by Fingerprint

```rust
// New API that routes by fingerprint
fn push_with_fingerprint(&self, worker: &WorkerState<T>, state: T, fp: u64) {
    let home_numa = self.home_numa(fp);
    if home_numa == worker.numa_node {
        // Fast path: local NUMA, push to local queue
        worker.worker.push(state);
    } else {
        // Cross-NUMA: push to home NUMA's injector
        self.numa_injectors[home_numa].push(state);
    }
}
```

#### 4. Work Stealing: Prefer Local NUMA Injector

```rust
fn pop_slow_path(&self, worker: &mut WorkerState<T>) -> Option<T> {
    // 1. Try local NUMA's injector first (states homed here)
    if let Steal::Success(item) = self.numa_injectors[worker.numa_node].steal() {
        return Some(item);
    }

    // 2. Try stealing from same-NUMA workers
    if let Some(item) = self.try_steal_local_numa(worker) {
        return Some(item);
    }

    // 3. Try remote NUMA injectors (load balancing)
    for numa in 0..self.num_numa_nodes {
        if numa != worker.numa_node {
            if let Steal::Success(item) = self.numa_injectors[numa].steal() {
                return Some(item);
            }
        }
    }

    // 4. Try remote NUMA workers (last resort)
    self.try_steal_remote_numa(worker)
}
```

#### 5. Runtime Integration

```rust
// In worker loop, after fingerprint check:
for (state, fp) in new_states_with_fps {
    queues.push_with_fingerprint(&worker_state, state, fp);
}
```

### Benefits

| Metric | Before | After |
|--------|--------|-------|
| Local fingerprint checks | 17% | 100% |
| Remote memory access | 83% | ~5% (work stealing) |
| Expected throughput | 5-9M/min | 15-25M/min |

### Implementation Order

1. Add `home_numa()` and deterministic shard assignment to fingerprint store
2. Add per-NUMA injectors to work-stealing queues
3. Add `push_with_fingerprint()` API
4. Update runtime to pass fingerprints when pushing
5. Test on 384-core system

### Correctness

- Each fingerprint maps to exactly one NUMA (deterministic hash)
- States are routed to that NUMA's injector
- Workers on that NUMA check fingerprints in local shards
- No duplicate exploration: fingerprint checked once on home NUMA
- Load balancing: workers can steal from remote NUMAs when idle
