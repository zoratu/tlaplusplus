# Many-Core Scaling Bottleneck Analysis

## Problem Statement

When running on 128-core systems with 96 workers, CPU utilization is extremely low (~5-7% per core, ~90% idle). This indicates a severe scaling bottleneck preventing effective use of available parallelism.

## Root Cause Analysis

### Worker Loop Architecture (src/runtime.rs:504-618)

Current worker pattern:
```rust
loop {
    match queue.pop() {  // Pop ONE state
        Ok(Some(state)) => {
            // Process single state
            // Generate successors
            // Batch fingerprint check
            // Enqueue new states
        }
        Ok(None) => {
            // Queue empty - SLEEP
            std::thread::sleep(Duration::from_millis(poll_sleep_ms));
        }
    }
}
```

### Bottleneck Mechanisms

1. **Queue Starvation**
   - Each worker processes ONE state per iteration
   - With 96 workers competing for work, queue depletes rapidly
   - Workers spend ~90% of time sleeping waiting for queue refill
   - Counter-grid model has low branching factor (~2 successors per state)

2. **Synchronization Overhead**
   - Every `queue.pop()` acquires a lock
   - Every `queue.push()` acquires a lock
   - 96 workers × frequent lock acquisition = high contention

3. **Work Imbalance**
   - Simple models generate few successors per state
   - Queue empties before workers can saturate
   - No work-stealing or local queue buffering

## Measured Performance

**Test Configuration:**
- System: 128 cores, 247GB RAM
- Workers: 96
- Model: counter-grid (max-x=20000, max-y=20000, max-sum=40000)

**Observed:**
- Per-core utilization: ~5-7% (target: >80%)
- Idle time: ~90%
- Process total CPU: ~935% (equivalent to 9-10 cores active)
- Runtime duration: 5+ minutes (should be <1 minute at full utilization)

## Proposed Solutions

### 1. Bulk Dequeue Pattern (High Priority)

**Current:** Workers pop 1 state
**Proposed:** Workers pop N states (e.g., 64 states)

```rust
// Worker loop modification
let mut local_queue: VecDeque<State> = VecDeque::with_capacity(64);

loop {
    // Refill local queue
    if local_queue.is_empty() {
        match queue.pop_bulk(64) {  // NEW: bulk pop
            Ok(states) => local_queue.extend(states),
            Ok(empty) => { sleep(); continue; }
        }
    }

    // Process from local queue
    while let Some(state) = local_queue.pop_front() {
        // Process state
        // Batch successors
    }
}
```

**Benefits:**
- Reduces lock contention from O(states) to O(states/64)
- Workers stay busy longer between queue interactions
- Better cache locality processing related states

### 2. Work Stealing (Medium Priority)

Implement per-worker local queues with work stealing:
- Each worker has a local deque
- Workers push to own queue
- Steal from random worker when local queue empty
- Only fall back to global queue when all local queues empty

**Reference:** Rayon's work-stealing algorithm

### 3. Adaptive Polling (Low Priority)

Replace fixed sleep with adaptive backoff:
```rust
let mut backoff = Duration::from_micros(1);
loop {
    match queue.pop() {
        Ok(Some(state)) => {
            backoff = Duration::from_micros(1);  // Reset
            // Process
        }
        Ok(None) => {
            std::thread::sleep(backoff);
            backoff = (backoff * 2).min(Duration::from_millis(10));
        }
    }
}
```

### 4. Model-Specific Optimizations

For low-branching models, reduce worker count:
- Use `workers = sqrt(total_cores)` for low-branching models
- Use `workers = total_cores` for high-branching models
- Auto-detect branching factor and tune dynamically

## Short-Term Workaround

For immediate testing, use models with higher branching factors:
- Increase counter-grid bounds: max-x=50000, max-y=50000
- Use TLA models with complex state transitions
- Reduce worker count to match available work: `--workers 16`

## Testing Plan

1. Implement bulk dequeue pattern
2. Benchmark on 128-core system with counter-grid model
3. Measure:
   - Per-core utilization (target: >80%)
   - Throughput (states/sec)
   - Scalability (throughput vs worker count)
4. Compare against TLC on same hardware

## References

- Counter-grid model: src/models/counter_grid.rs
- Worker loop: src/runtime.rs:504-618
- Queue implementation: src/storage/queue.rs
- Remote testing: scripts/remote_bench.sh
