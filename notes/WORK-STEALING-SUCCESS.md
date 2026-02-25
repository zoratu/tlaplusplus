# Work-Stealing Queues: 80%+ Utilization Achieved!

## Implementation

Switched from channel-based queues to **work-stealing queues** using crossbeam-deque.

### Key Design Decisions

1. **Per-worker owned queues**: Each worker thread owns its `Worker<T>` (not shared via Arc)
2. **Shared stealers**: Only `Stealer<T>` handles are shared for work stealing
3. **Global injector**: For initial work distribution
4. **Lock-free operations**: Zero contention on common path (local push/pop)

### Architecture

```rust
pub struct WorkStealingQueues<T> {
    global: Injector<T>,              // Global queue for initial work
    stealers: Vec<Stealer<T>>,        // Shared stealers (thread-safe)
    // Workers NOT stored here - owned by threads
}

pub struct WorkerState<T> {
    worker: Worker<T>,                // Owned by worker thread
    id: usize,
}
```

### Algorithm

Each worker:
1. **Pop from local queue** (LIFO) - cache friendly, zero contention
2. **Steal from global queue** if local empty
3. **Steal from other workers** (random) if global empty
4. **Sleep briefly** if nothing available, then retry
5. **Exit** when no work and no active workers

## Results

### Test Configuration
- System: 128 cores (AWS EC2)
- Workers: 120 threads on cores 2-127
- Workload: High-branching model (branching factor 200)
- Depth: 5 levels

### CPU Utilization

**Before (Crossbeam Channel Queue)**:
- Per-core: ~4-5% utilization
- Thread states: 11 Running, 111 Sleeping
- Total CPU: ~500% (4 cores worth)

**After (Work-Stealing Queues)**:
- **Per-core (cores 2-127): 83.96% user CPU** ✓
- **Combined (usr+sys): 90.66%** ✓
- Thread states: **120 Running, 2 Sleeping** ✓
- Total CPU: **~10,996% (86 cores worth)** ✓

### Key Metrics

```
30-second sustained measurement (cores 2-127):
  Cores measured: 126
  Average %usr: 83.96%  ← EXCEEDS 80% TARGET ✓
  Average %sys:  6.69%
  Combined:     90.66%
  %idle:         9.34%

Process stats (PID 118627):
  %CPU: 10,996 (86 cores actively utilized)
  Memory: 163GB (state space exploration)

Thread states:
  120 R (Running)  ← 98% of workers active
    2 S (Sleeping)

System-wide average (multiple 5-second samples):
  %usr: 89.46%
  %sys:  1.69%
  %idle: 8.83%
```

## Why It Works

### Zero Contention on Common Path

Workers push/pop from their **own** local queue:
```rust
// Fast path - no locks, no atomic operations beyond the worker's internal buffer
worker_state.worker.push(state);  // O(1), no contention
let state = worker_state.worker.pop();  // O(1), no contention
```

### Automatic Load Balancing

When a worker runs out of work, it steals from others:
```rust
// Pseudo-random stealing reduces contention
for other_worker in shuffle(all_workers) {
    if let Some(work) = other_worker.steal() {
        return work;
    }
}
```

### Cache Locality

LIFO pop from local queue = recently pushed states are hot in cache.

### Completion Detection

```rust
fn should_finish() -> bool {
    // No active workers = no new work being generated
    self.active_workers.load() == 0
}
```

## Conclusion

**Work-stealing queues achieve 84% per-core CPU utilization on 120 cores** ✓

**Goal achieved**: 83.96% user CPU exceeds the 80% target on 100+ cores.

The synchronous architecture CAN scale to 100+ cores when using:
1. Per-worker local queues (owned, not shared)
2. Lock-free steal operations
3. LIFO local pop (cache locality)
4. FIFO stealing (load balancing)

### Performance Comparison

| Metric | Channel Queue | Work-Stealing | Improvement |
|--------|--------------|---------------|-------------|
| Active threads | 11/122 (9%) | 120/122 (98%) | **11x** |
| Per-core %usr | ~4% | 84% | **21x** |
| Active cores | ~4 cores | ~86 cores | **21x** |
| %sys (contention) | 2-3% | 6.7% | Acceptable |

## Implementation Files

- `src/storage/work_stealing_queues.rs` - Core implementation
- `src/runtime.rs` - Integration with worker threads
- `Cargo.toml` - Added `crossbeam-deque = "0.8.6"`

## Previous Failed Approaches

1. Bulk dequeue: 10-15% utilization
2. In-memory fingerprint store: No improvement
3. Mutex+Condvar queue: 97% sys time (kernel contention)
4. Crossbeam channel: 5% utilization (95% workers sleeping)

Work-stealing was the **only approach that achieved the goal**.
