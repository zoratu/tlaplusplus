//! Atomic shared counters used by workers to publish per-batch progress.
//!
//! `RunStats` (the public final-snapshot view) lives in `runtime/mod.rs`
//! alongside `RunOutcome` and the other re-exports. `AtomicRunStats` is
//! the live, lock-free aggregation that workers update via Relaxed CAS;
//! the snapshot is read once at shutdown and converted into `RunStats`.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub(super) struct AtomicRunStats {
    pub(super) states_generated: AtomicU64,
    pub(super) states_processed: AtomicU64,
    pub(super) states_distinct: AtomicU64,
    pub(super) duplicates: AtomicU64,
    pub(super) enqueued: AtomicU64,
    pub(super) checkpoints: AtomicU64,
}

impl AtomicRunStats {
    /// Create stats initialized from checkpoint values (for resume)
    pub(super) fn from_checkpoint(
        states_generated: u64,
        states_processed: u64,
        states_distinct: u64,
        duplicates: u64,
        enqueued: u64,
        checkpoints: u64,
    ) -> Self {
        Self {
            states_generated: AtomicU64::new(states_generated),
            states_processed: AtomicU64::new(states_processed),
            states_distinct: AtomicU64::new(states_distinct),
            duplicates: AtomicU64::new(duplicates),
            enqueued: AtomicU64::new(enqueued),
            checkpoints: AtomicU64::new(checkpoints),
        }
    }

    pub(super) fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64) {
        (
            self.states_generated.load(Ordering::Relaxed),
            self.states_processed.load(Ordering::Relaxed),
            self.states_distinct.load(Ordering::Relaxed),
            self.duplicates.load(Ordering::Relaxed),
            self.enqueued.load(Ordering::Relaxed),
            self.checkpoints.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn default_is_all_zeroes() {
        let s = AtomicRunStats::default();
        assert_eq!(s.snapshot(), (0, 0, 0, 0, 0, 0));
    }

    #[test]
    fn from_checkpoint_preserves_all_six_counters() {
        // Distinct values catch field-order swap mutations (e.g. processed
        // and distinct accidentally aliased).
        let s = AtomicRunStats::from_checkpoint(11, 22, 33, 44, 55, 66);
        assert_eq!(s.snapshot(), (11, 22, 33, 44, 55, 66));
    }

    #[test]
    fn snapshot_is_field_aligned() {
        // Bump each counter in isolation; snapshot must surface in the
        // documented (gen, proc, dist, dup, enq, ckpt) order.
        let s = AtomicRunStats::default();
        s.states_generated.fetch_add(1, Ordering::Relaxed);
        assert_eq!(s.snapshot(), (1, 0, 0, 0, 0, 0));
        s.states_processed.fetch_add(2, Ordering::Relaxed);
        assert_eq!(s.snapshot(), (1, 2, 0, 0, 0, 0));
        s.states_distinct.fetch_add(4, Ordering::Relaxed);
        assert_eq!(s.snapshot(), (1, 2, 4, 0, 0, 0));
        s.duplicates.fetch_add(8, Ordering::Relaxed);
        assert_eq!(s.snapshot(), (1, 2, 4, 8, 0, 0));
        s.enqueued.fetch_add(16, Ordering::Relaxed);
        assert_eq!(s.snapshot(), (1, 2, 4, 8, 16, 0));
        s.checkpoints.fetch_add(32, Ordering::Relaxed);
        assert_eq!(s.snapshot(), (1, 2, 4, 8, 16, 32));
    }

    #[test]
    fn concurrent_increments_sum_correctly() {
        // Atomic increments from N threads must total N*K — kills any
        // mutation that swaps fetch_add → store or relaxes the type.
        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 5_000;
        let s = Arc::new(AtomicRunStats::default());
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let s = Arc::clone(&s);
            handles.push(std::thread::spawn(move || {
                for _ in 0..PER_THREAD {
                    s.states_generated.fetch_add(1, Ordering::Relaxed);
                    s.duplicates.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let (gen_count, proc_count, dist, dup, enq, ckpt) = s.snapshot();
        assert_eq!(gen_count, THREADS * PER_THREAD);
        assert_eq!(dup, THREADS * PER_THREAD);
        assert_eq!(proc_count, 0);
        assert_eq!(dist, 0);
        assert_eq!(enq, 0);
        assert_eq!(ckpt, 0);
    }
}
