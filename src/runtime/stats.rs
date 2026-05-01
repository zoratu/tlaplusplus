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
