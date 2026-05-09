//! Cross-node distributed work-stealing handler wiring (T6).
//!
//! Spawns the inbound message handler, the bloom-and-termination
//! broadcaster, and the steal-trigger task into the
//! `DistributedWorkStealer`'s tokio handle. Short-circuits cleanly on
//! singleton clusters (the trigger checks `should_initiate_steal`
//! internally), so unconditional spawn is safe.
//!
//! ### Critical: `local_pending_fn` ordering bridge (T6)
//!
//! `pending_count()` over-reports because workers only flush their
//! per-worker popped counter at exit. To keep the steal trigger from
//! firing on stale "still has work" snapshots, we wrap it in a closure
//! that first consults `has_pending_work()` (which inspects deques + per-worker
//! active flags) and reports 0 if the queue is genuinely empty.

use crate::distributed::handler::LocalPendingFn;
use crate::storage::spillable_work_stealing::SpillableWorkStealingQueues;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::EngineConfig;

/// Build the cross-node "local pending count" bridge closure used by both
/// the inbound steal handler (to apply the steal-victim threshold) and the
/// steal-trigger task (to detect a starved local queue).
///
/// `SpillableWorkStealingQueues::pending_count()` over-reports because
/// workers only flush their per-worker popped counter at exit. To keep the
/// steal trigger from firing on stale "still has work" snapshots, this
/// closure first consults `has_pending_work()` (which inspects the actual
/// deques + per-worker active flags) and reports 0 when the queue is
/// genuinely empty.
///
/// Extracted so the bridge logic can be unit-tested without spinning up the
/// full distributed handler.
pub(super) fn make_local_pending_fn<S>(
    queue: &Arc<SpillableWorkStealingQueues<S>>,
) -> LocalPendingFn
where
    S: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + Clone + 'static,
{
    let queue_for_pending = Arc::clone(queue);
    Arc::new(move || {
        if !queue_for_pending.has_pending_work() {
            0
        } else {
            queue_for_pending.pending_count()
        }
    })
}

/// Spawn distributed-mode handlers when `config.distributed_stealer` is set.
///
/// This is a no-op when distributed mode is disabled.
pub(super) fn spawn_handlers_if_any<S>(
    config: &EngineConfig,
    queue: &Arc<SpillableWorkStealingQueues<S>>,
    stop: &Arc<AtomicBool>,
) where
    S: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + Clone + 'static,
{
    let Some(ref stealer) = config.distributed_stealer else {
        return;
    };

    // `stealer.transport()` already returns `&Arc<dyn Transport>` so the
    // clone trivially produces the trait object the handler tasks expect.
    let handler_transport = Arc::clone(stealer.transport());
    let handler_stealer = Arc::clone(stealer);
    let handler_stop = Arc::clone(stop);
    let tokio_handle = stealer.tokio_handle();

    // Closure used by both the inbound handler (to honor the steal-victim
    // threshold) and the steal-trigger task (to detect a starved local queue).
    // See `make_local_pending_fn` for the load-bearing visibility note.
    let local_pending_fn = make_local_pending_fn(queue);

    // Spawn inbound handler with the steal/donate channels
    if let (Some(stolen_tx), Some(donate_rx)) =
        (&config.stolen_states_tx, &config.donate_states_rx)
    {
        crate::distributed::handler::spawn_inbound_handler(
            &tokio_handle,
            handler_transport.clone(),
            handler_stealer.clone(),
            stolen_tx.clone(),
            donate_rx.clone(),
            handler_stop.clone(),
            Arc::clone(&local_pending_fn),
        );
    }

    // Spawn bloom exchange and termination broadcaster
    crate::distributed::handler::spawn_bloom_and_termination_task(
        &tokio_handle,
        handler_transport.clone(),
        handler_stealer.clone(),
        handler_stop.clone(),
        100, // check every 100ms
    );

    // Spawn the cross-node steal-trigger (T6).
    // Only meaningful when the cluster has >1 node, but the trigger
    // itself short-circuits in `should_initiate_steal` for singleton
    // clusters, so spawning unconditionally is harmless.
    crate::distributed::handler::spawn_steal_trigger_task(
        &tokio_handle,
        handler_transport,
        handler_stealer,
        handler_stop,
        local_pending_fn,
        100, // poll every 100ms; 250ms idle window before firing
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::spillable_work_stealing::SpillableConfig;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    fn temp_spill_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tlapp-runtime-distributed-{prefix}-{nanos}-{}",
            std::process::id()
        ))
    }

    fn build_queue(prefix: &str) -> Arc<SpillableWorkStealingQueues<u64>> {
        let dir = temp_spill_dir(prefix);
        let cfg = SpillableConfig {
            max_inmem_items: 1_000,
            spill_dir: dir,
            spill_batch: 100,
            load_existing: false,
            worker_spill_buffer_size: 50,
            worker_channel_bound: 4,
            defer_segment_deletion: false,
            compression_enabled: false,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };
        let (queues, _workers) =
            SpillableWorkStealingQueues::<u64>::new(2, vec![0, 0], cfg).expect("queue");
        queues
    }

    #[test]
    fn spawn_handlers_is_noop_when_distributed_stealer_is_none() {
        // Without a configured stealer the function must early-return
        // cleanly; constructing the queue and stop flag is enough to
        // exercise the entire happy short-circuit path.
        let config = EngineConfig::default();
        assert!(config.distributed_stealer.is_none());
        let queue = build_queue("noop");
        let stop = Arc::new(AtomicBool::new(false));
        // Must not panic.
        spawn_handlers_if_any(&config, &queue, &stop);
        // Stop flag must be left untouched.
        assert!(!stop.load(Ordering::Acquire));
    }

    #[test]
    fn local_pending_fn_reports_zero_on_empty_queue() {
        // When `has_pending_work` returns false, the bridge must report 0
        // even if `pending_count` is stale (the load-bearing invariant).
        let queue = build_queue("empty");
        assert!(!queue.has_pending_work(), "freshly built queue is empty");
        let f = make_local_pending_fn(&queue);
        assert_eq!(f(), 0, "empty queue must report 0 pending");
    }

    #[test]
    fn local_pending_fn_reflects_pushed_items() {
        // After pushing items, the bridge must report a nonzero count
        // (specifically, queue.pending_count()).
        let queue = build_queue("pushed");
        for i in 0..32u64 {
            queue.push_global(i);
        }
        let f = make_local_pending_fn(&queue);
        // Exact value tracks `pending_count()`; we only need a positive
        // sanity bound to kill mutations that flip the if-branch.
        assert!(
            f() > 0,
            "queue with 32 items should report >0 pending, got {}",
            f()
        );
    }

    #[test]
    fn local_pending_fn_is_cloneable_arc() {
        // The closure is wrapped in an Arc so multiple handlers can hold
        // it. Verify that calling through clones returns the same value.
        let queue = build_queue("clone");
        for i in 0..8u64 {
            queue.push_global(i);
        }
        let f1 = make_local_pending_fn(&queue);
        let f2 = Arc::clone(&f1);
        assert_eq!(f1(), f2(), "Arc-cloned closures must agree");
    }
}
