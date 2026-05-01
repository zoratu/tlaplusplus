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

    let handler_transport = Arc::clone(stealer.transport());
    let handler_stealer = Arc::clone(stealer);
    let handler_stop = Arc::clone(stop);
    let tokio_handle = stealer.tokio_handle();

    // Closure used by both the inbound handler (to honor the steal-victim
    // threshold) and the steal-trigger task (to detect a starved local queue).
    //
    // NOTE: `pending_count()` is an approximation derived from
    // `global_pushed - global_popped`, where `global_popped` is only
    // updated when workers flush their local counters (currently only at
    // worker exit). That makes `pending_count()` over-report by
    // potentially the entire local-popped count, which would prevent the
    // steal trigger from firing even when the local queue is genuinely
    // empty. We bridge this by ALSO checking `has_pending_work()` (which
    // inspects the actual deques + per-worker active flags); when it
    // returns false, we report 0 to the trigger.
    let queue_for_pending = Arc::clone(queue);
    let local_pending_fn: LocalPendingFn = Arc::new(move || {
        if !queue_for_pending.has_pending_work() {
            0
        } else {
            queue_for_pending.pending_count()
        }
    });

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
