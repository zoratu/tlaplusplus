use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;

use super::protocol::Message;
use super::proxy::DistributedFingerprintProxy;
use super::transport::ClusterTransport;

/// Spawn the inbound message handler as a tokio task.
///
/// This task receives messages from the transport layer, processes them,
/// and feeds new states back to the local worker queues via the proxy.
///
/// For `FingerprintBatch` messages:
/// 1. Check each fingerprint against the local fingerprint store
/// 2. States with new fingerprints are enqueued to the proxy's inbound channel
///    for local workers to pick up and explore
/// 3. Send `FingerprintAck` back to the sender with a bitmap of new vs seen
///
/// For `TerminationToken` messages:
/// - Update peer idle status in the proxy
/// - If all peers + self are idle, set global termination
///
/// For `Stop` messages:
/// - Set the stop flag to halt local exploration
pub fn spawn_inbound_handler(
    transport: Arc<ClusterTransport>,
    proxy: Arc<DistributedFingerprintProxy>,
    fp_store: Arc<UnifiedFingerprintStore>,
    stop: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        loop {
            if stop.load(Ordering::Acquire) {
                break;
            }

            let msg = match transport.recv().await {
                Some((_from, msg)) => msg,
                None => {
                    // Transport shut down
                    break;
                }
            };

            match msg {
                Message::FingerprintBatch {
                    from_node,
                    batch_id,
                    entries,
                } => {
                    handle_fingerprint_batch(
                        &proxy,
                        &fp_store,
                        &transport,
                        from_node,
                        batch_id,
                        entries,
                    )
                    .await;
                }

                Message::TerminationToken {
                    initiator,
                    round: _,
                    all_idle,
                } => {
                    if all_idle {
                        // The initiator is reporting that it sees all nodes idle.
                        // Mark all-idle for this peer.
                        proxy.set_peer_idle(initiator, true);
                    } else {
                        proxy.set_peer_idle(initiator, false);
                    }

                    // Check if we can declare global termination
                    if proxy.all_nodes_idle() {
                        proxy.set_globally_terminated();
                    }
                }

                Message::Stop { node_id, message } => {
                    eprintln!(
                        "[cluster] received stop from node {}: {}",
                        node_id, message
                    );
                    stop.store(true, Ordering::Release);
                }

                Message::Heartbeat {
                    node_id,
                    states_generated: _,
                    states_distinct: _,
                } => {
                    // Heartbeats are informational — could update peer stats
                    // For now, just acknowledge the peer is alive
                    let _ = node_id;
                }

                Message::Join { .. } | Message::Leave { .. } | Message::FingerprintAck { .. } => {
                    // Join/Leave: dynamic membership changes not yet supported
                    // FingerprintAck: currently unused (fire-and-forget batching)
                }
            }
        }
    });
}

/// Handle an inbound fingerprint batch: check each fingerprint against the
/// local store, enqueue new states for exploration, and send an ack back.
async fn handle_fingerprint_batch(
    proxy: &DistributedFingerprintProxy,
    fp_store: &UnifiedFingerprintStore,
    transport: &ClusterTransport,
    from_node: u32,
    batch_id: u64,
    entries: Vec<(u64, Vec<u8>)>,
) {
    let mut new_bitmap = Vec::with_capacity(entries.len());
    let fps: Vec<u64> = entries.iter().map(|(fp, _)| *fp).collect();
    let mut seen = vec![false; fps.len()];

    // Batch check-and-insert against the local fingerprint store.
    // Use worker_id=0 affinity since we're in the handler task.
    if let Err(e) = fp_store.contains_or_insert_batch_with_affinity(&fps, &mut seen, 0) {
        eprintln!(
            "[cluster] fingerprint batch check failed for batch {} from node {}: {}",
            batch_id, from_node, e
        );
        return;
    }

    for (idx, (fp, compressed_state)) in entries.into_iter().enumerate() {
        let is_new = !seen[idx];
        new_bitmap.push(is_new);
        if is_new {
            // This is a new state — enqueue it for local exploration
            proxy.enqueue_inbound(fp, compressed_state);
        }
    }

    // Send acknowledgment back to the sender
    let ack = Message::FingerprintAck {
        batch_id,
        new_bitmap,
    };
    if let Err(e) = transport.send(from_node, &ack).await {
        eprintln!(
            "[cluster] failed to send FingerprintAck for batch {} to node {}: {}",
            batch_id, from_node, e
        );
    }
}

/// Spawn a periodic termination-check task that broadcasts this node's
/// idle status to all peers.
///
/// Runs every `interval_ms` milliseconds. When the local node is idle
/// (all workers idle + all queues empty + all outbound batches flushed),
/// it broadcasts a `TerminationToken` with `all_idle=true`.
pub fn spawn_termination_broadcaster(
    transport: Arc<ClusterTransport>,
    proxy: Arc<DistributedFingerprintProxy>,
    stop: Arc<AtomicBool>,
    interval_ms: u64,
) {
    tokio::spawn(async move {
        let mut round = 0u64;
        let interval = tokio::time::Duration::from_millis(interval_ms);

        loop {
            tokio::time::sleep(interval).await;

            if stop.load(Ordering::Acquire) {
                break;
            }

            if proxy.is_globally_terminated() {
                break;
            }

            // First, flush any expired batches from all workers
            proxy.flush_all();

            let is_idle = proxy.pending_count() == 0;
            // Note: locally_idle must be set externally by the runtime
            // based on worker idle status + queue emptiness

            let token = Message::TerminationToken {
                initiator: proxy.node_id(),
                round,
                all_idle: is_idle,
            };

            if let Err(e) = transport.broadcast(&token).await {
                eprintln!("[cluster] failed to broadcast termination token: {}", e);
            }

            round += 1;
        }
    });
}
