//! Inbound message handler for the independent-exploration distributed model checker.
//!
//! Handles three message types on the hot path:
//! - `StealRequest`: pop states from local donation channel, serialize, send back
//! - `StealResponse`: forward stolen states to the work stealer for local processing
//! - `BloomExchange`: merge remote bloom filter into the work stealer
//!
//! Also handles termination tokens, stop signals, and heartbeats.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::Sender;

use super::protocol::Message;
use super::transport::ClusterTransport;
use super::work_stealer::DistributedWorkStealer;

/// Inbound state received from a steal response.
/// Contains the compressed serialized state bytes.
pub struct StolenState {
    pub compressed_state: Vec<u8>,
}

/// Closure that returns the current local pending-state count.
///
/// Passed into the inbound handler so the steal-victim path can decide
/// whether to donate or reply EMPTY based on actual queue depth (not just
/// the size of the donation channel, which is bounded and gets fed on a
/// timer-like schedule).
pub type LocalPendingFn = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Spawn the inbound message handler as a tokio task.
///
/// This task receives messages from the transport layer and processes them:
///
/// - `StealRequest`: If our local backlog is above the victim threshold, pop
///   states from the donation channel and send them back. Otherwise reply
///   with an empty state list so the requester moves on to another peer.
/// - `StealResponse`: Forward states to the stolen-work channel for workers
///   to pick up. Always decrements `pending_steal_requests` so termination
///   detection can advance.
/// - `BloomExchange`: Merge the remote bloom into the work stealer
/// - `TerminationToken`: Update peer idle status
/// - `Stop`: Set the stop flag to halt exploration
pub fn spawn_inbound_handler(
    handle: &tokio::runtime::Handle,
    transport: Arc<ClusterTransport>,
    stealer: Arc<DistributedWorkStealer>,
    stolen_tx: Sender<StolenState>,
    donate_rx: crossbeam_channel::Receiver<Vec<u8>>,
    stop: Arc<AtomicBool>,
    local_pending: LocalPendingFn,
) {
    let transport_for_handler = Arc::clone(&transport);
    handle.spawn(async move {
        loop {
            if stop.load(Ordering::Acquire) {
                break;
            }

            let msg = match transport_for_handler.recv().await {
                Some((_from, msg)) => msg,
                None => break, // Transport shut down
            };

            match msg {
                Message::StealRequest {
                    from_node,
                    max_items,
                } => {
                    let local = local_pending();
                    let mut states: Vec<Vec<u8>> = Vec::new();
                    if stealer.can_donate(local) {
                        states.reserve(max_items as usize);
                        for _ in 0..max_items {
                            match donate_rx.try_recv() {
                                Ok(compressed) => states.push(compressed),
                                Err(_) => break,
                            }
                        }
                    }
                    // else: reply EMPTY so requester moves on to another peer
                    let donated_count = states.len() as u64;

                    let response = Message::StealResponse { states };
                    if let Err(e) = transport_for_handler.send(from_node, &response).await {
                        eprintln!(
                            "[cluster] failed to send steal response to node {}: {}",
                            from_node, e
                        );
                    }

                    if donated_count > 0 {
                        stealer
                            .states_donated
                            .fetch_add(donated_count, Ordering::Relaxed);
                    }
                }

                Message::StealResponse { states } => {
                    let count = states.len();
                    if count > 0 {
                        stealer.note_work_received();
                    } else {
                        stealer.note_empty_steal_response();
                    }
                    for compressed in states {
                        let _ = stolen_tx.try_send(StolenState {
                            compressed_state: compressed,
                        });
                    }
                    if count > 0 {
                        stealer
                            .states_stolen
                            .fetch_add(count as u64, Ordering::Relaxed);
                    }
                    // ALWAYS decrement pending-steal counter when we get a
                    // response (empty or not). Otherwise termination detection
                    // would stall behind the in-flight count.
                    stealer.end_steal();
                }

                Message::BloomExchange {
                    from_node: _,
                    bloom_data,
                } => {
                    stealer.merge_remote_bloom(&bloom_data);
                }

                Message::TerminationToken {
                    initiator,
                    round: _,
                    all_idle,
                } => {
                    stealer.set_peer_idle(initiator, all_idle);
                    if stealer.all_nodes_idle() {
                        stealer.set_globally_terminated();
                    }
                }

                Message::Stop { node_id, message } => {
                    eprintln!("[cluster] received stop from node {}: {}", node_id, message);
                    stop.store(true, Ordering::Release);
                }

                Message::Heartbeat { node_id, .. } => {
                    let _ = node_id;
                }

                Message::Join { .. } | Message::Leave { .. } => {
                    // Dynamic membership not yet supported
                }
            }
        }
    });
}

/// Spawn a periodic bloom exchange + termination check task.
///
/// This task runs on a timer and:
/// 1. Triggers bloom filter exchange with peers
/// 2. Broadcasts this node's idle status for termination detection
pub fn spawn_bloom_and_termination_task(
    handle: &tokio::runtime::Handle,
    transport: Arc<ClusterTransport>,
    stealer: Arc<DistributedWorkStealer>,
    stop: Arc<AtomicBool>,
    interval_ms: u64,
) {
    handle.spawn(async move {
        let mut round = 0u64;
        let interval = tokio::time::Duration::from_millis(interval_ms);

        loop {
            tokio::time::sleep(interval).await;

            if stop.load(Ordering::Acquire) {
                break;
            }

            if stealer.is_globally_terminated() {
                break;
            }

            // Trigger bloom exchange if enough time has elapsed
            stealer.maybe_exchange_bloom();

            // Broadcast termination status
            let token = Message::TerminationToken {
                initiator: stealer.node_id(),
                round,
                all_idle: stealer.all_nodes_idle(),
            };

            if let Err(e) = transport.broadcast(&token).await {
                eprintln!("[cluster] failed to broadcast termination token: {}", e);
            }

            round += 1;
        }
    });
}

/// Spawn the cross-node steal trigger task.
///
/// Polls every `poll_interval_ms` (default 100ms). When the local node has
/// been starved of work for longer than the configured idle window AND the
/// local pending count is at the low-water mark, the task picks the first
/// live peer (in a clock-rotated order) and sends a `StealRequest`.
///
/// Failure handling:
/// - If the transport `send` errors out, the peer is marked down for the
///   configured cooldown (default 30s) and the in-flight counter is rolled
///   back so termination detection isn't blocked by a dead peer.
/// - If the response never arrives (peer process died, TCP black hole), the
///   in-flight counter would otherwise stall. We arm a `tokio::time::sleep`
///   alongside the steal so a stuck request will time out and roll back the
///   counter.
///
/// IMPORTANT: this task does NOT touch any local queue lock or hold any
/// long-running mutex. The deduplication of received states happens in the
/// existing worker fast-path (`process_batch`); this task only fires a
/// network request.
pub fn spawn_steal_trigger_task(
    handle: &tokio::runtime::Handle,
    transport: Arc<ClusterTransport>,
    stealer: Arc<DistributedWorkStealer>,
    stop: Arc<AtomicBool>,
    local_pending: LocalPendingFn,
    poll_interval_ms: u64,
) {
    let transport = Arc::clone(&transport);
    handle.spawn(async move {
        let interval = tokio::time::Duration::from_millis(poll_interval_ms);
        loop {
            tokio::time::sleep(interval).await;

            if stop.load(Ordering::Acquire) {
                break;
            }
            if stealer.is_globally_terminated() {
                break;
            }

            let local = local_pending();
            if !stealer.should_initiate_steal(local) {
                continue;
            }

            // Cap concurrent in-flight steals — one at a time per node is
            // plenty (the response carries a batch of 4096 states).
            if stealer.pending_steal_count() > 0 {
                continue;
            }

            let peers = stealer.live_peers_shuffled();
            let Some(&peer_id) = peers.first() else {
                continue;
            };

            stealer.begin_steal();
            let req = Message::StealRequest {
                from_node: stealer.node_id(),
                max_items: stealer.steal_batch_size(),
            };

            // Send. If the send itself fails, mark peer down and roll back
            // the in-flight counter immediately. Otherwise leave it set; the
            // inbound handler will decrement when the response arrives.
            //
            // Arm a separate timeout: if the response never lands, the
            // timeout task rolls back so termination detection isn't blocked
            // by a dead/black-hole peer.
            match transport.send(peer_id, &req).await {
                Ok(()) => {
                    let stealer_for_to = Arc::clone(&stealer);
                    let snapshot_count_before = stealer.states_stolen.load(Ordering::Relaxed)
                        + stealer.steal_responses_empty.load(Ordering::Relaxed);
                    let timeout = super::work_stealer::DEFAULT_STEAL_TIMEOUT;
                    tokio::spawn(async move {
                        tokio::time::sleep(timeout).await;
                        let snapshot_count_after =
                            stealer_for_to.states_stolen.load(Ordering::Relaxed)
                                + stealer_for_to.steal_responses_empty.load(Ordering::Relaxed);
                        if snapshot_count_after == snapshot_count_before
                            && stealer_for_to.pending_steal_count() > 0
                        {
                            // Response never arrived — peer is stuck.
                            // Mark it down and roll back so termination
                            // detection isn't held hostage.
                            eprintln!(
                                "[cluster] steal request to node {} timed out after {:?}",
                                peer_id, timeout
                            );
                            stealer_for_to.mark_peer_down(peer_id);
                            stealer_for_to.end_steal();
                        }
                    });
                }
                Err(e) => {
                    eprintln!(
                        "[cluster] failed to send steal request to node {}: {}",
                        peer_id, e
                    );
                    stealer.mark_peer_down(peer_id);
                    stealer.end_steal();
                }
            }
        }
    });
}
