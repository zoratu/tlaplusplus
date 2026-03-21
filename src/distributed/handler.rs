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

/// Spawn the inbound message handler as a tokio task.
///
/// This task receives messages from the transport layer and processes them:
///
/// - `StealRequest`: Pop states from the donation channel and send back
/// - `StealResponse`: Forward states to the stolen-work channel for workers to pick up
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
                    // Pop up to max_items states from the donation channel
                    let mut states = Vec::with_capacity(max_items as usize);
                    for _ in 0..max_items {
                        match donate_rx.try_recv() {
                            Ok(compressed) => states.push(compressed),
                            Err(_) => break,
                        }
                    }
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
