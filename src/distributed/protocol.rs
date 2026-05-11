use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Messages exchanged between cluster nodes.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Message {
    // --- Work stealing (independent exploration) ---
    /// Request to steal work from a peer node.
    StealRequest {
        from_node: u32,
        /// Maximum number of states to steal.
        max_items: u32,
    },
    /// Response to a steal request with serialized states.
    StealResponse {
        /// Compressed serialized states (each is a zstd-compressed bincode blob).
        states: Vec<Vec<u8>>,
    },
    /// Bloom filter exchange for probabilistic cross-node dedup.
    BloomExchange {
        from_node: u32,
        /// Serialized BloomSnapshot (bincode).
        bloom_data: Vec<u8>,
    },

    // --- Cluster management ---
    /// A node joining the cluster.
    Join {
        node_id: u32,
        addr: SocketAddr,
        num_cores: u32,
    },
    /// A node leaving gracefully.
    Leave { node_id: u32 },
    /// Periodic heartbeat carrying progress counters.
    Heartbeat {
        node_id: u32,
        states_generated: u64,
        states_distinct: u64,
    },
    /// Stop exploration (invariant violation found).
    Stop { node_id: u32, message: String },
    /// Termination token for distributed termination detection.
    ///
    /// Phase 2 of T10.2 extends this message with two `Option<u64>` fields
    /// (`inflight_partition_edges`, `inflight_red_probes`). Both are
    /// `Option`-typed so older nodes that send the message without the
    /// streaming fields are still understood by newer nodes — bincode
    /// happily decodes a missing-but-trailing `Option::None`. New nodes
    /// initialize both to `None` when streaming is disabled, preserving
    /// the old wire shape on the no-streaming path.
    TerminationToken {
        initiator: u32,
        round: u64,
        all_idle: bool,
        /// Sum of all nodes' in-flight `PartitionEdge` counters at the
        /// moment they last forwarded the token. `None` means the sender
        /// is not running streaming-DFS mode.
        #[serde(default)]
        inflight_partition_edges: Option<u64>,
        /// Sum of all nodes' in-flight `RedDfsProbe` counters at the
        /// moment they last forwarded the token. `None` means the sender
        /// is not running streaming-DFS mode.
        #[serde(default)]
        inflight_red_probes: Option<u64>,
    },

    // --- T10.2 phase 2 streaming nested-DFS protocol ---
    /// A cross-partition DFS edge. The sender encountered `state_fp` while
    /// expanding a frame; the receiver owns this fingerprint's partition
    /// (`partition_id = state_fp mod partition_count`) and is responsible
    /// for entering blue DFS on it (or, if the slot is already Cyan/Blue/
    /// Red, recording the back-edge for cycle detection).
    PartitionEdge {
        from_node: u32,
        from_worker: u32,
        owner_node: u32,
        owner_worker: u32,
        /// zstd-compressed bincode of the full state, as sent on the wire.
        /// Receivers decode lazily — most edges land on already-coloured
        /// fingerprints and never require state materialization.
        state_blob: Vec<u8>,
        /// Fingerprint of `state_blob` (sender's home), so receiver does
        /// not have to recompute on every message.
        state_fp: u64,
        /// Action label index that produced this edge, for trace
        /// reconstruction. Use `u16::MAX` as the "unknown action" sentinel.
        via_action: u16,
        /// The sender's `blue_path_fps` depth when emitting this edge,
        /// used by the receiver for fairness-threshold computation.
        sender_depth: u32,
    },

    /// Acknowledgement that a `PartitionEdge` was processed (added to the
    /// receiver's frontier OR found already-colored). Required for
    /// termination detection: the global "all queues empty" predicate has
    /// to count in-flight partition edges, mirroring the existing
    /// `pending_steals` counter on `DistributedStealer`.
    PartitionEdgeAck {
        from_node: u32,
        owner_node: u32,
        state_fp: u64,
        outcome: PartitionEdgeOutcome,
    },

    /// Cross-partition red-DFS probe. The red search may itself cross
    /// partitions; this carries the search to the partition owning
    /// `target_fp`. Carries a TTL via `trail.len() <= MAX_RED_DFS_HOPS`;
    /// if the trail length saturates, the receiver returns
    /// `RedDfsOutcome::RedDfsTruncated`.
    RedDfsProbe {
        from_node: u32,
        from_worker: u32,
        owner_node: u32,
        owner_worker: u32,
        /// The accepting state that launched this red DFS (used by the
        /// receiver to detect "this cycle returns to my own seed", which
        /// is the lasso-witness condition).
        seed_fp: u64,
        /// The state to test for Cyan (witness) or to continue red-DFS
        /// expansion from.
        target_fp: u64,
        /// Fingerprints on the red trail so far, in DFS push order.
        trail: Vec<u64>,
    },

    /// Response to a `RedDfsProbe`. The trail extension lets the requester
    /// stitch together a multi-partition cycle witness without round-tripping
    /// individual states.
    RedDfsResponse {
        from_node: u32,
        seed_fp: u64,
        outcome: RedDfsOutcome,
        /// Fingerprints to append to the requester's trail to reconstruct
        /// the witness path. Empty when the outcome is `NotFound`.
        trail_extension: Vec<u64>,
    },

    /// Trace-reconstruction request. When fairness fails and the cycle
    /// straddles partitions, the lasso reporter sends this message to peer
    /// partitions to fetch the full state blobs corresponding to a list
    /// of fingerprints. One round trip per peer is fine — trace
    /// reconstruction is one-shot per violation, not on the hot path.
    RequestStateBlob {
        from_node: u32,
        owner_node: u32,
        fps: Vec<u64>,
    },

    /// Response to `RequestStateBlob`. `blobs[i]` corresponds to
    /// `fps[i]`; missing entries (e.g. the peer evicted the state) are
    /// returned as empty `Vec<u8>` and the requester falls back to
    /// reporting the lasso without that frame's full state.
    StateBlobResponse {
        from_node: u32,
        owner_node: u32,
        fps: Vec<u64>,
        blobs: Vec<Vec<u8>>,
    },
}

/// Outcome of processing a `PartitionEdge` on its home partition. Carried
/// in the `PartitionEdgeAck` so the sender can update its DFS state
/// without a second round trip.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq)]
pub enum PartitionEdgeOutcome {
    /// Receiver CAS'd White->Cyan; this is a fresh DFS root for them.
    ClaimedFresh,
    /// Receiver observed an already-colored slot (Blue or Red); no DFS
    /// work to do, but the edge is reported for completeness.
    AlreadyVisited,
    /// Receiver observed Cyan: this is a back-edge into the *receiver's*
    /// blue path. Sender records it as a candidate cycle witness and may
    /// trigger red-DFS.
    BackEdgeToCyan,
}

/// Outcome of a cross-partition red-DFS probe.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RedDfsOutcome {
    /// The probe found a Cyan slot reachable from `target_fp`. The witness
    /// fingerprint is the Cyan slot itself.
    FoundCyanWitness(u64),
    /// The probe finished without finding a witness; no cycle through this
    /// branch.
    NotFound,
    /// The probe hit the TTL bound (`trail.len() >= MAX_RED_DFS_HOPS`); the
    /// caller should fall back to the Tarjan path for this SCC.
    RedDfsTruncated,
}

/// TTL on cross-partition red-DFS probes. Mirrors the design doc's
/// `max_red_dfs_hops`.
pub const MAX_RED_DFS_HOPS: usize = 1024;

/// Wire format: `[4-byte big-endian length][bincode payload]`.
///
/// The 4-byte length prefix allows framing on a TCP stream so the receiver
/// knows exactly how many bytes to read for each message.
pub fn encode_message(msg: &Message) -> Result<Vec<u8>> {
    let payload = bincode::serialize(msg).context("failed to bincode-serialize cluster message")?;
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decode a message from a bincode payload (without the 4-byte length prefix).
///
/// Callers are expected to have already read the length prefix and extracted
/// exactly that many bytes before calling this function.
pub fn decode_message(bytes: &[u8]) -> Result<Message> {
    bincode::deserialize(bytes).context("failed to bincode-deserialize cluster message")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_heartbeat() {
        let msg = Message::Heartbeat {
            node_id: 7,
            states_generated: 1_000_000,
            states_distinct: 500_000,
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        assert_eq!(len, encoded.len() - 4);
        let decoded = decode_message(&encoded[4..]).unwrap();
        match decoded {
            Message::Heartbeat {
                node_id,
                states_generated,
                states_distinct,
            } => {
                assert_eq!(node_id, 7);
                assert_eq!(states_generated, 1_000_000);
                assert_eq!(states_distinct, 500_000);
            }
            _ => panic!("expected Heartbeat"),
        }
    }

    #[test]
    fn roundtrip_steal_request() {
        let msg = Message::StealRequest {
            from_node: 2,
            max_items: 512,
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::StealRequest {
                from_node,
                max_items,
            } => {
                assert_eq!(from_node, 2);
                assert_eq!(max_items, 512);
            }
            _ => panic!("expected StealRequest"),
        }
    }

    #[test]
    fn roundtrip_steal_response() {
        let msg = Message::StealResponse {
            states: vec![vec![1, 2, 3], vec![4, 5, 6]],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::StealResponse { states } => {
                assert_eq!(states.len(), 2);
                assert_eq!(states[0], vec![1, 2, 3]);
            }
            _ => panic!("expected StealResponse"),
        }
    }

    #[test]
    fn roundtrip_bloom_exchange() {
        let msg = Message::BloomExchange {
            from_node: 1,
            bloom_data: vec![0xFF; 100],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::BloomExchange {
                from_node,
                bloom_data,
            } => {
                assert_eq!(from_node, 1);
                assert_eq!(bloom_data.len(), 100);
            }
            _ => panic!("expected BloomExchange"),
        }
    }

    #[test]
    fn roundtrip_stop() {
        let msg = Message::Stop {
            node_id: 0,
            message: "invariant Inv violated".to_string(),
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::Stop { node_id, message } => {
                assert_eq!(node_id, 0);
                assert_eq!(message, "invariant Inv violated");
            }
            _ => panic!("expected Stop"),
        }
    }

    #[test]
    fn roundtrip_termination_token() {
        let msg = Message::TerminationToken {
            initiator: 1,
            round: 99,
            all_idle: true,
            inflight_partition_edges: None,
            inflight_red_probes: None,
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::TerminationToken {
                initiator,
                round,
                all_idle,
                inflight_partition_edges,
                inflight_red_probes,
            } => {
                assert_eq!(initiator, 1);
                assert_eq!(round, 99);
                assert!(all_idle);
                assert_eq!(inflight_partition_edges, None);
                assert_eq!(inflight_red_probes, None);
            }
            _ => panic!("expected TerminationToken"),
        }
    }

    #[test]
    fn roundtrip_termination_token_with_streaming_counters() {
        let msg = Message::TerminationToken {
            initiator: 3,
            round: 17,
            all_idle: true,
            inflight_partition_edges: Some(42),
            inflight_red_probes: Some(7),
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::TerminationToken {
                inflight_partition_edges,
                inflight_red_probes,
                ..
            } => {
                assert_eq!(inflight_partition_edges, Some(42));
                assert_eq!(inflight_red_probes, Some(7));
            }
            _ => panic!("expected TerminationToken"),
        }
    }

    #[test]
    fn roundtrip_partition_edge() {
        let msg = Message::PartitionEdge {
            from_node: 2,
            from_worker: 5,
            owner_node: 3,
            owner_worker: 11,
            state_blob: vec![0xCA, 0xFE, 0xBA, 0xBE],
            state_fp: 0x1122334455667788,
            via_action: 0x0042,
            sender_depth: 17,
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::PartitionEdge {
                from_node,
                from_worker,
                owner_node,
                owner_worker,
                state_blob,
                state_fp,
                via_action,
                sender_depth,
            } => {
                assert_eq!(from_node, 2);
                assert_eq!(from_worker, 5);
                assert_eq!(owner_node, 3);
                assert_eq!(owner_worker, 11);
                assert_eq!(state_blob, vec![0xCA, 0xFE, 0xBA, 0xBE]);
                assert_eq!(state_fp, 0x1122334455667788);
                assert_eq!(via_action, 0x0042);
                assert_eq!(sender_depth, 17);
            }
            _ => panic!("expected PartitionEdge"),
        }
    }

    #[test]
    fn roundtrip_partition_edge_ack() {
        for outcome in [
            PartitionEdgeOutcome::ClaimedFresh,
            PartitionEdgeOutcome::AlreadyVisited,
            PartitionEdgeOutcome::BackEdgeToCyan,
        ] {
            let msg = Message::PartitionEdgeAck {
                from_node: 4,
                owner_node: 1,
                state_fp: 0xDEADBEEFDEADBEEF,
                outcome,
            };
            let encoded = encode_message(&msg).unwrap();
            let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
            let decoded = decode_message(&encoded[4..4 + len]).unwrap();
            match decoded {
                Message::PartitionEdgeAck {
                    from_node,
                    owner_node,
                    state_fp,
                    outcome: got,
                } => {
                    assert_eq!(from_node, 4);
                    assert_eq!(owner_node, 1);
                    assert_eq!(state_fp, 0xDEADBEEFDEADBEEF);
                    assert_eq!(got, outcome);
                }
                _ => panic!("expected PartitionEdgeAck"),
            }
        }
    }

    #[test]
    fn roundtrip_red_dfs_probe() {
        let msg = Message::RedDfsProbe {
            from_node: 0,
            from_worker: 1,
            owner_node: 2,
            owner_worker: 3,
            seed_fp: 0xAAAA,
            target_fp: 0xBBBB,
            trail: vec![0x1, 0x2, 0x3, 0x4],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::RedDfsProbe {
                seed_fp,
                target_fp,
                trail,
                ..
            } => {
                assert_eq!(seed_fp, 0xAAAA);
                assert_eq!(target_fp, 0xBBBB);
                assert_eq!(trail, vec![0x1, 0x2, 0x3, 0x4]);
            }
            _ => panic!("expected RedDfsProbe"),
        }
    }

    #[test]
    fn roundtrip_red_dfs_response_witness() {
        let msg = Message::RedDfsResponse {
            from_node: 1,
            seed_fp: 0xAAAA,
            outcome: RedDfsOutcome::FoundCyanWitness(0xCCCC),
            trail_extension: vec![0x10, 0x20],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::RedDfsResponse {
                outcome,
                trail_extension,
                ..
            } => {
                match outcome {
                    RedDfsOutcome::FoundCyanWitness(fp) => assert_eq!(fp, 0xCCCC),
                    other => panic!("expected witness, got {:?}", other),
                }
                assert_eq!(trail_extension, vec![0x10, 0x20]);
            }
            _ => panic!("expected RedDfsResponse"),
        }
    }

    #[test]
    fn roundtrip_red_dfs_response_truncated() {
        let msg = Message::RedDfsResponse {
            from_node: 1,
            seed_fp: 0xAAAA,
            outcome: RedDfsOutcome::RedDfsTruncated,
            trail_extension: Vec::new(),
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::RedDfsResponse { outcome, .. } => match outcome {
                RedDfsOutcome::RedDfsTruncated => (),
                other => panic!("expected truncated, got {:?}", other),
            },
            _ => panic!("expected RedDfsResponse"),
        }
    }

    #[test]
    fn roundtrip_request_state_blob() {
        let msg = Message::RequestStateBlob {
            from_node: 0,
            owner_node: 1,
            fps: vec![0xA, 0xB, 0xC],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::RequestStateBlob { fps, .. } => assert_eq!(fps, vec![0xA, 0xB, 0xC]),
            _ => panic!("expected RequestStateBlob"),
        }
    }

    #[test]
    fn roundtrip_state_blob_response() {
        let msg = Message::StateBlobResponse {
            from_node: 1,
            owner_node: 0,
            fps: vec![0xA, 0xB],
            blobs: vec![vec![1, 2, 3], vec![]],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::StateBlobResponse { fps, blobs, .. } => {
                assert_eq!(fps, vec![0xA, 0xB]);
                assert_eq!(blobs, vec![vec![1, 2, 3], vec![]]);
            }
            _ => panic!("expected StateBlobResponse"),
        }
    }
}
