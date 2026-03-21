use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Messages exchanged between cluster nodes.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Message {
    /// Batch of (fingerprint, compressed_state) pairs to check-and-insert.
    FingerprintBatch {
        from_node: u32,
        batch_id: u64,
        /// Each entry is `(fingerprint, zstd-compressed serialized state bytes)`.
        entries: Vec<(u64, Vec<u8>)>,
    },
    /// Response: bitmap indicating which entries were new (true) vs already seen (false).
    FingerprintAck {
        batch_id: u64,
        new_bitmap: Vec<bool>,
    },
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
    /// Termination token for distributed termination detection
    /// (Dijkstra-Scholten / ring-based).
    TerminationToken {
        initiator: u32,
        round: u64,
        all_idle: bool,
    },
}

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
        // First 4 bytes are length prefix
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
    fn roundtrip_fingerprint_batch() {
        let msg = Message::FingerprintBatch {
            from_node: 2,
            batch_id: 42,
            entries: vec![(0xDEADBEEF, vec![1, 2, 3]), (0xCAFEBABE, vec![4, 5])],
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::FingerprintBatch {
                from_node,
                batch_id,
                entries,
            } => {
                assert_eq!(from_node, 2);
                assert_eq!(batch_id, 42);
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].0, 0xDEADBEEF);
            }
            _ => panic!("expected FingerprintBatch"),
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
        };
        let encoded = encode_message(&msg).unwrap();
        let len = u32::from_be_bytes(encoded[..4].try_into().unwrap()) as usize;
        let decoded = decode_message(&encoded[4..4 + len]).unwrap();
        match decoded {
            Message::TerminationToken {
                initiator,
                round,
                all_idle,
            } => {
                assert_eq!(initiator, 1);
                assert_eq!(round, 99);
                assert!(all_idle);
            }
            _ => panic!("expected TerminationToken"),
        }
    }
}
