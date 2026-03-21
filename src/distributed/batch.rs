use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Per-destination batch accumulator with size and time-based flush triggers.
///
/// Workers push `(fingerprint, compressed_state_bytes)` entries destined for
/// remote nodes. When a destination batch reaches `batch_size` entries OR
/// the oldest entry has been waiting longer than `batch_timeout`, the batch
/// is flushed and returned to the caller for sending.
pub struct BatchAccumulator {
    /// Maximum entries per batch before forced flush.
    batch_size: usize,
    /// Maximum age (milliseconds) of the oldest entry before forced flush.
    batch_timeout_ms: u64,
    /// Per-destination buffers. Key is the target node_id.
    buffers: HashMap<u32, PendingBatch>,
    /// Monotonically increasing batch ID generator.
    next_batch_id: AtomicU64,
}

/// A batch being accumulated for a single destination node.
struct PendingBatch {
    entries: Vec<(u64, Vec<u8>)>,
    first_insert: Instant,
}

/// A completed batch ready to be sent.
pub struct ReadyBatch {
    /// Target node ID.
    pub dest_node: u32,
    /// Unique batch identifier for correlating with FingerprintAck.
    pub batch_id: u64,
    /// The fingerprint + compressed state entries.
    pub entries: Vec<(u64, Vec<u8>)>,
}

impl BatchAccumulator {
    /// Create a new accumulator with the given flush thresholds.
    pub fn new(batch_size: usize, batch_timeout_ms: u64) -> Self {
        BatchAccumulator {
            batch_size,
            batch_timeout_ms,
            buffers: HashMap::new(),
            next_batch_id: AtomicU64::new(0),
        }
    }

    /// Push an entry for a destination node.
    ///
    /// Returns `Some(ReadyBatch)` if the push caused the destination's buffer
    /// to reach `batch_size`, triggering an immediate flush. Otherwise `None`.
    pub fn push(
        &mut self,
        dest_node: u32,
        fp: u64,
        compressed_state: Vec<u8>,
    ) -> Option<ReadyBatch> {
        let pending = self
            .buffers
            .entry(dest_node)
            .or_insert_with(|| PendingBatch {
                entries: Vec::with_capacity(self.batch_size),
                first_insert: Instant::now(),
            });
        pending.entries.push((fp, compressed_state));

        if pending.entries.len() >= self.batch_size {
            Some(self.flush_node(dest_node))
        } else {
            None
        }
    }

    /// Flush any destination buffers whose oldest entry has exceeded the
    /// timeout. Returns all flushed batches.
    pub fn flush_expired(&mut self) -> Vec<ReadyBatch> {
        let now = Instant::now();
        let timeout = std::time::Duration::from_millis(self.batch_timeout_ms);
        let expired_nodes: Vec<u32> = self
            .buffers
            .iter()
            .filter(|(_, pending)| {
                !pending.entries.is_empty() && now.duration_since(pending.first_insert) >= timeout
            })
            .map(|(&node, _)| node)
            .collect();

        let mut batches = Vec::new();
        for node in expired_nodes {
            batches.push(self.flush_node(node));
        }
        batches
    }

    /// Flush all non-empty destination buffers regardless of size/time.
    /// Useful during shutdown or termination.
    pub fn flush_all(&mut self) -> Vec<ReadyBatch> {
        let nodes: Vec<u32> = self
            .buffers
            .keys()
            .copied()
            .filter(|n| {
                self.buffers
                    .get(n)
                    .map(|p| !p.entries.is_empty())
                    .unwrap_or(false)
            })
            .collect();

        let mut batches = Vec::new();
        for node in nodes {
            batches.push(self.flush_node(node));
        }
        batches
    }

    /// Number of entries currently buffered across all destinations.
    pub fn pending_count(&self) -> usize {
        self.buffers.values().map(|p| p.entries.len()).sum()
    }

    /// Flush a single destination's buffer and return it as a ReadyBatch.
    fn flush_node(&mut self, dest_node: u32) -> ReadyBatch {
        let pending = self
            .buffers
            .get_mut(&dest_node)
            .expect("flush_node called for unknown dest");
        let entries = std::mem::take(&mut pending.entries);
        pending.first_insert = Instant::now();
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);
        ReadyBatch {
            dest_node,
            batch_id,
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_on_batch_size() {
        let mut acc = BatchAccumulator::new(3, 1000);
        assert!(acc.push(1, 100, vec![1]).is_none());
        assert!(acc.push(1, 200, vec![2]).is_none());
        let batch = acc.push(1, 300, vec![3]);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.dest_node, 1);
        assert_eq!(batch.entries.len(), 3);
        assert_eq!(acc.pending_count(), 0);
    }

    #[test]
    fn separate_destinations() {
        let mut acc = BatchAccumulator::new(2, 1000);
        assert!(acc.push(1, 100, vec![1]).is_none());
        assert!(acc.push(2, 200, vec![2]).is_none());
        // Node 1 still has 1 entry, node 2 has 1 entry.
        assert_eq!(acc.pending_count(), 2);

        // Fill node 1.
        let batch = acc.push(1, 101, vec![3]);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().dest_node, 1);
        assert_eq!(acc.pending_count(), 1); // only node 2's entry remains
    }

    #[test]
    fn flush_all_returns_everything() {
        let mut acc = BatchAccumulator::new(100, 1000);
        acc.push(1, 10, vec![1]);
        acc.push(2, 20, vec![2]);
        acc.push(1, 11, vec![3]);
        assert_eq!(acc.pending_count(), 3);

        let batches = acc.flush_all();
        assert_eq!(batches.len(), 2);
        assert_eq!(acc.pending_count(), 0);

        let total_entries: usize = batches.iter().map(|b| b.entries.len()).sum();
        assert_eq!(total_entries, 3);
    }

    #[test]
    fn batch_ids_are_unique() {
        let mut acc = BatchAccumulator::new(1, 1000);
        let b1 = acc.push(1, 10, vec![1]).unwrap();
        let b2 = acc.push(1, 20, vec![2]).unwrap();
        assert_ne!(b1.batch_id, b2.batch_id);
    }
}
