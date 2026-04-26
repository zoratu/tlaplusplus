// In-memory zstd-compressed segment ring (T8).
//
// Sits between the hot lock-free work-stealing deques and the disk-backed
// overflow queue. When the hot queue exceeds its in-memory item budget, the
// spill coordinator compresses incoming batches with zstd and stores the
// resulting opaque byte buffer in a FIFO ring. When the hot queue runs dry,
// the loader pops the oldest compressed segment, decompresses it, and pushes
// the items back to the hot queue.
//
// Why this layer exists:
// - Serialized TLA+ states compress 3-10x with zstd (lots of repeated keys
//   and small integer/bool fields). A 256MB compressed-segment ring can hold
//   ~1-2GB-equivalent of pending state without touching disk.
// - Compression keeps a much larger working set in memory than the raw hot
//   queue would, so disk spilling becomes a true overflow path rather than
//   a hot-path bottleneck for medium-size models.
// - Transparent: the hot queue, fingerprint store, and worker dispatch loop
//   are unchanged. Items move through this layer in their original form.
//
// The hot queue itself remains uncompressed and lock-free. Only items that
// would otherwise have spilled to disk get compressed in memory first.

use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Default compression level. Level 1 = fastest (~500MB/s compress on Graviton)
/// while still achieving ~3x ratio on serialized TLA+ states. Level 3 gives
/// slightly better ratios but doubles the compress time; level 1 is the right
/// choice for a hot path that needs to be net-positive on wall-clock.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 1;

/// Default in-memory budget for compressed segments. 256MB of compressed
/// data typically holds 1-2GB-equivalent of states, deferring disk I/O for
/// the vast majority of medium-size model checks while bounding overall
/// resident memory.
pub const DEFAULT_MAX_COMPRESSED_BYTES: usize = 256 * 1024 * 1024;

/// One compressed batch of items, opaque to callers.
pub struct CompressedSegment {
    /// zstd-compressed bincode-serialized Vec<T>.
    bytes: Vec<u8>,
    /// Number of items inside (so callers can adjust pending counters).
    item_count: usize,
    /// Original (uncompressed) serialized size, for ratio reporting.
    original_bytes: usize,
}

impl CompressedSegment {
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    pub fn compressed_size(&self) -> usize {
        self.bytes.len()
    }

    pub fn original_size(&self) -> usize {
        self.original_bytes
    }
}

#[derive(Default)]
pub struct CompressionStats {
    pub segments_compressed: AtomicU64,
    pub segments_decompressed: AtomicU64,
    pub items_compressed: AtomicU64,
    pub items_decompressed: AtomicU64,
    pub bytes_compressed: AtomicU64,
    pub bytes_uncompressed: AtomicU64,
    pub compress_time_us: AtomicU64,
    pub decompress_time_us: AtomicU64,
    /// Number of times a push was rejected because the ring was full.
    pub push_rejected_full: AtomicU64,
}

#[derive(Clone, Copy, Debug)]
pub struct CompressionStatsSnapshot {
    pub segments_compressed: u64,
    pub segments_decompressed: u64,
    pub items_compressed: u64,
    pub items_decompressed: u64,
    pub bytes_compressed: u64,
    pub bytes_uncompressed: u64,
    pub compress_time_us: u64,
    pub decompress_time_us: u64,
    pub push_rejected_full: u64,
    /// Live (currently resident) compressed bytes.
    pub current_compressed_bytes: u64,
    pub current_segments: u64,
    pub current_items: u64,
}

impl CompressionStats {
    pub fn snapshot(
        &self,
        current_compressed_bytes: usize,
        current_segments: usize,
        current_items: usize,
    ) -> CompressionStatsSnapshot {
        CompressionStatsSnapshot {
            segments_compressed: self.segments_compressed.load(Ordering::Relaxed),
            segments_decompressed: self.segments_decompressed.load(Ordering::Relaxed),
            items_compressed: self.items_compressed.load(Ordering::Relaxed),
            items_decompressed: self.items_decompressed.load(Ordering::Relaxed),
            bytes_compressed: self.bytes_compressed.load(Ordering::Relaxed),
            bytes_uncompressed: self.bytes_uncompressed.load(Ordering::Relaxed),
            compress_time_us: self.compress_time_us.load(Ordering::Relaxed),
            decompress_time_us: self.decompress_time_us.load(Ordering::Relaxed),
            push_rejected_full: self.push_rejected_full.load(Ordering::Relaxed),
            current_compressed_bytes: current_compressed_bytes as u64,
            current_segments: current_segments as u64,
            current_items: current_items as u64,
        }
    }
}

impl CompressionStatsSnapshot {
    pub fn ratio(&self) -> f64 {
        if self.bytes_compressed == 0 {
            1.0
        } else {
            self.bytes_uncompressed as f64 / self.bytes_compressed as f64
        }
    }
}

/// FIFO ring of compressed-in-memory segments.
///
/// Push/pop are coarsely synchronized (parking_lot::Mutex over a VecDeque).
/// This is fine because:
///   - Pushes happen from the single spill coordinator thread (no contention).
///   - Pops happen from the single loader thread plus the rare worker fast
///     path during pop_for_worker; the lock holds for ~nanoseconds (just the
///     pop_front), with decompression happening outside the lock.
pub struct CompressedSegmentRing {
    segments: Mutex<VecDeque<Arc<CompressedSegment>>>,
    /// Sum of compressed bytes across all resident segments. Maintained
    /// atomically alongside segments to avoid lock-and-iterate.
    current_compressed_bytes: AtomicUsize,
    /// Sum of items across all resident segments. Used for pending-count
    /// reporting without touching the lock.
    current_items: AtomicUsize,
    /// Hard cap on resident compressed bytes. Pushes that would cross this
    /// threshold are rejected (caller falls back to disk spill).
    max_compressed_bytes: usize,
    /// zstd compression level (1-22). Level 1 is the right default for
    /// hot-path compression.
    compression_level: i32,
    pub stats: CompressionStats,
}

impl CompressedSegmentRing {
    pub fn new(max_compressed_bytes: usize, compression_level: i32) -> Self {
        Self {
            segments: Mutex::new(VecDeque::new()),
            current_compressed_bytes: AtomicUsize::new(0),
            current_items: AtomicUsize::new(0),
            max_compressed_bytes,
            compression_level,
            stats: CompressionStats::default(),
        }
    }

    /// Outstanding items across all resident compressed segments.
    pub fn pending_items(&self) -> usize {
        self.current_items.load(Ordering::Acquire)
    }

    /// Outstanding compressed bytes (the metric the budget caps).
    pub fn pending_bytes(&self) -> usize {
        self.current_compressed_bytes.load(Ordering::Acquire)
    }

    pub fn segment_count(&self) -> usize {
        self.segments.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.current_items.load(Ordering::Acquire) == 0
    }

    /// Snapshot all stats including live ring counters.
    pub fn snapshot_stats(&self) -> CompressionStatsSnapshot {
        let segs = self.segments.lock().len();
        self.stats
            .snapshot(self.pending_bytes(), segs, self.pending_items())
    }

    /// Try to push a batch into the ring. Returns:
    ///   - `Ok(None)` if the batch was accepted (compressed and stored).
    ///   - `Ok(Some(items))` if the ring is full or compression failed;
    ///     caller should spill those items elsewhere (typically the disk
    ///     overflow queue). The original items are returned uncompressed
    ///     so no work is wasted and no items are dropped.
    ///   - `Err(...)` only for catastrophic internal errors that we can't
    ///     attribute to the batch (none today; reserved for future use).
    ///
    /// The "full" check is conservative: if accepting the batch would push
    /// the resident bytes above `max_compressed_bytes`, we reject *without*
    /// actually compressing, so the caller's fallback path is cheap.
    pub fn try_push_batch<T>(&self, items: Vec<T>) -> Result<Option<Vec<T>>>
    where
        T: Serialize,
    {
        if items.is_empty() {
            return Ok(None);
        }

        // Budget check first — cheap, avoids wasted compression on a full ring.
        let current = self.current_compressed_bytes.load(Ordering::Acquire);
        if current >= self.max_compressed_bytes {
            self.stats
                .push_rejected_full
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Some(items));
        }

        let item_count = items.len();
        let start = Instant::now();
        let serialized = match bincode::serialize(&items) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "compressed-segment serialize failed; returning batch to caller: {}",
                    e
                );
                return Ok(Some(items));
            }
        };
        let original_bytes = serialized.len();
        let compressed = match zstd::encode_all(serialized.as_slice(), self.compression_level) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "compressed-segment zstd encode failed; returning batch to caller: {}",
                    e
                );
                return Ok(Some(items));
            }
        };
        let elapsed_us = start.elapsed().as_micros() as u64;

        // Re-check budget after compression — another push may have raced
        // ahead. If we'd cross the cap now, reject and return the original
        // items so the caller spills them to disk.
        let compressed_size = compressed.len();
        let pre_add = self
            .current_compressed_bytes
            .fetch_add(compressed_size, Ordering::AcqRel);
        if pre_add + compressed_size > self.max_compressed_bytes && pre_add > 0 {
            // Roll back the reservation. The first push to an empty ring is
            // always accepted (so we never have a deadlock where a single
            // huge batch is permanently rejected).
            self.current_compressed_bytes
                .fetch_sub(compressed_size, Ordering::AcqRel);
            self.stats
                .push_rejected_full
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Some(items));
        }

        let segment = Arc::new(CompressedSegment {
            bytes: compressed,
            item_count,
            original_bytes,
        });

        {
            let mut guard = self.segments.lock();
            guard.push_back(segment);
        }

        self.current_items.fetch_add(item_count, Ordering::AcqRel);
        self.stats
            .segments_compressed
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .items_compressed
            .fetch_add(item_count as u64, Ordering::Relaxed);
        self.stats
            .bytes_uncompressed
            .fetch_add(original_bytes as u64, Ordering::Relaxed);
        self.stats
            .bytes_compressed
            .fetch_add(compressed_size as u64, Ordering::Relaxed);
        self.stats
            .compress_time_us
            .fetch_add(elapsed_us, Ordering::Relaxed);

        Ok(None)
    }

    /// Pop the oldest (FIFO) segment, returning `None` if the ring is empty.
    /// Decompression is the caller's responsibility (so the lock is held only
    /// for the pop_front).
    pub fn pop_oldest_segment(&self) -> Option<Arc<CompressedSegment>> {
        let segment = {
            let mut guard = self.segments.lock();
            guard.pop_front()?
        };
        let compressed_size = segment.bytes.len();
        let item_count = segment.item_count;
        self.current_compressed_bytes
            .fetch_sub(compressed_size, Ordering::AcqRel);
        self.current_items.fetch_sub(item_count, Ordering::AcqRel);
        Some(segment)
    }

    /// Decompress a segment back to its `Vec<T>` form.
    pub fn decompress_segment<T>(&self, segment: &CompressedSegment) -> Result<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let start = Instant::now();
        let decompressed = zstd::decode_all(segment.bytes.as_slice())
            .map_err(|e| anyhow!("compressed-segment zstd decode failed: {}", e))?;
        let items: Vec<T> = bincode::deserialize(&decompressed)
            .map_err(|e| anyhow!("compressed-segment deserialize failed: {}", e))?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.stats
            .segments_decompressed
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .items_decompressed
            .fetch_add(items.len() as u64, Ordering::Relaxed);
        self.stats
            .decompress_time_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        Ok(items)
    }

    /// Drain every resident segment to a Vec, decompressing as we go.
    /// Used during checkpoint flush to push compressed-in-memory data out
    /// to disk before persisting.
    pub fn drain_all_decompressed<T>(&self) -> Result<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let mut out = Vec::new();
        while let Some(segment) = self.pop_oldest_segment() {
            let items: Vec<T> = self.decompress_segment(&segment)?;
            out.extend(items);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    struct Item {
        a: u64,
        b: String,
        c: Vec<i32>,
    }

    fn sample_batch(n: usize, seed: u64) -> Vec<Item> {
        (0..n)
            .map(|i| Item {
                a: seed + i as u64,
                b: format!("item-{}-{}", seed, i),
                c: (0..16).map(|j| (i as i32) + j).collect(),
            })
            .collect()
    }

    #[test]
    fn round_trip_preserves_items() {
        let ring = CompressedSegmentRing::new(64 * 1024 * 1024, DEFAULT_COMPRESSION_LEVEL);
        let batch = sample_batch(100, 0);
        let original = batch.clone();
        let res = ring.try_push_batch(batch).unwrap();
        assert!(res.is_none(), "small batch must fit");
        assert_eq!(ring.pending_items(), 100);
        assert_eq!(ring.segment_count(), 1);

        let segment = ring.pop_oldest_segment().expect("must pop the segment");
        let restored: Vec<Item> = ring.decompress_segment(&segment).unwrap();
        assert_eq!(restored, original);
        assert_eq!(ring.pending_items(), 0);
        assert_eq!(ring.segment_count(), 0);
        assert_eq!(ring.pending_bytes(), 0);

        let snap = ring.snapshot_stats();
        assert_eq!(snap.segments_compressed, 1);
        assert_eq!(snap.segments_decompressed, 1);
        assert!(snap.bytes_compressed > 0);
        assert!(snap.bytes_uncompressed >= snap.bytes_compressed);
        assert!(snap.ratio() >= 1.0);
    }

    #[test]
    fn fifo_order_across_multiple_segments() {
        let ring = CompressedSegmentRing::new(64 * 1024 * 1024, DEFAULT_COMPRESSION_LEVEL);
        for seed in 0..5 {
            let batch = sample_batch(20, seed * 100);
            assert!(ring.try_push_batch(batch).unwrap().is_none());
        }
        assert_eq!(ring.segment_count(), 5);
        assert_eq!(ring.pending_items(), 100);

        for seed in 0..5 {
            let seg = ring.pop_oldest_segment().expect("segment present");
            let items: Vec<Item> = ring.decompress_segment(&seg).unwrap();
            assert_eq!(items.len(), 20);
            // First item of each batch encodes the seed in `a`.
            assert_eq!(items[0].a, seed * 100);
        }
        assert!(ring.is_empty());
    }

    #[test]
    fn budget_rejects_when_full() {
        // Tiny budget: first batch must succeed (we always accept the first
        // push to an empty ring). Subsequent pushes that would cross the cap
        // return the items back to the caller.
        let ring = CompressedSegmentRing::new(256, DEFAULT_COMPRESSION_LEVEL);
        let batch = sample_batch(50, 0);
        assert!(
            ring.try_push_batch(batch).unwrap().is_none(),
            "first batch always accepted"
        );

        let next = sample_batch(50, 1);
        let returned = ring
            .try_push_batch(next)
            .unwrap()
            .expect("ring should be full");
        assert_eq!(returned.len(), 50);
        assert_eq!(ring.segment_count(), 1);
        assert!(ring.snapshot_stats().push_rejected_full >= 1);
    }

    #[test]
    fn drain_all_returns_items_in_fifo_order() {
        let ring = CompressedSegmentRing::new(64 * 1024 * 1024, DEFAULT_COMPRESSION_LEVEL);
        for seed in 0..3 {
            assert!(
                ring.try_push_batch(sample_batch(10, seed * 10))
                    .unwrap()
                    .is_none()
            );
        }
        let drained: Vec<Item> = ring.drain_all_decompressed().unwrap();
        assert_eq!(drained.len(), 30);
        assert_eq!(drained[0].a, 0);
        assert_eq!(drained[10].a, 10);
        assert_eq!(drained[20].a, 20);
        assert!(ring.is_empty());
    }

    #[test]
    fn empty_batch_is_noop() {
        let ring = CompressedSegmentRing::new(64 * 1024 * 1024, DEFAULT_COMPRESSION_LEVEL);
        let res = ring.try_push_batch::<Item>(Vec::new()).unwrap();
        assert!(res.is_none());
        assert_eq!(ring.segment_count(), 0);
        assert_eq!(ring.snapshot_stats().segments_compressed, 0);
    }
}
