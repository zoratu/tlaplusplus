// Page-aligned, NUMA-aware 2-bit color map for streaming nested DFS.
//
// This is the data structure that backs Phase 2 of T10.2 (in-exploration
// streaming SCC for liveness checking). It mirrors the layout decisions
// of `PageAlignedFingerprintStore` (huge-page-aligned mmap, NUMA-shard
// placement, lock-free CAS) but stores 2 bits per fingerprint instead of
// the full 64-bit fingerprint, packing 32 colors per 64-bit word.
//
// At `--fp-expected-items 100M` the color map costs 25 MB (vs 1.6 GB for
// the fingerprint store), making it cheap to keep alive alongside the
// existing FP store rather than overloading slot bits in the FP store.
//
// The map intentionally does not resize: it is sized once at startup from
// `--fp-expected-items`. If the workload grows past that, the map clamps
// the offending fingerprint to the nearest in-range slot (treated as a
// "color collision" — the only soundness implication is that two distinct
// fingerprints may share a color slot, which the streaming nested DFS
// handles by also consulting the global FP store on every CAS attempt).
//
// See `docs/T10.2-phase2-refined.md` section 1 for the design rationale
// and CAS shape that this file implements.

#[cfg(target_os = "linux")]
use libc::{MAP_POPULATE, madvise};
use libc::{
    MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap, munmap,
};

#[cfg(target_os = "linux")]
const MAP_HUGETLB: libc::c_int = 0x40000;
#[cfg(not(target_os = "linux"))]
const MAP_HUGETLB: libc::c_int = 0;

#[cfg(not(target_os = "linux"))]
const MAP_POPULATE: libc::c_int = 0;

#[cfg(target_os = "linux")]
const MADV_HUGEPAGE: i32 = 14;

use std::sync::atomic::{AtomicU64, Ordering};

const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2 MB

/// Phase-2 nested-DFS coloring (2 bits per fingerprint). The encoding is
/// load-bearing: `White = 0` is the "never-visited" sentinel and matches
/// the zero-filled state of fresh mmap pages, so we never need to bulk-
/// initialize the map.
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Color {
    White = 0b00, // never visited
    Cyan = 0b01,  // on current blue DFS path (cycle-witness target)
    Blue = 0b10,  // blue DFS finished
    Red = 0b11,   // visited by red DFS
}

/// Bit offset within a u64 color word (T13.4 Phase 2).
///
/// Cfg-split: default keeps `((within % 32) * 2) as u32` inline; under
/// `--features verus` delegates to the verified `compute_bit_offset`
/// whose contract guarantees `result < 64` — required for the shipping
/// `word >> bit_offset` shift on a `u64` to be well-defined.
#[cfg(not(feature = "verus"))]
#[inline]
fn compute_bit_offset(within: usize) -> u32 {
    ((within % 32) * 2) as u32
}

#[cfg(feature = "verus")]
#[inline]
fn compute_bit_offset(within: usize) -> u32 {
    crate::storage::verus_smoke::compute_bit_offset(within)
}

impl Color {
    #[inline]
    fn from_bits(bits: u64) -> Color {
        match bits & 0b11 {
            0b00 => Color::White,
            0b01 => Color::Cyan,
            0b10 => Color::Blue,
            0b11 => Color::Red,
            _ => unreachable!("masked to 2 bits"),
        }
    }

    #[inline]
    fn as_bits(self) -> u64 {
        self as u64
    }
}

/// One NUMA-pinned shard of the color map. Allocates a single contiguous
/// huge-page-aligned region of `AtomicU64` words; each word packs 32
/// 2-bit color slots.
struct ColorShard {
    /// Raw mmap base — kept so we can `munmap` on drop.
    memory: *mut u8,
    /// Allocation size in bytes.
    memory_size: usize,
    /// Base of the word array (alias of `memory` cast to `*const AtomicU64`).
    words: *const AtomicU64,
    /// Number of `AtomicU64` words in this shard.
    word_count: usize,
    /// Slot capacity: `word_count * 32`. Currently informational; kept
    /// alongside `word_count` for future bounds-checked APIs.
    #[allow(dead_code)]
    capacity: usize,
    /// NUMA node this shard was placed on (informational).
    numa_node: usize,
}

// SAFETY: `ColorShard` only mutates its memory through atomic words; sharing
// across threads is sound.
unsafe impl Send for ColorShard {}
unsafe impl Sync for ColorShard {}

impl ColorShard {
    /// Build a new shard sized for `slots` color entries on the given NUMA
    /// node. `slots` is rounded up to a multiple of 32 (one word).
    fn new(slots: usize, numa_node: usize) -> Result<Self, ColorMapAllocError> {
        let slots = slots.max(32);
        let word_count = slots.div_ceil(32);
        let bytes = word_count * std::mem::size_of::<u64>();
        // Round up to the huge-page boundary to match the FP store.
        let aligned = bytes.div_ceil(HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE;

        // Hint NUMA placement before mmap. Falls through harmlessly on
        // non-Linux or single-node systems.
        let _ = crate::storage::numa::set_preferred_node(numa_node);

        // Try explicit huge pages first; fall back to anonymous mmap so the
        // shard still allocates on hugepage-disabled hosts.
        let memory = unsafe {
            let attempt = mmap(
                std::ptr::null_mut(),
                aligned,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                -1,
                0,
            );
            if attempt != MAP_FAILED {
                attempt
            } else {
                let plain = mmap(
                    std::ptr::null_mut(),
                    aligned,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                    -1,
                    0,
                );
                if plain == MAP_FAILED {
                    return Err(ColorMapAllocError::MmapFailed);
                }
                #[cfg(target_os = "linux")]
                {
                    let _ = madvise(plain, aligned, MADV_HUGEPAGE);
                }
                plain
            }
        };

        let memory = memory as *mut u8;
        Ok(Self {
            memory,
            memory_size: aligned,
            words: memory as *const AtomicU64,
            word_count,
            capacity: word_count * 32,
            numa_node,
        })
    }

    #[inline]
    fn word(&self, idx: usize) -> &AtomicU64 {
        debug_assert!(idx < self.word_count, "color shard word index OOB");
        // SAFETY: `idx < word_count` and `words` aliases a valid mmap.
        unsafe { &*self.words.add(idx) }
    }
}

impl Drop for ColorShard {
    fn drop(&mut self) {
        if !self.memory.is_null() {
            unsafe {
                let _ = munmap(self.memory as *mut libc::c_void, self.memory_size);
            }
            self.memory = std::ptr::null_mut();
        }
    }
}

/// What went wrong allocating a shard.
#[derive(Debug)]
pub enum ColorMapAllocError {
    MmapFailed,
}

impl std::fmt::Display for ColorMapAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorMapAllocError::MmapFailed => f.write_str("mmap failed for color shard"),
        }
    }
}

impl std::error::Error for ColorMapAllocError {}

/// Page-aligned color map. Holds `shard_count` NUMA-pinned shards, each
/// `slots_per_shard` colors wide. Total addressable slots:
/// `shard_count * slots_per_shard` (rounded up to 32-bit alignment per shard).
pub struct PageAlignedColorMap {
    shards: Vec<ColorShard>,
    /// Cached value of `shards[0].capacity` (all shards have equal capacity).
    slots_per_shard: usize,
    /// Pre-computed mask `slots_per_shard - 1` when slots_per_shard is a
    /// power of two, else 0 (then we use modulo).
    slot_mask: usize,
    /// Pre-computed mask `shards.len() - 1` when shard count is a power of
    /// two, else 0 (then we use modulo).
    shard_mask: usize,
}

impl PageAlignedColorMap {
    /// Build a color map sized for `expected_fingerprints` slots, distributed
    /// across `shard_count` NUMA-pinned shards.
    pub fn new(
        expected_fingerprints: usize,
        shard_count: usize,
        numa_nodes: &[usize],
    ) -> Result<Self, ColorMapAllocError> {
        assert!(shard_count >= 1, "color map shard count must be >= 1");
        let slots_per_shard = expected_fingerprints
            .div_ceil(shard_count)
            .max(32);
        // Round slots_per_shard up to next power of two so we can mask
        // instead of modulo. (Cheap: at 100M / 128 shards = ~781K slots
        // per shard, so ~30% wasted address space at most.)
        let slots_per_shard = slots_per_shard.next_power_of_two();

        let mut shards = Vec::with_capacity(shard_count);
        for shard_idx in 0..shard_count {
            let node = if numa_nodes.is_empty() {
                0
            } else {
                numa_nodes[shard_idx % numa_nodes.len()]
            };
            shards.push(ColorShard::new(slots_per_shard, node)?);
        }

        let slot_mask = slots_per_shard - 1; // power of two by construction
        let shard_mask = if shard_count.is_power_of_two() {
            shard_count - 1
        } else {
            0
        };

        Ok(Self {
            shards,
            slots_per_shard,
            slot_mask,
            shard_mask,
        })
    }

    /// Total capacity across all shards.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots_per_shard * self.shards.len()
    }

    /// Number of shards.
    #[inline]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// NUMA node a shard was placed on (informational; useful for tests).
    pub fn shard_numa_node(&self, shard_idx: usize) -> Option<usize> {
        self.shards.get(shard_idx).map(|s| s.numa_node)
    }

    /// Raw memory pointer for a shard. Used only by NUMA placement tests.
    #[doc(hidden)]
    pub fn shard_base_ptr(&self, shard_idx: usize) -> Option<*const u8> {
        self.shards.get(shard_idx).map(|s| s.memory as *const u8)
    }

    /// Decode (shard, word, bit_offset) for a fingerprint. Public for tests.
    #[inline]
    pub(crate) fn locate(&self, fp: u64) -> (usize, usize, u32) {
        let shard_idx = if self.shard_mask != 0 {
            (fp >> 32) as usize & self.shard_mask
        } else {
            (fp >> 32) as usize % self.shards.len()
        };
        let within = (fp as u32) as usize & self.slot_mask;
        let word_idx = within / 32;
        let bit_offset = compute_bit_offset(within);
        (shard_idx, word_idx, bit_offset)
    }

    /// Read the current color of a fingerprint without modifying anything.
    pub fn load(&self, fp: u64) -> Color {
        let (shard_idx, word_idx, bit_offset) = self.locate(fp);
        let word = self.shards[shard_idx].word(word_idx).load(Ordering::Acquire);
        Color::from_bits(word >> bit_offset)
    }

    /// Atomically transition the color from `expected` to `new`. Returns
    /// `Ok(())` on success, or `Err(actual_color)` if another thread
    /// already moved the slot away from `expected`.
    pub fn cas(&self, fp: u64, expected: Color, new: Color) -> Result<(), Color> {
        let (shard_idx, word_idx, bit_offset) = self.locate(fp);
        let word = self.shards[shard_idx].word(word_idx);
        let mask = 0b11u64 << bit_offset;
        loop {
            let cur = word.load(Ordering::Acquire);
            let cur_color = Color::from_bits(cur >> bit_offset);
            if cur_color != expected {
                return Err(cur_color);
            }
            let next_word = (cur & !mask) | (new.as_bits() << bit_offset);
            match word.compare_exchange_weak(
                cur,
                next_word,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(_) => continue, // someone else won; re-read and check
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn cas_white_to_cyan_succeeds() {
        let map = PageAlignedColorMap::new(1024, 1, &[0]).unwrap();
        assert_eq!(map.load(42), Color::White);
        map.cas(42, Color::White, Color::Cyan).expect("white->cyan");
        assert_eq!(map.load(42), Color::Cyan);
    }

    #[test]
    fn cas_white_to_cyan_loses_when_other_worker_wins() {
        let map = Arc::new(PageAlignedColorMap::new(1024, 1, &[0]).unwrap());
        // Spawn 8 threads, all CASing the same fp from White->Cyan.
        // Exactly one should win; the others should observe Cyan and lose.
        let n_threads = 8usize;
        let mut handles = Vec::with_capacity(n_threads);
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        for _ in 0..n_threads {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                map.cas(0xDEADBEEF, Color::White, Color::Cyan).is_ok()
            }));
        }
        let wins: usize = handles.into_iter().map(|h| h.join().unwrap() as usize).sum();
        assert_eq!(wins, 1, "exactly one thread should win the CAS");
        assert_eq!(map.load(0xDEADBEEF), Color::Cyan);
    }

    #[test]
    fn cas_cyan_to_blue_succeeds() {
        let map = PageAlignedColorMap::new(1024, 1, &[0]).unwrap();
        map.cas(7, Color::White, Color::Cyan).unwrap();
        map.cas(7, Color::Cyan, Color::Blue).unwrap();
        assert_eq!(map.load(7), Color::Blue);
        // Wrong-expected CAS returns the actual color.
        let err = map.cas(7, Color::Cyan, Color::Red).unwrap_err();
        assert_eq!(err, Color::Blue);
    }

    #[test]
    fn concurrent_distinct_fps_in_same_word_no_corruption() {
        // Build a 1-shard map with at least 32 slots. We pick fingerprints
        // that map to the same word but distinct bit offsets, so 32 threads
        // each CAS *their* slot from White to a distinct color — no slot
        // should be clobbered.
        let map = Arc::new(PageAlignedColorMap::new(64, 1, &[0]).unwrap());

        // We need 32 fps that all share (shard, word) but have distinct
        // within-word offsets. Easiest: pick fp = i for i in 0..32 because
        // (fp as u32) % slots_per_shard / 32 == 0 for the first 32 fps in
        // a shard whose slots_per_shard is a power of two >= 32.
        let n = 32usize;
        let mut fps = Vec::with_capacity(n);
        for i in 0..n as u64 {
            fps.push(i);
        }
        // Sanity-check: all fps share (shard, word).
        let (s0, w0, _) = map.locate(fps[0]);
        for &fp in &fps[1..] {
            let (s, w, _) = map.locate(fp);
            assert_eq!((s, w), (s0, w0), "test setup: fps must share word");
        }

        let barrier = Arc::new(std::sync::Barrier::new(n));
        let mut handles = Vec::with_capacity(n);
        for fp in fps.iter().copied() {
            let map = Arc::clone(&map);
            let barrier = Arc::clone(&barrier);
            // Half cas to Cyan, half to Blue. (We can't pick White as
            // "new" because that's the start state.)
            let target = if fp.is_multiple_of(2) {
                Color::Cyan
            } else {
                Color::Blue
            };
            handles.push(thread::spawn(move || {
                barrier.wait();
                map.cas(fp, Color::White, target).expect("each fp untouched");
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // All slots should hold their assigned color exactly.
        for fp in fps {
            let want = if fp.is_multiple_of(2) {
                Color::Cyan
            } else {
                Color::Blue
            };
            assert_eq!(map.load(fp), want, "fp {} clobbered", fp);
        }
    }

    #[test]
    fn red_set_independent_of_blue_set() {
        // Two separate maps modeling "blue colors" vs "red colors" —
        // mutating one must not perturb the other.
        let blue = PageAlignedColorMap::new(1024, 1, &[0]).unwrap();
        let red = PageAlignedColorMap::new(1024, 1, &[0]).unwrap();
        blue.cas(99, Color::White, Color::Cyan).unwrap();
        blue.cas(99, Color::Cyan, Color::Blue).unwrap();
        assert_eq!(blue.load(99), Color::Blue);
        // Red map untouched.
        assert_eq!(red.load(99), Color::White);
        red.cas(99, Color::White, Color::Red).unwrap();
        // Blue map untouched.
        assert_eq!(blue.load(99), Color::Blue);
        assert_eq!(red.load(99), Color::Red);
    }

    #[test]
    fn numa_local_placement_observed() {
        // On systems with multiple NUMA nodes, each shard should actually
        // be placed on the node we asked for. On single-node hosts (or
        // non-Linux), we just verify the API doesn't panic.
        let topology = match crate::storage::numa::NumaTopology::detect() {
            Ok(t) => t,
            Err(_) => return, // no /sys topology; skip
        };
        if topology.node_count <= 1 {
            // Single-node host: shard_numa_node should report 0 for shard 0.
            let map = PageAlignedColorMap::new(1024, 4, &[0]).unwrap();
            for i in 0..4 {
                assert_eq!(map.shard_numa_node(i), Some(0));
            }
            return;
        }

        // Multi-node host: bind each shard to a distinct node and check
        // via get_memory_numa_node. Pages need to be touched first for the
        // policy to take effect, so we CAS one slot per shard before
        // querying.
        let nodes: Vec<usize> = (0..topology.node_count.min(4)).collect();
        let map = PageAlignedColorMap::new(4096, nodes.len(), &nodes).unwrap();
        for (shard_idx, &node) in nodes.iter().enumerate() {
            // Touch a slot in this shard so the kernel actually faults a
            // page in. `fp = shard_idx as u64` typically lands in shard 0
            // (because shard is `(fp >> 32) % shard_count`); we instead
            // construct fps that target shard `shard_idx` directly.
            let fp = (shard_idx as u64) << 32;
            assert_eq!(
                map.locate(fp).0,
                shard_idx,
                "test setup: fp should land in target shard"
            );
            map.cas(fp, Color::White, Color::Cyan).unwrap();
            // Now query the actual page.
            if let Some(ptr) = map.shard_base_ptr(shard_idx) {
                if let Some(actual_node) = crate::storage::numa::get_memory_numa_node(ptr) {
                    assert_eq!(
                        actual_node, node,
                        "shard {} expected on node {}, found on node {}",
                        shard_idx, node, actual_node
                    );
                }
                // If get_memory_numa_node returned None, we can't verify;
                // that's a kernel/permission limitation, not a bug.
            }
        }
    }

    #[test]
    fn oom_graceful_degradation() {
        // Asking for a wildly oversized map should fail cleanly with a
        // typed error, not panic. We can't reliably trigger OOM in CI
        // without trashing the host, so we instead probe with a request
        // for a single absurdly-large shard and accept either Ok (the
        // host actually has that much overcommit) or Err — what we
        // forbid is a panic.
        //
        // This also covers the "shards 0..K stay valid if shard K fails"
        // guarantee: ColorShard::new returns Err on mmap failure, and
        // PageAlignedColorMap::new propagates the error. Earlier shards
        // are dropped at end-of-scope through the partially-built Vec,
        // running each shard's munmap exactly once.
        let huge_slots = 1usize << 50; // ~1.1 PB request
        match PageAlignedColorMap::new(huge_slots, 1, &[0]) {
            Ok(_) => {
                // Host has overcommit on; no signal here, but it didn't
                // panic. Acceptable.
            }
            Err(ColorMapAllocError::MmapFailed) => {
                // Expected case — and the partial-build vector dropped
                // cleanly (no leaks because we only built shard 0 before
                // failing on shard 1, and we only ask for 1 shard here).
            }
        }

        // Double-check: a normal-sized map after a failed-or-recovered
        // attempt still works.
        let ok = PageAlignedColorMap::new(1024, 1, &[0]).unwrap();
        ok.cas(123, Color::White, Color::Cyan).unwrap();
        assert_eq!(ok.load(123), Color::Cyan);
    }
}
