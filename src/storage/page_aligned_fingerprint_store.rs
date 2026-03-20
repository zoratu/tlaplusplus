// Page-aligned, NUMA-aware fingerprint store with huge page support
//
// This implementation achieves 95%+ CPU utilization by eliminating TLB thrashing:
// - Pre-allocated 2MB-aligned memory regions
// - NUMA-aware shard placement
// - Open-addressed hash table for cache efficiency
// - Atomic operations for lock-free concurrency
//
// Performance characteristics:
// - Before (DashMap): 48M page table entries → 99.998% TLB miss rate → 20-32% sys time
// - After (This): 98K page table entries → 99% TLB hit rate → 2-5% sys time
// - Net gain: 15-20% more user CPU

use crate::storage::async_fingerprint_writer::FingerprintPersistMsg;
use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE, madvise, mmap, munmap};
use parking_lot::Mutex;

// Platform-specific constants
#[cfg(target_os = "linux")]
const MAP_HUGETLB: libc::c_int = 0x40000;
#[cfg(target_os = "linux")]
const MAP_POPULATE: libc::c_int = 0x8000;

// Fallback for non-Linux platforms
#[cfg(not(target_os = "linux"))]
const MAP_HUGETLB: libc::c_int = 0;
#[cfg(not(target_os = "linux"))]
const MAP_POPULATE: libc::c_int = 0;
use serde::{Deserialize, Serialize};
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};

const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB
const MADV_HUGEPAGE: i32 = 14; // From linux/mman.h
const MADV_RANDOM: i32 = 1; // From linux/mman.h
const MADV_DONTNEED: i32 = 4; // From linux/mman.h

/// MADV_PAGEOUT - Linux 5.4+, asks kernel to page out (swap/writeback) the pages
#[cfg(target_os = "linux")]
#[allow(dead_code)]
const MADV_PAGEOUT: i32 = 21;
#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
const MADV_PAGEOUT: i32 = -1; // Not available

/// Load factor threshold for triggering resize (75% - earlier to reduce contention)
const RESIZE_LOAD_THRESHOLD: f64 = 0.75;
/// How often to check load factor (every N inserts)
const RESIZE_CHECK_INTERVAL: u64 = 1_000;
/// Maximum spin iterations before yielding during resize wait (reserved for future backoff)
#[allow(dead_code)]
const MAX_SPIN_BEFORE_YIELD: u32 = 64;
/// Maximum yields before sleeping (reserved for future backoff)
#[allow(dead_code)]
const MAX_YIELDS_BEFORE_SLEEP: u32 = 16;
/// Maximum retry iterations in fingerprint operations before warning
const MAX_RETRIES_BEFORE_WARN: u32 = 10_000;
/// Maximum retry iterations before panic (something is very wrong)
const MAX_RETRIES_BEFORE_PANIC: u32 = 1_000_000;

#[derive(Clone, Debug)]
pub struct FingerprintStoreConfig {
    pub shard_count: usize,
    pub expected_items: usize,
    pub shard_size_mb: usize, // Memory per shard in MB
}

impl Default for FingerprintStoreConfig {
    fn default() -> Self {
        Self {
            shard_count: 128, // Power of 2 for fast modulo
            expected_items: 100_000_000,
            shard_size_mb: 1024, // 1GB per shard = 128GB total
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FingerprintStats {
    pub checks: u64,
    pub hits: u64,
    pub inserts: u64,
    pub batch_calls: u64,
    pub batch_items: u64,
    pub collisions: u64, // Linear probing collisions
}

#[derive(Default)]
struct FingerprintStatsAtomic {
    checks: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    batch_calls: AtomicU64,
    batch_items: AtomicU64,
    collisions: AtomicU64,
}

/// Hash table entry - cache-line aligned for optimal access
#[repr(C, align(16))]
struct HashTableEntry {
    /// Fingerprint value (0 = empty slot)
    fp: AtomicU64,
    /// Entry state: 0=empty, 1=occupied
    state: AtomicU8,
    /// Padding to avoid false sharing
    _pad: [u8; 7],
}

impl HashTableEntry {
    #[allow(dead_code)]
    const fn new() -> Self {
        Self {
            fp: AtomicU64::new(0),
            state: AtomicU8::new(0),
            _pad: [0; 7],
        }
    }
}

/// Single shard of the fingerprint store with dynamic resize support
struct FingerprintShard {
    /// Pre-allocated 2MB-aligned memory region (atomic for resize)
    memory: AtomicPtr<u8>,
    /// Size of allocated memory in bytes
    memory_size: AtomicUsize,
    /// Hash table entries (pointer into memory, atomic for resize)
    table: AtomicPtr<HashTableEntry>,
    /// Capacity (number of entries, atomic for resize)
    capacity: AtomicUsize,
    /// Current count of fingerprints
    count: AtomicU64,
    /// NUMA node this shard is allocated on
    numa_node: usize,
    /// Seqlock for resize coordination (odd = resizing, even = stable)
    seq: AtomicU64,
    /// Counter for periodic load factor checks
    check_counter: AtomicU64,
    /// Mutex for resize coordination (only one thread can resize)
    resize_lock: Mutex<()>,
    /// Old table pointer during resize (for RCU-style reads)
    /// Readers can check this table during resize to find existing fingerprints
    old_table: AtomicPtr<HashTableEntry>,
    /// Old table capacity during resize
    old_capacity: AtomicUsize,
    /// New table pointer during resize (for lock-free inserts)
    /// Workers can insert here during resize instead of waiting
    new_table: AtomicPtr<HashTableEntry>,
    /// New table capacity during resize
    new_capacity: AtomicUsize,
    /// Old memory regions waiting to be freed (after readers drain)
    #[allow(clippy::type_complexity)]
    old_memory: Mutex<Vec<(*mut u8, usize)>>,
    /// Configured backing file path for file-backed mmap attempts
    /// (None = always anonymous mmap).
    backing_path: Option<PathBuf>,
    /// Whether the current mapping in `memory` is file-backed.
    mapping_is_file_backed: AtomicBool,
    /// File descriptor for backing file (kept open for madvise operations)
    backing_fd: Mutex<Option<RawFd>>,
    /// Shard identifier (for file naming during resize)
    shard_id: usize,
    /// Cursor for incremental rehash: entries [0..rehash_cursor) have been moved
    /// to the new table. Workers advance this atomically in batches.
    rehash_cursor: AtomicUsize,
    /// Adaptive batch size for incremental rehash, determined at resize start
    /// based on table occupancy. Larger batches when table is sparse (less CAS
    /// contention), smaller when dense.
    rehash_batch_size: AtomicUsize,
    /// Number of rehash batches processed by workers (not the resize thread)
    rehash_worker_batches: AtomicUsize,
    /// Number of rehash batches processed by the resize thread
    rehash_resize_batches: AtomicUsize,
}

unsafe impl Send for FingerprintShard {}
unsafe impl Sync for FingerprintShard {}

impl FingerprintShard {
    /// Create a new shard with huge page allocation or file-backed mmap
    ///
    /// When `backing_dir` is Some, uses file-backed mmap so the kernel can page
    /// cold fingerprints to disk under memory pressure. When None, uses anonymous
    /// mmap with huge pages (original behavior).
    fn new(
        size_mb: usize,
        numa_node: usize,
        backing_dir: Option<&Path>,
        shard_id: usize,
    ) -> Result<Self> {
        let memory_size = size_mb * 1024 * 1024;

        // Set NUMA preference for this allocation
        if let Err(e) = crate::storage::numa::set_preferred_node(numa_node) {
            eprintln!("Warning: Failed to set NUMA node {}: {}", numa_node, e);
        }

        // Try file-backed mmap if backing_dir is provided
        let (memory, backing_path, mapping_is_file_backed, backing_fd) = if let Some(dir) =
            backing_dir
        {
            let file_path = dir.join(format!("shard-{}.fp", shard_id));
            match unsafe { allocate_file_backed(memory_size, &file_path) } {
                Ok((ptr, fd)) => (ptr, Some(file_path), true, Some(fd)),
                Err(e) => {
                    eprintln!(
                        "Warning: File-backed mmap failed for shard {} ({}), falling back to anonymous mmap",
                        shard_id, e
                    );
                    let ptr = unsafe { allocate_huge_pages(memory_size) };
                    (ptr, Some(file_path), false, None)
                }
            }
        } else {
            let ptr = unsafe { allocate_huge_pages(memory_size) };
            (ptr, None, false, None)
        };

        // Calculate capacity (leave 10% headroom for open addressing)
        let entry_size = std::mem::size_of::<HashTableEntry>();
        let capacity = (memory_size / entry_size) * 9 / 10;

        // Initialize hash table in allocated memory
        let table = memory as *mut HashTableEntry;
        unsafe {
            // Zero-initialize all entries
            std::ptr::write_bytes(table, 0, capacity);
        }

        Ok(Self {
            memory: AtomicPtr::new(memory),
            memory_size: AtomicUsize::new(memory_size),
            table: AtomicPtr::new(table),
            capacity: AtomicUsize::new(capacity),
            count: AtomicU64::new(0),
            numa_node,
            seq: AtomicU64::new(0),
            check_counter: AtomicU64::new(0),
            resize_lock: Mutex::new(()),
            old_table: AtomicPtr::new(std::ptr::null_mut()),
            old_capacity: AtomicUsize::new(0),
            new_table: AtomicPtr::new(std::ptr::null_mut()),
            new_capacity: AtomicUsize::new(0),
            old_memory: Mutex::new(Vec::new()),
            backing_path,
            mapping_is_file_backed: AtomicBool::new(mapping_is_file_backed),
            backing_fd: Mutex::new(backing_fd),
            shard_id,
            rehash_cursor: AtomicUsize::new(0),
            rehash_batch_size: AtomicUsize::new(4096),
            rehash_worker_batches: AtomicUsize::new(0),
            rehash_resize_batches: AtomicUsize::new(0),
        })
    }

    /// Compute adaptive rehash batch size based on table occupancy.
    /// Sparse tables (<50% full) use larger batches (16384) since there are
    /// fewer entries to move and less CAS contention. Dense tables (>75% full)
    /// use smaller batches (1024) to reduce contention and yield more CPU to workers.
    fn compute_rehash_batch_size(&self) -> usize {
        let load = self.load_factor();
        if load < 0.50 {
            16384
        } else if load > 0.75 {
            1024
        } else {
            4096
        }
    }

    /// Participate in incremental rehash: move a batch of entries from
    /// old table to new table. Returns true if there's more work to do.
    /// When `is_resize_thread` is true, the batch is counted as resize-thread work.
    fn rehash_batch_counted(&self, is_resize_thread: bool) -> bool {
        let old_cap = self.old_capacity.load(Ordering::Acquire);
        if old_cap == 0 {
            return false;
        }

        let old_table_ptr = self.old_table.load(Ordering::Acquire);
        let new_table_ptr = self.new_table.load(Ordering::Acquire);
        if old_table_ptr.is_null() || new_table_ptr.is_null() {
            return false;
        }

        let new_cap = self.new_capacity.load(Ordering::Acquire);
        if new_cap == 0 {
            return false;
        }

        let batch_size = self.rehash_batch_size.load(Ordering::Relaxed);

        // Claim a batch of entries to rehash
        let start = self
            .rehash_cursor
            .fetch_add(batch_size, Ordering::AcqRel);
        if start >= old_cap {
            return false; // All batches claimed
        }
        let end = (start + batch_size).min(old_cap);

        // Track which thread type processed this batch
        if is_resize_thread {
            self.rehash_resize_batches.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rehash_worker_batches.fetch_add(1, Ordering::Relaxed);
        }

        let old_slice = unsafe { std::slice::from_raw_parts(old_table_ptr, old_cap) };
        let new_slice = unsafe { std::slice::from_raw_parts(new_table_ptr, new_cap) };

        for i in start..end {
            let fp = old_slice[i].fp.load(Ordering::Acquire);
            if fp != 0 {
                // Insert into new table using CAS
                let mut index = (fp as usize) % new_cap;
                loop {
                    let new_entry = &new_slice[index];
                    let stored = new_entry.fp.load(Ordering::Acquire);
                    if stored == fp {
                        break; // Already present
                    }
                    if stored == 0 {
                        match new_entry.fp.compare_exchange(
                            0,
                            fp,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => {
                                new_entry.state.store(1, Ordering::Release);
                                break;
                            }
                            Err(actual) if actual == fp => break,
                            Err(_) => {}
                        }
                    }
                    index = (index + 1) % new_cap;
                }
            }
        }

        end < old_cap // true = more batches remain
    }

    /// Participate in incremental rehash (called by workers).
    fn rehash_batch(&self) -> bool {
        self.rehash_batch_counted(false)
    }

    /// Check if incremental rehash is complete
    fn is_rehash_complete(&self) -> bool {
        let old_cap = self.old_capacity.load(Ordering::Acquire);
        old_cap == 0 || self.rehash_cursor.load(Ordering::Acquire) >= old_cap
    }

    /// Finalize resize after all entries have been rehashed
    fn finalize_resize(&self) {
        let old_memory = self.memory.load(Ordering::Acquire);
        let old_memory_size = self.memory_size.load(Ordering::Acquire);
        let new_table = self.new_table.load(Ordering::Acquire);
        let new_cap = self.new_capacity.load(Ordering::Acquire);
        let new_memory = new_table as *mut u8;
        let entry_size = std::mem::size_of::<HashTableEntry>();
        let new_memory_size = new_cap * entry_size;

        // Swap in new table
        self.table.store(new_table, Ordering::Release);
        self.capacity.store(new_cap, Ordering::Release);
        self.memory.store(new_memory, Ordering::Release);
        self.memory_size.store(new_memory_size, Ordering::Release);

        // Mark resize complete (even seq number)
        self.seq.fetch_add(1, Ordering::AcqRel);

        // Clear old/new table pointers
        self.old_table
            .store(std::ptr::null_mut(), Ordering::Release);
        self.old_capacity.store(0, Ordering::Release);
        self.new_table
            .store(std::ptr::null_mut(), Ordering::Release);
        self.new_capacity.store(0, Ordering::Release);
        self.rehash_cursor.store(0, Ordering::Release);

        // Queue old memory for cleanup
        self.old_memory.lock().push((old_memory, old_memory_size));

        let new_backing_fd: Option<(RawFd, PathBuf)> = None;
        self.finish_resize_backing_transition(new_backing_fd);
    }

    /// Get current capacity (thread-safe)
    #[inline]
    fn get_capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    /// Get current table pointer (thread-safe)
    #[inline]
    fn get_table(&self) -> *mut HashTableEntry {
        self.table.load(Ordering::Acquire)
    }

    /// Check if resize is needed and trigger if so
    fn maybe_resize(&self) {
        // Only check periodically to avoid overhead
        let counter = self.check_counter.fetch_add(1, Ordering::Relaxed);
        if counter % RESIZE_CHECK_INTERVAL != 0 {
            return;
        }

        let load = self.load_factor();
        if load < RESIZE_LOAD_THRESHOLD {
            return;
        }

        // Try to acquire resize lock (non-blocking)
        if let Some(_guard) = self.resize_lock.try_lock() {
            // Double-check load factor after acquiring lock
            if self.load_factor() >= RESIZE_LOAD_THRESHOLD {
                self.resize();
            }
        }
        // If we can't get the lock, another thread is resizing - that's fine
    }

    /// Resize the shard to double capacity
    fn resize(&self) {
        let old_capacity = self.get_capacity();
        let new_capacity = old_capacity * 2;
        let entry_size = std::mem::size_of::<HashTableEntry>();
        let new_memory_size = new_capacity * entry_size;

        eprintln!(
            "Resizing fingerprint shard {} from {} to {} entries ({} MB -> {} MB)",
            self.numa_node,
            old_capacity,
            new_capacity,
            old_capacity * entry_size / (1024 * 1024),
            new_memory_size / (1024 * 1024)
        );

        // Set NUMA preference for new allocation
        let _ = crate::storage::numa::set_preferred_node(self.numa_node);

        // Allocate new table (file-backed or anonymous)
        let (new_memory, new_backing_fd) = if let Some(ref backing_path) = self.backing_path {
            // File-backed: create new file, then rename after rehash
            let dir_path = backing_path.parent().unwrap_or(Path::new("."));
            let new_file_path = dir_path.join(format!("shard-{}-new.fp", self.shard_id));
            match unsafe { allocate_file_backed(new_memory_size, &new_file_path) } {
                Ok((ptr, fd)) => (ptr, Some((fd, new_file_path))),
                Err(e) => {
                    eprintln!(
                        "Warning: File-backed resize failed for shard {} ({}), using anonymous mmap",
                        self.shard_id, e
                    );
                    (unsafe { allocate_huge_pages(new_memory_size) }, None)
                }
            }
        } else {
            (unsafe { allocate_huge_pages(new_memory_size) }, None)
        };
        let new_table = new_memory as *mut HashTableEntry;

        // Zero-initialize new table
        unsafe {
            std::ptr::write_bytes(new_table, 0, new_capacity);
        }

        // Get old table pointer BEFORE marking resize in progress
        let old_table = self.get_table();

        // Store old table for RCU-style reads during resize
        // Readers can check this table while resize is in progress
        self.old_table.store(old_table, Ordering::Release);
        self.old_capacity.store(old_capacity, Ordering::Release);

        // Store new table for lock-free inserts during resize
        // Workers can insert here instead of waiting for resize to complete
        self.new_table.store(new_table, Ordering::Release);
        self.new_capacity.store(new_capacity, Ordering::Release);

        // Reset rehash cursor and batch counters for incremental migration
        self.rehash_cursor.store(0, Ordering::Release);
        self.rehash_worker_batches.store(0, Ordering::Relaxed);
        self.rehash_resize_batches.store(0, Ordering::Relaxed);

        // Determine adaptive batch size based on current occupancy
        let adaptive_batch = self.compute_rehash_batch_size();
        self.rehash_batch_size.store(adaptive_batch, Ordering::Release);

        // Mark resize in progress (odd seq number)
        // After this, workers will:
        //   1. Check old_table for reads (existing fingerprints)
        //   2. Insert into new_table (new fingerprints)
        //   3. Participate in incremental rehash via rehash_batch()
        self.seq.fetch_add(1, Ordering::AcqRel);

        // Incremental rehash: the resize thread rehashes in batches,
        // yielding between batches so workers can make progress.
        // Workers also participate by calling rehash_batch() on each operation.
        let rehash_start = std::time::Instant::now();
        while !self.is_rehash_complete() {
            self.rehash_batch_counted(true);
            // Yield after each batch to let workers run
            std::thread::yield_now();
        }

        let rehash_elapsed = rehash_start.elapsed();
        let resize_batches = self.rehash_resize_batches.load(Ordering::Relaxed);
        let worker_batches = self.rehash_worker_batches.load(Ordering::Relaxed);
        eprintln!(
            "Rehash complete for shard {} in {:.1}ms ({} entries, batch_size={}, resize_thread={} batches, workers={} batches)",
            self.shard_id,
            rehash_elapsed.as_secs_f64() * 1000.0,
            self.count.load(Ordering::Relaxed),
            adaptive_batch,
            resize_batches,
            worker_batches,
        );

        // Finalize: swap tables, clear pointers, queue old memory
        self.finalize_resize();

        // Handle file-backed transition if applicable
        let _ = new_backing_fd; // finalize_resize handles cleanup

        eprintln!(
            "Resize complete: new load factor: {:.1}%",
            self.load_factor() * 100.0
        );

        // Try to free old memory regions (safe if seq has advanced by 2+)
        self.cleanup_old_memory();
    }

    /// Clean up old memory regions that are safe to free
    fn cleanup_old_memory(&self) {
        let mut old_regions = self.old_memory.lock();
        // Keep only the most recent one (current readers might still use it)
        // Free older ones
        while old_regions.len() > 1 {
            let (ptr, size) = old_regions.remove(0);
            unsafe {
                munmap(ptr as *mut libc::c_void, size);
            }
        }
    }

    /// Check if fingerprint exists (read-only, no insert)
    ///
    /// Returns true if fingerprint exists, false otherwise.
    /// This is completely non-blocking.
    fn contains(&self, fp: u64) -> bool {
        let fp = if fp == 0 { 1 } else { fp };

        loop {
            let seq_before = self.seq.load(Ordering::Acquire);
            if seq_before % 2 == 1 {
                // RESIZE IN PROGRESS — participate in incremental rehash
                // before doing our own lookup. This distributes rehash work
                // across all workers instead of blocking on the resize thread.
                self.rehash_batch();

                // Check if rehash is now complete — try to finalize
                if self.is_rehash_complete() {
                    if let Some(_guard) = self.resize_lock.try_lock() {
                        // Double-check under lock
                        if self.seq.load(Ordering::Acquire) % 2 == 1 && self.is_rehash_complete() {
                            self.finalize_resize();
                        }
                    }
                    continue; // Retry with normal path
                }

                // Check old table
                let old_table_ptr = self.old_table.load(Ordering::Acquire);
                let old_cap = self.old_capacity.load(Ordering::Acquire);

                if !old_table_ptr.is_null() && old_cap > 0 {
                    let old_table = unsafe { std::slice::from_raw_parts(old_table_ptr, old_cap) };
                    let mut index = (fp as usize) % old_cap;
                    let mut probes = 0u64;

                    while probes < old_cap as u64 {
                        let stored_fp = old_table[index].fp.load(Ordering::Acquire);
                        if stored_fp == fp {
                            return true;
                        }
                        if stored_fp == 0 {
                            break;
                        }
                        index = (index + 1) % old_cap;
                        probes += 1;
                    }
                }

                // Check new table
                let new_table_ptr = self.new_table.load(Ordering::Acquire);
                let new_cap = self.new_capacity.load(Ordering::Acquire);

                if !new_table_ptr.is_null() && new_cap > 0 {
                    let new_table = unsafe { std::slice::from_raw_parts(new_table_ptr, new_cap) };
                    let mut index = (fp as usize) % new_cap;
                    let mut probes = 0u64;

                    while probes < new_cap as u64 {
                        let stored_fp = new_table[index].fp.load(Ordering::Acquire);
                        if stored_fp == fp {
                            return true;
                        }
                        if stored_fp == 0 {
                            break;
                        }
                        index = (index + 1) % new_cap;
                        probes += 1;
                    }
                }

                return false;
            }

            // NORMAL PATH - check main table
            let capacity = self.get_capacity();
            let table_ptr = self.get_table();
            let table = unsafe { std::slice::from_raw_parts(table_ptr, capacity) };

            let mut index = (fp as usize) % capacity;
            let mut probes = 0u64;

            while probes < capacity as u64 {
                let stored_fp = table[index].fp.load(Ordering::Acquire);
                if stored_fp == fp {
                    return true;
                }
                if stored_fp == 0 {
                    break;
                }
                index = (index + 1) % capacity;
                probes += 1;
            }

            // Check if resize happened during our operation
            let seq_after = self.seq.load(Ordering::Acquire);
            if seq_before == seq_after {
                return false;
            }
            // Resize happened, retry
        }
    }

    /// Check if fingerprint exists or insert it (lock-free with resize support)
    ///
    /// Returns true if fingerprint already existed, false if newly inserted
    /// This is completely non-blocking during resize.
    fn contains_or_insert(&self, fp: u64, stats: &FingerprintStatsAtomic) -> bool {
        // Never store 0 (reserved for empty slots)
        let fp = if fp == 0 { 1 } else { fp };

        let mut retries = 0u32;
        loop {
            retries += 1;
            if retries == MAX_RETRIES_BEFORE_WARN {
                eprintln!(
                    "Warning: fingerprint {:#x} (stats) exceeded {} retries in shard {}",
                    fp, MAX_RETRIES_BEFORE_WARN, self.numa_node
                );
            }
            if retries > MAX_RETRIES_BEFORE_PANIC {
                panic!(
                    "Fingerprint {:#x} (stats) stuck after {} retries in shard {}",
                    fp, MAX_RETRIES_BEFORE_PANIC, self.numa_node
                );
            }
            if retries > 0 && retries % 1000 == 0 {
                std::thread::yield_now();
            }
            let seq_before = self.seq.load(Ordering::Acquire);
            if seq_before % 2 == 1 {
                // RESIZE IN PROGRESS - use lock-free path through old+new tables

                // Step 1: Check old table (RCU read)
                let old_table_ptr = self.old_table.load(Ordering::Acquire);
                let old_cap = self.old_capacity.load(Ordering::Acquire);

                if !old_table_ptr.is_null() && old_cap > 0 {
                    let old_table = unsafe { std::slice::from_raw_parts(old_table_ptr, old_cap) };
                    let mut index = (fp as usize) % old_cap;
                    let mut probes = 0u64;

                    while probes < old_cap as u64 {
                        let stored_fp = old_table[index].fp.load(Ordering::Acquire);
                        if stored_fp == fp {
                            return true;
                        }
                        if stored_fp == 0 {
                            break;
                        }
                        index = (index + 1) % old_cap;
                        probes += 1;
                    }
                }

                // Step 2: Check/insert into new table (non-blocking!)
                let new_table_ptr = self.new_table.load(Ordering::Acquire);
                let new_cap = self.new_capacity.load(Ordering::Acquire);

                if !new_table_ptr.is_null() && new_cap > 0 {
                    let new_table =
                        unsafe { std::slice::from_raw_parts_mut(new_table_ptr, new_cap) };
                    let mut index = (fp as usize) % new_cap;
                    let mut probes = 0u64;

                    while probes < new_cap as u64 {
                        let entry = &new_table[index];
                        let stored_fp = entry.fp.load(Ordering::Acquire);

                        if stored_fp == fp {
                            return true;
                        }

                        if stored_fp == 0 {
                            match entry.fp.compare_exchange(
                                0,
                                fp,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => {
                                    entry.state.store(1, Ordering::Release);
                                    self.count.fetch_add(1, Ordering::Relaxed);
                                    if probes > 0 {
                                        stats.collisions.fetch_add(probes, Ordering::Relaxed);
                                    }
                                    return false;
                                }
                                Err(actual) if actual == fp => {
                                    return true;
                                }
                                Err(_) => {}
                            }
                        }

                        index = (index + 1) % new_cap;
                        probes += 1;
                    }

                    // New table full during resize — yield to let the resize thread
                    // make progress instead of spinning on CAS
                    if retries % 1000 == 0 {
                        eprintln!(
                            "Warning: fp {:#x} (stats) retry {} - new table full during resize",
                            fp, retries
                        );
                    }
                    if retries < 10 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                    continue;
                }

                // Check if resize completed
                let seq_recheck = self.seq.load(Ordering::Acquire);
                if seq_recheck % 2 == 0 {
                    continue;
                }
                // Resize still in progress — yield instead of spinning
                if retries < 10 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                continue;
            }

            // NORMAL PATH - no resize in progress
            let capacity = self.get_capacity();
            let table_ptr = self.get_table();
            let table = unsafe { std::slice::from_raw_parts_mut(table_ptr, capacity) };

            let mut index = (fp as usize) % capacity;
            let mut probes = 0u64;
            let mut result = None;

            while probes < capacity as u64 {
                let entry = &table[index];
                let stored_fp = entry.fp.load(Ordering::Acquire);

                if stored_fp == fp {
                    result = Some(true);
                    break;
                }

                if stored_fp == 0 {
                    // Check seq BEFORE CAS to avoid inserting while resize may have started
                    let seq_check = self.seq.load(Ordering::Acquire);
                    if seq_check != seq_before {
                        break; // Resize started, retry
                    }

                    match entry
                        .fp
                        .compare_exchange(0, fp, Ordering::AcqRel, Ordering::Acquire)
                    {
                        Ok(_) => {
                            entry.state.store(1, Ordering::Release);
                            self.count.fetch_add(1, Ordering::Relaxed);
                            if probes > 0 {
                                stats.collisions.fetch_add(probes, Ordering::Relaxed);
                            }
                            result = Some(false);
                            break;
                        }
                        Err(actual) if actual == fp => {
                            result = Some(true);
                            break;
                        }
                        Err(_) => {}
                    }
                }

                index = (index + 1) % capacity;
                probes += 1;
            }

            let seq_after = self.seq.load(Ordering::Acquire);
            if seq_before != seq_after {
                continue;
            }

            if let Some(existed) = result {
                if !existed {
                    self.maybe_resize();
                }
                return existed;
            }

            // Table is full - trigger resize and retry
            // Use try_lock to avoid blocking all workers
            if let Some(_guard) = self.resize_lock.try_lock() {
                if self.load_factor() >= 0.80 {
                    if retries > 1 {
                        eprintln!(
                            "Triggering resize (stats) after {} retries (load={:.1}%)",
                            retries,
                            self.load_factor() * 100.0
                        );
                    }
                    self.resize();
                }
            } else {
                // Someone else is resizing - yield and retry
                if retries % 1000 == 0 {
                    eprintln!(
                        "Warning: fp {:#x} (stats) retry {} - waiting for resize",
                        fp, retries
                    );
                }
                std::thread::yield_now();
            }
        }
    }

    /// Check if fingerprint exists or insert it (lock-free, no stats, with resize support)
    ///
    /// Returns true if fingerprint already existed, false if newly inserted
    /// This version skips stats updates for maximum performance
    ///
    /// IMPORTANT: This is completely non-blocking during resize:
    /// - Check old table for existing fingerprints
    /// - Check/insert into new table (no waiting!)
    fn contains_or_insert_no_stats(&self, fp: u64) -> bool {
        // Never store 0 (reserved for empty slots)
        let fp = if fp == 0 { 1 } else { fp };

        let mut retries = 0u32;
        let mut resize_path_count = 0u32;
        let mut normal_path_count = 0u32;
        let mut seq_mismatch_count = 0u32;
        loop {
            retries += 1;
            if retries == MAX_RETRIES_BEFORE_WARN {
                eprintln!(
                    "Warning: fingerprint {:#x} exceeded {} retries in shard {} (seq={}, load={:.1}%, resize_path={}, normal_path={}, seq_mismatch={})",
                    fp,
                    MAX_RETRIES_BEFORE_WARN,
                    self.numa_node,
                    self.seq.load(Ordering::Relaxed),
                    self.load_factor() * 100.0,
                    resize_path_count,
                    normal_path_count,
                    seq_mismatch_count
                );
            }
            if retries > MAX_RETRIES_BEFORE_PANIC {
                panic!(
                    "Fingerprint {:#x} stuck in infinite loop after {} retries in shard {} (seq={}, capacity={}, count={}, load={:.1}%, resize_path={}, normal_path={}, seq_mismatch={})",
                    fp,
                    MAX_RETRIES_BEFORE_PANIC,
                    self.numa_node,
                    self.seq.load(Ordering::Relaxed),
                    self.get_capacity(),
                    self.len(),
                    self.load_factor() * 100.0,
                    resize_path_count,
                    normal_path_count,
                    seq_mismatch_count
                );
            }
            // Exponential backoff to reduce contention at high worker counts
            // This helps prevent livelock when many workers are competing
            if retries > 100 {
                // After 100 retries, yield to let other threads (especially the
                // resize thread) make progress instead of spinning
                std::thread::yield_now();
            }
            if retries > 0 && retries % 1000 == 0 {
                std::thread::yield_now();
            }
            // Read seqlock - if odd, resize in progress
            let seq_before = self.seq.load(Ordering::Acquire);
            if seq_before % 2 == 1 {
                resize_path_count += 1;

                // If we've been stuck in resize path for many retries, yield aggressively
                // This helps the resize thread get CPU time to complete
                if resize_path_count > 100 && resize_path_count % 10 == 0 {
                    std::thread::yield_now();
                }

                // RESIZE IN PROGRESS — participate in incremental rehash
                self.rehash_batch();

                // Check if rehash is now complete — try to finalize
                if self.is_rehash_complete() {
                    if let Some(_guard) = self.resize_lock.try_lock() {
                        if self.seq.load(Ordering::Acquire) % 2 == 1 && self.is_rehash_complete() {
                            self.finalize_resize();
                        }
                    }
                    continue; // Retry with normal path
                }

                // Step 1: Check old table (RCU read)
                let old_table_ptr = self.old_table.load(Ordering::Acquire);
                let old_cap = self.old_capacity.load(Ordering::Acquire);

                if !old_table_ptr.is_null() && old_cap > 0 {
                    let old_table = unsafe { std::slice::from_raw_parts(old_table_ptr, old_cap) };
                    let mut index = (fp as usize) % old_cap;
                    let mut probes = 0u64;

                    while probes < old_cap as u64 {
                        let stored_fp = old_table[index].fp.load(Ordering::Acquire);
                        if stored_fp == fp {
                            // Found in old table - already seen
                            return true;
                        }
                        if stored_fp == 0 {
                            break;
                        }
                        index = (index + 1) % old_cap;
                        probes += 1;
                    }
                }

                // Step 2: Check/insert into new table (non-blocking!)
                let new_table_ptr = self.new_table.load(Ordering::Acquire);
                let new_cap = self.new_capacity.load(Ordering::Acquire);

                if !new_table_ptr.is_null() && new_cap > 0 {
                    let new_table =
                        unsafe { std::slice::from_raw_parts_mut(new_table_ptr, new_cap) };
                    let mut index = (fp as usize) % new_cap;
                    let mut probes = 0u64;

                    while probes < new_cap as u64 {
                        let entry = &new_table[index];
                        let stored_fp = entry.fp.load(Ordering::Acquire);

                        if stored_fp == fp {
                            // Found in new table
                            return true;
                        }

                        if stored_fp == 0 {
                            // Try to insert
                            match entry.fp.compare_exchange(
                                0,
                                fp,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => {
                                    entry.state.store(1, Ordering::Release);
                                    self.count.fetch_add(1, Ordering::Relaxed);
                                    return false; // New fingerprint
                                }
                                Err(actual) if actual == fp => {
                                    return true; // Concurrent insert
                                }
                                Err(_) => {
                                    // Slot taken, continue probing
                                }
                            }
                        }

                        index = (index + 1) % new_cap;
                        probes += 1;
                    }

                    // New table is getting full during resize - very rare
                    // This can happen if:
                    // 1. Many workers inserting concurrently fill the table
                    // 2. The resize thread is slow to complete
                    if retries % 1000 == 0 {
                        eprintln!(
                            "Warning: fp {:#x} retry {} - new table full during resize (cap={}, seq={})",
                            fp,
                            retries,
                            new_cap,
                            self.seq.load(Ordering::Relaxed)
                        );
                    }
                    if retries < 10 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                    continue;
                }

                // new_table not yet set up OR resize just completed and cleared it
                let seq_recheck = self.seq.load(Ordering::Acquire);
                if seq_recheck % 2 == 0 {
                    continue;
                }
                // Still resizing but new_table not ready - yield
                if retries < 10 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                continue;
            }

            // NORMAL PATH - no resize in progress
            normal_path_count += 1;
            let capacity = self.get_capacity();
            let table_ptr = self.get_table();
            let table = unsafe { std::slice::from_raw_parts_mut(table_ptr, capacity) };

            let mut index = (fp as usize) % capacity;
            let mut probes = 0u64;
            let mut result = None;

            while probes < capacity as u64 {
                let entry = &table[index];
                let stored_fp = entry.fp.load(Ordering::Acquire);

                if stored_fp == fp {
                    result = Some(true);
                    break;
                }

                if stored_fp == 0 {
                    // Check seq BEFORE CAS to avoid inserting while resize may have started
                    let seq_check = self.seq.load(Ordering::Acquire);
                    if seq_check != seq_before {
                        break; // Resize started, retry
                    }

                    match entry
                        .fp
                        .compare_exchange(0, fp, Ordering::AcqRel, Ordering::Acquire)
                    {
                        Ok(_) => {
                            entry.state.store(1, Ordering::Release);
                            self.count.fetch_add(1, Ordering::Relaxed);
                            result = Some(false);
                            break;
                        }
                        Err(actual) if actual == fp => {
                            result = Some(true);
                            break;
                        }
                        Err(_) => {}
                    }
                }

                index = (index + 1) % capacity;
                probes += 1;
            }

            // Check if resize happened during our operation
            let seq_after = self.seq.load(Ordering::Acquire);
            if seq_before != seq_after {
                seq_mismatch_count += 1;
                continue;
            }

            if let Some(existed) = result {
                if !existed {
                    self.maybe_resize();
                }
                return existed;
            }

            // Table is full - trigger resize and retry
            // Use try_lock to avoid blocking all workers
            if let Some(_guard) = self.resize_lock.try_lock() {
                if self.load_factor() >= 0.80 {
                    if retries > 1 {
                        eprintln!(
                            "Triggering resize after {} retries (cap={}, count={}, load={:.1}%)",
                            retries,
                            capacity,
                            self.len(),
                            self.load_factor() * 100.0
                        );
                    }
                    self.resize();
                }
            } else {
                // Someone else is resizing - yield and retry
                // The resize path above will handle insertion into new table
                if retries % 1000 == 0 {
                    eprintln!(
                        "Warning: fp {:#x} retry {} - waiting for resize to complete",
                        fp, retries
                    );
                }
                std::thread::yield_now();
            }
        }
    }

    /// Get current count of fingerprints
    fn len(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get load factor (0.0 to 1.0)
    fn load_factor(&self) -> f64 {
        self.len() as f64 / self.get_capacity() as f64
    }

    fn has_file_backed_mapping(&self) -> bool {
        self.mapping_is_file_backed.load(Ordering::Acquire)
    }

    fn can_advise_cold(&self) -> bool {
        self.has_file_backed_mapping()
    }

    fn finish_resize_backing_transition(&self, new_backing: Option<(RawFd, PathBuf)>) {
        if let Some((new_fd, new_file_path)) = new_backing {
            if let Some(ref backing_path) = self.backing_path {
                if let Err(e) = std::fs::rename(&new_file_path, backing_path) {
                    eprintln!(
                        "Warning: Failed to rename {} -> {}: {}",
                        new_file_path.display(),
                        backing_path.display(),
                        e
                    );
                }
            }

            let old_fd = {
                let mut guard = self.backing_fd.lock();
                let old_fd = guard.take();
                *guard = Some(new_fd);
                old_fd
            };
            if let Some(old_fd) = old_fd {
                unsafe {
                    libc::close(old_fd);
                }
            }
            self.mapping_is_file_backed.store(true, Ordering::Release);
            return;
        }

        let old_fd = self.backing_fd.lock().take();
        if let Some(old_fd) = old_fd {
            unsafe {
                libc::close(old_fd);
            }
        }
        self.mapping_is_file_backed.store(false, Ordering::Release);
    }

    /// Advise the kernel that this shard's memory is cold and can be paged out
    ///
    /// For file-backed MAP_SHARED mappings:
    /// - MADV_PAGEOUT (Linux 5.4+): asks kernel to write dirty pages to file and reclaim
    /// - MADV_DONTNEED fallback: drops clean pages from RAM (data stays in file)
    ///
    /// For anonymous mappings, MADV_DONTNEED discards pages (they'll be zero on next access),
    /// so we skip it to avoid data loss.
    fn advise_cold(&self) {
        let memory = self.memory.load(Ordering::Acquire);
        let memory_size = self.memory_size.load(Ordering::Acquire);

        if memory.is_null() || memory_size == 0 {
            return;
        }

        // Only advise cold for file-backed mappings (safe - data is in the file)
        if !self.can_advise_cold() {
            return;
        }

        unsafe {
            // Try MADV_PAGEOUT first (Linux 5.4+)
            #[cfg(target_os = "linux")]
            {
                let ret = libc::madvise(memory as *mut libc::c_void, memory_size, MADV_PAGEOUT);
                if ret == 0 {
                    return;
                }
                // MADV_PAGEOUT not available, fall back to MADV_DONTNEED
            }
            // For file-backed MAP_SHARED, MADV_DONTNEED drops the page from RAM
            // but the data remains in the file and will be faulted back in on access
            libc::madvise(memory as *mut libc::c_void, memory_size, MADV_DONTNEED);
        }
    }

    /// Get the memory base address for NUMA diagnostics
    fn memory_base_addr(&self) -> *const u8 {
        self.memory.load(Ordering::Acquire)
    }
}

impl Drop for FingerprintShard {
    fn drop(&mut self) {
        // Free current memory
        let memory = self.memory.load(Ordering::Relaxed);
        let memory_size = self.memory_size.load(Ordering::Relaxed);
        if !memory.is_null() {
            unsafe {
                munmap(memory as *mut libc::c_void, memory_size);
            }
        }

        // Close backing file descriptor if file-backed
        // Note: we do NOT delete the backing file (allows resume)
        if let Some(fd) = self.backing_fd.lock().take() {
            unsafe {
                libc::close(fd);
            }
        }

        // Free any old memory regions
        let old_regions = self.old_memory.lock();
        for &(ptr, size) in old_regions.iter() {
            if !ptr.is_null() {
                unsafe {
                    munmap(ptr as *mut libc::c_void, size);
                }
            }
        }
    }
}

/// Page-aligned fingerprint store with NUMA awareness
pub struct PageAlignedFingerprintStore {
    shards: Vec<FingerprintShard>,
    #[allow(dead_code)]
    shard_mask: usize,
    stats: FingerprintStatsAtomic,
    /// Optional persistence channels (one per shard)
    persist_tx: Option<Vec<Sender<FingerprintPersistMsg>>>,
    /// Number of NUMA nodes
    num_numa_nodes: usize,
    /// Shards per NUMA node (for deterministic routing)
    shards_per_numa: usize,
}

impl PageAlignedFingerprintStore {
    /// Create new fingerprint store with NUMA-local shard placement
    ///
    /// Shards are deterministically assigned to NUMA nodes:
    /// - Shards 0..K on NUMA 0, K..2K on NUMA 1, etc.
    /// - Fingerprints are routed to their home NUMA via `home_numa(fp)`
    /// - This ensures fingerprint checks are local when states are routed correctly
    pub fn new(config: FingerprintStoreConfig, _worker_cpus: &[Option<usize>]) -> Result<Self> {
        Self::new_with_backing(config, _worker_cpus, None)
    }

    /// Create new fingerprint store with optional file-backed mmap
    ///
    /// When `backing_dir` is Some, shards use file-backed mmap so the kernel
    /// can page cold fingerprints to disk under memory pressure.
    pub fn new_with_backing(
        config: FingerprintStoreConfig,
        _worker_cpus: &[Option<usize>],
        backing_dir: Option<&Path>,
    ) -> Result<Self> {
        let shard_count = config.shard_count.max(1).next_power_of_two();

        // Detect NUMA topology
        let numa_topology = crate::storage::numa::NumaTopology::detect()?;
        let num_numa_nodes = numa_topology.node_count.max(1);

        // Deterministic NUMA assignment: shards are evenly distributed across NUMA nodes
        // Shards 0..K on NUMA 0, K..2K on NUMA 1, etc.
        let shards_per_numa = (shard_count + num_numa_nodes - 1) / num_numa_nodes;

        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "Creating {} shards with NUMA-local routing ({} nodes, {} shards/node)",
                shard_count, num_numa_nodes, shards_per_numa
            );
        }

        // Create shards with deterministic NUMA placement and staggered sizes
        // Staggering prevents thundering herd when all shards hit resize threshold together
        let mut shards = Vec::with_capacity(shard_count);
        for shard_id in 0..shard_count {
            // Deterministic: shard N belongs to NUMA (N / shards_per_numa)
            let numa_node = (shard_id / shards_per_numa).min(num_numa_nodes - 1);

            // Stagger sizes: vary by ±25% based on shard_id to spread out resizes
            // Using prime multipliers to ensure good distribution
            let stagger_factor = 1.0 + 0.25 * ((shard_id * 7 % 11) as f64 / 10.0 - 0.5);
            let staggered_size_mb =
                ((config.shard_size_mb as f64 * stagger_factor) as usize).max(64);

            let shard = FingerprintShard::new(staggered_size_mb, numa_node, backing_dir, shard_id)
                .context(format!(
                    "Failed to create shard {} on NUMA node {}",
                    shard_id, numa_node
                ))?;

            if std::env::var("TLAPP_VERBOSE").is_ok() {
                eprintln!(
                    "  Shard {:3}: {:.1} GB on NUMA node {} (capacity: {} fingerprints, stagger: {:.0}%)",
                    shard_id,
                    staggered_size_mb as f64 / 1024.0,
                    numa_node,
                    shard.get_capacity(),
                    stagger_factor * 100.0
                );
            }

            shards.push(shard);
        }

        let total_capacity: usize = shards.iter().map(|s| s.get_capacity()).sum();
        let total_memory_gb = (shard_count * config.shard_size_mb) as f64 / 1024.0;

        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "Total fingerprint capacity: {} entries ({:.1} GB)",
                total_capacity, total_memory_gb
            );
        }

        Ok(Self {
            shards,
            shard_mask: shard_count - 1,
            stats: FingerprintStatsAtomic::default(),
            persist_tx: None,
            num_numa_nodes,
            shards_per_numa,
        })
    }

    /// Get the home NUMA node for a fingerprint
    ///
    /// This determines which NUMA node "owns" this fingerprint.
    /// States should be routed to their fingerprint's home NUMA for local checking.
    #[inline]
    pub fn home_numa(&self, fp: u64) -> usize {
        // Mix bits to reduce correlation between NUMA routing and shard selection
        let mixed = (fp >> 32) ^ (fp >> 16) ^ fp;
        (mixed as usize) % self.num_numa_nodes
    }

    /// Get number of NUMA nodes
    #[inline]
    pub fn num_numa_nodes(&self) -> usize {
        self.num_numa_nodes
    }

    /// Enable persistence by providing senders for each shard
    pub fn enable_persistence(&mut self, persist_tx: Vec<Sender<FingerprintPersistMsg>>) {
        assert_eq!(
            persist_tx.len(),
            self.shards.len(),
            "Must provide one sender per shard"
        );
        self.persist_tx = Some(persist_tx);
    }

    /// Get the shard for a fingerprint using NUMA-local routing
    ///
    /// The fingerprint is routed to a shard on its home NUMA node:
    /// 1. Compute home NUMA from fingerprint
    /// 2. Select shard within that NUMA's range
    #[inline]
    #[allow(dead_code)]
    fn shard_for(&self, fp: u64) -> &FingerprintShard {
        let numa = self.home_numa(fp);
        // Select shard within this NUMA's range
        let shard_within_numa = (fp as usize) % self.shards_per_numa;
        let shard_id = (numa * self.shards_per_numa + shard_within_numa).min(self.shards.len() - 1);
        &self.shards[shard_id]
    }

    /// Get shard ID for a fingerprint (for external use)
    #[inline]
    pub fn shard_id_for(&self, fp: u64) -> usize {
        let numa = self.home_numa(fp);
        let shard_within_numa = (fp as usize) % self.shards_per_numa;
        (numa * self.shards_per_numa + shard_within_numa).min(self.shards.len() - 1)
    }

    /// Check if fingerprint exists (read-only, no insert)
    pub fn contains(&self, fp: u64) -> bool {
        // Route using original fp to match insert path
        // The shard normalizes fp=0 to fp=1 internally
        let shard_id = self.shard_id_for(fp);
        let shard = &self.shards[shard_id];
        shard.contains(fp)
    }

    /// Batch check for fingerprints (read-only, no insert)
    pub fn contains_batch(&self, fps: &[u64], seen: &mut Vec<bool>) {
        seen.clear();
        seen.resize(fps.len(), false);

        let num_shards = self.shards.len();

        // Group by shard for cache locality
        let mut shard_groups: Vec<Vec<(usize, u64)>> = vec![Vec::new(); num_shards];

        for (idx, &fp) in fps.iter().enumerate() {
            let shard_id = self.shard_id_for(fp);
            shard_groups[shard_id].push((idx, fp));
        }

        // Process each shard's fingerprints
        for (shard_id, group) in shard_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }

            let shard = &self.shards[shard_id];

            for &(idx, fp) in group {
                seen[idx] = shard.contains(fp);
            }
        }
    }

    /// Check if fingerprint exists or insert it
    pub fn contains_or_insert(&self, fp: u64) -> bool {
        // Chaos: simulate shard full - degrade gracefully by treating as "not seen"
        // This may cause duplicate exploration but won't crash the system
        #[cfg(feature = "failpoints")]
        if crate::fail_point_is_set!("fp_store_shard_full") {
            // Log once per ~1000 calls to avoid spam
            static WARN_COUNTER: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            if WARN_COUNTER.fetch_add(1, Ordering::Relaxed) % 1000 == 0 {
                eprintln!("Warning: fingerprint store shard full (chaos), degrading gracefully");
            }
            return false; // Treat as new - may cause duplicate work but won't crash
        }

        self.stats.checks.fetch_add(1, Ordering::Relaxed);
        let shard_id = self.shard_id_for(fp);
        let shard = &self.shards[shard_id];

        let existed = shard.contains_or_insert(fp, &self.stats);

        if existed {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.inserts.fetch_add(1, Ordering::Relaxed);

            // If persistence enabled, try to send (non-blocking)
            if let Some(ref persist_tx) = self.persist_tx {
                let _ = persist_tx[shard_id].try_send(FingerprintPersistMsg { fp });
                // If channel full, fingerprint is still in memory (safe)
            }
        }

        existed
    }

    /// Batch check and insert fingerprints
    ///
    /// Note: For better scalability at high worker counts, use
    /// `contains_or_insert_batch_with_affinity` which processes shards
    /// in a worker-specific order to reduce contention.
    pub fn contains_or_insert_batch(&self, fps: &[u64], seen: &mut Vec<bool>) -> Result<()> {
        self.contains_or_insert_batch_with_affinity(fps, seen, 0)
    }

    /// Batch check and insert fingerprints with worker affinity
    ///
    /// Each worker processes shards starting from a different offset (based on worker_id).
    /// This spreads contention across shards when many workers process similar fingerprints.
    ///
    /// Example with 4 shards and 4 workers:
    /// - Worker 0 processes shards: 0, 1, 2, 3
    /// - Worker 1 processes shards: 1, 2, 3, 0
    /// - Worker 2 processes shards: 2, 3, 0, 1
    /// - Worker 3 processes shards: 3, 0, 1, 2
    ///
    /// This reduces the probability of multiple workers hitting the same shard simultaneously.
    pub fn contains_or_insert_batch_with_affinity(
        &self,
        fps: &[u64],
        seen: &mut Vec<bool>,
        worker_id: usize,
    ) -> Result<()> {
        // NOTE: Stats updates removed from hot path to reduce atomic contention
        // Stats are now estimated from queue counters instead

        seen.clear();
        seen.resize(fps.len(), false);

        let num_shards = self.shards.len();

        // Group by shard for cache locality
        let mut shard_groups: Vec<Vec<(usize, u64)>> = vec![Vec::new(); num_shards];

        for (idx, &fp) in fps.iter().enumerate() {
            let shard_id = self.shard_id_for(fp);
            shard_groups[shard_id].push((idx, fp));
        }

        // Process shards in worker-specific order to reduce contention
        // Worker N starts at shard (N % num_shards) and wraps around
        let start_shard = worker_id % num_shards;

        for offset in 0..num_shards {
            let shard_id = (start_shard + offset) % num_shards;
            let group = &shard_groups[shard_id];

            if group.is_empty() {
                continue;
            }

            let shard = &self.shards[shard_id];

            for &(idx, fp) in group {
                let existed = shard.contains_or_insert_no_stats(fp);
                seen[idx] = existed;

                if !existed {
                    // If persistence enabled, try to send (non-blocking)
                    if let Some(ref persist_tx) = self.persist_tx {
                        let _ = persist_tx[shard_id].try_send(FingerprintPersistMsg { fp });
                    }
                }
            }
        }

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> FingerprintStats {
        FingerprintStats {
            checks: self.stats.checks.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            batch_calls: self.stats.batch_calls.load(Ordering::Relaxed),
            batch_items: self.stats.batch_items.load(Ordering::Relaxed),
            collisions: self.stats.collisions.load(Ordering::Relaxed),
        }
    }

    /// Get total count of fingerprints
    pub fn len(&self) -> u64 {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get average load factor across shards
    pub fn load_factor(&self) -> f64 {
        let sum: f64 = self.shards.iter().map(|s| s.load_factor()).sum();
        sum / self.shards.len() as f64
    }

    /// Get the number of shards
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Advise the kernel that all shard memory is cold and can be paged out
    ///
    /// Only effective for file-backed mappings. For anonymous mappings, this is a no-op.
    pub fn advise_cold(&self) {
        for shard in &self.shards {
            shard.advise_cold();
        }
    }

    /// Flush (no-op for in-memory store)
    pub fn flush(&self) -> Result<usize> {
        Ok(0)
    }

    /// Get one memory sample address per shard for NUMA diagnostics.
    pub fn memory_base_addrs(&self) -> Vec<*const u8> {
        self.shards
            .iter()
            .map(|shard| shard.memory_base_addr())
            .collect()
    }
}

/// Allocate file-backed mmap memory
///
/// Creates/truncates a file at `path`, extends it to `size` bytes (rounded up to
/// HUGE_PAGE_SIZE), and maps it with MAP_SHARED. The kernel can page cold data
/// to the file under memory pressure, making memory limits enforceable.
///
/// Returns (pointer, file_descriptor) on success.
unsafe fn allocate_file_backed(size: usize, path: &Path) -> Result<(*mut u8, RawFd)> {
    use std::ffi::CString;

    let aligned_size = (size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);

    // Create/truncate the backing file
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path: {:?}", path))?;
    let c_path =
        CString::new(path_str).map_err(|e| anyhow::anyhow!("Invalid path string: {}", e))?;

    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC,
            0o644,
        )
    };
    if fd < 0 {
        return Err(anyhow::anyhow!(
            "Failed to open backing file {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }

    // Extend file to the required size
    if unsafe { libc::ftruncate(fd, aligned_size as libc::off_t) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(anyhow::anyhow!(
            "Failed to ftruncate {} to {} bytes: {}",
            path.display(),
            aligned_size,
            err
        ));
    }

    // mmap with MAP_SHARED (NOT MAP_ANONYMOUS, NOT MAP_POPULATE)
    // MAP_SHARED allows the kernel to write dirty pages back to the file
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            aligned_size,
            PROT_READ | PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };

    if ptr == MAP_FAILED {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(anyhow::anyhow!(
            "Failed to mmap {} ({} bytes): {}",
            path.display(),
            aligned_size,
            err
        ));
    }

    // Advise random access pattern (hash table lookups are random)
    unsafe { libc::madvise(ptr, aligned_size, MADV_RANDOM) };

    // Request transparent huge pages for TLB efficiency
    unsafe { libc::madvise(ptr, aligned_size, MADV_HUGEPAGE) };

    if std::env::var("TLAPP_VERBOSE").is_ok() {
        eprintln!(
            "  Allocated {} MB file-backed mmap at {}",
            aligned_size / (1024 * 1024),
            path.display()
        );
    }

    Ok((ptr as *mut u8, fd))
}

/// Allocate memory with huge page support
unsafe fn allocate_huge_pages(size: usize) -> *mut u8 {
    // Round up to huge page size
    let aligned_size = (size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);

    // Try explicit huge pages first (requires hugetlbfs)
    let ptr = unsafe {
        mmap(
            std::ptr::null_mut(),
            aligned_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
            -1,
            0,
        )
    };

    if ptr != MAP_FAILED {
        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "  Allocated {} MB with explicit huge pages",
                aligned_size / (1024 * 1024)
            );
        }
        return ptr as *mut u8;
    }

    // Fallback to transparent huge pages (THP)
    let ptr = unsafe {
        mmap(
            std::ptr::null_mut(),
            aligned_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
            -1,
            0,
        )
    };

    if ptr == MAP_FAILED {
        panic!("Failed to allocate {} bytes of memory", aligned_size);
    }

    // Advise kernel to use huge pages
    let result = unsafe { madvise(ptr, aligned_size, MADV_HUGEPAGE) };
    if std::env::var("TLAPP_VERBOSE").is_ok() {
        if result == 0 {
            eprintln!(
                "  Allocated {} MB with transparent huge pages (THP)",
                aligned_size / (1024 * 1024)
            );
        } else {
            eprintln!(
                "  Allocated {} MB (THP advise failed, using regular pages)",
                aligned_size / (1024 * 1024)
            );
        }
    }

    ptr as *mut u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_backing_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tlapp-page-aligned-{prefix}-{nanos}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn test_shard_basic_operations() {
        let shard = FingerprintShard::new(64, 0, None, 0).unwrap();
        let stats = FingerprintStatsAtomic::default();

        // Insert new fingerprint
        assert!(!shard.contains_or_insert(12345, &stats));
        assert_eq!(shard.len(), 1);

        // Check existing fingerprint
        assert!(shard.contains_or_insert(12345, &stats));
        assert_eq!(shard.len(), 1);

        // Insert another
        assert!(!shard.contains_or_insert(67890, &stats));
        assert_eq!(shard.len(), 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn file_backed_resize_fallback_disables_cold_advice() {
        let backing_dir = temp_backing_dir("resize-fallback");
        std::fs::create_dir_all(&backing_dir).unwrap();

        let shard = FingerprintShard::new(1, 0, Some(&backing_dir), 0).unwrap();
        assert!(
            shard.has_file_backed_mapping(),
            "expected writable file-backed mmap in temp dir"
        );
        assert!(shard.can_advise_cold());
        assert!(shard.backing_fd.lock().is_some());

        shard.finish_resize_backing_transition(None);

        assert!(!shard.has_file_backed_mapping());
        assert!(!shard.can_advise_cold());
        assert!(shard.backing_fd.lock().is_none());

        let _ = std::fs::remove_dir_all(backing_dir);
    }

    #[test]
    #[serial]
    fn test_store_basic_operations() {
        let config = FingerprintStoreConfig {
            shard_count: 4,
            expected_items: 1000,
            shard_size_mb: 64,
        };

        let worker_cpus = vec![Some(0), Some(1), Some(2), Some(3)];
        let store = PageAlignedFingerprintStore::new(config, &worker_cpus).unwrap();

        // Insert fingerprints
        assert!(!store.contains_or_insert(100));
        assert!(!store.contains_or_insert(200));
        assert!(!store.contains_or_insert(300));

        // Check they exist
        assert!(store.contains_or_insert(100));
        assert!(store.contains_or_insert(200));
        assert!(store.contains_or_insert(300));

        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_batch_operations() {
        let config = FingerprintStoreConfig {
            shard_count: 4,
            expected_items: 1000,
            shard_size_mb: 64,
        };

        let worker_cpus = vec![Some(0), Some(1), Some(2), Some(3)];
        let store = PageAlignedFingerprintStore::new(config, &worker_cpus).unwrap();

        let fps = vec![1, 2, 3, 4, 5];
        let mut seen = Vec::new();

        // First batch - all new
        store.contains_or_insert_batch(&fps, &mut seen).unwrap();
        assert_eq!(seen, vec![false, false, false, false, false]);

        // Second batch - all existing
        store.contains_or_insert_batch(&fps, &mut seen).unwrap();
        assert_eq!(seen, vec![true, true, true, true, true]);

        // Mixed batch
        let mixed = vec![1, 100, 2, 200, 3];
        store.contains_or_insert_batch(&mixed, &mut seen).unwrap();
        assert_eq!(seen, vec![true, false, true, false, true]);
    }

    #[test]
    #[serial]
    fn test_edge_case_fingerprints() {
        // Regression test for fp=0 edge case found by fuzzing
        // fp=0 is normalized to fp=1 internally, but routing must be consistent
        let config = FingerprintStoreConfig {
            shard_count: 4,
            expected_items: 1000,
            shard_size_mb: 64,
        };

        let worker_cpus = vec![Some(0), Some(1), Some(2), Some(3)];
        let store = PageAlignedFingerprintStore::new(config, &worker_cpus).unwrap();

        // Test fp=0 via single insert
        assert!(!store.contains_or_insert(0), "fp=0 should be new");
        assert!(store.contains(0), "fp=0 should exist after insert");
        assert!(
            store.contains_or_insert(0),
            "fp=0 should exist on second insert"
        );

        // Test fp=u64::MAX
        assert!(!store.contains_or_insert(u64::MAX), "fp=MAX should be new");
        assert!(store.contains(u64::MAX), "fp=MAX should exist after insert");

        // Test fp=0 and fp=u64::MAX via batch
        let fps = vec![0, 1, u64::MAX, u64::MAX - 1];
        let mut seen = Vec::new();

        // Reset with fresh store
        let config2 = FingerprintStoreConfig {
            shard_count: 4,
            expected_items: 1000,
            shard_size_mb: 64,
        };
        let store2 = PageAlignedFingerprintStore::new(config2, &worker_cpus).unwrap();

        store2.contains_or_insert_batch(&fps, &mut seen).unwrap();
        assert_eq!(seen, vec![false, false, false, false], "All should be new");

        // Verify all exist via contains
        for &fp in &fps {
            assert!(
                store2.contains(fp),
                "fp={} should exist after batch insert",
                fp
            );
        }

        // Second batch should find all
        store2.contains_or_insert_batch(&fps, &mut seen).unwrap();
        assert_eq!(seen, vec![true, true, true, true], "All should exist");
    }

    #[test]
    #[serial]
    fn test_concurrent_inserts() {
        use std::sync::Arc;
        use std::thread;

        let config = FingerprintStoreConfig {
            shard_count: 8,
            expected_items: 10000,
            shard_size_mb: 64,
        };

        let worker_cpus: Vec<_> = (0..8).map(Some).collect();
        let store = Arc::new(PageAlignedFingerprintStore::new(config, &worker_cpus).unwrap());

        // Spawn 8 threads, each inserting 1000 fingerprints
        let handles: Vec<_> = (0..8)
            .map(|thread_id| {
                let store = Arc::clone(&store);
                thread::spawn(move || {
                    for i in 0..1000 {
                        let fp = (thread_id * 1000 + i) as u64;
                        store.contains_or_insert(fp);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have exactly 8000 fingerprints
        assert_eq!(store.len(), 8000);

        // All fingerprints should be findable
        for thread_id in 0..8 {
            for i in 0..1000 {
                let fp = (thread_id * 1000 + i) as u64;
                assert!(store.contains_or_insert(fp));
            }
        }
    }

    #[test]
    fn test_shard_resize() {
        // Create small shard (2MB = ~118K capacity with 90% headroom)
        // With 16-byte entries: 2MB / 16 bytes = 131072 entries
        // 90% of that = 117964 capacity
        let shard = FingerprintShard::new(2, 0, None, 0).unwrap();
        let stats = FingerprintStatsAtomic::default();

        let initial_capacity = shard.get_capacity();
        eprintln!("Initial capacity: {}", initial_capacity);

        // Fill to 95% to ensure resize triggers (threshold is 85%, checked every 10K inserts)
        // The check at 110K inserts will see 93% load and trigger resize
        let target_count = (initial_capacity as f64 * 0.95) as u64;
        eprintln!(
            "Inserting {} fingerprints to trigger resize...",
            target_count
        );

        for i in 1..=target_count {
            let existed = shard.contains_or_insert(i, &stats);
            assert!(!existed, "Fingerprint {} should be new", i);
        }

        let final_capacity = shard.get_capacity();
        let final_count = shard.len();
        let final_load = shard.load_factor();

        eprintln!(
            "After inserts: capacity={}, count={}, load={:.2}%",
            final_capacity,
            final_count,
            final_load * 100.0
        );

        // Verify resize happened
        assert!(
            final_capacity > initial_capacity,
            "Expected resize: initial={}, final={}",
            initial_capacity,
            final_capacity
        );

        // Verify capacity doubled
        assert_eq!(
            final_capacity,
            initial_capacity * 2,
            "Capacity should double on resize"
        );

        // Verify load factor decreased
        assert!(
            final_load < 0.85,
            "Load factor should be below 85% after resize, got {:.2}%",
            final_load * 100.0
        );

        // Verify all fingerprints still exist
        for i in 1..=target_count {
            let existed = shard.contains_or_insert(i, &stats);
            assert!(existed, "Fingerprint {} should exist after resize", i);
        }
    }

    #[test]
    fn test_concurrent_resize() {
        use std::sync::Arc;
        use std::thread;

        // Create small shard to trigger resize quickly
        let shard = Arc::new(FingerprintShard::new(2, 0, None, 0).unwrap());
        let initial_capacity = shard.get_capacity();

        // Target 150% of initial capacity to ensure resize
        let total_inserts = (initial_capacity as f64 * 1.5) as u64;
        let num_threads = 8;
        let inserts_per_thread = total_inserts / num_threads;

        eprintln!(
            "Concurrent resize test: {} threads x {} inserts = {} total",
            num_threads, inserts_per_thread, total_inserts
        );

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let shard = Arc::clone(&shard);
                let stats = Arc::new(FingerprintStatsAtomic::default());
                thread::spawn(move || {
                    let base = thread_id * inserts_per_thread + 1;
                    for i in 0..inserts_per_thread {
                        let fp = base + i;
                        shard.contains_or_insert(fp, &stats);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let final_capacity = shard.get_capacity();
        let final_count = shard.len();

        eprintln!(
            "Concurrent resize complete: capacity {} -> {}, count={}",
            initial_capacity, final_capacity, final_count
        );

        // Verify resize happened
        assert!(
            final_capacity >= initial_capacity * 2,
            "Expected at least one resize"
        );

        // Verify all fingerprints exist
        let stats = FingerprintStatsAtomic::default();
        for thread_id in 0..num_threads {
            let base = thread_id * inserts_per_thread + 1;
            for i in 0..inserts_per_thread {
                let fp = base + i;
                assert!(
                    shard.contains_or_insert(fp, &stats),
                    "Fingerprint {} should exist",
                    fp
                );
            }
        }
    }

    /// Property-based tests for fingerprint store
    mod proptests {
        use super::{FingerprintShard, FingerprintStatsAtomic};
        use proptest::prelude::*;
        use std::collections::HashSet;

        proptest! {
            /// Insert then contains always returns true (no false negatives)
            #[test]
            fn insert_then_contains(fps in prop::collection::vec(1u64..1_000_000, 1..100)) {
                let shard = FingerprintShard::new(64, 0, None, 0).unwrap();
                let stats = FingerprintStatsAtomic::default();

                // Insert all
                for &fp in &fps {
                    shard.contains_or_insert(fp, &stats);
                }

                // Verify all exist
                for &fp in &fps {
                    prop_assert!(
                        shard.contains_or_insert(fp, &stats),
                        "Inserted fingerprint {} must be found",
                        fp
                    );
                }
            }

            /// Duplicate inserts return true (second insert sees it exists)
            #[test]
            fn duplicate_insert_returns_true(fp: u64) {
                let shard = FingerprintShard::new(16, 0, None, 0).unwrap();
                let stats = FingerprintStatsAtomic::default();

                // First insert - should not exist yet
                let first = shard.contains_or_insert(fp, &stats);
                prop_assert!(!first, "First insert should return false (not previously present)");

                // Second insert - should exist now
                let second = shard.contains_or_insert(fp, &stats);
                prop_assert!(second, "Second insert should return true (already present)");
            }

            /// Count matches unique fingerprints inserted
            #[test]
            fn count_matches_unique_inserts(fps in prop::collection::vec(any::<u64>(), 1..200)) {
                let shard = FingerprintShard::new(64, 0, None, 0).unwrap();
                let stats = FingerprintStatsAtomic::default();
                let unique: HashSet<_> = fps.iter().copied().collect();

                for &fp in &fps {
                    shard.contains_or_insert(fp, &stats);
                }

                prop_assert_eq!(
                    shard.len() as usize,
                    unique.len(),
                    "Count should match unique fingerprints"
                );
            }
        }
    }
}
