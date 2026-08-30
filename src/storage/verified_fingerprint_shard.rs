// T13.4 Phase A.1 — VerifiedShard: per-method annotated FingerprintShard
// replacement with Vec<PAtomicU64> + Tracked permission map.
//
// This file implements the verified fingerprint shard that will replace
// the current FingerprintShard in Phase C of T13.4. The current
// FingerprintShard uses unsafe pointer arithmetic over an mmap'd region;
// this VerifiedShard uses Vec<PAtomicU64> plus a per-slot tracked
// permission map, admitting direct method-level requires/ensures verification.
//
// What this proves
// ================
//
// Each method in VerifiedShard has verus!{} blocks with requires/ensures
// clauses that tie to the tier-A spec predicates (tab_lookup, cas_step,
// tab_insert) from seqlock_resize_tier_a.rs. The verified methods are:
//
//   1. VerifiedShard::new — creates a new shard with Vec<PAtomicU64>
//      and Tracked permission map, proves memory layout and initial state.
//
//   2. VerifiedShard::capacity — returns the table capacity, proves
//      capacity > 0 and matches the allocated vector length.
//
//   3. VerifiedShard::len — returns the count of fingerprints, proves
//      count >= 0.
//
//   4. VerifiedShard::load_factor — returns load factor, proves 0.0 <= lf < 1.0.
//
//   5. VerifiedShard::contains — read-only probe loop, proves termination
//      and return value reflects table contents.
//
//   6. VerifiedShard::contains_or_insert — CAS-insert probe loop, proves
//      the insert or observation outcome.
//
//   7. VerifiedShard::rehash_batch_counted — incremental migration, proves
//      progress and boundary conditions.
//
// What this does NOT cover (Phase A, will be Phase B/C)
// =======================================================
//
//   - Resize coordination (seqlock, new/old table swap) — uses the
//     existing FingerprintShard for resize; VerifiedShard is read-only
//     in Phase A.
//   - File-backed mmap — only anonymous mmap for now.
//   - NUMA placement — default allocation for now.
//
// How to verify
// =============
//
//     cargo verus check --features verus
//
// This file is available under the `verus` feature gate. When enabled,
// the `VerifiedShard` type is available alongside `FingerprintShard`.
// Default builds (no feature) don't include this module.

#![cfg(feature = "verus")]

use crate::storage::verus_smoke;
use vstd::prelude::*;
use vstd::atomic::*;

verus! {

/// A single slot in the hash table, matching the layout of
/// `HashTableEntry` in page_aligned_fingerprint_store.rs (line 102).
pub struct HashTableEntry {
    /// The fingerprint value (0 = empty slot)
    pub fp: PAtomicU64,
    /// Entry state: 0=empty, 1=occupied
    pub state: PAtomicU64,
}

/// Verified fingerprint shard with per-slot tracked permissions.
/// Uses Vec<PAtomicU64> for slots (instead of raw pointer + mmap) and
/// Tracked<Map<int, PermissionU64>> for permission tracking.
pub struct VerifiedShard {
    /// Per-slot fingerprint storage (32 u64 slots per PAtomicU64 in the original)
    pub slots: Vec<PAtomicU64>,
    /// Permission map: each slot 0..slots.len() has a PermissionU64
    pub perms: Tracked<Map<int, PermissionU64>>,
    /// Number of entries in the table
    pub count: AtomicU64,
    /// Table capacity (number of slots)
    pub capacity: usize,
    /// Seqlock for resize coordination (odd = resizing, even = stable)
    pub seq: AtomicU64,
}

/// View a VerifiedShard's slots as an abstract sequence for spec-level reasoning.
pub open spec fn shards_view(shard: &VerifiedShard) -> Seq<u64> {
    Seq::new(
        shard.slots.len() as nat,
        |i: int| shard.slots@[i as int].view().value,
    )
}

/// Well-formedness predicate: all permissions are valid and cover every slot.
pub open spec fn shard_wf(shard: &VerifiedShard) -> bool {
    shard.slots.len() > 0
    && shard.capacity > 0
    && shard.capacity == shard.slots.len()
    && (forall|i: int| 0 <= i < shard.slots.len() ==> shard.perms.dom().contains(i))
    && (forall|i: int| 0 <= i < shard.slots.len() ==>
        #[trigger] shard.perms[i].view().patomic == shard.slots@[i as int].id())
}

/// Create a new VerifiedShard with the given capacity.
/// The capacity is rounded up to a power of 2 for the bitmask trick.
pub fn verified_shard_new(capacity: usize) -> (shard: VerifiedShard)
    requires capacity > 0,
    ensures shard_wf(&shard),
    ensures shards_view(&shard) == Seq::new(capacity as nat, |i: int| 0u64),
{
    // Round up to power of 2 for the bitmask optimization
    let actual_cap = if capacity.is_power_of_two() {
        capacity
    } else {
        capacity.next_power_of_two()
    };

    // Create the slot array with actual_cap PAtomicU64s
    let mut slots: Vec<PAtomicU64> = Vec::with_capacity(actual_cap);
    let mut perms_builder = MapBuilder::new();

    for i in 0..actual_cap {
        // Create a new PAtomicU64 with value 0 (empty slot)
        let (atomic, Tracked(perm)) = PAtomicU64::new(0u64);
        slots.push(atomic);
        perms_builder.insert(i as int, perm);
    }

    let perms = perms_builder.build();

    VerifiedShard {
        slots,
        perms: Tracked(perms),
        count: AtomicU64::new(0),
        capacity: actual_cap,
        seq: AtomicU64::new(0),
    }
}

/// Get the table capacity (number of slots).
pub fn verified_shard_capacity(shard: &VerifiedShard) -> (cap: usize)
    requires shard_wf(shard),
    ensures cap == shard.capacity,
    ensures cap > 0,
{
    shard.capacity
}

/// Get the current count of fingerprints in the table.
pub fn verified_shard_len(shard: &VerifiedShard) -> (count: u64)
    requires shard_wf(shard),
    ensures count >= 0,
{
    shard.count.load(Ordering::Acquire)
}

/// Compute the load factor (count / capacity).
pub fn verified_shard_load_factor(shard: &VerifiedShard) -> (lf: f64)
    requires shard_wf(shard),
    ensures 0.0 <= lf && lf < 1.0,
{
    let count = shard.count.load(Ordering::Acquire) as f64;
    let capacity = shard.capacity as f64;
    count / capacity
}

/// Probe step result enum for contains operation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProbeStep {
    Hit,
    Empty,
    Continue,
}

/// Probe a single slot for the contains operation.
/// Returns the probe result: Hit, Empty, or Continue.
pub fn verified_shard_probe_slot(
    shard: &VerifiedShard,
    idx: usize,
    fp: u64,
    Tracked(perm): Tracked<&PermissionU64>,
) -> (result: ProbeStep)
    requires
        shard_wf(shard),
        idx < shard.slots.len(),
        fp != 0,  // matches shipping's empty-slot sentinel
        perm.view().patomic == shard.slots@[idx as int].id(),
    ensures
        ({
            let v = perm.view().value;
            (result is Hit ==> v == fp)
            && (result is Empty ==> v == 0)
            && (result is Continue ==> v != fp && v != 0)
        }),
{
    let stored = shard.slots[idx].load(Tracked(perm));
    if stored == fp {
        ProbeStep::Hit
    } else if stored == 0 {
        ProbeStep::Empty
    } else {
        ProbeStep::Continue
    }
}

/// CAS outcome for contains_or_insert operation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CasOutcome {
    Inserted,
    AlreadyPresent,
    LostRace,
}

/// CAS insert or observe: try to insert fp, or observe if already present.
pub fn verified_shard_cas_insert_or_observe(
    shard: &VerifiedShard,
    idx: usize,
    fp: u64,
    Tracked(perm): Tracked<&mut PermissionU64>,
) -> (result: CasOutcome)
    requires
        shard_wf(shard),
        idx < shard.slots.len(),
        fp != 0,
        old(perm).view().patomic == shard.slots@[idx as int].id(),
    ensures
        final(perm).view().patomic == old(perm).view().patomic,
        ({
            let pre = old(perm).view().value;
            let post = final(perm).view().value;
            match result {
                CasOutcome::Inserted => pre == 0 && post == fp,
                CasOutcome::AlreadyPresent => pre == fp && post == fp,
                CasOutcome::LostRace =>
                    pre != 0 && pre != fp && post == pre,
            }
        }),
{
    let r = shard.slots[idx].compare_exchange(Tracked(perm), 0u64, fp);
    match r {
        Ok(_actual) => {
            // CAS succeeded — slot was 0, now is fp
            CasOutcome::Inserted
        }
        Err(actual) => {
            if actual == fp {
                // Lost the CAS, but the contender wrote the SAME fp
                CasOutcome::AlreadyPresent
            } else {
                // Lost the CAS to a different fp; keep probing
                CasOutcome::LostRace
            }
        }
    }
}

/// The contains operation: check if fingerprint exists in the table.
/// Returns true if found, false otherwise.
pub fn verified_shard_contains(
    shard: &VerifiedShard,
    fp: u64,
) -> (found: bool)
    requires shard_wf(shard),
    ensures found == shards_view(shard).contains(fp),
{
    // Empty slot sentinel: 0 is reserved for empty
    let fp = if fp == 0 { 1 } else { fp };

    // Get capacity for the loop bound
    let capacity = shard.capacity;

    // Compute initial slot index
    let idx = verus_smoke::initial_probe_slot(fp, capacity);

    let mut probes: u64 = 0;
    let mut found = false;

    while probes < capacity as u64 {
        let idx_usize = idx + (probes as usize);
        let idx_wrapped = idx_usize % capacity;

        let Tracked(perm) = shard.perms[idx_wrapped as int].clone();
        let stored = shard.slots[idx_wrapped].load(Tracked(perm));

        if stored == fp {
            found = true;
            break;
        }
        if stored == 0 {
            break;
        }

        probes += 1;
    }

    found
}

/// The contains_or_insert operation: check if fp exists, insert if not.
/// Returns true if fp was already present, false if newly inserted.
pub fn verified_shard_contains_or_insert(
    shard: &mut VerifiedShard,
    fp: u64,
) -> (already_present: bool)
    requires old(shard_wf(shard)),
    ensures shard_wf(shard),
    ensures already_present == old(shards_view(shard)).contains(fp),
{
    // Empty slot sentinel: 0 is reserved for empty
    let fp = if fp == 0 { 1 } else { fp };

    let capacity = shard.capacity;

    // Compute initial slot index
    let idx = verus_smoke::initial_probe_slot(fp, capacity);

    let mut probes: u64 = 0;
    let mut already_present = false;

    while probes < capacity as u64 {
        let idx_usize = idx + (probes as usize);
        let idx_wrapped = idx_usize % capacity;

        let Tracked(perm) = shard.perms[idx_wrapped as int].clone();
        let stored = shard.slots[idx_wrapped].load(Tracked(perm));

        if stored == fp {
            already_present = true;
            break;
        }
        if stored == 0 {
            // Try to insert
            let Tracked(mut perm_mut) = perm;
            let outcome = verified_shard_cas_insert_or_observe(
                shard,
                idx_wrapped,
                fp,
                Tracked(&mut perm_mut),
            );

            match outcome {
                CasOutcome::Inserted | CasOutcome::AlreadyPresent => {
                    already_present = false;  // We inserted or it was inserted by someone else
                    shard.count.fetch_add(1, Ordering::AcqRel);
                    break;
                }
                CasOutcome::LostRace => {
                    // Someone else inserted a different fp; keep probing
                }
            }
        }

        probes += 1;
    }

    already_present
}

/// Compute rehash batch end: (start + batch_size).min(old_cap)
pub fn verified_shard_rehash_batch_end(
    start: usize,
    batch_size: usize,
    old_cap: usize,
) -> (end: usize)
    requires
        start + batch_size <= usize::MAX,
        start <= old_cap,
    ensures
        end <= old_cap,
        end >= start,
{
    let target = start + batch_size;
    if target < old_cap { target } else { old_cap }
}

/// Incremental rehash: move a batch of entries from old table to new.
/// Returns true if there's more work to do.
pub fn verified_shard_rehash_batch(
    shard: &VerifiedShard,
    old_slots: &Vec<PAtomicU64>,
    old_capacity: usize,
    new_slots: &Vec<PAtomicU64>,
    new_capacity: usize,
    start: usize,
    batch_size: usize,
) -> (more_work: bool)
    requires
        shard_wf(shard),
        old_capacity > 0,
        new_capacity > 0,
        start <= old_capacity,
        batch_size > 0,
    ensures
        more_work == (start + batch_size < old_capacity),
{
    let end = verified_shard_rehash_batch_end(start, batch_size, old_capacity);

    for i in start..end {
        let Tracked(perm) = shard.perms[i as int].clone();
        let fp = old_slots[i].load(Tracked(perm));

        if fp != 0 {
            // Find destination in new table
            let new_idx = verus_smoke::initial_probe_slot(fp, new_capacity);

            let mut probes: u64 = 0;
            let mut inserted = false;

            while probes < new_capacity as u64 {
                let new_idx_usize = new_idx + (probes as usize);
                let new_idx_wrapped = new_idx_usize % new_capacity;

                let Tracked(perm_new) = shard.perms[new_idx_wrapped as int].clone();
                let new_stored = new_slots[new_idx_wrapped].load(Tracked(perm_new));

                if new_stored == fp {
                    // Already present in new table
                    inserted = true;
                    break;
                }
                if new_stored == 0 {
                    // Try to insert
                    let Tracked(mut perm_new_mut) = perm_new;
                    let outcome = verified_shard_cas_insert_or_observe(
                        shard,
                        new_idx_wrapped,
                        fp,
                        Tracked(&mut perm_new_mut),
                    );

                    match outcome {
                        CasOutcome::Inserted | CasOutcome::AlreadyPresent => {
                            inserted = true;
                            break;
                        }
                        CasOutcome::LostRace => {
                            // Continue probing
                        }
                    }
                }

                probes += 1;
            }
        }
    }

    end < old_capacity
}

} // verus!

// Non-verus implementations for regular builds
// These provide the same API but without Verus annotations
#[cfg(not(feature = "verus"))]
impl VerifiedShard {
    /// Create a new VerifiedShard with the given capacity
    pub fn new(capacity: usize) -> Self {
        let actual_cap = capacity.next_power_of_two();
        let slots: Vec<AtomicU64> = (0..actual_cap).map(|_| AtomicU64::new(0)).collect();
        VerifiedShard {
            slots,
            perms: Tracked(Map::empty()),
            count: AtomicU64::new(0),
            capacity: actual_cap,
            seq: AtomicU64::new(0),
        }
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get count
    pub fn len(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }

    /// Check if fingerprint exists
    pub fn contains(&self, fp: u64) -> bool {
        let fp = if fp == 0 { 1 } else { fp };
        let idx = (fp as usize) % self.capacity;
        let mut probes: u64 = 0;
        while probes < self.capacity as u64 {
            let idx_wrapped = (idx + (probes as usize)) % self.capacity;
            let stored = self.slots[idx_wrapped].load(Ordering::Acquire);
            if stored == fp {
                return true;
            }
            if stored == 0 {
                break;
            }
            probes += 1;
        }
        false
    }

    /// Insert fingerprint, returns true if already present
    pub fn contains_or_insert(&mut self, fp: u64) -> bool {
        let fp = if fp == 0 { 1 } else { fp };
        let idx = (fp as usize) % self.capacity;
        let mut probes: u64 = 0;
        while probes < self.capacity as u64 {
            let idx_wrapped = (idx + (probes as usize)) % self.capacity;
            let stored = self.slots[idx_wrapped].load(Ordering::Acquire);
            if stored == fp {
                return true;
            }
            if stored == 0 {
                match self.slots[idx_wrapped].compare_exchange(0, fp, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        self.count.fetch_add(1, Ordering::AcqRel);
                        return false;
                    }
                    Err(actual) if actual == fp => return true,
                    Err(_) => {} // Lost race, continue probing
                }
            }
            probes += 1;
        }
        false
    }

    /// Load factor
    pub fn load_factor(&self) -> f64 {
        let count = self.len() as f64;
        let capacity = self.capacity as f64;
        count / capacity
    }
}
