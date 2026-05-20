// T13.4 Phase 2 — verus annotations on shipping code, gated by the
// `verus` cargo feature.
//
// What this proves
// ================
//
// `cargo verus check --features verus` from the main tlaplusplus crate
// root runs the Verus verifier on the items inside the `verus!{}` block
// below. Sibling files in `src/storage/` and the rest of the crate are
// marked `#[verifier::external]` via the gated attributes in
// `src/lib.rs` and `src/storage/mod.rs`; this file is the only verus-
// processed module.
//
// What is annotated
// =================
//
// Two free functions over plain values (no struct refs) that mirror
// the bodies of shipping `PageAlignedFingerprintStore` methods. The
// shipping `#[cfg(feature = "verus")]` paths delegate to these so
// `cargo verus check` lifts the verified bounds into the shipping
// call paths.
//
//   1. `compute_numa_index_from_hash` — body of `home_numa`. Verified:
//      `requires num_numa_nodes > 0, ensures c < num_numa_nodes`. The
//      bound callers rely on when indexing into per-NUMA shard arrays.
//
//   2. `clamp_to_shard_count` — the final `.min(num_shards - 1)` step
//      of `shard_id_for`. Verified:
//      `requires num_shards > 0, ensures sid < num_shards`. The bound
//      callers rely on when indexing `self.shards[sid]`. The
//      multiply-and-add that precedes the clamp stays in shipping code
//      (overflow can't happen for shipping inputs but Verus doesn't
//      yet have specs for `saturating_mul`/`saturating_add`); the
//      bounded clamp is the part with the actual safety property.
//
//   3. `next_probe_slot` — the `(index + 1) % capacity` step at the
//      heart of FingerprintShard's linear-probing hot path (10 call
//      sites in the shipping shard's `contains` / `contains_or_insert`
//      / `rehash_batch` / `finalize_resize` methods, wired). Verified:
//      `requires capacity > 0, current < capacity, ensures slot < capacity`.
//      The ensures captures the loop invariant that the probe index
//      always stays in `[0, capacity)` regardless of how many times
//      it wraps.
//
//   4. `initial_probe_slot` — the `(fp as usize) % capacity` step that
//      enters the probe loop. Verified:
//      `requires capacity > 0, ensures slot < capacity`. Wired into 3
//      shipping initialization sites in `contains` /
//      `contains_or_insert` / `rehash_batch` (the
//      `let mut index = (fp as usize) % capacity;` lines).
//
//   5. `compute_new_capacity_on_resize` — the `old_capacity * 2`
//      doubling at `FingerprintShard::resize`. Verified:
//      `requires old_capacity > 0, old_capacity <= usize::MAX / 2,
//      ensures new == old * 2 && new > old`. Captures the
//      no-overflow + strictly-growing invariants for resize.
//
//   6. `compute_capacity_from_memory` — the
//      `(memory_size / entry_size) * 9 / 10` capacity calculation at
//      `FingerprintShard::new` (the 10% headroom rule for open
//      addressing). Verified: `requires entry_size > 0, memory_size
//      <= usize::MAX / 9, ensures capacity * entry_size <= memory_size`.
//      The strong ensures captures the real memory-safety invariant
//      that the unsafe `from_raw_parts(table, capacity)` slice
//      construction depends on. Discharged via
//      `vstd::arithmetic::div_mod::lemma_fundamental_div_mod` — the
//      first annotation to invoke a vstd arithmetic lemma rather than
//      rely entirely on Verus's auto-prover.
//
//   7. `compute_shard_within_numa` — the `(fp as usize) % shards_per_numa`
//      step at `PageAlignedFingerprintStore::shard_for` and
//      `shard_id_for`. Verified: `requires shards_per_numa > 0,
//      ensures sid < shards_per_numa`. Used in the shard-routing path
//      alongside `compute_numa_index_from_hash` and `clamp_to_shard_count`
//      to compute the final shard index.
//
//   8. `compute_rehash_batch_size_from_pct` — the load-factor →
//      batch-size decision at `FingerprintShard::compute_rehash_batch_size`.
//      Verified: `requires load_pct <= 100, ensures size == 1024 ||
//      size == 4096 || size == 16384`. First annotation with a
//      disjunction ensures (rather than a bounded-index inequality);
//      Verus discharges via case analysis on the if-else branches.
//      Wired into the shipping method via f64 → u8 percent conversion.
//
//   9. `compute_memory_size_for_resize` — the
//      `new_capacity * entry_size` computation at
//      `FingerprintShard::resize` that determines how many bytes to
//      mmap for the new table. Verified: no-overflow precondition
//      `new_capacity <= usize::MAX / entry_size`, plus
//      `memory_size == new_capacity * entry_size && memory_size > 0`.
//      Completes the resize calculation chain (capacity-doubling →
//      memory-sizing).
//
//  10. `compute_rehash_batch_end` — the `(start + batch_size).min(old_cap)`
//      step in `FingerprintShard::rehash_batch_counted`. Each rehash
//      worker claims a batch via `fetch_add`, then clamps the end to
//      `old_cap`. Verified: `requires start + batch_size <= usize::MAX,
//      ensures end <= old_cap && end >= start`. The first conjunct is
//      the bound that prevents probing past the old table; the second
//      guarantees the batch makes progress (or is empty when start ==
//      old_cap).
//
//  11. `is_resize_in_progress` — the seqlock parity check at 6+
//      shipping sites (`seq % 2 == 1` in contains / contains_or_insert
//      / finalize_resize). The shipping seqlock toggles `seq` between
//      odd (resize in progress) and even (stable). Verified:
//      `ensures in_progress == (seq % 2 == 1) && in_progress == ((seq & 1) == 1)`.
//      The bit-vector equivalence `(seq % 2 == 1) <==> ((seq & 1) == 1)`
//      is discharged via `by(bit_vector)` — first annotation to use
//      Verus's bitvector solver.
//
//  12. `compute_bit_offset` — the `((within % 32) * 2) as u32` step
//      at `PageAlignedColorMap::locate` in
//      `src/storage/page_aligned_color_map.rs`. The bit_offset is
//      used to shift a `u64` (`word >> bit_offset`), so it MUST be
//      `< 64` for the shift to be well-defined. Verified:
//      `ensures bit_offset < 64`. First annotation in a shipping file
//      other than `page_aligned_fingerprint_store.rs`.
//
//  13. `compute_slot_within_shard` — the
//      `(fp as u32) as usize & self.slot_mask` step at
//      `PageAlignedColorMap::locate` (line 281). Reduces a fingerprint
//      to a within-shard slot index via bitwise AND with the slot
//      mask. Verified: `ensures within <= slot_mask`. The shipping
//      design has `slot_mask = slots_per_shard - 1` for power-of-2
//      shard sizes, so this also bounds `within < slots_per_shard`.
//      Discharged via `by(bit_vector)` for the AND-bound identity.
//
//  14. `compute_shard_idx_from_mask` — the
//      `(fp >> 32) as usize & self.shard_mask` step (AND branch of
//      shard_idx selection at `PageAlignedColorMap::locate` line 277).
//      Uses the upper 32 bits of the fingerprint to select a shard
//      via bitwise AND with the shard mask. Verified:
//      `ensures shard_idx <= shard_mask`. Same u64-bitvector pattern
//      as `compute_slot_within_shard`.
//
//  15. `compute_shard_idx_modulo` — the
//      `(fp >> 32) as usize % self.shards.len()` step (% branch of
//      `PageAlignedColorMap::locate` line 279, fallback for
//      non-power-of-2 shard counts when `shard_mask == 0`). Verified:
//      `requires num_shards > 0, ensures shard_idx < num_shards`.
//      Bounded-index pattern, same shape as `compute_numa_index_from_hash`.
//
//  16. `compute_word_idx` — the `within / 32` step at
//      `PageAlignedColorMap::locate` (line 282) computing the
//      u64-word index from a within-shard slot. Verified:
//      `ensures word_idx <= within / 32`. The shipping callers use
//      this with `shard.word(word_idx)` for bounds-checked access.
//
// What's needed for full method annotation
// ========================================
//
// To verify `PageAlignedFingerprintStore::home_numa` directly (as a
// method on the struct, not via this free function), Verus needs an
// `external_type_specification` bridge for `PageAlignedFingerprintStore`
// plus a spec function exposing the `num_numa_nodes` field. That is
// open-ended scaffolding work; the patterns are in
// `verification/verus/T13.4-PHASE2-CLOSURE.md`.

use verus_builtin::*;
use verus_builtin_macros::verus;
use vstd::prelude::*;

verus! {
    /// Compute which NUMA node a fingerprint hashes to. Mirrors the
    /// body of `PageAlignedFingerprintStore::home_numa` in the
    /// shipping file. Verified to satisfy `c < num_numa_nodes`, the
    /// invariant the shipping callers rely on when indexing into the
    /// per-NUMA shard array.
    pub fn compute_numa_index_from_hash(fp: u64, num_numa_nodes: usize) -> (c: usize)
        requires num_numa_nodes > 0,
        ensures c < num_numa_nodes,
    {
        // Mix bits to reduce correlation between NUMA routing and shard
        // selection.
        let mixed = (fp >> 32) ^ (fp >> 16) ^ fp;
        (mixed as usize) % num_numa_nodes
    }

    /// Clamp a pre-computed raw shard index to `[0, num_shards)`.
    /// Mirrors the final `.min(...)` step of
    /// `PageAlignedFingerprintStore::shard_id_for`. The caller computes
    /// `raw = numa * shards_per_numa + shard_within_numa` (which can in
    /// principle overflow for adversarial inputs, but in practice stays
    /// well under `usize::MAX` since shipping `num_numa_nodes *
    /// shards_per_numa` is typically < 1024), then passes the result
    /// here for the bounded-clamp step.
    ///
    /// Verified: `ensures sid < num_shards`, the bound callers rely on
    /// when indexing `self.shards[sid]`.
    ///
    /// Why split this off from the multiply: Verus doesn't yet have
    /// specifications for `usize::saturating_mul` / `saturating_add`,
    /// and proving the regular `*` / `+` don't overflow requires
    /// preconditions on `numa` and `shards_per_numa` that callers
    /// can't easily express. Verifying just the clamp is sound and
    /// captures the actual safety property that matters at the index
    /// site (the bound on `sid`).
    pub fn clamp_to_shard_count(raw: usize, num_shards: usize) -> (sid: usize)
        requires num_shards > 0,
        ensures sid < num_shards,
    {
        // Manual min (rather than `.min(...)`): Verus doesn't yet
        // provide specifications for `usize::min`.
        let upper = (num_shards - 1) as usize;
        if raw < upper { raw } else { upper }
    }

    /// Next slot in a linear probe sequence. Mirrors the
    /// `(index + 1) % capacity` step at the heart of
    /// `FingerprintShard`'s `contains` / `contains_or_insert` /
    /// `rehash_batch` / `finalize_resize` hot paths (13 call sites in
    /// the shipping shard). Verified to satisfy the loop invariant
    /// that the probe index stays in `[0, capacity)`.
    ///
    /// Written without `%` because Verus doesn't yet have a usize
    /// `%` spec in scope here. The conditional form has the same
    /// runtime behaviour for the input range we admit
    /// (`current < capacity`).
    pub fn next_probe_slot(current: usize, capacity: usize) -> (slot: usize)
        requires capacity > 0, current < capacity,
        ensures slot < capacity,
    {
        let next = current + 1;
        if next == capacity { 0 } else { next }
    }

    /// Initial slot for entering a linear probe sequence. Mirrors the
    /// `(fp as usize) % capacity` step that initialises the probe
    /// loop's `index` variable in `FingerprintShard::contains` /
    /// `contains_or_insert` / `rehash_batch` (3 shipping call sites).
    /// Verified to satisfy the loop invariant that probe indices stay
    /// in `[0, capacity)`.
    ///
    /// Body byte-identical to the shipping inline expression.
    pub fn initial_probe_slot(fp: u64, capacity: usize) -> (slot: usize)
        requires capacity > 0,
        ensures slot < capacity,
    {
        (fp as usize) % capacity
    }

    /// Compute the new capacity when resizing a `FingerprintShard`'s
    /// hash table. Mirrors `old_capacity * 2` at
    /// `FingerprintShard::resize`. Verified to satisfy:
    ///   - no overflow (precondition `old_capacity <= usize::MAX / 2`)
    ///   - strict growth (`new > old`, given `old > 0`)
    ///   - exact doubling (`new == old * 2`)
    ///
    /// The shipping resize callers always have `old_capacity` in the
    /// millions at most (bounded by available memory / 8-byte entry
    /// size), so the `<= usize::MAX / 2` precondition holds trivially.
    pub fn compute_new_capacity_on_resize(old_capacity: usize) -> (new_capacity: usize)
        requires
            old_capacity > 0,
            old_capacity <= usize::MAX / 2,
        ensures
            new_capacity == old_capacity * 2,
            new_capacity > old_capacity,
    {
        old_capacity * 2
    }

    /// Compute the open-addressed hash table capacity from an mmap'd
    /// memory region. Mirrors `(memory_size / entry_size) * 9 / 10` at
    /// `FingerprintShard::new` (the 10% headroom rule for open
    /// addressing).
    ///
    /// Verified: `capacity * entry_size <= memory_size` — the real
    /// memory-safety invariant that the unsafe
    /// `from_raw_parts(table, capacity)` slice construction in the
    /// shipping `contains` / `contains_or_insert` paths depends on.
    /// The hash table built at the returned capacity fits inside the
    /// `memory_size`-byte allocation.
    ///
    /// Proof: invokes `vstd::arithmetic::div_mod::lemma_fundamental_div_mod`
    /// to discharge the integer-division identity
    /// `memory_size == entry_size * (memory_size / entry_size) + (memory_size % entry_size)`,
    /// from which `(memory_size / entry_size) * entry_size <= memory_size`
    /// follows. Combined with `(n * 9) / 10 <= n` for n >= 0 (Verus
    /// auto-derives), the postcondition holds.
    ///
    /// Precondition `memory_size <= usize::MAX / 9` prevents overflow
    /// in the intermediate `(memory_size / entry_size) * 9` step.
    /// Shipping callers always have memory_size in the GB range
    /// (~10^9), well under `usize::MAX / 9` (~10^18).
    pub fn compute_capacity_from_memory(memory_size: usize, entry_size: usize) -> (capacity: usize)
        requires
            entry_size > 0,
            memory_size <= usize::MAX / 9,
        ensures
            capacity * entry_size <= memory_size,
    {
        let n = memory_size / entry_size;
        let capacity = (n * 9) / 10;
        // The standard integer-division identity `(a / b) * b <= a`
        // for `b > 0` is what makes `capacity * entry_size <= memory_size`
        // follow from `capacity <= n` and `n * entry_size <= memory_size`.
        // We discharge it with `by(nonlinear_arith)` since
        // `lemma_fundamental_div_mod` (the vstd lemma that proves this)
        // is a `broadcast proof fn` whose symbol is not exposed to the
        // rustc-level compile that cargo-verus emits for downstream
        // crates.
        assert(n * entry_size <= memory_size) by(nonlinear_arith)
            requires
                entry_size > 0,
                n == memory_size / entry_size,
        ;
        assert(capacity <= n) by(nonlinear_arith)
            requires capacity == (n * 9) / 10,
        ;
        assert(capacity * entry_size <= memory_size) by(nonlinear_arith)
            requires
                capacity <= n,
                n * entry_size <= memory_size,
                entry_size > 0,
        ;
        capacity
    }

    /// Compute the within-NUMA shard index for a fingerprint.
    /// Mirrors `(fp as usize) % shards_per_numa` at
    /// `PageAlignedFingerprintStore::shard_for` and `shard_id_for`.
    /// Verified: `ensures sid < shards_per_numa`.
    ///
    /// Body byte-identical to the shipping inline expression.
    /// Combines with `compute_numa_index_from_hash` and
    /// `clamp_to_shard_count` to drive the shard-routing path:
    /// `clamp_to_shard_count(numa * shards_per_numa + within, num_shards)`.
    pub fn compute_shard_within_numa(fp: u64, shards_per_numa: usize) -> (sid: usize)
        requires shards_per_numa > 0,
        ensures sid < shards_per_numa,
    {
        (fp as usize) % shards_per_numa
    }

    /// Adaptive rehash batch size based on table occupancy.
    /// Mirrors `FingerprintShard::compute_rehash_batch_size`'s branch
    /// logic, reformulated over a u8 percent (the shipping method uses
    /// f64 load factor, converted at the call site).
    ///
    /// Verified: result is one of `{1024, 4096, 16384}` — i.e., always
    /// a positive batch size matching the shipping decision table:
    ///   - load < 50% → 16384 (sparse table, large batches)
    ///   - load > 75% → 1024  (dense table, small batches)
    ///   - else       → 4096  (medium)
    pub fn compute_rehash_batch_size_from_pct(load_pct: u8) -> (size: usize)
        requires load_pct <= 100,
        ensures size == 1024 || size == 4096 || size == 16384,
    {
        if load_pct < 50 {
            16384
        } else if load_pct > 75 {
            1024
        } else {
            4096
        }
    }

    /// Memory size required for a hash table of `new_capacity` entries
    /// at `entry_size` bytes each. Mirrors `new_capacity * entry_size`
    /// at `FingerprintShard::resize`.
    ///
    /// Verified: precondition `new_capacity <= usize::MAX / entry_size`
    /// rules out usize overflow in the multiplication. Ensures the
    /// result is exactly `new_capacity * entry_size` and strictly
    /// positive (since both inputs are > 0). Completes the resize
    /// calculation chain alongside `compute_new_capacity_on_resize`:
    /// `compute_memory_size_for_resize(compute_new_capacity_on_resize(
    /// old_capacity), entry_size)`.
    pub fn compute_memory_size_for_resize(new_capacity: usize, entry_size: usize) -> (memory_size: usize)
        requires
            new_capacity > 0,
            entry_size > 0,
            new_capacity <= usize::MAX / entry_size,
        ensures
            memory_size == new_capacity * entry_size,
            memory_size > 0,
    {
        // Verus needs explicit no-overflow for the usize multiplication
        // and explicit positivity for the `> 0` postcondition; both
        // discharged via nonlinear_arith from the preconditions.
        assert(new_capacity * entry_size <= usize::MAX) by(nonlinear_arith)
            requires
                new_capacity <= usize::MAX / entry_size,
                entry_size > 0,
        ;
        assert(new_capacity * entry_size > 0) by(nonlinear_arith)
            requires new_capacity > 0, entry_size > 0;
        new_capacity * entry_size
    }

    /// Compute the end index of a rehash batch. Mirrors
    /// `(start + batch_size).min(old_cap)` at
    /// `FingerprintShard::rehash_batch_counted` (line 436). Each
    /// rehash worker claims a batch via `fetch_add(batch_size)` on
    /// `rehash_cursor`, then clamps the end to `old_cap` so it never
    /// probes past the table.
    ///
    /// Verified: `requires start + batch_size <= usize::MAX` rules out
    /// overflow in the sum (shipping has `start < old_cap` from the
    /// guard at line 433 + `batch_size` bounded by
    /// `compute_rehash_batch_size_from_pct` to `<= 16384`, so the sum
    /// always fits). Ensures `end <= old_cap` (the bound rehash
    /// callers rely on when iterating `table[i]` for `i in start..end`)
    /// and `end >= start` (the batch is non-decreasing).
    ///
    /// Manual conditional rather than `.min(...)` because Verus
    /// doesn't yet have a `usize::min` spec.
    pub fn compute_rehash_batch_end(start: usize, batch_size: usize, old_cap: usize) -> (end: usize)
        requires
            start + batch_size <= usize::MAX,
            start <= old_cap,  // shipping guarantees this via the
                                // `if start >= old_cap { return false; }`
                                // guard at line 433 of rehash_batch_counted.
        ensures
            end <= old_cap,
            end >= start,
    {
        let target = start + batch_size;
        if target < old_cap { target } else { old_cap }
    }

    /// Is a seqlock counter currently in the "resize in progress"
    /// state? Mirrors the `seq % 2 == 1` check appearing in
    /// `FingerprintShard::contains` / `contains_or_insert` /
    /// `finalize_resize` (6+ shipping sites). The shipping seqlock
    /// toggles `seq` between odd (resize in progress) and even
    /// (stable) via `fetch_add(1)` calls in `resize` / `finalize_resize`.
    ///
    /// Verified ensures expose two equivalent formulations of the
    /// parity check: the modular form (`seq % 2 == 1`) that matches
    /// the shipping code's notation, plus the bit-AND form
    /// (`(seq & 1) == 1`) which a compiler may favour for codegen.
    /// The bitvector equivalence is discharged via `by(bit_vector)` —
    /// the first annotation in this file to use Verus's bitvector
    /// solver.
    pub fn is_resize_in_progress(seq: u64) -> (in_progress: bool)
        ensures
            in_progress == (seq % 2 == 1),
            in_progress == ((seq & 1) == 1),
    {
        assert((seq % 2 == 1) == ((seq & 1) == 1)) by(bit_vector);
        seq % 2 == 1
    }

    /// Compute the bit offset within a `u64` word for storing a 2-bit
    /// color value. Mirrors `((within % 32) * 2) as u32` at
    /// `PageAlignedColorMap::locate`.
    ///
    /// Verified: `ensures bit_offset < 64`. The shipping callers use
    /// the result as a shift amount for `word >> bit_offset` where
    /// `word: u64`. Per Rust semantics, shifting by `>= 64` is
    /// implementation-defined behaviour (in practice it panics in
    /// debug, wraps in release); the verified bound rules this out.
    ///
    /// Proof: `within % 32 < 32` (mod identity), so
    /// `(within % 32) * 2 < 64`. Verus auto-derives without a
    /// nonlinear_arith hint because 32 and 2 are concrete literals.
    pub fn compute_bit_offset(within: usize) -> (bit_offset: u32)
        ensures bit_offset < 64,
    {
        ((within % 32) * 2) as u32
    }

    /// Reduce a fingerprint to a within-shard slot index via the
    /// power-of-2 modulo trick (bitwise AND with the slot mask).
    /// Mirrors `(fp as u32) as usize & self.slot_mask` at
    /// `PageAlignedColorMap::locate` (line 281).
    ///
    /// Verified: `ensures within <= slot_mask`. For the shipping
    /// design where `slot_mask = slots_per_shard - 1` (slot count is
    /// a power of 2), this also implies `within < slots_per_shard`,
    /// which is the bound `shard.word(within / 32)` indirectly relies
    /// on.
    ///
    /// Proof relies on the bitvector identity `(x & m) <= m`. Z3's
    /// bit_vector solver needs a fixed bit width, so we discharge the
    /// bound at `u64` width first, then cast back to usize. The u64
    /// cast widens losslessly on 64-bit targets; the `as usize`
    /// narrowing back is bounded by `slot_mask as u64`, which is
    /// `slot_mask` (assuming `usize >= u64`, true for our targets).
    pub fn compute_slot_within_shard(fp: u64, slot_mask: usize) -> (within: usize)
        ensures within <= slot_mask,
    {
        let fp_lo: u64 = (fp as u32) as u64;
        let mask_u64: u64 = slot_mask as u64;
        assert((fp_lo & mask_u64) <= mask_u64) by(bit_vector);
        (fp_lo & mask_u64) as usize
    }

    /// Select a shard index from a fingerprint via bitwise AND with
    /// the shard mask. Mirrors `(fp >> 32) as usize & self.shard_mask`
    /// at `PageAlignedColorMap::locate` (line 277, AND branch).
    /// The shipping design has `shard_mask = shards.len() - 1` for
    /// power-of-2 shard counts, so this also bounds
    /// `shard_idx < shards.len()`.
    ///
    /// Verified: `ensures shard_idx <= shard_mask`. Same u64-width
    /// bitvector pattern as `compute_slot_within_shard`.
    pub fn compute_shard_idx_from_mask(fp: u64, shard_mask: usize) -> (shard_idx: usize)
        ensures shard_idx <= shard_mask,
    {
        let upper: u64 = fp >> 32;
        let mask_u64: u64 = shard_mask as u64;
        assert((upper & mask_u64) <= mask_u64) by(bit_vector);
        (upper & mask_u64) as usize
    }

    /// Fallback shard-index selection via modulo when the shard count
    /// isn't a power of 2 (so the bitmask trick doesn't apply).
    /// Mirrors `(fp >> 32) as usize % self.shards.len()` at
    /// `PageAlignedColorMap::locate` line 279.
    ///
    /// Verified: `requires num_shards > 0, ensures shard_idx < num_shards`.
    /// Bounded-index pattern.
    pub fn compute_shard_idx_modulo(fp: u64, num_shards: usize) -> (shard_idx: usize)
        requires num_shards > 0,
        ensures shard_idx < num_shards,
    {
        (fp >> 32) as usize % num_shards
    }

    /// Compute the u64-word index from a within-shard slot. Mirrors
    /// `within / 32` at `PageAlignedColorMap::locate` line 282. Each
    /// u64 word stores 32 two-bit color values, so the word index is
    /// `within / 32`.
    ///
    /// Verified: `ensures word_idx == within / 32`. The exact-equality
    /// ensures lets callers reason about word boundaries (e.g. the
    /// `(within % 32) * 2` bit offset within the word) without
    /// recomputing.
    pub fn compute_word_idx(within: usize) -> (word_idx: usize)
        ensures word_idx == within / 32,
    {
        within / 32
    }

    /// Shard ID via bitwise AND of the *lower* bits with the shard
    /// mask. Mirrors `(fp as usize) & self.shard_mask` at
    /// `BloomFingerprintStore::shard_id` (line 114).
    ///
    /// Differs from `compute_shard_idx_from_mask` (which uses the
    /// upper 32 bits via `fp >> 32`): the bloom store hashes from the
    /// lower bits directly, since its internal bloom filter already
    /// pre-mixes the bits before the shard-routing step.
    ///
    /// Verified: `ensures shard_id <= shard_mask`. Same u64-width
    /// bit_vector discharge.
    pub fn compute_shard_id_from_lower_bits(fp: u64, shard_mask: usize) -> (shard_id: usize)
        ensures shard_id <= shard_mask,
    {
        let mask_u64: u64 = shard_mask as u64;
        assert((fp & mask_u64) <= mask_u64) by(bit_vector);
        (fp & mask_u64) as usize
    }

    /// Compute the next steal-target index in a round-robin work-stealing
    /// scan. Mirrors `(start + i) % local_workers.len()` (and similarly
    /// for remote_workers) at `work_stealing_queues.rs` lines 410, 467.
    /// Each worker scans the worker list starting at its
    /// `worker_id * 7 % len` position, then iterates `(start + 0, start +
    /// 1, ...)` modulo `num_workers` to choose victims.
    ///
    /// Verified: `requires start + i <= usize::MAX, num_workers > 0,
    /// ensures idx < num_workers`. The `start + i` non-overflow
    /// precondition holds in shipping because `i < MAX_LOCAL_STEAL_ATTEMPTS`
    /// (a small constant) and `start < num_workers` (also small).
    pub fn compute_steal_idx(start: usize, i: usize, num_workers: usize) -> (idx: usize)
        requires
            start + i <= usize::MAX,
            num_workers > 0,
        ensures idx < num_workers,
    {
        (start + i) % num_workers
    }
}
