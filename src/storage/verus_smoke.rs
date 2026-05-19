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
//      <= usize::MAX / 9, ensures capacity <= memory_size / entry_size`.
//      The stronger property `capacity * entry_size <= memory_size`
//      (the one that directly justifies the unsafe `from_raw_parts`
//      slice construction) needs `vstd::arithmetic::div_mod` lemmas
//      to discharge the integer-division identity; the weaker bound
//      here doesn't require that machinery and follows from the
//      `(x * 9) / 10 <= x` monotonicity that Verus auto-derives.
//
//   7. `compute_shard_within_numa` — the `(fp as usize) % shards_per_numa`
//      step at `PageAlignedFingerprintStore::shard_for` and
//      `shard_id_for`. Verified: `requires shards_per_numa > 0,
//      ensures sid < shards_per_numa`. Used in the shard-routing path
//      alongside `compute_numa_index_from_hash` and `clamp_to_shard_count`
//      to compute the final shard index.
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
    /// Verified: `capacity <= memory_size / entry_size`. Each entry
    /// fits in the allocation; combined with the shipping
    /// `entry_size` constant being `size_of::<HashTableEntry>()`, this
    /// bounds how many entries are reachable through the table
    /// pointer.
    ///
    /// The stronger safety property `capacity * entry_size <= memory_size`
    /// (which would directly justify the unsafe `from_raw_parts` slice
    /// construction) needs Verus's `vstd::arithmetic::div_mod` lemmas
    /// to discharge the integer-division identity `(a / b) * b <= a`;
    /// the weaker bound here doesn't require that machinery. Wiring
    /// the strong form is a follow-up using `lemma_fundamental_div_mod`.
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
            capacity <= memory_size / entry_size,
    {
        (memory_size / entry_size) * 9 / 10
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
}
