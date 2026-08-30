#[cfg_attr(feature = "verus", verifier::external)]
pub mod async_fingerprint_writer;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod auto_switching_fingerprint_store;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod bloom_fingerprint_store;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod channel_queue;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod compressed_segments;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod fingerprint_store;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod hybrid_fingerprint_store;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod numa;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod page_aligned_color_map;
// T13.4 full lift Phase A.1 — verified parallel shard struct.
// Contains verus!{} blocks for full method-level annotation.
// Compiled only by cargo verus check --features verus.
#[cfg(feature = "verus")]
pub mod verified_fingerprint_shard;
// T13.4 full lift Phase A.2 — equivalence tests for VerifiedShard vs FingerprintShard.
// Tests that both implementations produce identical results for the same fingerprint stream.
#[cfg(test)]
pub mod verified_fingerprint_shard_equivalence;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod page_aligned_fingerprint_store;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod queue;

// T13.4 Phase 2 — verus annotations on shipping code.
// Only compiled when the `verus` cargo feature is on. Contains
// `verus!{}` blocks that `cargo verus check --features verus` verifies.
// Sibling-module placement (rather than nested inside
// `page_aligned_fingerprint_store`) keeps the rest of the storage tree
// marked `#[verifier::external]` without losing the verified items.
#[cfg(feature = "verus")]
pub mod verus_smoke;
// T13.4 full lift Phase A.1 — verified parallel shard struct. See
// `verification/verus/T13.4-FULL-LIFT-PLAN.md`.
#[cfg(feature = "verus")]
pub mod verified_fp_shard;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod s3_persistence;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod simple_blocking_queue;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod spillable_work_stealing;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod unified_fingerprint_store;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod work_stealing_queues;
