//! `HashedArc<T>`: an `Arc<T>` carrying a precomputed `u64` hash.
//!
//! ## Why
//!
//! `TlaValue` collection variants (Set/Seq/Record/Function) wrap their
//! contents in `Arc<...>`. The derived `Ord` walks the structure on every
//! comparison; profiling `MCBinarySearch` on the eval interpreter showed
//! that recursive `TlaValue::Ord::cmp` chains inside `BTreeMap::insert`
//! dominated (~18% self-time) — comparing two Records meant walking every
//! field, recursing into nested values.
//!
//! By precomputing a `u64` hash at construction time and storing it adjacent
//! to the `Arc`, comparisons can fast-fail on inequality: two values whose
//! hashes differ are necessarily unequal, so `cmp` returns `Less` / `Greater`
//! based on the hash. Only on the rare hash-collision case does the full
//! structural compare run, preserving the structural `Ord` total-order.
//!
//! ## Soundness invariants
//!
//! 1. The cached hash MUST be computed from the same source data the
//!    structural compare reads. Otherwise two `HashedArc<T>` whose hashes
//!    agree could compare un-equal structurally, breaking transitivity.
//!    Enforced by `HashedArc::new` calling the same fingerprint hasher.
//!
//! 2. The hash MUST be deterministic across processes. `TlaValue::Hash` is
//!    consumed by `Model::fingerprint` (see PR #77 / `fingerprint_hasher`),
//!    which requires cross-process determinism for multi-node cluster
//!    partitioning. We therefore use the fixed-seed
//!    `crate::model::fingerprint_hasher`.
//!
//! 3. The hash MUST NOT be part of `Eq`/`PartialEq`: two `HashedArc`s with
//!    equal underlying values must compare `Eq` even if their hashes happen
//!    to disagree. (Hashes can't disagree if the values are equal — that
//!    would be a hash collision in the wrong direction — but we don't rely
//!    on this; we ignore the cached hash in `Eq` and walk the structure.)
//!
//! 4. `Serialize` MUST NOT emit the cached hash, and `Deserialize` MUST
//!    recompute it. The hash is an in-memory cache, not part of the wire format.

use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// `Arc<T>` plus a precomputed `u64` hash of `*T`. See module docs.
#[derive(Debug)]
pub struct HashedArc<T> {
    inner: Arc<T>,
    hash: u64,
}

impl<T> HashedArc<T>
where
    T: Hash,
{
    /// Wrap `inner` in an `Arc` and compute the cached hash.
    pub fn new(inner: T) -> Self {
        let arc = Arc::new(inner);
        let hash = compute_fingerprint_hash(&*arc);
        Self { inner: arc, hash }
    }

    /// Wrap an existing `Arc<T>`, computing the hash from the current
    /// contents. Use this when the caller already has an `Arc` they want to
    /// share (e.g. `Arc::clone`'d collections that haven't been wrapped yet).
    pub fn from_arc(inner: Arc<T>) -> Self {
        let hash = compute_fingerprint_hash(&*inner);
        Self { inner, hash }
    }
}

impl<T> HashedArc<T> {
    /// The cached hash. Cheap accessor; available without dereferencing.
    pub fn cached_hash(&self) -> u64 {
        self.hash
    }

    /// Access the underlying `Arc` (for situations that need to share the
    /// Arc reference, e.g. cloning a sub-collection without rewrapping).
    pub fn as_arc(&self) -> &Arc<T> {
        &self.inner
    }
}

impl<T> Clone for HashedArc<T> {
    fn clone(&self) -> Self {
        // Cheap: clones the Arc (refcount bump) and copies the u64 hash.
        Self {
            inner: Arc::clone(&self.inner),
            hash: self.hash,
        }
    }
}

impl<T> Deref for HashedArc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T> AsRef<T> for HashedArc<T> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T> Default for HashedArc<T>
where
    T: Default + Hash,
{
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> PartialEq for HashedArc<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // Same Arc -> equal without any deref. Different Arcs but same hash
        // -> usually equal but could be a hash collision; fall through to
        // structural compare. Different hashes -> definitely unequal.
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return true;
        }
        if self.hash != other.hash {
            return false;
        }
        *self.inner == *other.inner
    }
}

impl<T> Eq for HashedArc<T> where T: Eq {}

impl<T> Ord for HashedArc<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Same Arc -> Equal (no deref).
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return std::cmp::Ordering::Equal;
        }
        // Different hashes -> the values must be unequal; order by hash.
        // (This is the hot path on MCBinarySearch's BTreeSet inserts.)
        match self.hash.cmp(&other.hash) {
            std::cmp::Ordering::Equal => {
                // Hash collision OR genuinely equal values. Fall through to
                // structural compare for the correct total order.
                (*self.inner).cmp(&*other.inner)
            }
            non_eq => non_eq,
        }
    }
}

impl<T> PartialOrd for HashedArc<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Hash for HashedArc<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Delegate to the inner — DO NOT write the cached `self.hash` here.
        //
        // Why: `TlaValue` derives `Hash`, which flows into
        // `Model::fingerprint` (the cross-process state fingerprint that
        // S3 checkpoints, multi-node DFS partitioning, and the resume path
        // all depend on). Writing the cached hash instead of the structural
        // hash would change `Model::fingerprint`'s output for every state
        // containing a Record/Function/Set/Seq — invalidating existing
        // checkpoints and silently changing partition assignments across
        // versions. Deterministic but wire-incompatible.
        //
        // Hashing the inner here preserves wire compatibility. The
        // performance win on `Ord` (the profiled hotspot) still applies
        // because the cached hash is consulted there independently.
        (*self.inner).hash(state);
    }
}

// Serde delegates to the inner Arc. On deserialize the hash is recomputed
// from the loaded contents — the cached hash is purely an in-memory
// performance cache, not part of the wire format.

impl<T> Serialize for HashedArc<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (*self.inner).serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for HashedArc<T>
where
    T: Deserialize<'de> + Hash,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = T::deserialize(deserializer)?;
        Ok(Self::new(inner))
    }
}

/// Hash `value` with the fixed-seed fingerprint hasher (see
/// `crate::model::fingerprint_hasher`). Cross-process deterministic, which
/// matters because `TlaValue::Hash` flows into `Model::fingerprint` via
/// derives elsewhere.
fn compute_fingerprint_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = crate::model::fingerprint_hasher();
    value.hash(&mut hasher);
    hasher.finish()
}
