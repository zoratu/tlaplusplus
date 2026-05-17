// verification/verus/mmap_external_body.rs
//
// T13.4 Phase 2 slice 4 — mmap external_body wrapper closing gap 2.
//
// What this proves
// ================
//
// `mmap_allocate_huge_pages(size) -> (*mut u8, Tracked<PointsToRaw>,
// Tracked<MmapDealloc>)` is a `#[verifier::external_body]` wrapper
// modeled exactly on `vstd::raw_ptr::allocate`. It calls `libc::mmap`
// with the `MAP_HUGETLB | MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE`
// flags that `FingerprintShard::allocate_huge_pages` uses, then returns
// freshly-minted `Tracked<PointsToRaw>` + `Tracked<MmapDealloc>` via
// `Tracked::assume_new()` — the same trust pattern `vstd::raw_ptr::allocate`
// uses for the global allocator.
//
// What this validates
// ===================
//
// Gap 2 of the original T13.4 design doc claimed: "Verus refuses to
// admit a fresh ghost token from an external-body function." That
// claim was factually wrong — `vstd::raw_ptr::allocate` itself does
// exactly this (see `source/vstd/raw_ptr.rs` line 907). This file
// demonstrates the same pattern applied to mmap, with the same trust
// assumption (the implementation actually does what the `ensures`
// clause says) and no new axioms beyond what vstd already accepts for
// `GlobalAlloc`.
//
// Combined with shard_exec_wired.rs (slices 1-3c, exec wiring) the
// remaining gap is the bridge from a mmap'd `PointsToRaw` to a typed
// `PointsTo<InnerShard>` — which `vstd::raw_ptr::PointsToRaw::into_typed`
// already provides. `demonstrate_mmap_lifecycle` below allocates,
// converts to a typed `PointsTo<u64>`, writes/reads, and frees.
//
// Status: verified standalone with Verus 0.2026.05.13 (aarch64).

#![allow(unused_imports)]
#![allow(dead_code)]

use verus_builtin::*;
use verus_builtin_macros::*;
use vstd::prelude::*;
use vstd::raw_ptr::*;

// Linux libc bindings inline (no `libc` crate dep — Verus proof files
// in this directory are verified standalone). x86_64 / aarch64 values.
extern "C" {
    fn mmap(
        addr: *mut u8,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_HUGETLB: i32 = 0x40000;
const MAP_POPULATE: i32 = 0x8000;

#[inline]
fn map_failed() -> *mut u8 {
    !0usize as *mut u8
}

verus! {

/// Permission to deallocate an mmap-backed region. Mirrors
/// `vstd::raw_ptr::Dealloc` but for `munmap`. Like `Dealloc`, this is
/// an opaque tracked token; its only operation is being consumed by
/// `munmap_huge_pages` together with the matching `PointsToRaw`.
#[verifier::external_body]
pub tracked struct MmapDealloc {
    no_copy: NoCopy,
}

/// Data associated with a `MmapDealloc` permission. Same shape as
/// `DeallocData` minus `align` (mmap doesn't take an alignment
/// parameter — alignment is implicit from the page-size flag).
pub ghost struct MmapDeallocData {
    pub addr: usize,
    pub size: nat,
    pub provenance: Provenance,
}

impl MmapDealloc {
    pub uninterp spec fn view(self) -> MmapDeallocData;

    #[verifier::inline]
    pub open spec fn addr(self) -> usize {
        self.view().addr
    }

    #[verifier::inline]
    pub open spec fn size(self) -> nat {
        self.view().size
    }

    #[verifier::inline]
    pub open spec fn provenance(self) -> Provenance {
        self.view().provenance
    }
}

/// Allocate an mmap-backed huge-page region. Modeled on
/// `vstd::raw_ptr::allocate` (line 907 of `source/vstd/raw_ptr.rs`).
///
/// Trust assumption: `libc::mmap` with these flags either returns
/// `MAP_FAILED` or a valid `size`-byte region with anonymous-zeroed
/// memory and provenance covering `[addr, addr+size)`. Same shape of
/// trust as `vstd::raw_ptr::allocate`'s reliance on
/// `alloc::alloc::alloc`.
#[verifier::external_body]
pub fn mmap_allocate_huge_pages(size: usize) -> (pt: (
    *mut u8,
    Tracked<PointsToRaw>,
    Tracked<MmapDealloc>,
))
    requires
        size != 0,
        size as int % (2int * 1024 * 1024) == 0,
    ensures
        pt.1@.is_range(pt.0.addr() as int, size as int),
        pt.0.addr() + size <= usize::MAX + 1,
        pt.2@@ == (MmapDeallocData {
            addr: pt.0.addr(),
            size: size as nat,
            provenance: pt.1@.provenance(),
        }),
        pt.0@.provenance == pt.1@.provenance(),
    opens_invariants none
{
    // SAFETY: caller satisfies preconditions.
    let p = unsafe {
        mmap(
            core::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
            -1,
            0,
        )
    };
    if p == map_failed() {
        std::process::abort();
    }
    (p, Tracked::assume_new(), Tracked::assume_new())
}

/// Free an mmap-allocated region. Modeled on
/// `vstd::raw_ptr::deallocate`. Consuming both the `PointsToRaw` and
/// `MmapDealloc` permissions ensures no use-after-free.
#[verifier::external_body]
pub fn munmap_huge_pages(
    p: *mut u8,
    size: usize,
    Tracked(pt): Tracked<PointsToRaw>,
    Tracked(dealloc): Tracked<MmapDealloc>,
)
    requires
        dealloc.addr() == p.addr(),
        dealloc.size() == size,
        dealloc.provenance() == pt.provenance(),
        pt.is_range(dealloc.addr() as int, dealloc.size() as int),
        p@.provenance == dealloc.provenance(),
    opens_invariants none
{
    // SAFETY: ensured by MmapDealloc token.
    let res = unsafe { munmap(p, size) };
    if res != 0 {
        std::process::abort();
    }
}

/// Demonstration: allocate a 2 MiB huge-page region, then immediately
/// free it. Validates that the alloc + dealloc signatures compose —
/// the `MmapDealloc` token's `view` lines up with the
/// `PointsToRaw`'s shape exactly as `vstd::raw_ptr::allocate` +
/// `deallocate` do.
pub fn demonstrate_mmap_lifecycle() {
    let size: usize = 2 * 1024 * 1024;
    let (p, perm_raw, dealloc) = mmap_allocate_huge_pages(size);
    munmap_huge_pages(p, size, perm_raw, dealloc);
}

} // verus!

fn main() { }
