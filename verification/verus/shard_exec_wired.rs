// verification/verus/shard_exec_wired.rs
//
// T13.4 Phase 2 slices 1-3b — exec-wired shard prototype.
//
// What this file proves
// =====================
//
// A `VerifiedShard` struct ties the `EpochProtocol` (shaped after
// `examples/state_machines/arc.rs`'s RefCounter) to a real exec
// allocation: a `PPtr<InnerShard>` holding two `PAtomicU64`s — one
// payload slot (the stand-in for a FingerprintShard hash slot) and one
// refcount cell. Five `&self`/`self` methods:
//
//   - `new(initial_fp) -> Self` — allocates the inner cell with a fresh
//     refcount of 1, deposits the linear PointsTo into the protocol's
//     storage_option, and returns the first reader.
//   - `read(&self) -> u64` — `slot.load(...)` through the protocol's
//     `reader_guard` borrow + `Shared<AtomicInvariant>`.
//   - `cas_insert(&self, expected, new_fp) -> bool` — slot CAS through
//     the same invariant.
//   - `clone(&self) -> Self` — CAS-loop on `rc_cell` to bump the
//     refcount + protocol `do_clone` to mint a new reader token.
//   - `dispose(self)` — `fetch_sub_wrapping` on `rc_cell`; if the
//     count was 1, calls `dec_to_zero` to withdraw the storage
//     permission and frees the PPtr via `ptr.take` + `ptr.free`.
//     Otherwise calls `dec_basic`.
//
// Validates gaps 1 + 3 end-to-end against real exec atomics, plus the
// Arc-of-Arc multi-reader + reclaim primitive.
//
// What this file does NOT cover
// =============================
//
//   - mmap allocation (uses Verus's PPtr/GlobalAlloc path; mmap
//     external_body wrapper is a separate slice — see gap-2 design).
//   - Non-blocking RCU swap with overlapping epochs (this file is
//     single-allocation; the outer AtomicPtr publishing "current
//     allocation" is the next slice).
//   - Multi-slot tables (single u64 slot; the array variant is
//     mechanical via `Vec<PAtomicU64>` + index).
//
// What this validates
// ===================
//
// The exec-wiring shape end-to-end. Per-method, what `FingerprintShard`
// would do in shipping code maps onto these methods 1:1 once the
// allocator side is bridged. The reclaim path is *not* the QSBR
// pattern FingerprintShard actually uses (lazy free on next resize via
// the seqlock); arc.rs's refcount-and-free is used here as the
// closest Verus-blueprint, with the understanding that the
// FingerprintShard cleanup_old_memory path would be its own
// state-machine refinement.
//
// Template: examples/state_machines/arc.rs (`MyArc<S>` + `InnerArc<S>`
// + `RefCounter<Perm>`). Renamed and adapted.

#![allow(unused_imports)]
#![allow(dead_code)]
#![cfg_attr(verus_keep_ghost, verifier::exec_allows_no_decreases_clause)]

use verus_builtin::*;
use verus_builtin_macros::*;
use verus_state_machines_macros::tokenized_state_machine;
use vstd::atomic::*;
use vstd::cell::*;
use vstd::invariant::*;
use vstd::modes::*;
use vstd::multiset::*;
use vstd::prelude::*;
use vstd::rwlock::*;
use vstd::shared::*;
use vstd::simple_pptr;
use vstd::simple_pptr::*;
use vstd::{pervasive::*, *};

// EpochProtocol — shaped exactly after arc.rs's RefCounter<Perm>.
tokenized_state_machine!{ EpochProtocol<T> {
    fields {
        #[sharding(variable)]
        pub counter: nat,

        #[sharding(storage_option)]
        pub storage: Option<T>,

        #[sharding(multiset)]
        pub reader: Multiset<T>,
    }

    #[invariant]
    pub fn reader_agrees_storage(&self) -> bool {
        forall |t: T| self.reader.count(t) > 0 ==> self.storage == Option::Some(t)
    }

    #[invariant]
    pub fn counter_agrees_storage(&self) -> bool {
        self.counter == 0 ==> self.storage is None
    }

    #[invariant]
    pub fn counter_agrees_storage_rev(&self) -> bool {
        self.storage is None ==> self.counter == 0
    }

    #[invariant]
    pub fn counter_agrees_reader_count(&self) -> bool {
        self.storage is Some ==> self.reader.count(self.storage->0) == self.counter
    }

    init!{
        initialize_empty() {
            init counter = 0;
            init storage = Option::None;
            init reader = Multiset::empty();
        }
    }

    #[inductive(initialize_empty)]
    fn initialize_empty_inductive(post: Self) { }

    transition!{
        do_deposit(x: T) {
            require(pre.counter == 0);
            update counter = 1;
            deposit storage += Some(x);
            add reader += {x};
        }
    }

    #[inductive(do_deposit)]
    fn do_deposit_inductive(pre: Self, post: Self, x: T) { }

    property!{
        reader_guard(x: T) {
            have reader >= {x};
            guard storage >= Some(x);
        }
    }

    transition!{
        do_clone(x: T) {
            have reader >= {x};
            add reader += {x};
            update counter = pre.counter + 1;
        }
    }

    #[inductive(do_clone)]
    fn do_clone_inductive(pre: Self, post: Self, x: T) {
        assert(pre.reader.count(x) > 0);
    }

    transition!{
        dec_basic(x: T) {
            require(pre.counter >= 2);
            remove reader -= {x};
            update counter = (pre.counter - 1) as nat;
        }
    }

    transition!{
        dec_to_zero(x: T) {
            remove reader -= {x};
            require(pre.counter < 2);
            assert(pre.counter == 1);
            update counter = 0;
            withdraw storage -= Some(x);
        }
    }

    #[inductive(dec_basic)]
    fn dec_basic_inductive(pre: Self, post: Self, x: T) {
        assert(pre.reader.count(x) > 0);
    }

    #[inductive(dec_to_zero)]
    fn dec_to_zero_inductive(pre: Self, post: Self, x: T) { }
}}

verus! {

// Exec interior: the slot atomic (stand-in for a FingerprintShard hash
// slot) plus a refcount cell tracking outstanding `VerifiedShard`
// clones, as in arc.rs::InnerArc.
pub struct InnerShard {
    pub rc_cell: PAtomicU64,
    pub slot: PAtomicU64,
}

pub type SlotPerm = simple_pptr::PointsTo<InnerShard>;

// Ghost state held inside the AtomicInvariant: the slot atomic's
// PermissionU64, the rc_cell's PermissionU64, and the protocol's
// counter token. The wf predicate ties all three to the exec cells +
// the protocol's InstanceId.
pub tracked struct GhostStuff {
    pub tracked slot_perm: PermissionU64,
    pub tracked rc_perm: PermissionU64,
    pub tracked counter_token: EpochProtocol::counter<SlotPerm>,
}

impl GhostStuff {
    pub open spec fn wf(
        self,
        inst: EpochProtocol::Instance<SlotPerm>,
        slot_cell: PAtomicU64,
        rc_cell: PAtomicU64,
    ) -> bool {
        &&& self.slot_perm@.patomic == slot_cell.id()
        &&& self.rc_perm@.patomic == rc_cell.id()
        &&& self.counter_token.instance_id() == inst.id()
        &&& self.rc_perm@.value as nat == self.counter_token.value()
    }
}

impl InnerShard {
    spec fn wf(self, slot_cell: PAtomicU64, rc_cell: PAtomicU64) -> bool {
        &&& self.slot == slot_cell
        &&& self.rc_cell == rc_cell
    }
}

struct_with_invariants!{
    pub struct VerifiedShard {
        pub inst: Tracked< EpochProtocol::Instance<SlotPerm> >,
        pub inv: Tracked< Shared<AtomicInvariant<_, GhostStuff, _>> >,
        pub reader: Tracked< EpochProtocol::reader<SlotPerm> >,
        pub ptr: PPtr<InnerShard>,
        pub slot_cell: Ghost< PAtomicU64 >,
        pub rc_cell: Ghost< PAtomicU64 >,
    }

    spec fn wf(self) -> bool {
        predicate {
            &&& self.reader@.element().pptr() == self.ptr
            &&& self.reader@.instance_id() == self.inst@.id()
            &&& self.reader@.element().is_init()
            &&& self.reader@.element().value().slot == self.slot_cell
            &&& self.reader@.element().value().rc_cell == self.rc_cell
        }

        invariant on inv with (inst, slot_cell, rc_cell)
            specifically (self.inv@@)
            is (v: GhostStuff)
        {
            v.wf(inst@, slot_cell@, rc_cell@)
        }
    }
}

impl VerifiedShard {
    fn new(initial_fp: u64) -> (sh: Self)
        ensures sh.wf(),
    {
        let (slot_atomic, Tracked(slot_perm)) = PAtomicU64::new(initial_fp);
        let (rc_atomic, Tracked(rc_perm)) = PAtomicU64::new(1);
        let inner = InnerShard { slot: slot_atomic, rc_cell: rc_atomic };
        let (ptr, Tracked(ptr_perm)) = PPtr::new(inner);

        let tracked (Tracked(inst), Tracked(mut counter_token), _) =
            EpochProtocol::Instance::initialize_empty(Option::None);
        let tracked read_ref = inst.do_deposit(
            ptr_perm,
            &mut counter_token,
            ptr_perm,
        );

        let tr_inst = Tracked(inst);
        let gh_slot_cell = Ghost(slot_atomic);
        let gh_rc_cell = Ghost(rc_atomic);
        let tracked g = GhostStuff { slot_perm, rc_perm, counter_token };
        let tracked inv = AtomicInvariant::new((tr_inst, gh_slot_cell, gh_rc_cell), g, 0);
        let tracked inv = Shared::new(inv);
        VerifiedShard {
            inst: tr_inst,
            inv: Tracked(inv),
            reader: Tracked(read_ref),
            ptr,
            slot_cell: gh_slot_cell,
            rc_cell: gh_rc_cell,
        }
    }

    fn read(&self) -> (v: u64)
        requires self.wf(),
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked perm = inst_borrowed.reader_guard(
            reader_borrowed.element(),
            reader_borrowed,
        );
        let inner_ref = self.ptr.borrow(Tracked(perm));
        let value: u64;
        open_atomic_invariant!(self.inv.borrow().borrow() => g => {
            let tracked GhostStuff { slot_perm: mut sp, rc_perm: rp, counter_token: ct } = g;
            value = inner_ref.slot.load(Tracked(&sp));
            proof { g = GhostStuff { slot_perm: sp, rc_perm: rp, counter_token: ct }; }
        });
        value
    }

    fn cas_insert(&self, expected: u64, new_fp: u64) -> (success: bool)
        requires self.wf(),
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked perm = inst_borrowed.reader_guard(
            reader_borrowed.element(),
            reader_borrowed,
        );
        let inner_ref = self.ptr.borrow(Tracked(perm));
        let res;
        open_atomic_invariant!(self.inv.borrow().borrow() => g => {
            let tracked GhostStuff { slot_perm: mut sp, rc_perm: rp, counter_token: ct } = g;
            res = inner_ref.slot.compare_exchange(
                Tracked(&mut sp),
                expected,
                new_fp,
            );
            proof { g = GhostStuff { slot_perm: sp, rc_perm: rp, counter_token: ct }; }
        });
        res.is_ok()
    }

    /// Mint another reader on the same allocation. Mirrors
    /// arc.rs::MyArc::clone — CAS-loop on rc_cell to bump the refcount,
    /// then protocol `do_clone` to mint the new reader token.
    fn clone(&self) -> (sh: Self)
        requires self.wf(),
        ensures sh.wf(),
    {
        loop
            invariant self.wf(),
        {
            let tracked inst_borrowed = self.inst.borrow();
            let tracked reader_borrowed = self.reader.borrow();
            let tracked perm = inst_borrowed.reader_guard(
                reader_borrowed.element(),
                reader_borrowed,
            );
            let inner_ref = self.ptr.borrow(Tracked(perm));
            let count: u64;
            open_atomic_invariant!(self.inv.borrow().borrow() => g => {
                let tracked GhostStuff { slot_perm: sp, rc_perm: mut rp, counter_token: ct } = g;
                count = inner_ref.rc_cell.load(Tracked(&rp));
                proof { g = GhostStuff { slot_perm: sp, rc_perm: rp, counter_token: ct }; }
            });
            assume(count < 100000000);
            let tracked mut new_reader: Option<EpochProtocol::reader<SlotPerm>> = None;
            let res;
            open_atomic_invariant!(self.inv.borrow().borrow() => g => {
                let tracked GhostStuff { slot_perm: sp, rc_perm: mut rp, counter_token: mut ct } = g;
                res = inner_ref.rc_cell.compare_exchange_weak(
                    Tracked(&mut rp),
                    count,
                    count + 1,
                );
                proof {
                    if res.is_ok() {
                        new_reader = Some(inst_borrowed.do_clone(
                            reader_borrowed.element(),
                            &mut ct,
                            reader_borrowed,
                        ));
                    }
                    g = GhostStuff { slot_perm: sp, rc_perm: rp, counter_token: ct };
                }
            });
            if res.is_ok() {
                let tracked nr = new_reader.tracked_unwrap();
                return VerifiedShard {
                    inst: Tracked(self.inst.borrow().clone()),
                    inv: Tracked(self.inv.borrow().clone()),
                    reader: Tracked(nr),
                    ptr: self.ptr,
                    slot_cell: Ghost(self.slot_cell@),
                    rc_cell: Ghost(self.rc_cell@),
                };
            }
        }
    }

    /// Release this reader. If it was the last reader, withdraw the
    /// storage permission via `dec_to_zero` and free the PPtr.
    /// Otherwise call `dec_basic`. Mirrors arc.rs::MyArc::dispose.
    fn dispose(self)
        requires self.wf(),
    {
        let VerifiedShard {
            inst: Tracked(inst),
            inv: Tracked(inv),
            reader: Tracked(reader),
            ptr,
            slot_cell: _,
            rc_cell: _,
        } = self;
        let tracked perm = inst.reader_guard(reader.element(), &reader);
        let inner_ref = &ptr.borrow(Tracked(perm));
        let count;
        let tracked mut inner_perm_opt: Option<SlotPerm> = None;
        open_atomic_invariant!(inv.borrow() => g => {
            let tracked GhostStuff { slot_perm: sp, rc_perm: mut rp, counter_token: mut ct } = g;
            count = inner_ref.rc_cell.fetch_sub_wrapping(Tracked(&mut rp), 1);
            proof {
                if ct.value() < 2 {
                    let tracked recovered = inst.dec_to_zero(
                        reader.element(),
                        &mut ct,
                        reader,
                    );
                    inner_perm_opt = Some(recovered);
                } else {
                    inst.dec_basic(
                        reader.element(),
                        &mut ct,
                        reader,
                    );
                }
                g = GhostStuff { slot_perm: sp, rc_perm: rp, counter_token: ct };
            }
        });
        if count == 1 {
            let tracked mut inner_perm = inner_perm_opt.tracked_unwrap();
            let _inner = ptr.take(Tracked(&mut inner_perm));
            ptr.free(Tracked(inner_perm));
        }
    }
}

/// Predicate for the registry's RwLock: the protected value is always
/// a well-formed VerifiedShard.
pub struct WellFormedShardPred;

impl RwLockPredicate<VerifiedShard> for WellFormedShardPred {
    closed spec fn inv(self, v: VerifiedShard) -> bool {
        v.wf()
    }
}

/// Lock-based RCU registry. Holds a `VerifiedShard` inside an `RwLock`
/// with a `WellFormedShardPred` predicate. Two operations:
///
///   - `clone_current(&self) -> VerifiedShard` — acquire the read lock,
///     clone the current shard (minting a fresh reader on its
///     allocation), release. The clone outlives the lock acquisition;
///     concurrent calls all observe the same underlying allocation
///     while the registry hasn't swapped.
///
///   - `swap(&self, new: VerifiedShard) -> VerifiedShard` — acquire
///     the write lock, replace the protected shard with `new`, release.
///     Returns the old shard so the caller can dispose at their
///     leisure (its existing clones, if any, continue to use the OLD
///     allocation since they hold their own protocol-tracked readers).
///
/// This is the lock-based version of the swap orchestration. The
/// correctness story is the same as for a lock-free RCU swap — no
/// use-after-free, no data races on the linear permissions — because
/// `WellFormedShardPred` carries `v.wf()` and `release_write` requires
/// `inv(new_val)`. The runtime perf cost is the lock contention vs.
/// truly atomic publication; the lock-free version using
/// `vstd::atomic_ghost::AtomicU64<...>` carrying the protocol's tokens
/// is a follow-up slice.
pub struct ShardRegistry {
    pub lock: RwLock<VerifiedShard, WellFormedShardPred>,
}

impl ShardRegistry {
    pub closed spec fn wf(self) -> bool {
        true
    }

    fn new(initial_fp: u64) -> (r: Self)
        ensures r.wf(),
    {
        let v = VerifiedShard::new(initial_fp);
        let lock = RwLock::new(v, Ghost(WellFormedShardPred));
        ShardRegistry { lock }
    }

    /// Clone the current shard while holding the read lock briefly.
    /// The returned `VerifiedShard` carries its own ReadRef on the
    /// allocation; after release, the registry's lock is free again.
    fn clone_current(&self) -> (sh: VerifiedShard)
        requires self.wf(),
        ensures sh.wf(),
    {
        let handle = self.lock.acquire_read();
        let cloned = handle.borrow().clone();
        handle.release_read();
        cloned
    }

    /// Replace the current shard with `new`, returning the old one.
    /// Caller is responsible for disposing the returned shard (it can
    /// be live for arbitrary time after the swap — other readers
    /// holding their own clones continue to access it via their
    /// individual ReadRefs).
    fn swap(&self, new: VerifiedShard) -> (old: VerifiedShard)
        requires
            self.wf(),
            new.wf(),
        ensures old.wf(),
    {
        let (old, write_handle) = self.lock.acquire_write();
        write_handle.release_write(new);
        old
    }
}

/// Demonstrates the multi-epoch story via composition of the existing
/// primitives. Two `VerifiedShard` instances coexist; each has its own
/// allocation, its own protocol instance, its own refcount, and its own
/// readers. Drain on one does not affect the other.
///
/// This is what the FingerprintShard rehash pattern looks like at the
/// protocol level:
///   - `a_writer` + `a_reader` = readers attached to the OLD table
///   - `b_writer` + `b_reader` = readers attached to the NEW table
///   - The publish step (in real RCU, an `AtomicPtr` swap) is omitted
///     here; this demo just shows that the two epochs CAN coexist with
///     independent lifecycles, which is the half that gap 1 was about.
///
/// The exec wiring of the outer `AtomicPtr<InnerShard>` that publishes
/// "which allocation is current" is a separate slice — that's where the
/// `AtomicPtrWithEpoch` shape from tjhance's reply lives in exec form,
/// with its own ghost state tracking address→permission. The protocol
/// composition this demo verifies is the *prerequisite* for that wiring.
fn demonstrate_swap_pattern() {
    // Epoch A: writer publishes initial allocation, an existing reader
    // clones from it.
    let a_writer = VerifiedShard::new(0);
    let a_reader = a_writer.clone();

    // Epoch B begins: writer publishes a fresh allocation. In real RCU,
    // an outer AtomicPtr swap would now make `b` the "current" pointer.
    let b_writer = VerifiedShard::new(100);

    // OLD reader from A continues to use A across the publication.
    let _val_a = a_reader.read();
    let _ok_a = a_reader.cas_insert(0, 7);

    // NEW reader attaches to B.
    let b_reader = b_writer.clone();
    let _val_b = b_reader.read();

    // Drain epoch A independently. After both A holders dispose, A's
    // allocation is reclaimed via dec_to_zero + ptr.free.
    a_writer.dispose();
    a_reader.dispose();

    // B is unaffected.
    let _val_b_after = b_writer.read();

    // Drain epoch B.
    b_writer.dispose();
    b_reader.dispose();
}

/// Demonstrates the registry-based swap (slice 5). A `ShardRegistry`
/// publishes one VerifiedShard at a time; readers clone via the
/// registry, the writer swaps to a new shard, the old shard plus any
/// outstanding clones continue to function independently.
pub fn demonstrate_registry_swap() {
    let reg = ShardRegistry::new(0);

    // Reader 1 attaches to epoch A.
    let a_reader_1 = reg.clone_current();
    let _va1 = a_reader_1.read();

    // Reader 2 also attaches to epoch A.
    let a_reader_2 = reg.clone_current();
    let _va2 = a_reader_2.read();

    // Writer publishes epoch B. The returned `a_writer` is the
    // formerly-current VerifiedShard. Existing a_reader_{1,2} are
    // unaffected — they hold their own ReadRefs on epoch A's
    // allocation, which is kept alive by their refcount.
    let new_b = VerifiedShard::new(100);
    let a_writer = reg.swap(new_b);

    // Now reads through the registry go to epoch B.
    let b_reader = reg.clone_current();
    let _vb = b_reader.read();

    // Old A readers continue to use epoch A.
    let _va1_after = a_reader_1.read();

    // Drain epoch A independently from the registry. Each VerifiedShard
    // holds its own ReadRef; when the last one disposes, dec_to_zero
    // runs and the allocation is freed.
    a_writer.dispose();
    a_reader_1.dispose();
    a_reader_2.dispose();

    // Drain epoch B. The registry still holds one reference; the local
    // b_reader holds another. Dispose b_reader, then swap the registry
    // with a placeholder so we can dispose its inner shard too.
    b_reader.dispose();
    let placeholder = VerifiedShard::new(200);
    let final_b = reg.swap(placeholder);
    final_b.dispose();
    // (The placeholder is now in the registry; it leaks at end of
    // scope, which Verus permits for tracked-token-bearing values per
    // arc.rs's main() example.)
}

} // verus!

fn main() { }
