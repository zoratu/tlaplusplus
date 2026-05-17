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

} // verus!

fn main() { }
