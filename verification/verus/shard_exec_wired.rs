// verification/verus/shard_exec_wired.rs
//
// T13.4 Phase 2 slice 1 — exec-wired shard prototype.
//
// What this file proves
// =====================
//
// A `VerifiedShard` struct ties the `EpochProtocol` (the protocol skeleton
// from `atomic_ptr_with_epoch.rs`) to a real exec atomic: a `PAtomicU64`
// inside a `PPtr<InnerShard>` allocation. The `read(&self) -> u64`
// method verifies that:
//
//   - A reader can load the slot through `&self` (gap-3 demonstration).
//   - The load goes through the protocol's `read_ref_guards` to obtain a
//     `&PointsTo<InnerShard>` (gap-1 demonstration: the linear permission
//     stays parked in the protocol while shared `&` access is handed out).
//   - The slot's `PermissionU64` lives inside a `Shared<AtomicInvariant>`,
//     opened around the atomic load. This is the arc.rs pattern adapted
//     to our domain.
//
// The file does NOT cover:
//
//   - Resize (single-epoch only; the swap-pattern slice is next).
//   - mmap allocation (uses `PPtr::new` / `vstd::raw_ptr::allocate` path).
//   - CAS insert (single read path only; insert is the next slice after
//     resize).
//   - Multi-slot tables (single u64 slot; the array variant is trivial
//     to extend via `Vec<AtomicU64>` once the single-slot shape verifies).
//
// What this validates
// ===================
//
// The exec-wiring shape end-to-end. If this file verifies, the same
// pattern annotated onto the shipping `FingerprintShard` is mechanical
// (same struct skeleton, more fields). The remaining open work is then:
// (a) resize via per-epoch protocol instances, (b) mmap external_body
// for the allocation, (c) decide between AtomicInvariant per CAS vs
// logatom linearizers for the hot path.
//
// Template: examples/state_machines/arc.rs (MyArc<S> + InnerArc<S> +
// RefCounter<Perm>). The protocol is renamed (EpochProtocol<T>) and the
// inner cell carries our slot atomic.
//
// Status: pending Verus verification on the verification spot.

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

// EpochProtocol — same protocol as `atomic_ptr_with_epoch.rs`, inlined
// because Verus proof files in this directory are verified standalone
// and don't share a crate. Keep in sync with the upstream PoC if either
// file evolves.
tokenized_state_machine!{ EpochProtocol<T> {
    fields {
        #[sharding(storage_option)]
        pub stored: Option<T>,

        #[sharding(variable)]
        pub main_counter: Option<(nat, T)>,

        #[sharding(multiset)]
        pub read_ref: Multiset<T>,
    }

    init!{
        new() {
            init stored = None;
            init main_counter = None;
            init read_ref = Multiset::empty();
        }
    }

    transition!{
        publish(t: T) {
            require pre.main_counter.is_none();
            update main_counter = Some((0, t));
            deposit stored += Some(t);
        }
    }

    transition!{
        reclaim() {
            require let Some((count, t)) = pre.main_counter;
            require count == 0;
            update main_counter = None;
            withdraw stored -= Some(t);
        }
    }

    transition!{
        acquire_read() {
            require let Some((count, t)) = pre.main_counter;
            update main_counter = Some((count + 1, t));
            add read_ref += { t };
        }
    }

    transition!{
        release_read(t1: T) {
            remove read_ref -= { t1 };
            require let Some((count, t2)) = pre.main_counter;
            assert count >= 1;
            assert t1 == t2;
            update main_counter = Some(((count - 1) as nat, t1));
        }
    }

    property!{
        read_ref_guards(t: T) {
            have read_ref >= { t };
            guard stored >= Some(t);
        }
    }

    #[invariant]
    pub spec fn main_inv(&self) -> bool {
        match self.stored {
            None => {
                &&& self.main_counter.is_none()
                &&& self.read_ref =~= Multiset::empty()
            }
            Some(t) => {
                match self.main_counter {
                    Some((count, t1)) => {
                        &&& t == t1
                        &&& self.read_ref.count(t) == count
                        &&& (forall |t0: T| t0 != t ==> self.read_ref.count(t0) == 0)
                    }
                    None => false,
                }
            }
        }
    }

    #[inductive(new)]
    fn new_inductive(post: Self) { }
    #[inductive(publish)]
    fn publish_inductive(pre: Self, post: Self, t: T) { }
    #[inductive(reclaim)]
    fn reclaim_inductive(pre: Self, post: Self) { }
    #[inductive(acquire_read)]
    fn acquire_read_inductive(pre: Self, post: Self) { }
    #[inductive(release_read)]
    fn release_read_inductive(pre: Self, post: Self, t1: T) { }
}}

verus! {

// Exec interior: one atomic slot. Stand-in for one HashTableEntry from
// the shipping FingerprintShard. The multi-slot extension is just
// `slots: Vec<PAtomicU64>` with an index parameter on the operations.
pub struct InnerShard {
    pub slot: PAtomicU64,
}

pub type SlotPerm = simple_pptr::PointsTo<InnerShard>;

// Ghost state held inside the AtomicInvariant: the slot atomic's
// PermissionU64 (required to load/CAS the atomic) plus the protocol's
// main_counter token (required when the count needs to be mutated, e.g.
// future acquire/release transitions performed inside an exec method).
pub tracked struct GhostStuff {
    pub tracked slot_perm: PermissionU64,
    pub tracked counter_token: EpochProtocol::main_counter<SlotPerm>,
}

impl GhostStuff {
    pub open spec fn wf(
        self,
        inst: EpochProtocol::Instance<SlotPerm>,
        cell: PAtomicU64,
    ) -> bool {
        &&& self.slot_perm@.patomic == cell.id()
        &&& self.counter_token.instance_id() == inst.id()
    }
}

impl InnerShard {
    spec fn wf(self, cell: PAtomicU64) -> bool {
        self.slot == cell
    }
}

struct_with_invariants!{
    pub struct VerifiedShard {
        pub inst: Tracked< EpochProtocol::Instance<SlotPerm> >,
        pub inv: Tracked< Shared<AtomicInvariant<_, GhostStuff, _>> >,
        pub reader: Tracked< EpochProtocol::read_ref<SlotPerm> >,
        pub ptr: PPtr<InnerShard>,
        pub slot_cell: Ghost< PAtomicU64 >,
    }

    spec fn wf(self) -> bool {
        predicate {
            &&& self.reader@.element().pptr() == self.ptr
            &&& self.reader@.instance_id() == self.inst@.id()
            &&& self.reader@.element().is_init()
            &&& self.reader@.element().value().slot == self.slot_cell
        }

        invariant on inv with (inst, slot_cell)
            specifically (self.inv@@)
            is (v: GhostStuff)
        {
            v.wf(inst@, slot_cell@)
        }
    }
}

impl VerifiedShard {
    fn new(initial_fp: u64) -> (sh: Self)
        ensures sh.wf(),
    {
        let (slot_atomic, Tracked(slot_perm)) = PAtomicU64::new(initial_fp);
        let inner = InnerShard { slot: slot_atomic };
        let (ptr, Tracked(ptr_perm)) = PPtr::new(inner);

        let tracked (Tracked(inst), Tracked(mut counter_token), _) =
            EpochProtocol::Instance::new(None);
        proof {
            inst.publish(ptr_perm, ptr_perm, &mut counter_token);
        }
        let tracked read_ref = inst.acquire_read(&mut counter_token);

        let tr_inst = Tracked(inst);
        let gh_cell = Ghost(slot_atomic);
        let tracked g = GhostStuff { slot_perm, counter_token };
        let tracked inv = AtomicInvariant::new((tr_inst, gh_cell), g, 0);
        let tracked inv = Shared::new(inv);
        VerifiedShard {
            inst: tr_inst,
            inv: Tracked(inv),
            reader: Tracked(read_ref),
            ptr,
            slot_cell: gh_cell,
        }
    }

    fn read(&self) -> (v: u64)
        requires self.wf(),
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked perm = inst_borrowed.read_ref_guards(
            reader_borrowed.element(),
            reader_borrowed,
        );
        let inner_ref = self.ptr.borrow(Tracked(perm));
        let value: u64;
        open_atomic_invariant!(self.inv.borrow().borrow() => g => {
            let tracked GhostStuff { slot_perm: mut sp, counter_token: ct } = g;
            value = inner_ref.slot.load(Tracked(&sp));
            proof { g = GhostStuff { slot_perm: sp, counter_token: ct }; }
        });
        value
    }

    /// CAS the slot from `expected` to `new_fp`. Returns true if the
    /// swap succeeded. Models the FingerprintShard hot path's
    /// `compare_exchange_weak(0, fp)` on a slot.
    ///
    /// This is the write-path analog of `read`: same `&self` access,
    /// same `read_ref_guards` to obtain the `&PointsTo<InnerShard>`,
    /// same `open_atomic_invariant!` to access the slot's
    /// `PermissionU64`. The only difference is `compare_exchange` in
    /// place of `load`.
    fn cas_insert(&self, expected: u64, new_fp: u64) -> (success: bool)
        requires self.wf(),
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked perm = inst_borrowed.read_ref_guards(
            reader_borrowed.element(),
            reader_borrowed,
        );
        let inner_ref = self.ptr.borrow(Tracked(perm));
        let res;
        open_atomic_invariant!(self.inv.borrow().borrow() => g => {
            let tracked GhostStuff { slot_perm: mut sp, counter_token: ct } = g;
            res = inner_ref.slot.compare_exchange(
                Tracked(&mut sp),
                expected,
                new_fp,
            );
            proof { g = GhostStuff { slot_perm: sp, counter_token: ct }; }
        });
        res.is_ok()
    }
}

} // verus!

fn main() { }
