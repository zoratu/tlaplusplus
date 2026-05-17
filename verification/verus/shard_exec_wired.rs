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

// EpochProtocol — shaped after the RefCounter<Perm> protocol in
// examples/state_machines/arc.rs. Diverges from atomic_ptr_with_epoch.rs
// (which combines count + stored-value into a single
// `Option<(nat, T)>`); the split-field shape used here matches
// arc.rs's `counter: nat` + `storage: Option<T>` and lets `have reader`
// discharge precondition obligations on multi-reader transitions like
// `do_clone` and `dec_basic`.
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
    pub tracked counter_token: EpochProtocol::counter<SlotPerm>,
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
        pub reader: Tracked< EpochProtocol::reader<SlotPerm> >,
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
            EpochProtocol::Instance::initialize_empty(Option::None);
        let tracked read_ref = inst.do_deposit(
            ptr_perm,
            &mut counter_token,
            ptr_perm,
        );

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
        let tracked perm = inst_borrowed.reader_guard(
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

    /// Mint another reader on the same allocation. The clone shares
    /// `inst`, `inv`, `ptr`, `slot_cell` with `self` (the InstanceId and
    /// Shared invariant are Copy/clonable; the `PPtr` is Copy), and
    /// gets its own reader token minted via the protocol's `do_clone`
    /// transition. Mirrors arc.rs's `MyArc::clone` minus the exec rc_cell
    /// bump (our counter is purely ghost).
    fn clone(&self) -> (sh: Self)
        requires self.wf(),
        ensures sh.wf(),
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked mut new_reader_opt: Option<EpochProtocol::reader<SlotPerm>> = None;
        open_atomic_invariant!(self.inv.borrow().borrow() => g => {
            let tracked GhostStuff { slot_perm: sp, counter_token: mut ct } = g;
            proof {
                let tracked nr = inst_borrowed.do_clone(
                    reader_borrowed.element(),
                    &mut ct,
                    reader_borrowed,
                );
                new_reader_opt = Some(nr);
            }
            proof { g = GhostStuff { slot_perm: sp, counter_token: ct }; }
        });
        let tracked new_reader = new_reader_opt.tracked_unwrap();
        VerifiedShard {
            inst: Tracked(self.inst.borrow().clone()),
            inv: Tracked(self.inv.borrow().clone()),
            reader: Tracked(new_reader),
            ptr: self.ptr,
            slot_cell: Ghost(self.slot_cell@),
        }
    }

    /// CAS the slot from `expected` to `new_fp`. Returns true if the
    /// swap succeeded. Models the FingerprintShard hot path's
    /// `compare_exchange_weak(0, fp)` on a slot.
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
