// verification/verus/shard_multi_slot.rs
//
// T13.4 Phase 2 slice 6 — multi-slot table prototype.
//
// What this proves
// ================
//
// Extends `shard_exec_wired.rs`'s single-slot `VerifiedShard` to two
// slots. Validates that the per-slot AtomicInvariant bookkeeping
// composes for N slots: each `PAtomicU64` slot gets its own
// `PermissionU64` carried in `GhostStuff`, and read/CAS operations
// branch on an index parameter without losing soundness.
//
// Two slots is enough to demonstrate the multi-slot pattern; the
// N-slot extension is mechanical (each additional slot adds one
// `PAtomicU64` to `InnerShard`, one `PermissionU64` to `GhostStuff`,
// one `Ghost<PAtomicU64>` to `VerifiedShard`, and one match arm to
// `read_at` / `cas_insert_at`). For a true array (e.g. `Vec<PAtomicU64>`
// with `Map<int, PermissionU64>` ghost), see the comment at the
// bottom — that's a separate slice once we want to model
// `HashTableEntry[N]` directly.
//
// Reuses the same `EpochProtocol` shape as `shard_exec_wired.rs` /
// arc.rs (split-field `counter: nat` + `storage: Option<T>` +
// `reader: Multiset<T>` + do_deposit/do_clone/dec_basic/dec_to_zero).
//
// What this file does NOT cover
// =============================
//
// - `clone` / `dispose` (the slice 3a / 3b lifecycle). The single-slot
//   versions are already verified in shard_exec_wired.rs; the multi-slot
//   versions are mechanically the same — copy the AtomicInvariant
//   open/close pattern and adapt to the extra slot perms. Not duplicated
//   here to keep the file focused.
// - Registry / swap (slice 5). Same — the registry is a wrapper
//   independent of the slot count.
// - True N-slot via `Vec<PAtomicU64>` + `Map<int, PermissionU64>`. The
//   2-slot hardcoded version verifies the per-slot composition; the
//   array version is the next slice if/when we want to model
//   `HashTableEntry[N]` exactly.
//
// Status: verified standalone with Verus 0.2026.05.13 (aarch64).

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

// EpochProtocol — same shape as in shard_exec_wired.rs (arc.rs RefCounter).
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

// Two-slot exec interior. Stand-in for a fixed-N HashTableEntry array.
pub struct InnerShard {
    pub rc_cell: PAtomicU64,
    pub slot_0: PAtomicU64,
    pub slot_1: PAtomicU64,
}

pub type SlotPerm = simple_pptr::PointsTo<InnerShard>;

// Ghost state held inside the AtomicInvariant: a PermissionU64 per
// slot, the rc_cell's PermissionU64, and the protocol's counter token.
pub tracked struct GhostStuff {
    pub tracked slot_0_perm: PermissionU64,
    pub tracked slot_1_perm: PermissionU64,
    pub tracked rc_perm: PermissionU64,
    pub tracked counter_token: EpochProtocol::counter<SlotPerm>,
}

impl GhostStuff {
    pub open spec fn wf(
        self,
        inst: EpochProtocol::Instance<SlotPerm>,
        slot_0_cell: PAtomicU64,
        slot_1_cell: PAtomicU64,
        rc_cell: PAtomicU64,
    ) -> bool {
        &&& self.slot_0_perm@.patomic == slot_0_cell.id()
        &&& self.slot_1_perm@.patomic == slot_1_cell.id()
        &&& self.rc_perm@.patomic == rc_cell.id()
        &&& self.counter_token.instance_id() == inst.id()
        &&& self.rc_perm@.value as nat == self.counter_token.value()
    }
}

impl InnerShard {
    spec fn wf(
        self,
        slot_0_cell: PAtomicU64,
        slot_1_cell: PAtomicU64,
        rc_cell: PAtomicU64,
    ) -> bool {
        &&& self.slot_0 == slot_0_cell
        &&& self.slot_1 == slot_1_cell
        &&& self.rc_cell == rc_cell
    }
}

struct_with_invariants!{
    pub struct VerifiedShard {
        pub inst: Tracked< EpochProtocol::Instance<SlotPerm> >,
        pub inv: Tracked< Shared<AtomicInvariant<_, GhostStuff, _>> >,
        pub reader: Tracked< EpochProtocol::reader<SlotPerm> >,
        pub ptr: PPtr<InnerShard>,
        pub slot_0_cell: Ghost< PAtomicU64 >,
        pub slot_1_cell: Ghost< PAtomicU64 >,
        pub rc_cell: Ghost< PAtomicU64 >,
    }

    spec fn wf(self) -> bool {
        predicate {
            &&& self.reader@.element().pptr() == self.ptr
            &&& self.reader@.instance_id() == self.inst@.id()
            &&& self.reader@.element().is_init()
            &&& self.reader@.element().value().slot_0 == self.slot_0_cell
            &&& self.reader@.element().value().slot_1 == self.slot_1_cell
            &&& self.reader@.element().value().rc_cell == self.rc_cell
        }

        invariant on inv with (inst, slot_0_cell, slot_1_cell, rc_cell)
            specifically (self.inv@@)
            is (v: GhostStuff)
        {
            v.wf(inst@, slot_0_cell@, slot_1_cell@, rc_cell@)
        }
    }
}

impl VerifiedShard {
    fn new(initial_0: u64, initial_1: u64) -> (sh: Self)
        ensures sh.wf(),
    {
        let (slot_0_atomic, Tracked(slot_0_perm)) = PAtomicU64::new(initial_0);
        let (slot_1_atomic, Tracked(slot_1_perm)) = PAtomicU64::new(initial_1);
        let (rc_atomic, Tracked(rc_perm)) = PAtomicU64::new(1);
        let inner = InnerShard {
            slot_0: slot_0_atomic,
            slot_1: slot_1_atomic,
            rc_cell: rc_atomic,
        };
        let (ptr, Tracked(ptr_perm)) = PPtr::new(inner);

        let tracked (Tracked(inst), Tracked(mut counter_token), _) =
            EpochProtocol::Instance::initialize_empty(Option::None);
        let tracked read_ref = inst.do_deposit(
            ptr_perm,
            &mut counter_token,
            ptr_perm,
        );

        let tr_inst = Tracked(inst);
        let gh_0 = Ghost(slot_0_atomic);
        let gh_1 = Ghost(slot_1_atomic);
        let gh_rc = Ghost(rc_atomic);
        let tracked g = GhostStuff { slot_0_perm, slot_1_perm, rc_perm, counter_token };
        let tracked inv = AtomicInvariant::new((tr_inst, gh_0, gh_1, gh_rc), g, 0);
        let tracked inv = Shared::new(inv);
        VerifiedShard {
            inst: tr_inst,
            inv: Tracked(inv),
            reader: Tracked(read_ref),
            ptr,
            slot_0_cell: gh_0,
            slot_1_cell: gh_1,
            rc_cell: gh_rc,
        }
    }

    /// Read slot 0 or slot 1 depending on `idx`. `idx` is bound to
    /// `{0, 1}`; out-of-range indices panic at runtime.
    fn read_at(&self, idx: usize) -> (v: u64)
        requires
            self.wf(),
            idx < 2,
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked perm = inst_borrowed.reader_guard(
            reader_borrowed.element(),
            reader_borrowed,
        );
        let inner_ref = self.ptr.borrow(Tracked(perm));
        let value: u64;
        if idx == 0 {
            open_atomic_invariant!(self.inv.borrow().borrow() => g => {
                let tracked GhostStuff {
                    slot_0_perm: mut sp0,
                    slot_1_perm: sp1,
                    rc_perm: rp,
                    counter_token: ct,
                } = g;
                value = inner_ref.slot_0.load(Tracked(&sp0));
                proof {
                    g = GhostStuff {
                        slot_0_perm: sp0,
                        slot_1_perm: sp1,
                        rc_perm: rp,
                        counter_token: ct,
                    };
                }
            });
        } else {
            open_atomic_invariant!(self.inv.borrow().borrow() => g => {
                let tracked GhostStuff {
                    slot_0_perm: sp0,
                    slot_1_perm: mut sp1,
                    rc_perm: rp,
                    counter_token: ct,
                } = g;
                value = inner_ref.slot_1.load(Tracked(&sp1));
                proof {
                    g = GhostStuff {
                        slot_0_perm: sp0,
                        slot_1_perm: sp1,
                        rc_perm: rp,
                        counter_token: ct,
                    };
                }
            });
        }
        value
    }

    /// CAS the slot at `idx` from `expected` to `new_fp`. Returns true
    /// if the swap succeeded.
    fn cas_insert_at(&self, idx: usize, expected: u64, new_fp: u64) -> (success: bool)
        requires
            self.wf(),
            idx < 2,
    {
        let tracked inst_borrowed = self.inst.borrow();
        let tracked reader_borrowed = self.reader.borrow();
        let tracked perm = inst_borrowed.reader_guard(
            reader_borrowed.element(),
            reader_borrowed,
        );
        let inner_ref = self.ptr.borrow(Tracked(perm));
        let res;
        if idx == 0 {
            open_atomic_invariant!(self.inv.borrow().borrow() => g => {
                let tracked GhostStuff {
                    slot_0_perm: mut sp0,
                    slot_1_perm: sp1,
                    rc_perm: rp,
                    counter_token: ct,
                } = g;
                res = inner_ref.slot_0.compare_exchange(
                    Tracked(&mut sp0),
                    expected,
                    new_fp,
                );
                proof {
                    g = GhostStuff {
                        slot_0_perm: sp0,
                        slot_1_perm: sp1,
                        rc_perm: rp,
                        counter_token: ct,
                    };
                }
            });
        } else {
            open_atomic_invariant!(self.inv.borrow().borrow() => g => {
                let tracked GhostStuff {
                    slot_0_perm: sp0,
                    slot_1_perm: mut sp1,
                    rc_perm: rp,
                    counter_token: ct,
                } = g;
                res = inner_ref.slot_1.compare_exchange(
                    Tracked(&mut sp1),
                    expected,
                    new_fp,
                );
                proof {
                    g = GhostStuff {
                        slot_0_perm: sp0,
                        slot_1_perm: sp1,
                        rc_perm: rp,
                        counter_token: ct,
                    };
                }
            });
        }
        res.is_ok()
    }
}

/// Demonstrates that each slot is independently read/CAS-able.
pub fn demonstrate_multi_slot() {
    let sh = VerifiedShard::new(0, 100);

    let _v0 = sh.read_at(0);
    let _v1 = sh.read_at(1);

    let _ok0 = sh.cas_insert_at(0, 0, 42);
    let _ok1 = sh.cas_insert_at(1, 100, 200);

    let _v0_after = sh.read_at(0);
    let _v1_after = sh.read_at(1);
    // (Tracked tokens leak at end of scope — same convention as
    // arc.rs's main() example.)
}

// N-slot generalisation
// =====================
//
// For a true N-slot table (e.g. modeling `HashTableEntry[N]`):
//
//   pub struct InnerShard {
//       pub rc_cell: PAtomicU64,
//       pub slots: Vec<PAtomicU64>,
//   }
//
//   pub tracked struct GhostStuff {
//       pub tracked slot_perms: Map<int, PermissionU64>,
//       pub tracked rc_perm: PermissionU64,
//       pub tracked counter_token: ...,
//   }
//
// GhostStuff::wf adds:
//
//   forall |i: int| 0 <= i < slot_count ==>
//       #[trigger] self.slot_perms[i]@.patomic == inner.slots[i].id()
//
// Where `slot_count` and `slots` are threaded via the AtomicInvariant
// parameter pack. read_at / cas_insert_at take an index, extract the
// corresponding permission from `slot_perms`, then do the
// load / compare_exchange. The hardcoded N=2 case above demonstrates
// the per-slot composition; the Vec case is the same structurally,
// just with map-indexed permission lookup instead of branched
// destructuring.

} // verus!

fn main() { }
