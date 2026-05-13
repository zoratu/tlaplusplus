// verification/verus/atomic_ptr_with_epoch.rs
//
// T13.4 Gap 1 — proof-of-concept per verus-lang/verus#2437 comment from
// @tjhance: the RCU/seqlock atomic-pointer-swap pattern is verifiable
// today via `tokenized_state_machine!` + storage_option + counting
// permissions. No new `vstd` primitive needed.
//
// What this file proves
// =====================
//
// `EpochProtocol<T>` is a tokenized state machine modeling the
// single-epoch RCU swap pattern that maps directly onto gap 1 from
// `T13.2-T13.4-design.md`:
//
//   - `stored: Option<T>` (storage_option) — the linear permission token
//     for the current allocation. Deposited on publish (writer hands a
//     fresh permission to the protocol), withdrawn on reclaim (writer
//     receives it back once all readers have released).
//
//   - `main_counter: Option<(nat, T)>` (variable) — current epoch state.
//     `None` = no live allocation; `Some((n, t))` = `n` outstanding
//     readers of the allocation referenced by `t`.
//
//   - `read_ref: Multiset<T>` (multiset) — one token per outstanding
//     reader.
//
// Five proof-level operations exposed via `EpochInstance<T>`:
//
//   1. `new() -> (Instance, Counter)` — fresh protocol instance, no
//      allocation.
//   2. `publish(&mut counter, t)` — deposit the permission, go from no
//      allocation to "(0, t)". Models the writer's initial publish
//      sequence (mmap + AtomicPtr::store(new, Release)).
//   3. `reclaim(&mut counter) -> t` — withdraw the permission. Requires
//      `count == 0` (all readers drained). Models the post-grace-period
//      free.
//   4. `acquire_read(&mut counter) -> ReadRef` — atomically increment
//      reader count and mint a `ReadRef` token. Models `AtomicPtr::load`.
//   5. `release_read(&mut counter, ReadRef)` — decrement reader count
//      and consume the `ReadRef`. Models `PtrGuard::destroy` (or drop).
//   6. `read_guards(&ReadRef) -> &T` — borrow the stored linear
//      permission through a live `ReadRef`. This is the `guard`
//      instruction tjhance pointed at — gives `&T` access without
//      consuming the permission, for the lifetime of the `ReadRef`.
//
// The protocol invariant `main_inv` proves: the `read_ref` multiset's
// size equals the counter's `count`, and every read_ref token names
// the same `T` that's stored.
//
// What this file does NOT yet prove
// ==================================
//
// The single-epoch protocol matches tjhance's sketched `AtomicPtrWithEpoch`
// where `swap` BLOCKS until readers from the previous epoch are
// destroyed. The shipping `FingerprintShard` resize path uses a
// non-blocking multi-epoch pattern: OLD readers continue during the
// rehash; OLD is freed only on the NEXT resize via `cleanup_old_memory`
// (page_aligned_fingerprint_store.rs line 545).
//
// The multi-epoch generalisation is a tractable extension on top of
// this skeleton: replace `stored: Option<T>` with `epochs: Map<EpochId, T>`,
// tag each `ReadRef` with an `EpochId`, and parameterise `release_read`
// on which epoch's counter to decrement. The protocol shape is the
// same — keyed by epoch instead of single-Option. This file lands the
// single-epoch version first to validate the approach.
//
// Status: design-time skeleton; verifies the protocol invariant and
// each transition's inductive obligation.

#![allow(unused_imports)]
#![allow(dead_code)]

use verus_builtin::*;
use vstd::prelude::*;
use vstd::multiset::*;
use verus_state_machines_macros::tokenized_state_machine;

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

    // Writer publishes a new allocation. Goes from "no allocation" to
    // "(0 readers, t)" and deposits the linear permission `t` into the
    // protocol's storage slot.
    transition!{
        publish(t: T) {
            require pre.main_counter.is_none();
            update main_counter = Some((0, t));
            deposit stored += Some(t);
        }
    }

    // Writer reclaims the allocation. Requires count == 0 (the
    // grace-period precondition). Withdraws the linear permission.
    transition!{
        reclaim() {
            require let Some((count, t)) = pre.main_counter;
            require count == 0;
            update main_counter = None;
            withdraw stored -= Some(t);
        }
    }

    // Reader acquires a read reference. Increments the counter and mints
    // a `read_ref` token for the currently-stored value.
    transition!{
        acquire_read() {
            require let Some((count, t)) = pre.main_counter;
            update main_counter = Some((count + 1, t));
            add read_ref += { t };
        }
    }

    // Reader releases a read reference. Decrements the counter and
    // consumes the `read_ref` token.
    transition!{
        release_read(t1: T) {
            remove read_ref -= { t1 };
            require let Some((count, t2)) = pre.main_counter;
            assert count >= 1;
            assert t1 == t2;
            update main_counter = Some(((count - 1) as nat, t1));
        }
    }

    // The `guard` instruction: through a live `read_ref`, expose a
    // shared `&T` borrow of the stored permission. The shared reference
    // lives for the duration of the `read_ref` token's borrow.
    property!{
        read_ref_guards(t: T) {
            have read_ref >= { t };
            guard stored >= Some(t);
        }
    }

    // Any two outstanding `read_ref` tokens must refer to the same `T`
    // (the current epoch's allocation).
    property!{
        readers_agree(t1: T, t2: T) {
            have read_ref >= { t1 };
            have read_ref >= { t2 };
            assert t1 == t2;
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

verus!{

// Tracked wrapper structs lift the tokenized state machine's tokens into
// named Rust types so callers can pass them around without naming the
// macro-generated paths. Same pattern as the counting-permissions
// example at verus/examples/state_machines/counting.rs.

tracked struct EpochInstance<T> {
    tracked instance: EpochProtocol::Instance<T>,
}

tracked struct EpochCounter<T> {
    tracked token: EpochProtocol::main_counter<T>,
}

tracked struct ReadRef<T> {
    tracked token: EpochProtocol::read_ref<T>,
}

impl<T> EpochCounter<T> {
    pub closed spec fn instance_id(self) -> InstanceId {
        self.token.instance_id()
    }

    pub closed spec fn value(self) -> Option<(nat, T)> {
        self.token.value()
    }
}

impl<T> ReadRef<T> {
    pub closed spec fn instance_id(self) -> InstanceId {
        self.token.instance_id()
    }

    pub closed spec fn value(self) -> T {
        self.token.element()
    }
}

impl<T> EpochInstance<T> {
    pub closed spec fn id(self) -> InstanceId {
        self.instance.id()
    }

    /// Mint a fresh protocol instance. The returned counter starts in
    /// the "no allocation" state.
    proof fn new() -> (tracked res: (EpochInstance<T>, EpochCounter<T>))
        ensures
            res.1.instance_id() == res.0.id(),
            res.1.value() === None,
    {
        let tracked (Tracked(inst), Tracked(c), Tracked(_r)) =
            EpochProtocol::Instance::new(None);
        (EpochInstance { instance: inst }, EpochCounter { token: c })
    }

    /// Writer: publish a new allocation. Consumes the linear permission
    /// `t` (deposited into the protocol's `stored` slot) and transitions
    /// the counter from "no allocation" to "(0 readers, t)".
    proof fn publish(
        tracked &self,
        tracked counter: &mut EpochCounter<T>,
        tracked t: T,
    )
        requires
            old(counter).instance_id() == self.id(),
            old(counter).value() === None,
        ensures
            final(counter).instance_id() == self.id(),
            final(counter).value() === Some((0, t)),
    {
        self.instance.publish(t, t, &mut counter.token);
    }

    /// Writer: reclaim the allocation. Returns the original linear
    /// permission `t`. Requires `count == 0`, i.e. the grace period
    /// has elapsed and all readers have released.
    proof fn reclaim(
        tracked &self,
        tracked counter: &mut EpochCounter<T>,
    ) -> (tracked t: T)
        requires
            old(counter).instance_id() == self.id(),
            match old(counter).value() {
                None => false,
                Some((count, _)) => count == 0,
            },
        ensures
            final(counter).instance_id() == self.id(),
            final(counter).value() === None,
            t == old(counter).value().unwrap().1,
    {
        self.instance.reclaim(&mut counter.token)
    }

    /// Reader: acquire a read reference. Increments the counter and
    /// mints a `ReadRef` token tied to the currently-stored value.
    proof fn acquire_read(
        tracked &self,
        tracked counter: &mut EpochCounter<T>,
    ) -> (tracked read_ref: ReadRef<T>)
        requires
            old(counter).instance_id() == self.id(),
            old(counter).value().is_some(),
        ensures
            final(counter).instance_id() == self.id(),
            read_ref.instance_id() == self.id(),
            match old(counter).value() {
                None => false,
                Some((count, t)) =>
                    final(counter).value() == Some((count + 1, t))
                    && read_ref.value() == t,
            },
    {
        ReadRef { token: self.instance.acquire_read(&mut counter.token) }
    }

    /// Reader: release a read reference. Decrements the counter and
    /// consumes the `ReadRef`.
    proof fn release_read(
        tracked &self,
        tracked counter: &mut EpochCounter<T>,
        tracked read_ref: ReadRef<T>,
    )
        requires
            old(counter).instance_id() == self.id(),
            old(counter).value().is_some(),
            read_ref.instance_id() == self.id(),
        ensures
            final(counter).instance_id() == self.id(),
            match old(counter).value() {
                None => false,
                Some((count, t)) =>
                    count >= 1
                    && final(counter).value() == Some(((count - 1) as nat, t)),
            },
    {
        self.instance.release_read(read_ref.token.element(), &mut counter.token, read_ref.token)
    }

    /// Reader: borrow the stored linear permission through a live
    /// `ReadRef`. This is the `guard` instruction — gives `&T` access
    /// for the lifetime of the `ReadRef` without consuming the
    /// permission. In the eventual exec wiring, this is what lets the
    /// reader dereference the `HashTableEntry` slice through the loaded
    /// pointer.
    proof fn read_guards<'a>(
        tracked &self,
        tracked read_ref: &'a ReadRef<T>,
    ) -> (tracked borrowed_t: &'a T)
        requires
            read_ref.instance_id() == self.id(),
        ensures
            borrowed_t == read_ref.value(),
    {
        self.instance.read_ref_guards(read_ref.value(), &read_ref.token)
    }
}

}

fn main() { }
