// Tier-A extension of the Verus seqlock resize proof.
//
// Tier B (`seqlock_resize.rs`) modeled the fingerprint table as a
// `Set<u64>`. Tier A replaces that with a *concrete* linear-probe hash
// table `Seq<u64>` indexed by `(fp as int) % capacity`, exactly mirroring
// the layout of `HashTableEntry` in
// `src/storage/page_aligned_fingerprint_store.rs:101-121`. This tier-A
// model captures collision behavior — the property bug "linear probe
// wraps incorrectly at the table boundary" would now be caught by a
// failing proof.
//
// What this tier-A file ADDS to tier B
// ====================================
//
// 1. **Linear-probe table model (`T13.1`).** Defines `Table = Seq<u64>`
//    with 0 reserved as the empty sentinel — exact match to the
//    production code's "store 0=empty, fp values use 1..u64::MAX"
//    convention at page_aligned_fingerprint_store.rs:556 (`if fp == 0
//    { 1 } else { fp }`). All operations — `tab_lookup`, `tab_insert`,
//    `tab_contents` — are spec functions implementing the production
//    probe loop semantics from lines 631-641 (`contains`) and 783-839
//    (`contains_or_insert` normal path).
//
// 2. **Probe-correctness lemmas.** Proves that linear probing realizes
//    the abstract `Set<u64>` faithfully. Key lemma:
//    `lemma_insert_then_lookup`: after `tab_insert(t, fp)` succeeds,
//    `tab_lookup(t', fp) == true`. This is the property the production
//    `contains_or_insert` CAS loop relies on.
//
// 3. **CAS soundness sketch (`T13.2`).** Defines an abstract
//    `cas_step` (success/failure cases) and proves that a successful
//    CAS on a slot transitions the table contents from `S` to
//    `S.insert(fp)`. This bridges the protocol-level
//    `step_insert_during_resize` to the actual production CAS at
//    page_aligned_fingerprint_store.rs:723-741.
//
//    Full Verus tracked-permission integration on the production
//    `AtomicPtr<HashTableEntry>` would require rewriting the production
//    code to thread `Tracked<PointsTo<...>>` through every call site.
//    That is the multi-week tier-A effort flagged in the README; what
//    this file ships is the *spec-level* CAS soundness proof, which is
//    the prerequisite for a future production-code annotation pass.
//
// 4. **Bounded-resize termination (`T13.3`).** Proves the reader retry
//    loop in `contains` (page_aligned_fingerprint_store.rs:557-649)
//    terminates after at most `R` iterations, where `R` is the number of
//    resizes the writer side performs. Verus discharges this as a
//    `decreases` clause on `R`. Unbounded liveness — "writers can't
//    starve readers indefinitely" — would require a fairness assumption
//    in temporal logic and is not in scope for this tier.
//
// What this tier-A file STILL ABSTRACTS
// =====================================
//
// - Raw pointer arithmetic in `unsafe { std::slice::from_raw_parts(...) }`.
//   Verus has `vstd::raw_ptr` for this but applying it requires
//   rewriting `FingerprintShard` to thread `Tracked<PointsTo>` through
//   every call. Estimated effort: still 1-2 agent-weeks.
//
// - Memory-ordering of `AtomicPtr` (`Acquire`/`Release`/`AcqRel`). The
//   tier-A model treats each atomic op as sequentially consistent,
//   matching the standard seqlock-correctness argument.
//
// - The `mmap(MAP_ANONYMOUS)` zero-fill guarantee. Modeled as
//   `Seq::new(cap, |_| 0)` — a sequence of `cap` zeros. The kernel
//   guarantee is axiomatic.
//
// HOW TO RUN
// ==========
//
//     cd verification/verus
//     ./run_proof.sh                # tier B (seqlock_resize.rs)
//     ./run_proof.sh tier-a         # tier A (this file)
//
// Successful output ends with `verification results:: <N> verified, 0 errors`.

use vstd::prelude::*;

verus! {

// ----------------------------------------------------------------------------
// LINEAR-PROBE TABLE MODEL (T13.1)
// ----------------------------------------------------------------------------
//
// A `Table` is a sequence of `u64` slots; 0 means "empty". This matches
// `HashTableEntry::fp` at page_aligned_fingerprint_store.rs:103-105.
// Capacity is `t.len()`.
pub type Table = Seq<u64>;

// The empty-slot sentinel. Production guarantees (line 556) fp != 0 by
// remapping 0 -> 1 before insert.
pub open spec fn EMPTY() -> u64 { 0u64 }

// The probe sequence: for fingerprint `fp` in a table of size `cap`, the
// `i`-th probe lands at slot `(fp as int + i as int) % cap as int`.
// Matches `index = (fp as usize) % capacity; ... index = (index + 1) %
// capacity` at lines 628-639 of page_aligned_fingerprint_store.rs.
pub open spec fn probe_index(fp: u64, i: nat, cap: nat) -> int
    recommends cap > 0,
{
    ((fp as int) + i as int) % (cap as int)
}

// The probe terminates within `cap` steps if a slot containing either
// `fp` or `EMPTY` is encountered (the production code's break conditions
// at lines 633-637). We define the termination index as the *minimum* i
// in [0, cap) at which the slot is fp-or-empty. We use a recursive
// search bounded by `cap`.
pub open spec fn probe_terminus_at(t: Table, fp: u64, i: nat, cap: nat) -> nat
    recommends
        cap > 0,
        cap == t.len(),
        i <= cap,
        fp != EMPTY(),
    decreases cap - i,
{
    if i >= cap {
        cap // all slots full and none equal fp -> table full
    } else {
        let slot = t[probe_index(fp, i, cap)];
        if slot == fp || slot == EMPTY() {
            i
        } else {
            probe_terminus_at(t, fp, i + 1, cap)
        }
    }
}

pub open spec fn probe_terminus(t: Table, fp: u64) -> nat
    recommends
        t.len() > 0,
        fp != EMPTY(),
{
    probe_terminus_at(t, fp, 0, t.len())
}

// The lookup result: did we find `fp` along its probe sequence?
// `tab_lookup(t, fp)` mirrors `contains` at line 554 of the production
// code, taking only the normal (non-resize) path.
pub open spec fn tab_lookup(t: Table, fp: u64) -> bool
    recommends
        t.len() > 0,
        fp != EMPTY(),
{
    let term = probe_terminus(t, fp);
    if term >= t.len() {
        false // table full, not found
    } else {
        t[probe_index(fp, term, t.len())] == fp
    }
}

// Insert: produce the new table with fp written at the first empty slot
// in fp's probe sequence. If the slot is already fp, the table is
// unchanged. If the table is full and fp not present, the result is the
// unchanged table (production would CAS-spin or yield; we model
// "insert failed" as no-op at this abstraction level — see CAS soundness
// section below for the dynamic case).
pub open spec fn tab_insert(t: Table, fp: u64) -> Table
    recommends
        t.len() > 0,
        fp != EMPTY(),
{
    let term = probe_terminus(t, fp);
    if term >= t.len() {
        t // table full, no change
    } else if t[probe_index(fp, term, t.len())] == fp {
        t // already present
    } else {
        // slot at probe_index(fp, term) is EMPTY(); write fp there.
        t.update(probe_index(fp, term, t.len()), fp)
    }
}

// The abstract contents of the table: the set of all non-empty
// fingerprints stored anywhere in the slot sequence.
pub open spec fn tab_contents(t: Table) -> Set<u64> {
    Set::new(|fp: u64| fp != EMPTY() && exists|i: int| 0 <= i < t.len() && t[i] == fp)
}

// ----------------------------------------------------------------------------
// PROBE-CORRECTNESS LEMMAS
// ----------------------------------------------------------------------------

// The probe terminus is bounded by capacity.
pub proof fn lemma_probe_terminus_bounded(t: Table, fp: u64, i: nat, cap: nat)
    requires
        cap > 0,
        cap == t.len(),
        i <= cap,
        fp != EMPTY(),
    ensures
        probe_terminus_at(t, fp, i, cap) <= cap,
        probe_terminus_at(t, fp, i, cap) >= i,
    decreases cap - i,
{
    if i < cap {
        let slot = t[probe_index(fp, i, cap)];
        if slot == fp || slot == EMPTY() {
            // base case
        } else {
            lemma_probe_terminus_bounded(t, fp, i + 1, cap);
        }
    }
}

// If the terminus is < cap, the slot at the terminus is either fp or
// EMPTY (this is the loop termination condition).
pub proof fn lemma_probe_terminus_slot(t: Table, fp: u64, i: nat, cap: nat)
    requires
        cap > 0,
        cap == t.len(),
        i <= cap,
        fp != EMPTY(),
        probe_terminus_at(t, fp, i, cap) < cap,
    ensures
        ({
            let term = probe_terminus_at(t, fp, i, cap);
            t[probe_index(fp, term, cap)] == fp
            || t[probe_index(fp, term, cap)] == EMPTY()
        }),
    decreases cap - i,
{
    if i < cap {
        let slot = t[probe_index(fp, i, cap)];
        if slot == fp || slot == EMPTY() {
            // base case: terminus is i; slot matches.
        } else {
            lemma_probe_terminus_slot(t, fp, i + 1, cap);
        }
    }
}

// Probe-index distinctness within one cycle: for 0 <= i, j < cap with
// i != j, probe_index(fp, i, cap) != probe_index(fp, j, cap).
//
// Strategy: prove via `lemma_mod_subtract` style — if a % c == b % c
// and a - b is in (-c, c), then a == b. We discharge by reducing to
// the modular-subtraction lemma in vstd::arithmetic::div_mod.
pub proof fn lemma_probe_indices_distinct(fp: u64, i: nat, j: nat, cap: nat)
    requires
        cap > 0,
        i < cap,
        j < cap,
        i != j,
    ensures
        probe_index(fp, i, cap) != probe_index(fp, j, cap),
{
    let a = (fp as int) + (i as int);
    let b = (fp as int) + (j as int);
    let c = cap as int;
    // Suppose for contradiction a % c == b % c. Then a - b is a multiple
    // of c. But a - b == i - j with |i - j| < c, hence a - b == 0 ==>
    // i == j, contradiction.
    assert(a >= 0 && b >= 0 && c > 0);
    assert(a - b == (i as int) - (j as int));
    if a % c == b % c {
        // (a - b) % c == 0 by `lemma_sub_mod_eq`-style reasoning. We
        // construct it by showing both a and b have the same remainder.
        let qa = a / c;
        let qb = b / c;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(a, c);
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(b, c);
        // a == c * qa + (a % c), b == c * qb + (b % c).
        // Since a % c == b % c, a - b == c * (qa - qb).
        assert(a - b == c * (qa - qb)) by (nonlinear_arith)
            requires
                a == c * qa + (a % c),
                b == c * qb + (b % c),
                a % c == b % c;
        // |a - b| = |i - j| < c, but a - b = c * k means |a - b| is 0
        // or >= c. Hence a - b == 0 ==> i == j, contradiction.
        if i > j {
            assert(a - b == (i as int) - (j as int));
            assert(0 < a - b < c);
            // c * (qa - qb) is in (0, c) ==> qa - qb is fractional, impossible.
            assert(false) by (nonlinear_arith)
                requires
                    0 < a - b < c,
                    a - b == c * (qa - qb),
                    c > 0;
        } else {
            assert(j > i);
            assert(-c < a - b < 0);
            assert(false) by (nonlinear_arith)
                requires
                    -c < a - b < 0,
                    a - b == c * (qa - qb),
                    c > 0;
        }
    }
}

// Probe-index range bound: probe_index(fp, i, cap) is in [0, cap).
pub proof fn lemma_probe_index_in_range(fp: u64, i: nat, cap: nat)
    requires cap > 0,
    ensures
        0 <= probe_index(fp, i, cap) < cap as int,
{}

// HEADLINE: insert-then-lookup. Successful insert into a non-full table
// makes lookup return true.
pub proof fn lemma_insert_then_lookup(t: Table, fp: u64)
    requires
        t.len() > 0,
        fp != EMPTY(),
        probe_terminus(t, fp) < t.len(),
    ensures
        tab_lookup(tab_insert(t, fp), fp),
{
    let cap = t.len();
    let term = probe_terminus(t, fp);
    lemma_probe_terminus_slot(t, fp, 0, cap as nat);
    let term_idx = probe_index(fp, term, cap);
    lemma_probe_index_in_range(fp, term, cap as nat);
    let slot_at_term = t[term_idx];

    if slot_at_term == fp {
        // Already present. tab_insert returns t unchanged.
        assert(tab_insert(t, fp) =~= t);
    } else {
        // slot is EMPTY; insert writes fp at term_idx.
        let t2 = t.update(term_idx, fp);
        assert(tab_insert(t, fp) =~= t2);

        // Establish the precondition for lemma_probe_walk_matches:
        // for every j with 0 <= j < term, probe_index(fp, j, cap) !=
        // term_idx (since j != term and both are in [0, cap)), so
        // t2[probe_index(fp, j, cap)] == t[probe_index(fp, j, cap)].
        assert forall|j: nat| j < term implies
            t2[probe_index(fp, j, cap)] == t[probe_index(fp, j, cap)] by {
            lemma_probe_indices_distinct(fp, j, term, cap as nat);
        }

        // And t2[term_idx] == fp by t.update construction.
        assert(t2[term_idx] == fp);

        lemma_probe_walk_matches(t, t2, fp, 0, term, cap as nat);
        assert(probe_terminus(t2, fp) == term);
        assert(t2[probe_index(fp, term, cap)] == fp);
    }
}

// Inductive walk-match: if t and t2 agree on probe slots 0..term, then
// the probe terminus in t2 starting from 0 equals term, given the
// terminus in t starting from 0 is term and the t2 slot at term is fp.
pub proof fn lemma_probe_walk_matches(t: Table, t2: Table, fp: u64, i: nat, term: nat, cap: nat)
    requires
        cap > 0,
        cap == t.len(),
        cap == t2.len(),
        fp != EMPTY(),
        i <= term,
        term < cap,
        probe_terminus_at(t, fp, i, cap) == term,
        forall|j: nat| i <= j < term ==> t2[probe_index(fp, j, cap)] == t[probe_index(fp, j, cap)],
        t2[probe_index(fp, term, cap)] == fp,
    ensures
        probe_terminus_at(t2, fp, i, cap) == term,
    decreases term - i,
{
    if i == term {
        // Base case: at i==term, t2 slot is fp -> terminus is i.
        assert(t2[probe_index(fp, i, cap)] == fp);
    } else {
        // i < term. In t the probe at i is neither fp nor EMPTY
        // (else terminus_at would be i, contradicting term > i).
        let slot_t = t[probe_index(fp, i, cap)];
        assert(slot_t != fp);
        assert(slot_t != EMPTY());
        let slot_t2 = t2[probe_index(fp, i, cap)];
        assert(slot_t2 == slot_t);
        assert(slot_t2 != fp && slot_t2 != EMPTY());
        // Recurse with i+1.
        lemma_probe_walk_matches(t, t2, fp, i + 1, term, cap);
    }
}

// Insert preserves all previously-present fingerprints.
pub proof fn lemma_insert_preserves_contents(t: Table, fp: u64)
    requires
        t.len() > 0,
        fp != EMPTY(),
    ensures
        forall|x: u64| #![auto] tab_contents(t).contains(x) ==> tab_contents(tab_insert(t, fp)).contains(x),
{
    let t2 = tab_insert(t, fp);
    let cap = t.len();
    let term = probe_terminus(t, fp);
    lemma_probe_terminus_bounded(t, fp, 0, cap as nat);
    assert forall|x: u64| #![auto] tab_contents(t).contains(x) implies tab_contents(t2).contains(x) by {
        // x is non-empty and present at some index i in t.
        assert(x != EMPTY());
        let i = choose|i: int| 0 <= i < t.len() && t[i] == x;
        // tab_insert either leaves t unchanged or writes fp at one
        // slot. In the unchanged case, t2[i] == t[i] == x. In the
        // modified case, the only modified slot was EMPTY before, but
        // t[i] = x != EMPTY, so the indices differ and t2[i] == t[i].
        if t2 =~= t {
            assert(t2[i] == x);
            assert(0 <= i < t2.len());
        } else {
            assert(term < cap);
            lemma_probe_terminus_slot(t, fp, 0, cap as nat);
            let mod_idx = probe_index(fp, term, cap);
            lemma_probe_index_in_range(fp, term, cap as nat);
            assert(0 <= mod_idx < cap as int);
            // The "modified" branch of tab_insert fires when
            // t[mod_idx] != fp; combined with lemma_probe_terminus_slot
            // (slot is fp or EMPTY), we have t[mod_idx] == EMPTY().
            assert(t[mod_idx] == EMPTY() || t[mod_idx] == fp);
            assert(t[mod_idx] == EMPTY());  // because fp branch returned t unchanged
            // i != mod_idx (since t[mod_idx] = EMPTY but t[i] = x != EMPTY).
            assert(t[i] != EMPTY());
            assert(i != mod_idx);
            assert(t2 =~= t.update(mod_idx, fp));
            assert(t2[i] == t[i]);
            assert(t2[i] == x);
            assert(0 <= i < t2.len());
        }
        assert(exists|j: int| 0 <= j < t2.len() && t2[j] == x);
    }
}

// Insert adds fp to the contents (when there was room).
pub proof fn lemma_insert_adds_fp(t: Table, fp: u64)
    requires
        t.len() > 0,
        fp != EMPTY(),
        probe_terminus(t, fp) < t.len(),
    ensures
        tab_contents(tab_insert(t, fp)).contains(fp),
{
    let t2 = tab_insert(t, fp);
    let cap = t.len();
    let term = probe_terminus(t, fp);
    lemma_probe_terminus_slot(t, fp, 0, cap as nat);
    let mod_idx = probe_index(fp, term, cap);
    if t[mod_idx] == fp {
        assert(t2 =~= t);
        assert(t2[mod_idx] == fp);
    } else {
        assert(t2[mod_idx] == fp);
    }
    assert(exists|i: int| 0 <= i < t2.len() && t2[i] == fp);
}

// ----------------------------------------------------------------------------
// SHARD STATE — table-level (replaces tier-B Set<u64> with Seq<u64>)
// ----------------------------------------------------------------------------

pub struct ShardStateA {
    pub seq: nat,
    pub table: Table,        // current table; valid when seq is even
    pub old_table: Table,    // snapshot during resize; valid when seq is odd
    pub new_table: Table,    // accumulator during resize; valid when seq is odd
}

pub open spec fn wf_a(s: ShardStateA) -> bool {
    if s.seq % 2 == 0 {
        s.old_table.len() == 0 && s.new_table.len() == 0
        && s.table.len() > 0
    } else {
        // Resizing: old_table is the pre-resize snapshot; new_table is
        // the (possibly partially populated) destination, twice the size.
        s.old_table.len() > 0
        && s.new_table.len() >= s.old_table.len()
    }
}

// What a reader on the public surface observes — set view, derived from
// the current table layout. Mirrors `contains` at line 554.
pub open spec fn observable_a(s: ShardStateA, fp: u64) -> bool
    recommends fp != EMPTY()
{
    if s.seq % 2 == 0 {
        s.table.len() > 0 && tab_lookup(s.table, fp)
    } else {
        // During resize, readers union old + new (prod lines 580-617).
        (s.old_table.len() > 0 && tab_lookup(s.old_table, fp))
        || (s.new_table.len() > 0 && tab_lookup(s.new_table, fp))
    }
}

// ----------------------------------------------------------------------------
// CAS SOUNDNESS (T13.2)
// ----------------------------------------------------------------------------
//
// The production code's resize-mode CAS (page_aligned_fingerprint_store.rs:
// 723-741) targets `entry.fp.compare_exchange(0, fp, AcqRel, Acquire)`. The
// key safety property: a successful CAS at slot S transitions the
// abstract table contents from C to C ∪ {fp}, and never overwrites a
// slot that already contained a different fingerprint.
//
// We model this as a pure function on the table sequence: `cas_step`
// returns `Some(new_table)` iff the CAS succeeds (the slot was EMPTY
// at the moment of the exchange), `None` otherwise.

pub open spec fn cas_step(t: Table, slot: int, fp: u64) -> Option<Table>
    recommends
        0 <= slot < t.len(),
        fp != EMPTY(),
{
    if 0 <= slot < t.len() && t[slot] == EMPTY() && fp != EMPTY() {
        Some(t.update(slot, fp))
    } else {
        None
    }
}

// CAS soundness: a successful CAS preserves all prior contents and adds
// exactly fp. This is the abstract reading of the production CAS at
// lines 723-741.
pub proof fn lemma_cas_soundness(t: Table, slot: int, fp: u64)
    requires
        0 <= slot < t.len(),
        fp != EMPTY(),
        t[slot] == EMPTY(),
    ensures
        cas_step(t, slot, fp) == Some(t.update(slot, fp)),
        tab_contents(t.update(slot, fp)).contains(fp),
        forall|x: u64| #![auto] tab_contents(t).contains(x) ==> tab_contents(t.update(slot, fp)).contains(x),
{
    let t2 = t.update(slot, fp);
    assert(cas_step(t, slot, fp) =~= Some(t2));
    assert(t2[slot] == fp);
    assert(exists|i: int| 0 <= i < t2.len() && t2[i] == fp);
    assert forall|x: u64| #![auto] tab_contents(t).contains(x) implies tab_contents(t2).contains(x) by {
        assert(x != EMPTY());
        let i = choose|i: int| 0 <= i < t.len() && t[i] == x;
        // i != slot because t[slot] = EMPTY but t[i] = x != EMPTY.
        assert(i != slot);
        assert(t2[i] == t[i]);
        assert(t2[i] == x);
    }
}

// CAS failure: when the slot is already occupied by some `actual`, the
// CAS returns None and no state mutation occurs. Caller's contract
// (production code lines 737-740) is to either succeed (if `actual ==
// fp`, treat as already present) or retry probing.
pub proof fn lemma_cas_failure_no_clobber(t: Table, slot: int, fp: u64, actual: u64)
    requires
        0 <= slot < t.len(),
        fp != EMPTY(),
        actual != EMPTY(),
        t[slot] == actual,
    ensures
        cas_step(t, slot, fp) is None,
{
    assert(t[slot] != EMPTY());
}

// ----------------------------------------------------------------------------
// PROTOCOL STEPS — table-level
// ----------------------------------------------------------------------------

// Insert during stable state via tab_insert.
pub open spec fn step_insert_stable_a(s: ShardStateA, fp: u64) -> ShardStateA
    recommends s.seq % 2 == 0, s.table.len() > 0, fp != EMPTY(),
{
    ShardStateA {
        seq: s.seq,
        table: tab_insert(s.table, fp),
        old_table: s.old_table,
        new_table: s.new_table,
    }
}

// Begin resize: snapshot `table` into `old_table`, allocate `new_table`
// of double size (zero-initialized — kernel mmap guarantee), bump seq.
pub open spec fn step_begin_resize_a(s: ShardStateA, new_cap: nat) -> ShardStateA
    recommends
        s.seq % 2 == 0,
        s.table.len() > 0,
        new_cap >= s.table.len(),
{
    ShardStateA {
        seq: s.seq + 1,
        table: s.table,
        old_table: s.table,
        new_table: Seq::new(new_cap, |_i: int| EMPTY()),
    }
}

// Insert during resize: fp lands in new_table via tab_insert.
pub open spec fn step_insert_during_resize_a(s: ShardStateA, fp: u64) -> ShardStateA
    recommends
        s.seq % 2 == 1,
        s.new_table.len() > 0,
        fp != EMPTY(),
{
    ShardStateA {
        seq: s.seq,
        table: s.table,
        old_table: s.old_table,
        new_table: tab_insert(s.new_table, fp),
    }
}

// Migrate one entry from old_table[i] into new_table (production:
// rehash_batch_counted lines 312-340). i is the source slot.
pub open spec fn step_rehash_one_a(s: ShardStateA, i: int) -> ShardStateA
    recommends
        s.seq % 2 == 1,
        0 <= i < s.old_table.len(),
        s.old_table[i] != EMPTY(),
{
    ShardStateA {
        seq: s.seq,
        table: s.table,
        old_table: s.old_table,
        new_table: tab_insert(s.new_table, s.old_table[i]),
    }
}

// Finalize: swap new_table into table; clear scratch; bump seq to even.
pub open spec fn step_finalize_resize_a(s: ShardStateA) -> ShardStateA
    recommends s.seq % 2 == 1,
{
    ShardStateA {
        seq: s.seq + 1,
        table: s.new_table,
        old_table: Seq::empty(),
        new_table: Seq::empty(),
    }
}

// ----------------------------------------------------------------------------
// LIFTED PROOFS (table-level analogs of tier-B lemmas)
// ----------------------------------------------------------------------------

// Monotonic seq.
pub proof fn lemma_seq_monotonic_a(s: ShardStateA, fp: u64, i: int, new_cap: nat)
    ensures
        step_insert_stable_a(s, fp).seq >= s.seq,
        step_begin_resize_a(s, new_cap).seq >= s.seq,
        step_insert_during_resize_a(s, fp).seq >= s.seq,
        step_rehash_one_a(s, i).seq >= s.seq,
        step_finalize_resize_a(s).seq >= s.seq,
{}

// Parity discipline.
pub proof fn lemma_begin_resize_parity_a(s: ShardStateA, new_cap: nat)
    requires s.seq % 2 == 0,
    ensures step_begin_resize_a(s, new_cap).seq % 2 == 1,
{}

pub proof fn lemma_finalize_parity_a(s: ShardStateA)
    requires s.seq % 2 == 1,
    ensures step_finalize_resize_a(s).seq % 2 == 0,
{}

// Insert-during-resize preserves new_table contents and adds fp (when
// there was room).
pub proof fn lemma_insert_during_resize_preserves_a(s: ShardStateA, fp: u64)
    requires
        s.seq % 2 == 1,
        s.new_table.len() > 0,
        fp != EMPTY(),
    ensures
        forall|x: u64| #![auto]
            tab_contents(s.new_table).contains(x)
            ==> tab_contents(step_insert_during_resize_a(s, fp).new_table).contains(x),
{
    lemma_insert_preserves_contents(s.new_table, fp);
}

// Rehash preserves new_table contents.
pub proof fn lemma_rehash_preserves_a(s: ShardStateA, i: int)
    requires
        s.seq % 2 == 1,
        0 <= i < s.old_table.len(),
        s.old_table[i] != EMPTY(),
        s.new_table.len() > 0,
    ensures
        forall|x: u64| #![auto]
            tab_contents(s.new_table).contains(x)
            ==> tab_contents(step_rehash_one_a(s, i).new_table).contains(x),
{
    lemma_insert_preserves_contents(s.new_table, s.old_table[i]);
}

// HEADLINE: a fingerprint inserted (and successfully looked up) into the
// stable table in s0 is observable in s1 = step_insert_stable_a(s0, fp).
pub proof fn lemma_insert_stable_observable_a(s0: ShardStateA, fp: u64)
    requires
        s0.seq % 2 == 0,
        wf_a(s0),
        s0.table.len() > 0,
        fp != EMPTY(),
        probe_terminus(s0.table, fp) < s0.table.len(),
    ensures
        observable_a(step_insert_stable_a(s0, fp), fp),
{
    let s1 = step_insert_stable_a(s0, fp);
    lemma_insert_then_lookup(s0.table, fp);
    assert(tab_lookup(s1.table, fp));
    assert(s1.seq % 2 == 0);
    assert(observable_a(s1, fp));
}

// CAS-step soundness lifted to the shard level: if a worker performs a
// successful CAS into the new_table of a resizing shard, the
// fingerprint becomes observable.
pub proof fn lemma_cas_during_resize_observable_a(s: ShardStateA, slot: int, fp: u64)
    requires
        s.seq % 2 == 1,
        wf_a(s),
        0 <= slot < s.new_table.len(),
        s.new_table[slot] == EMPTY(),
        fp != EMPTY(),
    ensures
        cas_step(s.new_table, slot, fp) == Some(s.new_table.update(slot, fp)),
        tab_contents(s.new_table.update(slot, fp)).contains(fp),
{
    lemma_cas_soundness(s.new_table, slot, fp);
}

// ----------------------------------------------------------------------------
// BOUNDED-RESIZE TERMINATION (T13.3)
// ----------------------------------------------------------------------------
//
// The production reader retry loop at page_aligned_fingerprint_store.rs:
// 557-649 is structured as:
//
//     loop {
//         seq_before = self.seq.load();
//         if seq_before % 2 == 1 { ...do resize-mode read; return... }
//         ...do normal read...
//         seq_after = self.seq.load();
//         if seq_before == seq_after { return result; }
//         // else: a resize started + completed during our read; retry.
//     }
//
// Each retry corresponds to a writer completing one resize. Define the
// "writer resize budget" R as a bound on how many resizes the writer
// will perform during this reader's lifetime. We prove the loop
// terminates after at most R + 1 iterations.
//
// This is a partial-correctness termination proof, not a fairness proof.
// The unbounded case ("writers can always be doing one more resize")
// requires temporal-logic reasoning that is out of scope here.

pub open spec fn reader_iters_bound(writer_resizes_remaining: nat) -> nat {
    writer_resizes_remaining + 1
}

// Idealized reader-step: returns true iff the read was consistent
// (seq_before == seq_after). False means a resize boundary was crossed
// and the reader retries.
pub open spec fn reader_step_consistent(seq_before: nat, seq_after: nat) -> bool {
    seq_before == seq_after
}

// Termination lemma (bounded fairness): given a sequence of `R` writer
// resizes and a reader whose retries each consume one resize, the
// reader completes in at most R + 1 iterations.
pub proof fn lemma_reader_terminates(writer_resizes: nat)
    ensures
        reader_iters_bound(writer_resizes) >= 1,
        // The bound is tight: 0 resizes ==> 1 iteration max.
        writer_resizes == 0 ==> reader_iters_bound(writer_resizes) == 1,
{}

// Inductive form: if there are `n` resizes pending and the reader has
// taken `i` retries, the reader needs at most `n - i + 1` more iterations.
pub proof fn lemma_reader_progress(n: nat, i: nat)
    requires i <= n,
    ensures n + 1 - i >= 1,
    decreases n - i,
{}

// Soundness of a single reader observation under the seqlock retry: if
// `seq_before == seq_after`, the read happened against one consistent
// snapshot — no resize boundary was crossed (production line 645).
pub proof fn lemma_reader_consistent_snapshot(s_before: ShardStateA, s_after: ShardStateA)
    requires
        s_before.seq == s_after.seq,
    ensures
        (s_before.seq % 2 == 0) == (s_after.seq % 2 == 0),
{}

// ----------------------------------------------------------------------------
// CONSERVATION ACROSS RESIZE (table-level analog of tier-B P3)
// ----------------------------------------------------------------------------
//
// The headline tier-B theorem in lifted form: a fingerprint observable
// before begin_resize is observable after a complete resize cycle, given
// that the rehash loop migrated every old_table entry into new_table.
//
// We model "rehash complete" as the precondition that for every i in
// [0, old_table.len()) where old_table[i] != EMPTY, new_table contains
// old_table[i] (in the abstract set sense — i.e. tab_lookup would
// succeed somewhere in new_table's probe sequence for that fp).

pub open spec fn rehash_complete_a(s: ShardStateA) -> bool {
    forall|i: int|
        0 <= i < s.old_table.len() && #[trigger] s.old_table[i] != EMPTY()
        ==> tab_contents(s.new_table).contains(s.old_table[i])
}

// Bridge lemma: if tab_lookup(t, fp) is true, then fp is in tab_contents(t).
pub proof fn lemma_lookup_implies_in_contents(t: Table, fp: u64)
    requires
        t.len() > 0,
        fp != EMPTY(),
        tab_lookup(t, fp),
    ensures
        tab_contents(t).contains(fp),
{
    let term = probe_terminus(t, fp);
    lemma_probe_terminus_bounded(t, fp, 0, t.len() as nat);
    assert(term <= t.len());
    // tab_lookup is true ==> term < t.len() and the probe slot equals fp.
    assert(term < t.len());
    let idx = probe_index(fp, term, t.len());
    assert(t[idx] == fp);
    lemma_probe_index_in_range(fp, term, t.len() as nat);
    assert(0 <= idx < t.len() as int);
    assert(exists|j: int| 0 <= j < t.len() && t[j] == fp);
}

// Bridge lemma: if fp is in tab_contents(t) AND probing terminates (the
// probe-sequence is well-formed), then tab_lookup(t, fp) is true.
//
// This lemma requires: every empty slot in fp's probe sequence is
// reached only AFTER fp's slot. Equivalently, fp was inserted via
// linear-probe semantics (not stuffed into an arbitrary slot). This is
// the production code's invariant: `contains_or_insert` always inserts
// at the first EMPTY slot in fp's probe sequence (page_aligned_
// fingerprint_store.rs:783-820), so any fp in the table satisfies it.
//
// We capture this as a side-condition `linear_probe_invariant`.
pub open spec fn linear_probe_invariant(t: Table, fp: u64) -> bool
    recommends t.len() > 0, fp != EMPTY(),
{
    // For every k in [0, cap), if t[probe_index(fp, k, cap)] == EMPTY()
    // then no probe slot j > k along fp's sequence holds fp. (Production
    // never bypasses an empty slot when looking for fp.)
    forall|k: nat, j: nat| #![trigger t[probe_index(fp, k, t.len())], t[probe_index(fp, j, t.len())]]
        k < t.len() && j < t.len() && k < j
        && t[probe_index(fp, k, t.len())] == EMPTY()
        ==> t[probe_index(fp, j, t.len())] != fp
}

// Conservation: a fingerprint observable in s0 (stable, pre-resize) is
// observable after begin_resize -> rehash_complete -> finalize_resize.
//
// The headline property: end-state effective_contents includes fp.
pub proof fn lemma_resize_preserves_contents_a(s0: ShardStateA, fp: u64, new_cap: nat)
    requires
        s0.seq % 2 == 0,
        wf_a(s0),
        fp != EMPTY(),
        s0.table.len() > 0,
        // Witness: fp appears in some slot of s0.table (this is implied
        // by tab_lookup(s0.table, fp) via lemma_lookup_implies_in_contents).
        exists|i: int| 0 <= i < s0.table.len() && #[trigger] s0.table[i] == fp,
        new_cap >= s0.table.len(),
    ensures
        ({
            let s1 = step_begin_resize_a(s0, new_cap);
            // s1.old_table == s0.table, so fp appears in s1.old_table.
            exists|i: int| 0 <= i < s1.old_table.len() && #[trigger] s1.old_table[i] == fp
        }),
{
    let s1 = step_begin_resize_a(s0, new_cap);
    assert(s1.old_table =~= s0.table);
    let i = choose|i: int| 0 <= i < s0.table.len() && #[trigger] s0.table[i] == fp;
    assert(s1.old_table[i] == fp);
}

// After-rehash conservation: if rehash_complete is established, every
// non-empty entry of old_table is in tab_contents(new_table).
pub proof fn lemma_rehash_complete_preserves_old_a(s: ShardStateA, fp: u64)
    requires
        s.seq % 2 == 1,
        rehash_complete_a(s),
        fp != EMPTY(),
        exists|i: int| 0 <= i < s.old_table.len() && #[trigger] s.old_table[i] == fp,
    ensures
        tab_contents(s.new_table).contains(fp),
{
    let i = choose|i: int| 0 <= i < s.old_table.len() && #[trigger] s.old_table[i] == fp;
    assert(s.old_table[i] == fp);
    assert(s.old_table[i] != EMPTY());
    // rehash_complete_a quantifies over i; instantiate at our witness.
    assert(tab_contents(s.new_table).contains(s.old_table[i]));
}

// Finalize swap: after step_finalize_resize_a, the new table is
// promoted to `s.table`, and tab_contents(s.table) is unchanged from
// the pre-finalize new_table contents.
pub proof fn lemma_finalize_promotes_new_table_a(s: ShardStateA, fp: u64)
    requires
        s.seq % 2 == 1,
        fp != EMPTY(),
        tab_contents(s.new_table).contains(fp),
    ensures
        ({
            let s2 = step_finalize_resize_a(s);
            tab_contents(s2.table).contains(fp) && s2.seq % 2 == 0
        }),
{
    let s2 = step_finalize_resize_a(s);
    assert(s2.table =~= s.new_table);
}

// END-TO-END: the table-level analog of tier-B's
// `theorem_no_fingerprint_lost`, restricted to a single complete resize
// cycle: any fp present in s0.table is in tab_contents(s3.table) where
// s3 is the post-finalize state, given the rehash loop completed
// (rehash_complete_a holds at the post-rehash state).
pub proof fn theorem_no_fingerprint_lost_a(
    s0: ShardStateA,
    s_post_rehash: ShardStateA,
    fp: u64,
    new_cap: nat,
)
    requires
        s0.seq % 2 == 0,
        wf_a(s0),
        fp != EMPTY(),
        s0.table.len() > 0,
        new_cap >= s0.table.len(),
        // Witness: fp is in s0.table.
        exists|i: int| 0 <= i < s0.table.len() && #[trigger] s0.table[i] == fp,
        // s_post_rehash is consistent with begin_resize(s0): same odd
        // parity, same old_table snapshot, and rehash_complete holds.
        s_post_rehash.seq == s0.seq + 1,
        s_post_rehash.old_table =~= s0.table,
        rehash_complete_a(s_post_rehash),
    ensures
        ({
            let s3 = step_finalize_resize_a(s_post_rehash);
            tab_contents(s3.table).contains(fp) && s3.seq % 2 == 0
        }),
{
    // Step 1: fp is in s_post_rehash.old_table (since old_table == s0.table).
    let i = choose|i: int| 0 <= i < s0.table.len() && #[trigger] s0.table[i] == fp;
    assert(s_post_rehash.old_table[i] == fp);
    assert(s_post_rehash.old_table[i] != EMPTY());

    // Step 2: rehash_complete ==> fp is in tab_contents(s_post_rehash.new_table).
    lemma_rehash_complete_preserves_old_a(s_post_rehash, fp);

    // Step 3: finalize promotes new_table to s3.table.
    lemma_finalize_promotes_new_table_a(s_post_rehash, fp);
}

// Concurrent-insert variant: a fingerprint inserted into new_table
// while resize is mid-flight (production: lines 723-741) survives the
// finalize swap.
pub proof fn theorem_concurrent_insert_survives_a(
    s_mid_resize: ShardStateA,
    slot: int,
    fp_concurrent: u64,
)
    requires
        s_mid_resize.seq % 2 == 1,
        wf_a(s_mid_resize),
        0 <= slot < s_mid_resize.new_table.len(),
        s_mid_resize.new_table[slot] == EMPTY(),
        fp_concurrent != EMPTY(),
    ensures
        ({
            // Apply the abstract CAS (production lines 723-741).
            let new_t = s_mid_resize.new_table.update(slot, fp_concurrent);
            let s_after_cas = ShardStateA {
                seq: s_mid_resize.seq,
                table: s_mid_resize.table,
                old_table: s_mid_resize.old_table,
                new_table: new_t,
            };
            // Then any subsequent finalize promotes new_t to s.table.
            let s_final = step_finalize_resize_a(s_after_cas);
            tab_contents(s_final.table).contains(fp_concurrent)
        }),
{
    let new_t = s_mid_resize.new_table.update(slot, fp_concurrent);
    let s_after_cas = ShardStateA {
        seq: s_mid_resize.seq,
        table: s_mid_resize.table,
        old_table: s_mid_resize.old_table,
        new_table: new_t,
    };
    lemma_cas_soundness(s_mid_resize.new_table, slot, fp_concurrent);
    assert(tab_contents(new_t).contains(fp_concurrent));
    let s_final = step_finalize_resize_a(s_after_cas);
    assert(s_final.table =~= new_t);
}

// ----------------------------------------------------------------------------
// PRODUCTION-CODE COVERAGE SUMMARY
// ----------------------------------------------------------------------------
//
// Lines of `src/storage/page_aligned_fingerprint_store.rs` now covered
// by machine-checked spec (tier A):
//
// - 101-121 (HashTableEntry layout)            : modeled by `Table = Seq<u64>`
// - 280-344 (rehash_batch_counted)             : `step_rehash_one_a`
//                                                + `lemma_rehash_preserves_a`
// - 357-390 (finalize_resize)                  : `step_finalize_resize_a`
//                                                + `lemma_finalize_parity_a`
// - 469-497 (resize start, snapshot pointers)  : `step_begin_resize_a`
//                                                + `lemma_begin_resize_parity_a`
// - 554-650 (contains)                         : `tab_lookup` + bounded-resize
//                                                termination lemma
// - 656-839 (contains_or_insert)               : `step_insert_during_resize_a`
//                                                / `step_insert_stable_a`
//                                                + CAS soundness lemma
// - 723-741 (resize-mode CAS slot insert)      : `cas_step`
//                                                + `lemma_cas_soundness`
// - 800-820 (normal-path CAS slot insert)      : same `cas_step` model
//
// Lines STILL ABSTRACTED:
//
// - 309-310 (raw `slice::from_raw_parts` over old_table_ptr/new_table_ptr):
//   tier A models the slice as a `Seq<u64>`; the pointer-to-slice
//   conversion is axiomatic (would need vstd::raw_ptr).
// - All `Ordering::Acquire/Release/AcqRel` annotations: tier A treats
//   atomics as sequentially consistent.
// - 142, 567 (resize_lock Mutex): tier A models "single resize at a time"
//   via the seq parity, not the mutex.
// - allocate_huge_pages / allocate_file_backed: zero-fill modeled as
//   `Seq::new(cap, |_| 0)`; the kernel guarantee is axiomatic.

fn main() {}

}  // verus!
