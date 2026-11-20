# Tableau liveness checking — design

Status: design (pre-implementation). Addresses Codex review findings #1/#2/#4/#6
on PR #213. Supersedes the maximal-SCC + head-label approximation in
`graph_liveness_violation` / `graph_liveness_eventually_violation`.

## The problem with the current (approximate) checks

The shipped checks have two unsound approximations, both shared with the
pre-existing `<>[]Q` code:

- **Maximal-SCC vs recurrent subcycle (#1/#4).** Acceptance is tested on whole
  Tarjan SCCs of the full graph. But an infinite behaviour may stay forever in a
  *proper strongly-connected subset* of a maximal SCC. A `¬Q` sub-loop inside a
  maximal SCC that also contains a `Q` state is missed (`all_not_q` is false),
  so a real `<>Q`/`~>` violation is not reported. Symmetrically, applying the
  "SF enabled at SOME state" test to a maximal SCC can misjudge a subcycle.

- **Action label vs `<<A>>_v` subscript (#2/#6).** Fairness occurrence and
  enablement are matched by the action's head *label*, ignoring the fairness
  subscript. `WF_v(A)` means `A /\ (v' # v)` happens; an `A`-edge that leaves the
  subscript tuple `v` unchanged is NOT an `<<A>>_v` step. Counting it as "A taken"
  can mark an unfair cycle fair (miss) — or, where `v` is a strict subset of the
  variables `A` changes, mark a fair cycle unfair (false positive).

## Correct semantics (what TLC's LiveCheck computes)

A liveness property fails iff the behaviour graph, intersected with the fairness
assumptions, admits an infinite behaviour violating the property. For a
state-predicate property this reduces to an **accepting strongly-connected
subgraph (SCS)** reachable appropriately:

- `<>Q` (bare): fails iff an accepting SCS of `¬Q` states (or a `¬Q` terminal) is
  reachable from an initial state **along a `¬Q`-only path**. (If `Q` ever holds,
  `<>Q` is already satisfied, so the whole counterexample suffix — and its stem
  from Init — must be `¬Q`.)
- `[](P => <>Q)` / `P ~> Q` (incl. `TRUE ~> Q` ≡ `[]<>Q`): fails iff an accepting
  SCS of `¬Q` states is reachable from a state satisfying `P /\ ¬Q`, along a
  `¬Q`-only path. (The stem up to the `P`-state may contain `Q`.)
- `<>[]Q`: fails iff an accepting SCS that **contains a `¬Q` state** is reachable
  (the behaviour is `[]<>¬Q`). Here the SCS is in the *full* graph, not the `¬Q`
  subgraph.

"Accepting SCS" = a strongly-connected set `C` (every state recurs infinitely
often if the behaviour loops through all of `C`) that satisfies **every** fairness
constraint. On such a `C`:

- `WF_v(A)` is satisfied ⇔ `<<A>>_v` is **taken** somewhere in `C` **or**
  `<<A>>_v` is **disabled** at some state of `C`.
  → `C` **violates** `WF_v(A)` ⇔ `<<A>>_v` enabled at **every** state of `C`
  **and** no `<<A>>_v` edge within `C`.
- `SF_v(A)` is satisfied ⇔ `<<A>>_v` taken somewhere in `C` **or** `<<A>>_v`
  disabled at **every** state of `C`.
  → `C` **violates** `SF_v(A)` ⇔ `<<A>>_v` enabled at **some** state of `C`
  **and** no `<<A>>_v` edge within `C`.

The existing WF-every / SF-some verdicts are therefore **correct for a recurrent
SCS** — the bug is applying them to a maximal SCC rather than the SCS a behaviour
actually stays in, plus the subscript blindness in "taken"/"enabled".

### `<<A>>_v` primitives (subscript-aware)

Given the subscript variable list `v` (from `FairnessConstraint.vars`), states via
`state_by_fp`, and `A`-labelled edges via the action shard index:

- **taken `<<A>>_v` on edge `(f, t)`**: `(f, t)` is an `A`-labelled edge AND the
  projection of the `f`-state onto `v` differs from the `t`-state's. (When `v` is
  all variables, any edge between two distinct fingerprints already changes `v`,
  so this reduces to "an `A`-edge exists" — the common case.)
- **enabled `<<A>>_v` at state `s` (fp `f`)**: some `A`-labelled out-edge `(f, t)`
  (anywhere in the full graph) is a taken `<<A>>_v` step.

Wrapper-`Next` fairness (`WF_vars(Next)`) is NOT special-cased as "any internal
edge": the same projection rule applies — an internal edge counts as a taken
`<<Next>>_vars` step only when source ≠ target on `vars` (which, for full `vars`,
every edge between two *distinct* fingerprints already satisfies; a stutter
self-loop does not). (Codex review §5 corrected the earlier "any internal edge"
shortcut.)

## Algorithm (CORRECTED per Codex design review of 2026-11-20)

**Why a single Tarjan pass is wrong.** Tarjan returns *maximal* SCCs, not every
recurrent strongly-connected subgraph. Restricting to `¬Q` fixes the *property*
side, but not the *fairness* side: a fair recurrent set can be a *proper* sub-SCC
of a maximal `¬Q` SCC. This matters specifically for **strong fairness** — "SF
disabled at every recurrent state" is NOT monotone under adding vertices. Example:
maximal SCC `{x, y}`, `<<A>>_v` disabled at `x` but enabled at `y` (whose only `A`
step leaves the SCC), `SF_v(A)`, no internal `<<A>>_v` edge. The SF-some test
rejects `{x,y}` as unfair, but the sub-SCC `{x}` (stutter forever at `x`) is fair
— a real violation if `x` is `¬Q`. (WF-only *is* monotone, so a single pass is
complete for WF; SF needs the recursive search below.)

**Implicit stuttering.** Under `[Next]_vars`, stuttering is legal at EVERY state,
not only at deadlocks. So every singleton `{s}` is a recurrent candidate (the
behaviour `s, s, …`), and `<>Q` can be violated by stuttering forever at a single
`¬Q` state even when it has an edge to a `Q` state (if no fairness forces the
move). Terminal-only handling is insufficient. A stutter self-loop is NOT a
`<<A>>_v` step (source = target projection), so it never counts as "taken".

**Exact fair-subcycle search (recursive SF-pruning).** For a candidate SCC `S`
(within the restriction), decide whether it contains a fair recurrent subset:

```
fn accepting(S):                      # S strongly connected, within restriction
  if S has no internal edge and S is not an implicit-stutter singleton: return ∅
  for each WF_v(A):
    if enabled<<A>>_v at EVERY state of S and no internal <<A>>_v edge in S:
      return ∅                        # unrepairable: no subset can satisfy this WF
  for each SF_v(A):
    if enabled<<A>>_v at SOME state of S and no internal <<A>>_v edge in S:
      # every fair subset must avoid all states where A is SF-enabled
      S' = S minus { s : enabled<<A>>_v at s }
      return ⋃ over each SCC T of the subgraph induced by S': accepting(T)
  return S                            # no constraint fails: S is accepting
```

Termination: each recursion strictly shrinks the vertex set. Completeness: a
genuinely fair recurrent set is never removed — a WF-failing SCC admits no fair
subset; for an SF-failing SCC every fair subset must exclude the SF-enabled
states, so removing them preserves all fair subsets (Codex review §6).
"Internal `<<A>>_v` edge" and "enabled" use the subscript-aware primitives;
"enabled" is always a **global** out-edge (independent of `S`).

For each property `(p, q, is_bare)`:

1. **Restrict.** `R`:
   - `<>Q` / `~>` / `[](P=><>Q)`: `R = { fp : Q confidently false }`.
   - `<>[]Q`: `R =` all reachable states (no `¬Q` restriction; the accepted SCS
     must merely *contain* a `¬Q` state — see step 3).
2. **Maximal SCCs of the `R`-induced subgraph** via `TarjanSCC::find_sccs` with a
   successor closure that keeps only edges into `R`. (Full graph for `<>[]Q`.)
3. **Accepting subset.** For each maximal SCC, run `accepting(S)` above. A
   non-empty result is a bad recurrent set. For `<>[]Q`, additionally require the
   returned set to contain ≥1 `¬Q` state. Implicit-stutter singletons: every
   `R`-state is a singleton candidate; fold these in by letting `accepting`
   treat a lone `¬Q` state as recurrent (its stutter loop) unless fairness forces
   it to move (i.e. some WF/SF-enabled action with no self-satisfying option).
4. **Reachability.** A violation exists iff a bad-recurrent-set state is reachable
   along a `¬Q`-only path from the right start set:
   - `<>Q`: from an initial state (Init itself `¬Q`).
   - `~>` / `[](P=><>Q)`: from a `P /\ ¬Q` state.
   - `<>[]Q`: from Init through the FULL graph (no `P`, no `¬Q`-path restriction;
     relies on the graph containing only reachable states). Return a witness state.

### Correctness argument (sketch, corrected)

- Every violating behaviour has an infinite suffix confined to a finite set each
  of whose states recurs infinitely often ⇒ a recurrent strongly-connected set.
  The recursive `accepting` search enumerates exactly the *fair* such sets (it
  never discards a fair subset — see termination/completeness above). —
  *completeness*.
- An accepting set admits a concrete fair **closed walk** (not necessarily a
  simple cycle): because it is strongly connected and finite, a walk can visit
  every vertex AND every chosen internal fairness-witness edge, then repeat. It
  satisfies every WF/SF and stays in `R`. — *soundness*.
- `Some(false)`-only `¬Q`: an unevaluable predicate never enters `R`; it can only
  shrink the search — miss, never invent.

### Action-index semantic invariant (Codex review §4)

The subscript-aware "taken/enabled" primitives read `A`-labelled edges from the
action shard index. This is only correct if the index enumerates `(s, t)` for
EXACTLY the transitions satisfying the fairness action `A(s, t)` — including
disjunctive (`A == B \/ C`), aliased, and overlapping actions, regardless of the
head label an edge was generated under. Before relying on the label index, the
implementation must either (a) establish this invariant for the fairness action,
or (b) evaluate `A(s, t)` / `ENABLED <<A>>_v` directly against the model. This is
a prerequisite, verified during implementation (and a reason a `WF_v(A)` whose
`A` the index cannot faithfully represent must fall back to "unchecked").

## Data / API work

- Thread the fairness **subscript vars** and **states** into the acceptance test.
  Cleanest: do acceptance inside the `graph_liveness_*` methods on `TlaModel`
  (they already hold `state_by_fp` and `self.fairness_constraints`), replacing the
  per-maximal-SCC `fair_scc` precomputation. `check_fairness_on_scc_fp_sharded`
  stays for the diagnostics-only filter count, or is refactored to take the
  subscript + a state accessor.
- A subscript-projection helper: `project(state, &vars) -> Vec<TlaValue>` (or hash)
  using `TlaState::get`.
- Reuse `TarjanSCC::find_sccs` on `u64` fingerprints with a restricted-successor
  closure.

## Test matrix (all TLC-cross-checked)

Beyond the existing 8 specs, add:
- **Recurrent-subcycle (#1/#4):** a maximal SCC with a `¬Q` self-loop AND a
  reachable `Q` state, under a vacuous fairness constraint — must VIOLATE `<>Q`
  (currently missed). Holding companion where the `¬Q` sub-loop is unfair.
- **Partial subscript (#2/#6):** `WF_v(A)` with `v` a strict subset of the
  variables `A` changes; an `A`-edge leaving `v` unchanged must NOT count as
  taken (must VIOLATE where the maximal/label check would hold), and the inverse
  false-positive guard (must HOLD where the label check would false-flag).
- **SF subcycle:** `SF_v(A)` where the fair behaviour stays in a subcycle.
- Re-run all existing holding guards (no new false positives) + the Fermion
  repros + the unreduced DQSOwnershipLive.

## Rollout

Behind the same entry points (`needs_graph_liveness_check`,
`graph_liveness_violation`, `graph_liveness_eventually_violation`), so the runtime
wiring and the `fairness_extraction_incomplete` / `spec_has_non_fairness_liveness`
guards are unchanged. Validate via diff_tlc CI; Codex-review the implementation at
the milestone.
