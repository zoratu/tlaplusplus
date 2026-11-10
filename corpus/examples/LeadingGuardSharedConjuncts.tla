---- MODULE LeadingGuardSharedConjuncts ----
EXTENDS Integers
\* Isolates the SECOND #188 site only: the leading guard-disjunction split.
\*
\* The split lives in `split_action_body_disjuncts_uncached`, which the compiled
\* run-tla reaches via INSTANCE expansion of the callee body — a direct
\* (non-instance) action does not route through it, so this spec reuses
\* ClockDetector (whose `Receive` body is `/\ (\/ A \/ B) /\ LocalTick /\ seen'`,
\* a leading `\/`-group that is NOT the sole leading conjunct) via an instance
\* call. Crucially the argument is UNPRIMED (`{clock[i]}`, not `chan'[i]` as in
\* ClockDetectorMC), so the FIRST #188 site — the primed-arg staged-context read —
\* is a no-op here (`ctx.clone()` and the staged ctx agree for an unprimed arg).
\* Only `instance_leading_guard_split` can change this spec's behavior: with it,
\* the leading `\/` splits and the shared LocalTick/seen' conjuncts are dropped
\* from every branch but the last, so when guard A is taken clock does not advance
\* and the state graph collapses — diverging from TLC.
CONSTANT N
Proc == 1..N
VARIABLES clock, seen
Det == INSTANCE ClockDetector
Init ==
  /\ clock = [i \in Proc |-> 0]
  /\ seen  = [i \in Proc |-> {}]
Step(i) == Det!Receive(i, {clock[i]})
Next == \E i \in Proc : Step(i)
====
