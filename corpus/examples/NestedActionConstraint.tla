----------------------------- MODULE NestedActionConstraint -----------------------------
(***************************************************************************)
(* Regression for the "disabled action + post-state constraint" panic       *)
(* (found in the Chandra-Toueg EnvironmentController failure-detector spec). *)
(*                                                                          *)
(* `Next == /\ Act /\ Constraint` where:                                    *)
(*   - `Act` is `\E i : g /\ procPause' = .. /\ UNCHANGED failed /\ ( \/ .. )`*)
(*     -- outer assignments followed by a nested inner disjunction; and      *)
(*   - `Constraint` is a post-state predicate that READS a prime             *)
(*     (`\A i : failed'[i] = FALSE => ..`), assigning nothing.               *)
(*                                                                          *)
(* Once every process has moved (`moved[i] # "NO"`), `Act` is DISABLED. The  *)
(* compiled evaluator correctly returns zero successors for that state, but  *)
(* `execute_branch`'s inline path used to fall back to the interpreted       *)
(* evaluator UNCONDITIONALLY and PROPAGATE its error: evaluating             *)
(* `Constraint` against a state where `Act` staged no prime made `failed'`   *)
(* resolve to an unbound value, and `failed'[i]` raised "function            *)
(* application unsupported for ModelValue(...)", which surfaced as a         *)
(* next-state panic that stalled exploration at 1 state. The fix trusts the  *)
(* compiled empty result (guards legitimately blocked the transition) and    *)
(* swallows the interpreted cross-check's error, so this deadlocks cleanly   *)
(* with the same state count as TLC.                                         *)
(***************************************************************************)
EXTENDS Integers

CONSTANT N

Proc == 1..N

VARIABLES failed, procPause, moved

Init == /\ failed = [i \in Proc |-> FALSE]
        /\ procPause = [i \in Proc |-> 0]
        /\ moved = [i \in Proc |-> "NO"]

Act ==
   \E i \in Proc :
      /\ failed[i] = FALSE
      /\ moved[i] = "NO"
      /\ procPause' = [procPause EXCEPT ![i] = 0]
      /\ UNCHANGED failed
      /\ \/ moved' = [moved EXCEPT ![i] = "PREDICT"]
         \/ moved' = [moved EXCEPT ![i] = "RECEIVE"]

Constraint ==
   \A i \in Proc : (failed'[i] = FALSE => procPause'[i] <= 5)

Next == /\ Act
        /\ Constraint

Spec == Init /\ [][Next]_<<failed, procPause, moved>>

\* Holds throughout (procPause never leaves 0); the point is that reaching the
\* deadlock state -- where Act is disabled -- must not panic.
Inv == \A i \in Proc : procPause[i] <= 5
========================================================================================
