---- MODULE WrapperNextFairness ----
(*
 * T1.3 regression: WF_vars(Next) on a spec with an explicit
 * `Terminated /\ UNCHANGED vars` stutter disjunct must NOT report a
 * fairness violation. The terminated state forms a single-state SCC
 * with a self-loop; pre-fix, the fairness checker compared the
 * constraint name "Next" against the disjunct head label
 * ("Terminated") and reported a false positive.
 *)
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Step == /\ x < 3 /\ x' = x + 1
Done == /\ x = 3 /\ UNCHANGED vars

Next == Step \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
====
