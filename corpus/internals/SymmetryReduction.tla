---- MODULE SymmetryReduction ----
\* Differential-gate spec for SYMMETRY reduction under Permutations(Procs).
\* The state is fully symmetric under permutation of the model values in Procs:
\* full reachable set = |Val|^|Procs| = 3^3 = 27; quotienting by the process
\* permutations leaves the multisets of size 3 over {0,1,2} = C(5,3) = 10.
\* TLC applies the same symmetry reduction, so the distinct counts must agree
\* exactly — this gates our canonicalisation against TLC's.
EXTENDS Naturals, TLC

CONSTANT Procs

VARIABLE clock

vars == <<clock>>

Init == clock = [p \in Procs |-> 0]

Tick(p) == clock' = [clock EXCEPT ![p] = (clock[p] + 1) % 3]

Next == \E p \in Procs : Tick(p)

Spec == Init /\ [][Next]_vars

Inv == \A p \in Procs : clock[p] \in 0..2

Symmetry == Permutations(Procs)
================================================================================
