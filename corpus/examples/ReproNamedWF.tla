---- MODULE ReproNamedWF ----
\* Identical to ReproFairWF except the WF conjunct sits in a NAMED operator (Fairness) referenced from Spec.
\* <>[](x = 2) HOLDS.
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Fairness == WF_x(Next)
Spec == Init /\ [][Next]_x /\ Fairness
EvTwo == <>[](x = 2)
====
