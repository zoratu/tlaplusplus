---- MODULE FairnessTest ----
EXTENDS Naturals

VARIABLES x, y

TypeOK ==
    /\ x \in 0..2
    /\ y \in 0..2

Init ==
    /\ x = 0
    /\ y = 0

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ y' = y

IncY ==
    /\ y < 2
    /\ y' = y + 1
    /\ x' = x

Next ==
    \/ IncX
    \/ IncY

Spec == Init /\ [][Next]_<<x, y>> /\ WF_<<x, y>>(IncX) /\ WF_<<x, y>>(IncY)

====
