---- MODULE PorTwoCounters ----
EXTENDS Naturals

VARIABLES x, y

vars == <<x, y>>

Init == /\ x = 0
        /\ y = 0

IncX == /\ x < 3
        /\ x' = x + 1
        /\ UNCHANGED y

IncY == /\ y < 3
        /\ y' = y + 1
        /\ UNCHANGED x

Next == IncX \/ IncY

Spec == Init /\ [][Next]_vars

Inv == x \in 0..3 /\ y \in 0..3

================================================================================
