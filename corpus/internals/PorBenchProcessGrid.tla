---- MODULE PorBenchProcessGrid ----
EXTENDS Naturals

\* N independent processes each increment a local counter from 0 up to MAX.
\* Each process touches only its own variable, so they're all pairwise
\* independent.  Full state space is (MAX+1)^N.  POR with stubborn-set
\* seeded by lowest-index enabled action explores N*(MAX+1) - (N-1)
\* states (the "diagonal" path).

CONSTANT MAX

VARIABLES p1, p2, p3, p4

vars == <<p1, p2, p3, p4>>

Init == /\ p1 = 0
        /\ p2 = 0
        /\ p3 = 0
        /\ p4 = 0

Inc1 == /\ p1 < MAX
        /\ p1' = p1 + 1
        /\ UNCHANGED <<p2, p3, p4>>

Inc2 == /\ p2 < MAX
        /\ p2' = p2 + 1
        /\ UNCHANGED <<p1, p3, p4>>

Inc3 == /\ p3 < MAX
        /\ p3' = p3 + 1
        /\ UNCHANGED <<p1, p2, p4>>

Inc4 == /\ p4 < MAX
        /\ p4' = p4 + 1
        /\ UNCHANGED <<p1, p2, p3>>

Next == Inc1 \/ Inc2 \/ Inc3 \/ Inc4

Spec == Init /\ [][Next]_vars

Inv == p1 \in 0..MAX /\ p2 \in 0..MAX /\ p3 \in 0..MAX /\ p4 \in 0..MAX

================================================================================
