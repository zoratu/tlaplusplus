---- MODULE PorDependencyChain ----
EXTENDS Naturals

\* Three actions:
\*   A: writes p only
\*   B: writes q only
\*   C: reads q, writes r
\* So A is independent of B and C.  B and C share q (B writes, C reads).

VARIABLES p, q, r

vars == <<p, q, r>>

Init == /\ p = 0
        /\ q = 0
        /\ r = 0

A == /\ p < 2
     /\ p' = p + 1
     /\ UNCHANGED <<q, r>>

B == /\ q < 2
     /\ q' = q + 1
     /\ UNCHANGED <<p, r>>

C == /\ r < 2
     /\ r' = q
     /\ UNCHANGED <<p, q>>

Next == A \/ B \/ C

Spec == Init /\ [][Next]_vars

Inv == /\ p \in 0..2 /\ q \in 0..2 /\ r \in 0..2

================================================================================
