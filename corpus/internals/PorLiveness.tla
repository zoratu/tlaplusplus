---- MODULE PorLiveness ----
EXTENDS Naturals

VARIABLE x

vars == <<x>>

Init == x = 0

Tick == /\ x < 5
        /\ x' = x + 1

Next == Tick

\* Liveness property — POR must reject this.
Spec == Init /\ [][Next]_vars /\ WF_vars(Tick)
Liveness == <>(x = 5)

================================================================================
