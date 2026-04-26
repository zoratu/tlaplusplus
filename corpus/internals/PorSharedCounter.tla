---- MODULE PorSharedCounter ----
EXTENDS Naturals

VARIABLE x

vars == <<x>>

Init == x = 0

Inc == /\ x < 5
       /\ x' = x + 1

Dec == /\ x > 0
       /\ x' = x - 1

Next == Inc \/ Dec

Spec == Init /\ [][Next]_vars

Inv == x \in 0..5

================================================================================
