---- MODULE ViewTest ----
EXTENDS Naturals

VARIABLES x, y, timestamp

StateView == <<x, y>>

Init ==
    /\ x = 0
    /\ y = 0
    /\ timestamp = 0

Next ==
    /\ x + y < 15
    /\ \/ /\ x < 10
          /\ x' = x + 1
          /\ y' = y
       \/ /\ y < 10
          /\ y' = y + 1
          /\ x' = x
    /\ timestamp' = timestamp + 1

TypeOK ==
    /\ x \in 0..10
    /\ y \in 0..10
    /\ timestamp \in Nat

Spec == Init /\ [][Next]_<<x, y, timestamp>>

====
