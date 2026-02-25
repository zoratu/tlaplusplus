---- MODULE InstanceTest ----
EXTENDS Naturals

Helper == INSTANCE CoverageHelper WITH Node <- {1, 2, 3}

VARIABLES x

TypeOK == x \in Nat

Init == x = 0

Next ==
    \/ /\ x < 10
       /\ x' = x + 1
    \/ /\ x >= 10
       /\ x' = x

Spec == Init /\ [][Next]_<<x>>

TestModuleInstance ==
    /\ Helper!BoolToNat(TRUE) = 1
    /\ Helper!BoolToNat(FALSE) = 0
    /\ Helper!NatToBool(0) = FALSE
    /\ Helper!NatToBool(1) = TRUE
    /\ Helper!NatToBool(5) = TRUE

====
