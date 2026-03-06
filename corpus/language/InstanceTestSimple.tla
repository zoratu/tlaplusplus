---- MODULE InstanceTestSimple ----
EXTENDS Naturals

Helper == INSTANCE CoverageHelper WITH Node <- {1, 2, 3}

VARIABLES flag

TypeOK == flag \in BOOLEAN

Init == flag = FALSE

IncFlag ==
    /\ flag = FALSE
    /\ flag' = TRUE

StutterOnTrue ==
    /\ flag = TRUE
    /\ flag' = flag

Next ==
    \/ IncFlag
    \/ StutterOnTrue

Spec == Init /\ [][Next]_<<flag>>

TestModuleInstance ==
    /\ Helper!BoolToNat(TRUE) = 1
    /\ Helper!BoolToNat(FALSE) = 0
    /\ Helper!NatToBool(0) = FALSE
    /\ Helper!NatToBool(1) = TRUE
    /\ Helper!NatToBool(5) = TRUE

====
