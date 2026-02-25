---- MODULE LivenessTest ----
EXTENDS Naturals

VARIABLES x, y, flag

Init ==
    /\ x = 0
    /\ y = 0
    /\ flag = FALSE

IncX ==
    /\ x < 10
    /\ x' = x + 1
    /\ y' = y
    /\ flag' = flag

IncY ==
    /\ y < 10
    /\ y' = y + 1
    /\ x' = x
    /\ flag' = flag

SetFlag ==
    /\ x >= 5
    /\ y >= 5
    /\ ~flag
    /\ flag' = TRUE
    /\ UNCHANGED <<x, y>>

Next ==
    \/ IncX
    \/ IncY
    \/ SetFlag

TypeOK ==
    /\ x \in 0..10
    /\ y \in 0..10
    /\ flag \in BOOLEAN

EventuallyFlag ==
    <>(flag = TRUE)

EventuallyXIs10 ==
    <>(x = 10)

AlwaysNonNegative ==
    [](x >= 0 /\ y >= 0)

FlagImpliesProgress ==
    (x >= 5 /\ y >= 5) ~> flag = TRUE

Spec ==
    Init /\ [][Next]_<<x, y, flag>>

FairSpec ==
    Spec /\ WF_<<x,y,flag>>(IncX) /\ WF_<<x,y,flag>>(IncY) /\ WF_<<x,y,flag>>(SetFlag)

====
