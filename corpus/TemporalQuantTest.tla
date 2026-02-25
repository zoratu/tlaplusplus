---- MODULE TemporalQuantTest ----
EXTENDS Naturals

CONSTANTS MaxValue

VARIABLES x, y

TypeOK ==
    /\ x >= 0
    /\ x <= MaxValue
    /\ y >= 0
    /\ y <= MaxValue

Init ==
    /\ x = 0
    /\ y = 0

Inc ==
    /\ x < MaxValue
    /\ x' = x + 1
    /\ y' = y

Stutter ==
    /\ x = MaxValue
    /\ x' = x
    /\ y' = y

Next ==
    \/ Inc
    \/ Stutter

Spec == Init /\ [][Next]_<<x, y>>

\* Temporal property with \AA (universal quantification)
AlwaysInRange ==
    \AA n: (n \in 0..MaxValue) => [](n \in 0..MaxValue)

\* Temporal property with \EE (existential quantification)
EventuallyReachable ==
    \EE n: (n \in 0..MaxValue) /\ <>(x = n)

====
