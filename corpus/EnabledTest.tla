---- MODULE EnabledTest ----
EXTENDS Naturals

VARIABLES x, y

TypeOK ==
    /\ x \in Nat
    /\ y \in Nat

Init ==
    /\ x = 0
    /\ y = 0

IncX ==
    /\ x < 10
    /\ x' = x + 1
    /\ y' = y

IncY ==
    /\ y < 10
    /\ y' = y + 1
    /\ x' = x

StutterBoth ==
    /\ x >= 10
    /\ y >= 10
    /\ x' = x
    /\ y' = y

Next ==
    \/ IncX
    \/ IncY
    \/ StutterBoth

Spec == Init /\ [][Next]_<<x, y>>

TestEnabledIncX ==
    x < 10 => ENABLED IncX

TestEnabledIncY ==
    y < 10 => ENABLED IncY

TestEnabledStutter ==
    (x >= 10 /\ y >= 10) => ENABLED StutterBoth

====
