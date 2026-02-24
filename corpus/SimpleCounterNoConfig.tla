---- MODULE SimpleCounterNoConfig ----
EXTENDS Naturals

VARIABLES x, y

Init ==
    /\ x = 0
    /\ y = 0

IncrementX ==
    /\ x < 3
    /\ x' = x + 1
    /\ y' = y

IncrementY ==
    /\ y < 2
    /\ y' = y + 1
    /\ x' = x

Next ==
    \/ IncrementX
    \/ IncrementY

Inv == x + y <= 5

====
