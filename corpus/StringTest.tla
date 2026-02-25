---- MODULE StringTest ----
EXTENDS Naturals, Sequences

VARIABLES msg, count

Init ==
    /\ msg = "hello"
    /\ count = 0

StringEquality ==
    /\ "hello" = "hello"
    /\ "hello" # "world"

StringInSet ==
    /\ "a" \in {"a", "b", "c"}
    /\ "d" \notin {"a", "b", "c"}

Concat ==
    /\ msg = "hello"
    /\ msg' = "world"
    /\ count' = count + 1

Next ==
    \/ Concat
    \/ (count >= 1 /\ UNCHANGED <<msg, count>>)

TypeOK ==
    /\ count \in Nat
    /\ msg \in {"hello", "world"}

Inv ==
    /\ TypeOK
    /\ StringEquality
    /\ StringInSet

====
