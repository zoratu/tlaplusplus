---- MODULE LiveEventuallyHolds ----
\* Holding companion to ReproMissedCycle: here `Set` (x := 1) is the FAIR action.
\* Set is enabled at every x=0 state and under WF it must eventually be taken, so
\* x becomes 1 and stays. Therefore <>(x=1), [](y=0 => <>(x=1)) and (y=0)~>(x=1)
\* all HOLD. This is the false-positive guard for the Bug B eventually-shape
\* checks: our checker must NOT report a violation here (the x=0 Flip-only cycle
\* is UNFAIR — Set enabled but never taken — so it is excluded, not a bad cycle).
EXTENDS Naturals
VARIABLES x, y
vars == <<x, y>>
Init == x = 0 /\ y = 0
Flip == y' = 1 - y /\ UNCHANGED x
Set == x = 0 /\ x' = 1 /\ UNCHANGED y
Next == Flip \/ Set
Spec == Init /\ [][Next]_vars /\ WF_vars(Set)
EvSet == <>(x = 1)
LeadsSet == [](y = 0 => <>(x = 1))
LeadsSetTo == (y = 0) ~> (x = 1)
====
