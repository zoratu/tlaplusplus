---- MODULE ReproMissedCycle ----
\* y flips forever under WF(Flip); Set (x := 1) is NOT fair. The fair behavior Flip, Flip, ... never sets x,
\* so <>(x = 1) and [](y = 0 => <>(x = 1)) are VIOLATED (TLC finds the lasso).
EXTENDS Naturals
VARIABLES x, y
vars == <<x, y>>
Init == x = 0 /\ y = 0
Flip == y' = 1 - y /\ UNCHANGED x
Set == x = 0 /\ x' = 1 /\ UNCHANGED y
Next == Flip \/ Set
Spec == Init /\ [][Next]_vars /\ WF_vars(Flip)
EvSet == <>(x = 1)
LeadsSet == [](y = 0 => <>(x = 1))
LeadsSetTo == (y = 0) ~> (x = 1)
====
