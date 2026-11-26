---- MODULE ReproUnfairScc ----
\* y flips forever; Set (x := 1) is ALSO weakly fair. The Flip-only cycle at x = 0 is an UNFAIR SCC (Set stays
\* enabled and is never taken), so it must be DISCARDED, not reported. <>[](x = 1) HOLDS (TLC agrees).
EXTENDS Naturals
VARIABLES x, y
vars == <<x, y>>
Init == x = 0 /\ y = 0
Flip == y' = 1 - y /\ UNCHANGED x
Set == x = 0 /\ x' = 1 /\ UNCHANGED y
Next == Flip \/ Set
Spec == Init /\ [][Next]_vars /\ WF_vars(Set)
EvSetStable == <>[](x = 1)
====
