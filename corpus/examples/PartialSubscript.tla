---- MODULE PartialSubscript ----
EXTENDS Naturals
\* Partial-subscript discriminator (Codex tableau review #2/#6). Fairness is
\* WF_x(A) — subscript is the single variable x, a STRICT subset of what A can
\* change (A also toggles y). On the y-toggle loop at a fixed x (e.g. {(0,0),(0,1)})
\* action A IS taken, but those steps leave x UNCHANGED, so they are NOT <<A>>_x
\* steps. The head-LABEL check (ignoring the subscript) counts A as "taken" there,
\* marks the loop fair, and would FALSE-POSITIVE `<>(x=2)`. The subscript-aware
\* tableau sees <<A>>_x enabled at every loop state (A can advance x) but never
\* taken -> WF_x(A) violated -> loop UNFAIR -> excluded. So x must advance and
\* `<>(x=2)` HOLDS. TLC agrees (no violation).
VARIABLES x, y
vars == <<x, y>>
Init == x = 0 /\ y = 0
A == \/ (x < 2 /\ x' = x + 1 /\ y' = y)   \* advance x (changes the subscript x)
     \/ (x' = x /\ y' = 1 - y)            \* toggle y (x unchanged: NOT an <<A>>_x step)
Next == A
Spec == Init /\ [][Next]_vars /\ WF_x(A)
GoalX == <>(x = 2)
====
