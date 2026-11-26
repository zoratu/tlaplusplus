---- MODULE RecurrentSubcycle ----
EXTENDS Naturals
\* Recurrent-subcycle discriminator (Codex tableau review #1/#4). The whole
\* reachable graph {0,1,2} is ONE maximal SCC (0->1->2->0), and it CONTAINS a Q
\* state (x=2). The old maximal-SCC check tested `all_not_q` over the whole SCC,
\* found the Q state, and MISSED the violation. But the fair behavior can stay in
\* the ¬Q sub-loop {0,1} forever (toggle under WF_vars(Toggle), never take GoTo2,
\* which is not fair), so `<>(x=2)` is VIOLATED. TLC finds the lasso; the tableau
\* restricts to ¬Q = {0,1} and finds the fair sub-SCC.
VARIABLE x
vars == <<x>>
Init == x = 0
Toggle == \/ (x = 0 /\ x' = 1)
          \/ (x = 1 /\ x' = 0)
GoTo2 == x = 1 /\ x' = 2
Back == x = 2 /\ x' = 0
Next == Toggle \/ GoTo2 \/ Back
Spec == Init /\ [][Next]_vars /\ WF_vars(Toggle)
Reach2 == <>(x = 2)
====
