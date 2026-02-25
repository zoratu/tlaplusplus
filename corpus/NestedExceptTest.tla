---- MODULE NestedExceptTest ----
EXTENDS Naturals

VARIABLES matrix, counter

Init ==
    /\ matrix = [i \in 1..2 |-> [j \in 1..2 |-> 0]]
    /\ counter = 0

UpdateCell(i, j, val) ==
    /\ counter = 0
    /\ matrix' = [matrix EXCEPT ![i][j] = val]
    /\ counter' = counter + 1

IncrementCell(i, j) ==
    /\ counter = 1
    /\ matrix' = [matrix EXCEPT ![i][j] = @ + 1]
    /\ counter' = counter + 1

Next ==
    \/ UpdateCell(1, 1, 1)
    \/ IncrementCell(1, 1)
    \/ counter >= 2 /\ UNCHANGED <<matrix, counter>>

TypeOK ==
    /\ counter \in 0..3
    /\ matrix[1][1] \in 0..2
    /\ matrix[1][2] \in 0..2
    /\ matrix[2][1] \in 0..2
    /\ matrix[2][2] \in 0..2

MatrixCorrect ==
    /\ matrix[1][1] >= 0
    /\ matrix[1][2] >= 0
    /\ matrix[2][1] >= 0
    /\ matrix[2][2] >= 0

Spec == Init /\ [][Next]_<<matrix, counter>>

====
