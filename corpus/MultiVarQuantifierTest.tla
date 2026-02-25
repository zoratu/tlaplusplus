---- MODULE MultiVarQuantifierTest ----
EXTENDS Naturals

VARIABLES x, y

S == {1, 2, 3}

MultiVarForAll ==
    \A i, j \in S: i + j >= 2

MultiVarExists ==
    \E i, j \in S: i + j = 5

ThreeVarForAll ==
    \A i, j, k \in {1, 2}: i * j * k <= 8

ThreeVarExists ==
    \E i, j, k \in {1, 2}: i + j + k = 6

Init ==
    /\ x = 0
    /\ y = 0

Next ==
    /\ x < 3
    /\ y < 3
    /\ \E i, j \in S:
        /\ x' = i
        /\ y' = j

TypeOK ==
    /\ x \in 0..3
    /\ y \in 0..3

Inv ==
    /\ TypeOK
    /\ MultiVarForAll
    /\ MultiVarExists
    /\ ThreeVarForAll
    /\ ThreeVarExists

====
