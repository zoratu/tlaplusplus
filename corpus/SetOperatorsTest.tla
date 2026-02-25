---- MODULE SetOperatorsTest ----
EXTENDS Naturals, FiniteSets

VARIABLES step, currentSet

Base == {1, 2, 3}

AllSubsets == SUBSET Base

AllSubsetsCorrect ==
    /\ {} \in AllSubsets
    /\ {1} \in AllSubsets
    /\ {2} \in AllSubsets
    /\ {3} \in AllSubsets
    /\ {1, 2} \in AllSubsets
    /\ {1, 3} \in AllSubsets
    /\ {2, 3} \in AllSubsets
    /\ {1, 2, 3} \in AllSubsets
    /\ Cardinality(AllSubsets) = 8

SetOfSets == {{1}, {2, 3}, {1, 2, 3}}

UnionWorks ==
    /\ 1 \in {1, 2, 3}
    /\ 2 \in {1, 2, 3}
    /\ 3 \in {1, 2, 3}

NestedSets == {{1, 2}, {3, 4}}

TypeOK ==
    /\ step \in 0..5
    /\ currentSet \in SUBSET Base

Init ==
    /\ step = 0
    /\ currentSet = {}

Next ==
    /\ step < 5
    /\ step' = step + 1
    /\ \E s \in SUBSET Base: currentSet' = s

Inv ==
    /\ TypeOK
    /\ AllSubsetsCorrect
    /\ UnionWorks
    /\ 1 \in {1, 2, 3, 4}
    /\ 2 \in {1, 2, 3, 4}
    /\ 3 \in {1, 2, 3, 4}
    /\ 4 \in {1, 2, 3, 4}

Spec == Init /\ [][Next]_<<step, currentSet>>

====
