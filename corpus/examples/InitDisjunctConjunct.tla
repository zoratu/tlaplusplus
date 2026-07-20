---- MODULE InitDisjunctConjunct ----
EXTENDS Integers
CONSTANT C
VARIABLES x, y
\* Init is a CONJUNCTION whose middle conjunct is a guarded DISJUNCTION whose
\* branches contain a membership / equality over the state variable `y`. With
\* C = TRUE the first branch enumerates y in {10,20} (2 initial states) and the
\* second (~C) is pruned. A prior bug evaluated the `\/` conjunct as a boolean
\* guard — with `y` unbound that reads FALSE — so the membership was never
\* enumerated and only 1 initial state was produced (Prisoner{Solo,}LightUnknown).
Init == /\ x = 1
        /\ \/ C /\ y \in {10, 20}
           \/ ~C /\ y = 99
Next == UNCHANGED <<x, y>>
====
