---- MODULE OperatorSubstitutionTest ----
EXTENDS Naturals, Sequences

VARIABLES x, seq

Init ==
    /\ x = 0
    /\ seq = <<>>

Next ==
    /\ x < 5
    /\ LET newX == x + 1
       IN /\ x' = newX
          /\ seq' = Append(seq, newX)

TypeOK ==
    /\ x \in 0..10
    /\ Len(seq) <= 10
    /\ Len(seq) = x

====
