---- MODULE MultipleExtendsTest ----
EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC

VARIABLES x, seq, s

Init ==
    /\ x = 0
    /\ seq = <<1, 2, 3>>
    /\ s = {1, 2}

UsesNaturals ==
    x \in Nat

UsesIntegers ==
    /\ 0 - 5 = -5
    /\ -5 < 0

UsesSequences ==
    /\ Len(seq) >= 3
    /\ Head(seq) = 1

UsesFiniteSets ==
    Cardinality(s) >= 2

Next ==
    /\ x < 5
    /\ LET newX == x + 1
       IN /\ x' = newX
          /\ seq' = Append(seq, newX)
          /\ s' = s \union {newX}

TypeOK ==
    /\ x \in 0..10
    /\ seq \in Seq(Nat)
    /\ s \subseteq 0..10

Inv ==
    /\ x \in 0..10
    /\ Len(seq) >= 3
    /\ Head(seq) = 1
    /\ s \subseteq 0..10
    /\ Cardinality(s) >= 2
    /\ x \in Nat
    /\ 0 - 5 = -5
    /\ -5 < 0

Spec == Init /\ [][Next]_<<x, seq, s>>

====
