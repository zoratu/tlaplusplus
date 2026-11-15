---- MODULE ReproQuantWF ----
\* Each of two nodes independently counts 0 -> 1 under a QUANTIFIED weak-fairness conjunct.
\* <>[](\A n : x[n] = 1) HOLDS (every fair behavior sets both and then only stutters).
EXTENDS Naturals
CONSTANT N
VARIABLE x
Init == x = [n \in N |-> 0]
Set(n) == x[n] = 0 /\ x' = [x EXCEPT ![n] = 1]
Next == \E n \in N : Set(n)
Spec == Init /\ [][Next]_x /\ \A n \in N : WF_x(Set(n))
AllSet == <>[](\A n \in N : x[n] = 1)
====
