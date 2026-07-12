---------------------- MODULE NatOverrideMembership ----------------------
EXTENDS Naturals
CONSTANT MaxN
NatBound == 0 .. MaxN
VARIABLE x
Init == x = 1
Next == x' = x
\* With `Nat <- NatBound`, Clock == Nat \ {0} is the finite set 1..MaxN.
\* Regression: membership `x \in (Nat \ {0})` must not try to enumerate the
\* built-in infinite Nat (which spuriously failed TypeOK-style invariants).
Clock == Nat \ {0}
Inv == x \in Clock
Spec == Init /\ [][Next]_x
==========================================================================
