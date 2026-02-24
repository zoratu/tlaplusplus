---------------------------- MODULE CoverageHelper ----------------------------
EXTENDS Naturals, FiniteSets

CONSTANT Node

BoolToNat(b) == IF b THEN 1 ELSE 0
NatToBool(n) == n /= 0
IdentityNode(n) == n

PairSet == {<<a, b>> : a \in Node, b \in Node}

THEOREM PairSetType ==
    PairSet \subseteq (Node \X Node)

=============================================================================
