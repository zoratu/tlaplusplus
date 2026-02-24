------------------------ MODULE LanguageFeatureMatrix ------------------------
EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC

CONSTANTS Node, MaxDepth, Cap

LOCAL Helpers == INSTANCE CoverageHelper WITH Node <- Node

VARIABLES step, pos, frontier, seen, table, rec, pair

Vars == <<step, pos, frontier, seen, table, rec, pair>>

LOCAL Min2(a, b) == IF a < b THEN a ELSE b

NodeSeq(seq) ==
    \A i \in DOMAIN seq: seq[i] \in Node

HasUnseen ==
    \E n \in Node: n \notin seen

PickUnseenOrPos ==
    IF HasUnseen
        THEN CHOOSE n \in Node: n \notin seen
        ELSE pos

TrimToCap(seq) ==
    SubSeq(seq, 1, Min2(Len(seq), Cap))

TableDomain == DOMAIN table
RecomputedSeen == {n \in Node: table[n] > 0}
SeenIntersection == seen \intersect Node

Init ==
    /\ Node # {}
    /\ MaxDepth \in Nat
    /\ Cap \in Nat
    /\ Cap > 0
    /\ step = 0
    /\ pos = CHOOSE n \in Node: TRUE
    /\ frontier = <<pos>>
    /\ seen = {pos}
    /\ table = [n \in Node |-> IF n = pos THEN 1 ELSE 0]
    /\ rec = [flag |-> FALSE, count |-> 0]
    /\ pair = <<pos, 0>>

Visit ==
    /\ step < MaxDepth
    /\ LET fallback == PickUnseenOrPos
           nextPos ==
               CASE Len(frontier) = 0 -> fallback
                 [] Head(frontier) \in Node -> Head(frontier)
                 [] OTHER -> fallback
           grown ==
               IF Len(frontier) = 0
                   THEN <<nextPos>>
                   ELSE Append(Tail(frontier), nextPos)
           bounded == TrimToCap(grown)
           cleaned == SelectSeq(bounded, LAMBDA x: x \in Node)
       IN /\ pos' = nextPos
          /\ frontier' = cleaned
          /\ seen' = seen \union {nextPos}
          /\ table' = [table EXCEPT ![nextPos] = @ + 1]
          /\ rec' = [rec EXCEPT !.flag = ~@, !.count = @ + 1]
          /\ pair' = <<nextPos, step + 1>>
          /\ step' = step + 1

Rotate ==
    /\ step < MaxDepth
    /\ Len(frontier) > 0
    /\ frontier' = Append(Tail(frontier), Head(frontier))
    /\ step' = step + 1
    /\ pair' = <<pos, step + 1>>
    /\ UNCHANGED <<pos, seen, table, rec>>

Next ==
    \/ Visit
    \/ Rotate

TypeOK ==
    /\ step \in 0..MaxDepth
    /\ pos \in Node
    /\ frontier \in Seq(Node)
    /\ Len(frontier) <= Cap
    /\ seen \subseteq Node
    /\ table \in [Node -> Nat]
    /\ rec \in [flag: BOOLEAN, count: Nat]
    /\ pair \in Node \X (0..MaxDepth)

InvariantsHold ==
    /\ TypeOK
    /\ NodeSeq(frontier)
    /\ TableDomain = Node
    /\ SeenIntersection = seen
    /\ RecomputedSeen = seen
    /\ \A n \in Node: table[n] \in Nat
    /\ \E n \in Node: table[n] > 0
    /\ Helpers!NatToBool(Helpers!BoolToNat(rec.flag)) = rec.flag

CanVisitWhenNotDone ==
    step < MaxDepth => ENABLED Visit

StateConstraint ==
    /\ Len(frontier) <= Cap
    /\ step <= MaxDepth

ActionConstraint ==
    step' \in step..(step + 1)

Done ==
    step = MaxDepth

SafetyProperty ==
    []TypeOK

EventualDone ==
    <>Done

ProgressProperty ==
    (step < MaxDepth) ~> Done

Symmetry ==
    Permutations(Node)

ASSUME MaxDepth > 0 /\ Cap > 0

THEOREM HelperPairSetType ==
    Helpers!PairSet \subseteq (Node \X Node)

Spec ==
    Init /\ [][Next]_Vars

FairSpec ==
    Spec /\ WF_Vars(Visit) /\ SF_Vars(Rotate)

=============================================================================
