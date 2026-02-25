------------------------ MODULE InitNextTemporalQuant ------------------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS Node, MaxDepth

VARIABLES step, pos, trace

Vars == <<step, pos, trace>>

Visited(seq) ==
    {seq[i] : i \in DOMAIN seq}

AnyNode ==
    CHOOSE n \in Node: TRUE

Init ==
    /\ Node # {}
    /\ MaxDepth \in Nat
    /\ MaxDepth > 0
    /\ step = 0
    /\ pos = AnyNode
    /\ trace = <<>>

AltInit == Init

Advance ==
    /\ step < MaxDepth
    /\ LET newPos == IF Len(trace) = 0 THEN pos ELSE trace[Len(trace)]
       IN /\ step' = step + 1
          /\ pos' = newPos
          /\ trace' = Append(trace, newPos)

Reset ==
    /\ step > 0
    /\ step' = 0
    /\ pos' = AnyNode
    /\ trace' = <<>>

Next ==
    \/ Advance
    \/ Reset

AltNext == Advance

TypeOK ==
    /\ step \in 0..MaxDepth
    /\ pos \in Node
    /\ trace \in Seq(Node)

StepMonotonic ==
    step' >= step

Done ==
    step = MaxDepth

TemporalExistsEventuallyVisited ==
    \EE n: (n \in Node) /\ <>(n \in Visited(trace))

TemporalForAllAlwaysNode ==
    \AA n: (n \in Node) => [](n \in Node)

TemporalForAllBoundConstant ==
    \AA k: (k \in 0..MaxDepth) => []((k \in 0..MaxDepth) /\ (step <= MaxDepth))

Spec ==
    Init /\ [][Next]_Vars

=============================================================================
