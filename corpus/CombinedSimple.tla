---- MODULE CombinedSimple ----
EXTENDS Naturals

CONSTANTS MaxItems

VARIABLES items, count

ValidItem(i) ==
    /\ i.id >= 1
    /\ i.value >= 0

AllItemsValid == \A i \in items : ValidItem(i)

TypeOK ==
    AllItemsValid /\ count >= 0 /\ count <= MaxItems

Init ==
    /\ items = {[id |-> 1, value |-> 10], [id |-> 2, value |-> 20]}
    /\ count = Cardinality(items)

AddItem(id, val) ==
    /\ count < MaxItems
    /\ ~(\E i \in items : i.id = id)
    /\ items' = items \union {[id |-> id, value |-> val]}
    /\ count' = count + 1

RemoveItem(item) ==
    /\ item \in items
    /\ items' = items \ {item}
    /\ count' = count - 1

Next ==
    \/ \E id \in {3, 4, 5}, val \in {30, 40, 50} : AddItem(id, val)
    \/ \E i \in items : RemoveItem(i)

Spec == Init /\ [][Next]_<<items, count>>

CountMatchesSize == count = Cardinality(items)

====
