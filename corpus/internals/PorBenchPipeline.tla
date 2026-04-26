---- MODULE PorBenchPipeline ----
EXTENDS Naturals

\* Pipeline of three stages: producer → buffer → consumer.
\* Producer writes only `prod_count`. Consumer writes only `cons_count`.
\* Buffer (Move) reads both via shared `pending` (and writes pending).
\*
\* Producer and Consumer are mutually independent (disjoint vars).
\* Move depends on both because it touches the shared `pending`.

CONSTANT MAX

VARIABLES prod_count, cons_count, pending

vars == <<prod_count, cons_count, pending>>

Init == /\ prod_count = 0
        /\ cons_count = 0
        /\ pending = 0

Produce == /\ prod_count < MAX
           /\ prod_count' = prod_count + 1
           /\ UNCHANGED <<cons_count, pending>>

Move == /\ prod_count > pending
        /\ pending < MAX
        /\ pending' = pending + 1
        /\ UNCHANGED <<prod_count, cons_count>>

Consume == /\ cons_count < pending
           /\ cons_count' = cons_count + 1
           /\ UNCHANGED <<prod_count, pending>>

Next == Produce \/ Move \/ Consume

Spec == Init /\ [][Next]_vars

Inv == /\ prod_count \in 0..MAX
       /\ cons_count \in 0..MAX
       /\ pending \in 0..MAX
       /\ cons_count <= pending
       /\ pending <= prod_count

================================================================================
