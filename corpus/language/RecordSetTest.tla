---- MODULE RecordSetTest ----
EXTENDS Naturals, FiniteSets

VARIABLES step, currentConfig

\* Simple record set test
SimpleRecordSet == [x: {1, 2}]

TwoFieldRecordSet == [a: {1, 2}, b: {3, 4}]

ConfigSet == [ringSize: 1..2, replicationFactor: 1..2]

\* Test that the sets have correct cardinality
CardinalityTests ==
    Cardinality(SimpleRecordSet) = 2
    /\ Cardinality(TwoFieldRecordSet) = 4
    /\ Cardinality(ConfigSet) = 4

\* Test that the record set exists
SimpleRecordSetExists == SimpleRecordSet # {}

\* Test membership - helper definitions
RecordXOne == [x |-> 1]
RecordXTwo == [x |-> 2]
RecordAB1 == [a |-> 1, b |-> 3]
RecordAB2 == [a |-> 2, b |-> 4]

\* Test with explicit membership
TestMembership == RecordXOne \in SimpleRecordSet
    /\ RecordXTwo \in SimpleRecordSet
    /\ RecordAB1 \in TwoFieldRecordSet
    /\ RecordAB2 \in TwoFieldRecordSet

TypeOK ==
    step \in 0..5
    /\ currentConfig \in ConfigSet

Init ==
    step = 0
    /\ currentConfig = [ringSize |-> 1, replicationFactor |-> 1]

\* This is the key pattern from ConfigRollout.tla that was failing
Next ==
    \/ step < 5
       /\ \E cfg \in ConfigSet:
           currentConfig' = cfg
           /\ step' = step + 1
    \/ step = 5
       /\ UNCHANGED <<step, currentConfig>>

Inv == TypeOK /\ CardinalityTests /\ SimpleRecordSetExists /\ TestMembership

Spec == Init /\ [][Next]_<<step, currentConfig>>

====
