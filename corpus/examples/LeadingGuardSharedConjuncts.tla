---- MODULE LeadingGuardSharedConjuncts ----
EXTENDS Integers
\* Isolates the SECOND #188 site only: an action body of the shape
\*   /\ (\/ A \/ B)        <- leading guard-disjunction
\*   /\ <shared primed conjuncts>
\* The leading `\/`-group is NOT the sole leading conjunct (LocalTick and seen'
\* follow it), so `leading_conjunct_is_sole` must return false and the body must
\* stay one disjunct. Splitting it (the bug) absorbs the shared LocalTick/seen'
\* conjuncts into only the LAST branch, dropping them from the first — so the
\* first branch stages no primes and the state graph diverges. No INSTANCE here,
\* so the first #188 site (primed-arg staging) is not involved.
\* Chaff: instance_leading_guard_split. Mirrors ClockDetector!Receive's body.
VARIABLES clock, seen
Proc == 1..2
LocalTick(i) == clock' = [ clock EXCEPT ![i] = IF clock[i] >= 5 THEN 0 ELSE clock[i] + 1 ]
Step(i) ==
  /\ \/ /\ clock[i] % 2 = 0
        /\ clock[i] % 3 = 0
     \/ /\ clock[i] % 2 # 0
        /\ clock[i] % 3 # 0
  /\ LocalTick(i)
  /\ seen' = [ seen EXCEPT ![i] = clock[i] ]
Init ==
  /\ clock = [i \in Proc |-> 0]
  /\ seen  = [i \in Proc |-> 0]
Next == \E i \in Proc : Step(i)
====
