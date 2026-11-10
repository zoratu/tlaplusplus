---- MODULE PrimedArgMC ----
EXTENDS Integers
\* Isolates the FIRST #188 site only: the instance action `Det!Receive(i, chan'[i])`
\* reads the PRIMED argument `chan'[i]` — staged by the preceding `chan' = ...`
\* conjunct — so it must be evaluated against the caller's STAGED context. Against
\* the bare (unstaged) context `chan'[i]` is unbound and the step produces no/wrong
\* successor. The callee (PrimedArgInner!Receive) has no leading `\/`, so the
\* second #188 site (leading guard-disjunction split) is not involved here.
\* Chaff: instance_primed_arg_unstaged.
CONSTANT N
Proc == 1..N
VARIABLES clock, seen, chan
Det == INSTANCE PrimedArgInner
Init ==
  /\ clock = [i \in Proc |-> 0]
  /\ seen  = [i \in Proc |-> 0]
  /\ chan  = [i \in Proc |-> 0]
Step(i) ==
  /\ chan' = [ chan EXCEPT ![i] = clock[i] ]
  /\ Det!Receive(i, chan'[i])
Next == \E i \in Proc : Step(i)
====
