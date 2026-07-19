---- MODULE ClockDetectorMC ----
EXTENDS Integers
CONSTANT N
Proc == 1..N
VARIABLES clock, seen, chan
Det == INSTANCE ClockDetector
Init ==
  /\ clock = [i \in Proc |-> 0]
  /\ seen = [i \in Proc |-> {}]
  /\ chan = [i \in Proc |-> {}]
Step(i) ==
  /\ chan' = [ chan EXCEPT ![i] = {clock[i]} ]
  /\ Det!Receive(i, chan'[i])
Next == \E i \in Proc : Step(i)
====
