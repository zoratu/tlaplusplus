---- MODULE ClockDetector ----
EXTENDS Integers
CONSTANT Proc
VARIABLES clock, seen
LocalTick(i) == clock' = [ clock EXCEPT ![i] = IF clock[i] >= 5 THEN 0 ELSE clock[i] + 1 ]
Receive(i, incoming) ==
  /\ \/ /\ clock[i] % 2 = 0
        /\ clock[i] % 3 = 0
     \/ /\ clock[i] % 2 # 0
        /\ clock[i] % 3 # 0
  /\ LocalTick(i)
  /\ seen' = [ seen EXCEPT ![i] = incoming ]
====
