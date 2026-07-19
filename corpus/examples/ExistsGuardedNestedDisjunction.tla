---- MODULE ExistsGuardedNestedDisjunction ----
EXTENDS Integers
CONSTANT N
Proc == 1..N
VARIABLES failed, phase, moved
Init == /\ failed = [i \in Proc |-> FALSE]
        /\ phase = [i \in Proc |-> 0]
        /\ moved = [i \in Proc |-> "INIT"]
Reset == /\ moved' = [i \in Proc |-> "NO"]
         /\ UNCHANGED << failed, phase >>
\* `Step` is `\E i : /\ g1 /\ g2 /\ prime /\ UNCHANGED /\ (\/ A \/ B)`. The inner
\* `\/` lives inside the last `/\` conjunct: it must not be split off as a
\* top-level disjunct, which would drop the `moved[i]="NO"` guard and the
\* `phase'`/`UNCHANGED failed` conjuncts, firing A/B from every state and reading
\* an unstaged `failed'` in Guard (EnvironmentController's `ProcTick`).
Step == \E i \in Proc :
           /\ failed[i] = FALSE
           /\ moved[i] = "NO"
           /\ phase' = [phase EXCEPT ![i] = phase[i] + 1]
           /\ UNCHANGED failed
           /\ \/ moved' = [moved EXCEPT ![i] = "A"]
              \/ moved' = [moved EXCEPT ![i] = "B"]
Guard == \A i \in Proc : (failed'[i] = FALSE => phase'[i] <= 4)
Next ==
  /\ \/ Reset
     \/ Step
  /\ Guard
====
