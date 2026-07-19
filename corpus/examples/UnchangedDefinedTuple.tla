---- MODULE UnchangedDefinedTuple ----
EXTENDS Integers
CONSTANT N
Proc == 1..N
VARIABLES failed, ctr
\* A defined variable-tuple, the standard TLA+ idiom. `UNCHANGED frozen` must
\* stage the component prime `failed'` so the trailing Guard's `failed'[i]`
\* resolves; otherwise `failed'` reads as an unbound ModelValue and next_states
\* panics (Chandra-Toueg EnvironmentController's `UNCHANGED envVars`).
frozen == << failed >>
Init == /\ failed = [i \in Proc |-> FALSE]
        /\ ctr = 0
Act == \E i \in Proc :
          /\ failed[i] = FALSE
          /\ ctr < 3
          /\ ctr' = ctr + 1
          /\ UNCHANGED frozen
Guard == \A i \in Proc : (failed'[i] = FALSE => ctr' <= 3)
Next == Act /\ Guard
====
