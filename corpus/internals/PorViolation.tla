---- MODULE PorViolation ----
EXTENDS Naturals

\* Two independent processes each increment a shared counter (different vars
\* so they look independent statically — but the invariant looks at both,
\* so the violation is observable from any reachable state where count is
\* large enough).
\*
\* The state space includes states where count = 4. The invariant claims
\* count < 3, so the violation is real.

VARIABLES count, flag

vars == <<count, flag>>

Init == /\ count = 0
        /\ flag = FALSE

Tick == /\ count < 4
        /\ count' = count + 1
        /\ UNCHANGED flag

Toggle == /\ flag' = ~flag
          /\ UNCHANGED count

Next == Tick \/ Toggle

Spec == Init /\ [][Next]_vars

\* Tick raises count up to 4 — violates this invariant.
Inv == count < 3

================================================================================
