---- MODULE TraceMinimizationDiamond ----
EXTENDS Naturals

\* T9 fixture: a spec with multiple alternative paths to a violating state.
\*
\* Variables:
\*   - count: 0..N
\*   - noise: 0..M (orthogonal to the invariant)
\*   - phase: "A" or "B" (selects which "path" Tick follows)
\*
\* The invariant only constrains `count`.  `noise` and `phase` are pure
\* presentation noise from the invariant's perspective (Phase B should
\* mark them as such).
\*
\* The BFS-discovered trace can be longer than necessary if early states
\* explore Bump (changes noise) before reaching the violating count.
\* Phase A minimization should find a shorter direct-tick path.

VARIABLES count, noise, phase

vars == <<count, noise, phase>>

Init ==
    /\ count = 0
    /\ noise = 0
    /\ phase = "A"

\* Direct path: bump count straight to the violation.
Tick ==
    /\ count < 6
    /\ count' = count + 1
    /\ UNCHANGED <<noise, phase>>

\* Independent action that bumps a noise variable.  Has nothing to do
\* with the invariant.  Stays available even after count reaches the
\* violation, so the trace can keep growing.
Bump ==
    /\ noise < 4
    /\ noise' = noise + 1
    /\ UNCHANGED <<count, phase>>

\* Phase swapping.  Also unrelated to the invariant.
SwapPhase ==
    /\ phase' = IF phase = "A" THEN "B" ELSE "A"
    /\ UNCHANGED <<count, noise>>

Next == Tick \/ Bump \/ SwapPhase

Spec == Init /\ [][Next]_vars

\* Invariant: count must stay below 5.  Tick eventually drives it to 5.
Inv == count < 5

================================================================================
