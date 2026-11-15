---- MODULE ReproFairWF ----
\* Minimal: x counts 0 -> 2 under WF; then nothing is enabled (terminal state, stuttering only).
\* <>[](x = 2) HOLDS under WF_x(Next): every fair behavior reaches 2 and stutters there.
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Spec == Init /\ [][Next]_x /\ WF_x(Next)
EvTwo == <>[](x = 2)
LeadsTwo == [](x = 0 => <>(x = 2))
====
