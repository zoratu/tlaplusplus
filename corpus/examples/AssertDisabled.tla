---- MODULE AssertDisabled ----
EXTENDS Integers, TLC
\* The Assert sits in a branch whose guard (x > 100) is never satisfiable, so
\* conjunction short-circuit means it is never reached -> NO violation, and the
\* reachable state count is unaffected (x = 0,1,2,3 -> 4 states). The x=3
\* self-loop avoids a deadlock so TLC (run with -deadlock) also reports no
\* violation. Guards against the reached-assertion side channel producing FALSE
\* violations.
VARIABLE x
Init == x = 0
Next == \/ (x < 3 /\ x' = x + 1)
        \/ (x = 3 /\ x' = x)
        \/ (x > 100 /\ x' = x /\ Assert(FALSE, "must never fire"))
====
