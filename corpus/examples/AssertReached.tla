---- MODULE AssertReached ----
EXTENDS Integers, TLC
\* A reached Assert(FALSE) in the Next action is a safety violation: exploring
\* from x=2 takes x'=3 and Assert(x' < 3) fails. Previously the failing action
\* eval was silently swallowed as a disabled branch, masking the violation.
VARIABLE x
Init == x = 0
Next == /\ x < 3
        /\ x' = x + 1
        /\ Assert(x' < 3, "x' reached 3")
====
