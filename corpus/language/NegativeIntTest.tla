---- MODULE NegativeIntTest ----
EXTENDS Integers

VARIABLES x, y

NegativeRange == -5..5

RangeCorrect ==
    /\ -5 \in NegativeRange
    /\ 0 \in NegativeRange
    /\ 5 \in NegativeRange
    /\ -6 \notin NegativeRange
    /\ 6 \notin NegativeRange

NegativeArithmetic ==
    /\ -3 + 5 = 2
    /\ -3 - 5 = -8
    /\ -3 * 5 = -15
    /\ -3 < 0
    /\ -3 > -5

Init ==
    /\ x = -5
    /\ y = 0

Increment ==
    /\ x < 5
    /\ y >= -950
    /\ y <= 950
    /\ LET newX == x + 1
       IN /\ x' = newX
          /\ y' = y + newX

Decrement ==
    /\ x > -5
    /\ x < 0
    /\ y >= -950
    /\ x' = x - 1
    /\ y' = y - 1

Next ==
    \/ Increment
    \/ Decrement
    \/ (x = 5 /\ UNCHANGED <<x, y>>)

TypeOK ==
    /\ x \in -10..10
    /\ y \in -1000..1000

Inv ==
    /\ TypeOK
    /\ RangeCorrect
    /\ NegativeArithmetic

Spec == Init /\ [][Next]_<<x, y>>

====
