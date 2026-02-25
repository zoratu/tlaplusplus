---- MODULE RecursiveTest ----
EXTENDS Naturals

VARIABLES x, result

Factorial(n) ==
    IF n <= 1 THEN 1
    ELSE n * (n - 1)

Fib(n) ==
    IF n <= 1 THEN n
    ELSE n - 1 + n - 2

Sum(n) ==
    IF n = 0 THEN 0
    ELSE n + n - 1

TypeOK ==
    /\ x \in 0..10
    /\ result \in Nat

Init ==
    /\ x = 0
    /\ result = Factorial(0)

Next ==
    /\ x < 5
    /\ LET newX == x + 1
       IN /\ x' = newX
          /\ result' = Factorial(newX)

FactorialCorrect ==
    result = Factorial(x)

FibonacciWorks ==
    /\ Fib(0) = 0
    /\ Fib(1) = 1
    /\ Fib(2) >= 0

SumWorks ==
    Sum(5) >= 0

Spec == Init /\ [][Next]_<<x, result>>

====
