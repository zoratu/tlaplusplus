---- MODULE RecursiveTest ----
EXTENDS Naturals, Sequences

VARIABLES x, result

(* RECURSIVE declarations for operators that call themselves *)
RECURSIVE Factorial(_)
RECURSIVE SumSeq(_)
RECURSIVE SeqLen(_)

(* Factorial: n! = n * (n-1)! with base case 0! = 1 *)
Factorial(n) ==
    IF n <= 1 THEN 1
    ELSE n * Factorial(n - 1)

(* SumSeq: Sum all elements in a sequence *)
SumSeq(s) ==
    IF s = <<>> THEN 0
    ELSE Head(s) + SumSeq(Tail(s))

(* SeqLen: Compute length of a sequence recursively *)
SeqLen(s) ==
    IF s = <<>> THEN 0
    ELSE 1 + SeqLen(Tail(s))

(* Non-recursive helper for testing *)
Double(n) == n + n

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

(* Invariants to verify recursive operators work correctly *)
FactorialCorrect ==
    result = Factorial(x)

FactorialValues ==
    /\ Factorial(0) = 1
    /\ Factorial(1) = 1
    /\ Factorial(2) = 2
    /\ Factorial(3) = 6
    /\ Factorial(4) = 24
    /\ Factorial(5) = 120

SumSeqWorks ==
    /\ SumSeq(<<>>) = 0
    /\ SumSeq(<<1>>) = 1
    /\ SumSeq(<<1, 2>>) = 3
    /\ SumSeq(<<1, 2, 3>>) = 6
    /\ SumSeq(<<1, 2, 3, 4, 5>>) = 15

SeqLenWorks ==
    /\ SeqLen(<<>>) = 0
    /\ SeqLen(<<1>>) = 1
    /\ SeqLen(<<1, 2, 3>>) = 3

Spec == Init /\ [][Next]_<<x, result>>

====
