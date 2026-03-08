------------------------- MODULE WorkStealingTermination -------------------------
(*
 * Models the work-stealing termination detection protocol.
 *
 * This specification verifies that the parallel exploration correctly
 * detects termination when all work is complete:
 * - Per-worker active flags (cache-line padded)
 * - Per-NUMA idle counters
 * - Global injector queue
 *
 * Termination requires:
 * 1. All workers are idle
 * 2. Global injector queue is empty
 * 3. All local deques are empty
 *
 * The protocol must avoid:
 * - Premature termination (work still available)
 * - Missed work (items added while checking)
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumWorkers,        \* Number of worker threads
    NumNumaNodes,      \* Number of NUMA nodes
    MaxWorkItems       \* Maximum work items to model

VARIABLES
    globalQueue,       \* Global injector queue (set of work item IDs)
    localDeques,       \* Per-worker local deques (function: worker -> set of items)
    workerActive,      \* Per-worker active flag
    numaIdleCount,     \* Per-NUMA-node count of idle workers
    terminationCheck,  \* TRUE if a termination check is in progress
    checkPhase,        \* Phase of termination check: "idle", "checking_queues", "done"
    allWorkersIdle,    \* Cached result: all workers idle?
    allQueuesEmpty,    \* Cached result: all queues empty?
    terminated,        \* TRUE if termination was declared
    workCounter        \* Counter for generating unique work IDs

vars == <<globalQueue, localDeques, workerActive, numaIdleCount, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

Workers == 1..NumWorkers
NumaNodes == 0..(NumNumaNodes-1)
WorkerToNuma == [w \in Workers |-> (w-1) % NumNumaNodes]

TypeOK ==
    /\ globalQueue \in SUBSET (0..MaxWorkItems-1)
    /\ localDeques \in [Workers -> SUBSET (0..MaxWorkItems-1)]
    /\ workerActive \in [Workers -> BOOLEAN]
    /\ numaIdleCount \in [NumaNodes -> 0..NumWorkers]
    /\ terminationCheck \in BOOLEAN
    /\ checkPhase \in {"idle", "checking_queues", "done"}
    /\ allWorkersIdle \in BOOLEAN
    /\ allQueuesEmpty \in BOOLEAN
    /\ terminated \in BOOLEAN
    /\ workCounter \in (0..MaxWorkItems)

Init ==
    /\ globalQueue = {}
    /\ localDeques = [w \in Workers |-> {}]
    /\ workerActive = [w \in Workers |-> FALSE]
    /\ numaIdleCount = [n \in NumaNodes |-> Cardinality({w \in Workers : WorkerToNuma[w] = n})]
    /\ terminationCheck = FALSE
    /\ checkPhase = "idle"
    /\ allWorkersIdle = TRUE
    /\ allQueuesEmpty = TRUE
    /\ terminated = FALSE
    /\ workCounter = 0

-----------------------------------------------------------------------------
\* Work generation

\* Add work to global queue
AddGlobalWork ==
    /\ ~terminated
    /\ workCounter < MaxWorkItems
    /\ globalQueue' = globalQueue \union {workCounter}
    /\ workCounter' = workCounter + 1
    /\ UNCHANGED <<localDeques, workerActive, numaIdleCount, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated>>

\* Add work to a worker's local deque
AddLocalWork(w) ==
    /\ ~terminated
    /\ workCounter < MaxWorkItems
    /\ localDeques' = [localDeques EXCEPT ![w] = @ \union {workCounter}]
    /\ workCounter' = workCounter + 1
    /\ UNCHANGED <<globalQueue, workerActive, numaIdleCount, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated>>

-----------------------------------------------------------------------------
\* Worker operations

\* Worker becomes active (has work to do)
WorkerBecomeActive(w) ==
    /\ ~terminated
    /\ ~workerActive[w]
    /\ (globalQueue # {} \/ localDeques[w] # {})  \* Must have work available
    /\ workerActive' = [workerActive EXCEPT ![w] = TRUE]
    /\ numaIdleCount' = [numaIdleCount EXCEPT ![WorkerToNuma[w]] = @ - 1]
    /\ UNCHANGED <<globalQueue, localDeques, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

\* Worker processes work from global queue
WorkerProcessGlobal(w) ==
    /\ ~terminated
    /\ workerActive[w]
    /\ globalQueue # {}
    /\ \E item \in globalQueue :
        globalQueue' = globalQueue \ {item}
    /\ UNCHANGED <<localDeques, workerActive, numaIdleCount, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

\* Worker processes work from local deque
WorkerProcessLocal(w) ==
    /\ ~terminated
    /\ workerActive[w]
    /\ localDeques[w] # {}
    /\ \E item \in localDeques[w] :
        localDeques' = [localDeques EXCEPT ![w] = @ \ {item}]
    /\ UNCHANGED <<globalQueue, workerActive, numaIdleCount, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

\* Worker steals from another worker's deque
WorkerSteal(thief, victim) ==
    /\ ~terminated
    /\ thief # victim
    /\ workerActive[thief]
    /\ localDeques[victim] # {}
    /\ \E item \in localDeques[victim] :
        /\ localDeques' = [localDeques EXCEPT
             ![victim] = @ \ {item},
             ![thief] = @ \union {item}]
    /\ UNCHANGED <<globalQueue, workerActive, numaIdleCount, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

\* Worker becomes idle (no more local work)
WorkerBecomeIdle(w) ==
    /\ ~terminated
    /\ workerActive[w]
    /\ localDeques[w] = {}  \* Local deque empty
    /\ globalQueue = {}     \* Global queue empty (simplified)
    /\ workerActive' = [workerActive EXCEPT ![w] = FALSE]
    /\ numaIdleCount' = [numaIdleCount EXCEPT ![WorkerToNuma[w]] = @ + 1]
    /\ UNCHANGED <<globalQueue, localDeques, terminationCheck, checkPhase, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

-----------------------------------------------------------------------------
\* Termination detection

\* Begin termination check - snapshot worker states
BeginTerminationCheck ==
    /\ ~terminated
    /\ ~terminationCheck
    /\ checkPhase = "idle"
    /\ terminationCheck' = TRUE
    /\ checkPhase' = "checking_queues"
    \* Snapshot: are all workers idle?
    /\ allWorkersIdle' = (\A w \in Workers : ~workerActive[w])
    /\ UNCHANGED <<globalQueue, localDeques, workerActive, numaIdleCount, allQueuesEmpty, terminated, workCounter>>

\* Check queue emptiness
CheckQueuesEmpty ==
    /\ ~terminated
    /\ terminationCheck
    /\ checkPhase = "checking_queues"
    \* Check if all queues are empty
    /\ allQueuesEmpty' = (globalQueue = {} /\ \A w \in Workers : localDeques[w] = {})
    /\ checkPhase' = "done"
    /\ UNCHANGED <<globalQueue, localDeques, workerActive, numaIdleCount, allWorkersIdle, terminationCheck, terminated, workCounter>>

\* Complete termination check
CompleteTerminationCheck ==
    /\ ~terminated
    /\ terminationCheck
    /\ checkPhase = "done"
    /\ IF allWorkersIdle /\ allQueuesEmpty
       THEN
           \* Double-check: workers still idle and queues still empty?
           /\ IF (\A w \in Workers : ~workerActive[w]) /\
                 globalQueue = {} /\
                 (\A w \in Workers : localDeques[w] = {})
              THEN terminated' = TRUE
              ELSE terminated' = FALSE
       ELSE
           terminated' = FALSE
    /\ terminationCheck' = FALSE
    /\ checkPhase' = "idle"
    /\ UNCHANGED <<globalQueue, localDeques, workerActive, numaIdleCount, allWorkersIdle, allQueuesEmpty, workCounter>>

\* Abort termination check (e.g., worker became active)
AbortTerminationCheck ==
    /\ terminationCheck
    /\ checkPhase \in {"checking_queues", "done"}
    /\ \E w \in Workers : workerActive[w]  \* A worker is now active
    /\ terminationCheck' = FALSE
    /\ checkPhase' = "idle"
    /\ UNCHANGED <<globalQueue, localDeques, workerActive, numaIdleCount, allWorkersIdle, allQueuesEmpty, terminated, workCounter>>

-----------------------------------------------------------------------------
\* Helper definitions

TotalWork == Cardinality(globalQueue) + Cardinality(UNION {localDeques[w] : w \in Workers})

-----------------------------------------------------------------------------
\* Next state relation

Next ==
    \/ AddGlobalWork
    \/ \E w \in Workers : AddLocalWork(w)
    \/ \E w \in Workers : WorkerBecomeActive(w)
    \/ \E w \in Workers : WorkerProcessGlobal(w)
    \/ \E w \in Workers : WorkerProcessLocal(w)
    \/ \E thief, victim \in Workers : WorkerSteal(thief, victim)
    \/ \E w \in Workers : WorkerBecomeIdle(w)
    \/ BeginTerminationCheck
    \/ CheckQueuesEmpty
    \/ CompleteTerminationCheck
    \/ AbortTerminationCheck

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
\* Safety properties

\* CRITICAL: Termination only when no work remains
NoPrematureTermination ==
    terminated => (globalQueue = {} /\ \A w \in Workers : localDeques[w] = {})

\* All workers are idle when terminated
AllIdleWhenTerminated ==
    terminated => \A w \in Workers : ~workerActive[w]

\* NUMA idle counts are consistent with worker states
NumaCountsConsistent ==
    \A n \in NumaNodes :
        numaIdleCount[n] = Cardinality({w \in Workers : WorkerToNuma[w] = n /\ ~workerActive[w]})

-----------------------------------------------------------------------------
\* Invariants

SafetyInvariant ==
    /\ TypeOK
    /\ NoPrematureTermination
    /\ AllIdleWhenTerminated
    /\ NumaCountsConsistent

=============================================================================
