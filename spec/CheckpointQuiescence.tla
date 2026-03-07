------------------------ MODULE CheckpointQuiescence ------------------------
(*
 * Checkpoint quiescence coordination with fingerprint store mode switching.
 *
 * Models the interaction between checkpoint coordination and fingerprint
 * store mode switching in TLC. The key bug occurs when:
 * 1. Checkpoint requests pause
 * 2. Fingerprint store switch triggers (memory threshold)
 * 3. Switch takes write lock
 * 4. Workers waiting for read lock can't reach pause point
 * 5. Quiescence never achieved: paused=0/N
 *
 * Key properties:
 * - No checkpoint drain while any worker is active (not paused)
 * - Fingerprint store switch doesn't deadlock with pause coordination
 * - All workers eventually pause when requested (no starvation)
 *)

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    Workers,            \* Set of worker IDs
    MaxSteps            \* Maximum steps for state space bounding

VARIABLES
    (* Worker state *)
    workerState,        \* [Workers -> WorkerStates]

    (* PauseController state *)
    pauseRequested,     \* BOOLEAN - checkpoint requesting pause
    pausedWorkers,      \* Set of paused worker IDs

    (* FingerprintStore state *)
    fpMode,             \* {"exact", "switching", "hybrid"}
    rwLockState,        \* {"unlocked", "read_locked", "write_locked"}
    readLockHolders,    \* Set of workers holding read lock
    writeLockPending,   \* BOOLEAN - write lock requested but not granted

    (* CheckpointController state *)
    checkpointState,    \* {"idle", "requesting_pause", "waiting_quiescence",
                        \*  "draining", "resuming"}

    (* Model state *)
    stepCount           \* Nat - for state space bounding

vars == <<workerState, pauseRequested, pausedWorkers, fpMode, rwLockState,
          readLockHolders, writeLockPending, checkpointState, stepCount>>

-----------------------------------------------------------------------------
(* Type Definitions *)

WorkerStates == {"working", "pausing", "paused", "blocked_on_fp_lock"}

FPModes == {"exact", "switching", "hybrid"}

LockStates == {"unlocked", "read_locked", "write_locked"}

CheckpointStates == {"idle", "requesting_pause", "waiting_quiescence",
                     "draining", "resuming"}

-----------------------------------------------------------------------------
(* Helper Functions *)

NumPausedWorkers == Cardinality(pausedWorkers)

NumWorkers == Cardinality(Workers)

AllWorkersPaused == pausedWorkers = Workers

AnyWorkerActive ==
    \E w \in Workers : workerState[w] \in {"working", "pausing", "blocked_on_fp_lock"}

QuiescenceAchieved ==
    /\ pauseRequested
    /\ AllWorkersPaused

CanAcquireReadLock ==
    /\ rwLockState /= "write_locked"
    /\ ~writeLockPending

CanAcquireWriteLock ==
    /\ rwLockState = "unlocked"
    /\ readLockHolders = {}

WorkersBlockedOnLock ==
    {w \in Workers : workerState[w] = "blocked_on_fp_lock"}

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ workerState = [w \in Workers |-> "working"]
    /\ pauseRequested = FALSE
    /\ pausedWorkers = {}
    /\ fpMode = "exact"
    /\ rwLockState = "unlocked"
    /\ readLockHolders = {}
    /\ writeLockPending = FALSE
    /\ checkpointState = "idle"
    /\ stepCount = 0

-----------------------------------------------------------------------------
(* Worker Actions *)

(* Worker processes a state - needs to acquire read lock on FP store *)
WorkerProcessState(w) ==
    /\ workerState[w] = "working"
    /\ w \notin readLockHolders  \* Must not already hold the lock
    /\ ~pauseRequested
    /\ stepCount < MaxSteps
    /\ IF CanAcquireReadLock
       THEN /\ readLockHolders' = readLockHolders \union {w}
            /\ rwLockState' = "read_locked"
            /\ UNCHANGED workerState
       ELSE /\ workerState' = [workerState EXCEPT ![w] = "blocked_on_fp_lock"]
            /\ UNCHANGED <<rwLockState, readLockHolders>>
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<pauseRequested, pausedWorkers, fpMode, writeLockPending, checkpointState>>

(* Worker releases read lock after processing *)
WorkerReleaseReadLock(w) ==
    /\ w \in readLockHolders
    /\ readLockHolders' = readLockHolders \ {w}
    /\ rwLockState' = IF readLockHolders' = {} THEN "unlocked" ELSE "read_locked"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers, fpMode,
                   writeLockPending, checkpointState>>

(* Worker tries to pause when pause is requested *)
WorkerTryPause(w) ==
    /\ workerState[w] = "working"
    /\ pauseRequested
    /\ w \notin readLockHolders  \* Must not hold read lock to pause
    /\ workerState' = [workerState EXCEPT ![w] = "pausing"]
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<pauseRequested, pausedWorkers, fpMode, rwLockState,
                   readLockHolders, writeLockPending, checkpointState>>

(* Worker completes pause transition *)
WorkerCompletePause(w) ==
    /\ workerState[w] = "pausing"
    /\ workerState' = [workerState EXCEPT ![w] = "paused"]
    /\ pausedWorkers' = pausedWorkers \union {w}
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<pauseRequested, fpMode, rwLockState, readLockHolders,
                   writeLockPending, checkpointState>>

(* Blocked worker acquires read lock when available *)
WorkerUnblock(w) ==
    /\ workerState[w] = "blocked_on_fp_lock"
    /\ CanAcquireReadLock
    /\ workerState' = [workerState EXCEPT ![w] = "working"]
    /\ readLockHolders' = readLockHolders \union {w}
    /\ rwLockState' = "read_locked"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<pauseRequested, pausedWorkers, fpMode, writeLockPending, checkpointState>>

(*
 * BUG PATH: Blocked worker tries to pause but can't because it's waiting
 * for the read lock. It remains blocked_on_fp_lock.
 *
 * This models the bug: worker is stuck waiting for FP lock, cannot
 * reach pause point, so quiescence is never achieved.
 *)
WorkerBlockedTryPause(w) ==
    /\ workerState[w] = "blocked_on_fp_lock"
    /\ pauseRequested
    \* Worker is stuck - it needs the lock to make progress to pause point
    \* This action represents the worker's futile attempt
    /\ UNCHANGED vars

(* Worker resumes after checkpoint completes *)
WorkerResume(w) ==
    /\ workerState[w] = "paused"
    /\ ~pauseRequested
    /\ workerState' = [workerState EXCEPT ![w] = "working"]
    /\ pausedWorkers' = pausedWorkers \ {w}
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<pauseRequested, fpMode, rwLockState, readLockHolders,
                   writeLockPending, checkpointState>>

-----------------------------------------------------------------------------
(* PauseController Actions *)

RequestPause ==
    /\ checkpointState = "idle"
    /\ ~pauseRequested
    /\ pauseRequested' = TRUE
    /\ checkpointState' = "requesting_pause"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pausedWorkers, fpMode, rwLockState,
                   readLockHolders, writeLockPending>>

WaitForQuiescence ==
    /\ checkpointState = "requesting_pause"
    /\ checkpointState' = "waiting_quiescence"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers, fpMode,
                   rwLockState, readLockHolders, writeLockPending>>

-----------------------------------------------------------------------------
(* CheckpointController Actions *)

(* Checkpoint drain - only allowed when quiescence achieved *)
CheckpointDrain ==
    /\ checkpointState = "waiting_quiescence"
    /\ QuiescenceAchieved
    /\ checkpointState' = "draining"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers, fpMode,
                   rwLockState, readLockHolders, writeLockPending>>

(* Checkpoint completes and requests resume *)
CheckpointComplete ==
    /\ checkpointState = "draining"
    /\ checkpointState' = "resuming"
    /\ pauseRequested' = FALSE
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pausedWorkers, fpMode, rwLockState,
                   readLockHolders, writeLockPending>>

(* Checkpoint cycle completes *)
CheckpointFinish ==
    /\ checkpointState = "resuming"
    /\ pausedWorkers = {}  \* All workers resumed
    /\ checkpointState' = "idle"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers, fpMode,
                   rwLockState, readLockHolders, writeLockPending>>

-----------------------------------------------------------------------------
(* FingerprintStore Actions *)

(*
 * FP store requests switch to hybrid mode - takes write lock.
 * This can trigger during normal operation when memory threshold is reached.
 *
 * BUG TRIGGER: If this happens after pause is requested but before
 * workers have paused, and workers need the read lock to reach their
 * pause point, we get deadlock.
 *)
FPStoreRequestSwitch ==
    /\ fpMode = "exact"
    /\ ~writeLockPending
    /\ stepCount < MaxSteps
    /\ writeLockPending' = TRUE
    /\ fpMode' = "switching"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers, rwLockState,
                   readLockHolders, checkpointState>>

(* FP store acquires write lock when no readers *)
FPStoreAcquireWriteLock ==
    /\ writeLockPending
    /\ CanAcquireWriteLock
    /\ rwLockState' = "write_locked"
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers, fpMode,
                   readLockHolders, writeLockPending, checkpointState>>

(* FP store completes switch and releases write lock *)
FPStoreCompleteSwitch ==
    /\ fpMode = "switching"
    /\ rwLockState = "write_locked"
    /\ fpMode' = "hybrid"
    /\ rwLockState' = "unlocked"
    /\ writeLockPending' = FALSE
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED <<workerState, pauseRequested, pausedWorkers,
                   readLockHolders, checkpointState>>

-----------------------------------------------------------------------------
(* Next State *)

WorkerActions ==
    \E w \in Workers :
        \/ WorkerProcessState(w)
        \/ WorkerReleaseReadLock(w)
        \/ WorkerTryPause(w)
        \/ WorkerCompletePause(w)
        \/ WorkerUnblock(w)
        \/ WorkerResume(w)

PauseControllerActions ==
    \/ RequestPause
    \/ WaitForQuiescence

CheckpointControllerActions ==
    \/ CheckpointDrain
    \/ CheckpointComplete
    \/ CheckpointFinish

FPStoreActions ==
    \/ FPStoreRequestSwitch
    \/ FPStoreAcquireWriteLock
    \/ FPStoreCompleteSwitch

Next ==
    \/ WorkerActions
    \/ PauseControllerActions
    \/ CheckpointControllerActions
    \/ FPStoreActions

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Type Invariant *)

TypeOK ==
    /\ DOMAIN workerState = Workers
    /\ \A w \in Workers : workerState[w] \in WorkerStates
    /\ pauseRequested \in BOOLEAN
    /\ pausedWorkers \subseteq Workers
    /\ fpMode \in FPModes
    /\ rwLockState \in LockStates
    /\ readLockHolders \subseteq Workers
    /\ writeLockPending \in BOOLEAN
    /\ checkpointState \in CheckpointStates
    /\ stepCount \in Nat

-----------------------------------------------------------------------------
(* Safety Invariants *)

(*
 * CRITICAL SAFETY: No checkpoint drain while any worker is active.
 * Violation of this means data corruption - draining state while
 * workers might still be modifying it.
 *)
NoDrainWhileActive ==
    checkpointState = "draining" => ~AnyWorkerActive

(*
 * Consistency: paused workers set matches actual paused state.
 *)
PausedSetConsistent ==
    \A w \in Workers :
        (w \in pausedWorkers) <=> (workerState[w] = "paused")

(*
 * Read lock holders are working or blocked (waiting for more lock).
 * A paused worker should not hold a read lock.
 *)
ReadLockHoldersWorking ==
    \A w \in readLockHolders : workerState[w] \in {"working"}

(*
 * Write lock state consistency.
 *)
WriteLockConsistent ==
    /\ (rwLockState = "write_locked") => (readLockHolders = {})
    /\ (rwLockState = "unlocked") => (readLockHolders = {})
    /\ (rwLockState = "read_locked") => (readLockHolders /= {})

(*
 * Blocked workers are not holding any lock.
 *)
BlockedNotHoldingLock ==
    \A w \in Workers :
        workerState[w] = "blocked_on_fp_lock" => w \notin readLockHolders

Safety ==
    /\ TypeOK
    /\ NoDrainWhileActive
    /\ PausedSetConsistent
    /\ ReadLockHoldersWorking
    /\ WriteLockConsistent
    /\ BlockedNotHoldingLock

-----------------------------------------------------------------------------
(* Deadlock Detection - The Bug *)

(*
 * DEADLOCK STATE: The bug manifests as this condition:
 * - Pause is requested (checkpoint wants quiescence)
 * - Some workers are blocked waiting for FP read lock
 * - FP store holds or is waiting for write lock
 * - No worker can make progress to pause point
 *
 * This is a deadlock: checkpoint waits for quiescence, workers wait for
 * lock, lock holder waits for... nothing (but workers can't pause).
 *)
DeadlockState ==
    /\ pauseRequested
    /\ checkpointState \in {"requesting_pause", "waiting_quiescence"}
    /\ \E w \in Workers : workerState[w] = "blocked_on_fp_lock"
    /\ (rwLockState = "write_locked" \/ writeLockPending)
    /\ ~AllWorkersPaused

(*
 * This invariant should FAIL when the bug manifests.
 * TLC finding a counter-example here demonstrates the bug.
 *)
NoDeadlock == ~DeadlockState

(*
 * Alternative formulation: if we're waiting for quiescence and FP switch
 * is in progress, we can still make progress.
 *)
ProgressPossible ==
    (checkpointState = "waiting_quiescence" /\ fpMode = "switching")
    => \/ AllWorkersPaused
       \/ \E w \in Workers : workerState[w] = "working"

-----------------------------------------------------------------------------
(* Liveness Properties - require fairness *)

(*
 * Weak fairness on all actions.
 *)
Fairness ==
    /\ WF_vars(WorkerActions)
    /\ WF_vars(PauseControllerActions)
    /\ WF_vars(CheckpointControllerActions)
    /\ WF_vars(FPStoreActions)

(*
 * Strong fairness for critical actions to prevent starvation.
 *)
StrongFairness ==
    /\ SF_vars(\E w \in Workers : WorkerUnblock(w))
    /\ SF_vars(\E w \in Workers : WorkerCompletePause(w))

(*
 * Quiescence eventually achieved when pause requested.
 * This property should FAIL in the buggy model!
 *)
QuiescenceEventuallyAchieved ==
    pauseRequested ~> QuiescenceAchieved

(*
 * Workers eventually resume after checkpoint completes.
 *)
WorkersEventuallyResume ==
    checkpointState = "draining" ~> (checkpointState = "idle")

(*
 * FP store switch eventually completes.
 *)
FPSwitchEventuallyCompletes ==
    fpMode = "switching" ~> fpMode = "hybrid"

(*
 * Checkpoint cycle eventually completes (full liveness).
 *)
CheckpointEventuallyCompletes ==
    checkpointState = "requesting_pause" ~> checkpointState = "idle"

LiveSpec == Spec /\ Fairness /\ StrongFairness

-----------------------------------------------------------------------------
(* State Constraints for Bounded Model Checking *)

StateConstraint == stepCount <= MaxSteps

-----------------------------------------------------------------------------
(* Symmetry for Optimization - defined in .cfg file *)
(* TLC uses SYMMETRY directive in config file for worker permutation *)

=============================================================================
