----------------------------- MODULE QueueSegmentSync -----------------------------
(*
 * Models the queue segment S3 sync race condition.
 *
 * The bug:
 * - Checkpoint flush deletes consumed segments from local disk
 * - S3 upload runs on a timer (every 10s)
 * - Race: segments deleted locally before S3 uploads them
 * - On resume, segments are missing
 *
 * The fix:
 * - defer_segment_deletion flag prevents deletion during checkpoint
 * - Segments stay on local disk until S3 confirms upload
 * - Local pruning only deletes after S3 has the segment
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    MaxSegmentId,      \* Maximum segment ID to explore
    DeferDeletion      \* TRUE = fixed version, FALSE = buggy version

VARIABLES
    localSegments,     \* Set of segment IDs on local disk
    s3Segments,        \* Set of segment IDs uploaded to S3
    consumedSegments,  \* Set of segment IDs that have been consumed
    checkpointDone,    \* TRUE after checkpoint flush has run
    nextSegmentId      \* Next segment ID to allocate

vars == <<localSegments, s3Segments, consumedSegments, checkpointDone, nextSegmentId>>

TypeOK ==
    /\ localSegments \in SUBSET (0..MaxSegmentId)
    /\ s3Segments \in SUBSET (0..MaxSegmentId)
    /\ consumedSegments \in SUBSET (0..MaxSegmentId)
    /\ checkpointDone \in BOOLEAN
    /\ nextSegmentId \in (0..MaxSegmentId+1)

Init ==
    /\ localSegments = {}
    /\ s3Segments = {}
    /\ consumedSegments = {}
    /\ checkpointDone = FALSE
    /\ nextSegmentId = 0

-----------------------------------------------------------------------------
\* Spill a segment to disk
SpillSegment ==
    /\ nextSegmentId <= MaxSegmentId
    /\ localSegments' = localSegments \union {nextSegmentId}
    /\ nextSegmentId' = nextSegmentId + 1
    /\ UNCHANGED <<s3Segments, consumedSegments, checkpointDone>>

\* Consume (load) a segment into memory
ConsumeSegment(segId) ==
    /\ segId \in localSegments
    /\ segId \notin consumedSegments
    /\ consumedSegments' = consumedSegments \union {segId}
    /\ UNCHANGED <<localSegments, s3Segments, checkpointDone, nextSegmentId>>

ConsumeAnySegment == \E segId \in localSegments : ConsumeSegment(segId)

-----------------------------------------------------------------------------
\* Checkpoint flush - the critical operation
\* BUGGY: Deletes consumed segments immediately
\* FIXED: Keeps segments on disk for S3 to upload
CheckpointFlush ==
    /\ ~checkpointDone
    /\ consumedSegments # {}  \* Must have consumed something
    /\ IF DeferDeletion
       THEN
           \* FIXED: Don't delete, let S3 handle it
           localSegments' = localSegments
       ELSE
           \* BUGGY: Delete consumed segments immediately
           localSegments' = localSegments \ consumedSegments
    /\ checkpointDone' = TRUE
    /\ UNCHANGED <<s3Segments, consumedSegments, nextSegmentId>>

-----------------------------------------------------------------------------
\* S3 uploads any local segment (runs on timer)
S3Upload(segId) ==
    /\ segId \in localSegments
    /\ segId \notin s3Segments
    /\ s3Segments' = s3Segments \union {segId}
    /\ UNCHANGED <<localSegments, consumedSegments, checkpointDone, nextSegmentId>>

S3UploadAny == \E segId \in localSegments : S3Upload(segId)

\* Local prune after S3 confirms (only in fixed version)
LocalPrune(segId) ==
    /\ DeferDeletion
    /\ segId \in localSegments
    /\ segId \in s3Segments  \* Only prune after S3 has it
    /\ segId \in consumedSegments  \* And it's been consumed
    /\ localSegments' = localSegments \ {segId}
    /\ UNCHANGED <<s3Segments, consumedSegments, checkpointDone, nextSegmentId>>

LocalPruneAny == \E segId \in localSegments : LocalPrune(segId)

-----------------------------------------------------------------------------
Next ==
    \/ SpillSegment
    \/ ConsumeAnySegment
    \/ CheckpointFlush
    \/ S3UploadAny
    \/ LocalPruneAny

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* Safety: After checkpoint, all consumed segments must be recoverable
\* They must exist on local disk OR in S3
SegmentsRecoverable ==
    checkpointDone =>
        \A segId \in consumedSegments :
            (segId \in localSegments) \/ (segId \in s3Segments)

SafetyInvariant ==
    /\ TypeOK
    /\ SegmentsRecoverable

=============================================================================
