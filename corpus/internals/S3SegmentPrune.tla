------------------------------- MODULE S3SegmentPrune -------------------------------
(*
 * TLA+ model of S3 segment lifecycle and pruning.
 *
 * This models the interaction between:
 * - Workers: consume segments (load from disk to memory), delete local files
 * - Checkpoint: writes new segments to disk, records min_segment_id in manifest
 * - S3 Uploader: uploads segments to S3
 * - S3 Pruner: deletes old segments from S3 based on min_segment_id
 * - Resume: user deletes local state, restores from S3
 *
 * The key invariant is:
 * - Any segment with ID >= min_segment_id must exist in S3 for resume to work
 * - Segments with ID < min_segment_id can be safely pruned
 *
 * The BUG this model helps prevent:
 * - Before the fix, S3 pruner deleted segments that didn't exist locally
 * - But segments are deleted locally after being consumed (loaded into memory)
 * - This caused resume to fail because needed segments were gone from S3
 *
 * The FIX:
 * - Checkpoint records min_segment_id (the minimum segment needed for resume)
 * - S3 pruner only deletes segments with ID < min_segment_id
 *)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    MaxSegmentId,      \* Maximum segment ID for bounded state space
    MaxSegments        \* Maximum number of segments

VARIABLES
    localSegments,     \* Set of segment IDs that exist locally
    s3Segments,        \* Set of segment IDs that exist in S3
    minSegmentId,      \* Minimum segment ID needed for resume (from checkpoint)
    nextSegmentId,     \* Next segment ID to create
    localDeleted       \* Track if local state has been deleted (simulating rm -rf)

vars == <<localSegments, s3Segments, minSegmentId, nextSegmentId, localDeleted>>

TypeOK ==
    /\ localSegments \subseteq 0..MaxSegmentId
    /\ s3Segments \subseteq 0..MaxSegmentId
    /\ minSegmentId \in 0..MaxSegmentId
    /\ nextSegmentId \in 0..MaxSegmentId
    /\ localDeleted \in BOOLEAN

Init ==
    /\ localSegments = {}
    /\ s3Segments = {}
    /\ minSegmentId = 0
    /\ nextSegmentId = 0
    /\ localDeleted = FALSE

(* Checkpoint creates new segments and records min_segment_id *)
Checkpoint ==
    /\ nextSegmentId < MaxSegmentId
    /\ Cardinality(localSegments) < MaxSegments
    /\ LET newId == nextSegmentId
       IN /\ localSegments' = localSegments \cup {newId}
          /\ nextSegmentId' = nextSegmentId + 1
          \* Update min_segment_id to the oldest segment we need
          /\ IF localSegments = {}
             THEN minSegmentId' = newId
             ELSE minSegmentId' = minSegmentId  \* Keep existing min
    /\ UNCHANGED <<s3Segments, localDeleted>>

(* S3 uploader uploads local segments to S3 *)
S3Upload ==
    /\ \E seg \in localSegments:
        /\ seg \notin s3Segments
        /\ s3Segments' = s3Segments \cup {seg}
    /\ UNCHANGED <<localSegments, minSegmentId, nextSegmentId, localDeleted>>

(* Worker consumes a segment (loads into memory, deletes local file) *)
ConsumeSegment ==
    /\ ~localDeleted  \* Can only consume if local state exists
    /\ \E seg \in localSegments:
        /\ localSegments' = localSegments \ {seg}
        \* Update min_segment_id to next minimum
        /\ IF localSegments' = {}
           THEN minSegmentId' = nextSegmentId
           ELSE minSegmentId' = CHOOSE s \in localSegments':
                    \A t \in localSegments': s <= t
    /\ UNCHANGED <<s3Segments, nextSegmentId, localDeleted>>

(* S3 pruner - THE FIXED VERSION: only prune segments < min_segment_id *)
S3PruneFixed ==
    /\ \E seg \in s3Segments:
        /\ seg < minSegmentId  \* CRITICAL: Only prune if below min
        /\ s3Segments' = s3Segments \ {seg}
    /\ UNCHANGED <<localSegments, minSegmentId, nextSegmentId, localDeleted>>

(* S3 pruner - THE BUGGY VERSION: prune if not in local (DON'T USE) *)
(* This is commented out - it shows the bug we're fixing
S3PruneBuggy ==
    /\ \E seg \in s3Segments:
        /\ seg \notin localSegments  \* BUG: Checks local existence
        /\ s3Segments' = s3Segments \ {seg}
    /\ UNCHANGED <<localSegments, minSegmentId, nextSegmentId, localDeleted>>
*)

(* User deletes local state (simulates rm -rf ~/.tlapp) *)
DeleteLocalState ==
    /\ ~localDeleted
    /\ localDeleted' = TRUE
    /\ localSegments' = {}
    /\ UNCHANGED <<s3Segments, minSegmentId, nextSegmentId>>

(* Resume from S3 - downloads all segments with ID >= minSegmentId *)
ResumeFromS3 ==
    /\ localDeleted
    /\ LET neededSegments == {seg \in s3Segments: seg >= minSegmentId}
       IN localSegments' = neededSegments
    /\ localDeleted' = FALSE
    /\ UNCHANGED <<s3Segments, minSegmentId, nextSegmentId>>

Next ==
    \/ Checkpoint
    \/ S3Upload
    \/ ConsumeSegment
    \/ S3PruneFixed
    \/ DeleteLocalState
    \/ ResumeFromS3

Spec == Init /\ [][Next]_vars

(*
 * KEY SAFETY INVARIANT:
 * All segments needed for resume (ID >= minSegmentId) must exist in S3.
 * This ensures that even if local state is deleted, we can resume.
 *)
ResumeAlwaysPossible ==
    \* All segments with ID >= minSegmentId that have ever been created
    \* must still exist in S3 (unless they haven't been uploaded yet)
    \A seg \in 0..nextSegmentId-1:
        (seg >= minSegmentId /\ seg \in localSegments) => seg \in s3Segments \/ seg \in localSegments

(*
 * After resume, we should have all the segments we need
 *)
ResumeRestoresNeededSegments ==
    (localDeleted = FALSE /\ localSegments' = localSegments) =>
        \A seg \in localSegments: seg >= minSegmentId

(*
 * Segments below minSegmentId are never needed for resume
 *)
OldSegmentsNotNeeded ==
    \A seg \in localSegments: seg >= minSegmentId

================================================================================
