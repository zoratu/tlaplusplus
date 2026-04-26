// T1.5 soundness regression: Next bodies of the shape
//   /\ guard
//   /\ \/ A
//      \/ B
//   /\ shared_post
// must produce exactly the successors of the inner disjunction with the
// outer guard and shared post-condition applied. The pre-fix splitter
// (a) fabricated an extra "stutter" successor, and (b) dropped
// `shared_post` from one branch — leading to over-counting (e.g.
// ViewTest.tla reported 121 distinct states instead of TLC's 106).

use std::collections::BTreeMap;
use tlaplusplus::tla::{
    TlaDefinition, TlaValue, evaluate_next_states, parse_tla_module_text, tla_state,
};

#[test]
fn conjunctive_next_with_inner_disjunction_produces_exactly_inner_branches() {
    let module = parse_tla_module_text(
        r#"---- MODULE T1_5_Inline ----
EXTENDS Naturals

VARIABLES x, y, timestamp

Init ==
    /\ x = 0
    /\ y = 0
    /\ timestamp = 0

Next ==
    /\ x + y < 15
    /\ \/ /\ x < 10
          /\ x' = x + 1
          /\ y' = y
       \/ /\ y < 10
          /\ y' = y + 1
          /\ x' = x
    /\ timestamp' = timestamp + 1
====
"#,
    )
    .expect("inline T1.5 spec parses");

    let next_def: &TlaDefinition = module
        .definitions
        .get("Next")
        .expect("Next definition present");

    let state = tla_state([
        ("x", TlaValue::Int(0)),
        ("y", TlaValue::Int(0)),
        ("timestamp", TlaValue::Int(0)),
    ]);

    let successors = evaluate_next_states(&next_def.body, &module.definitions, &state)
        .expect("conjunctive next with inner disjunction must evaluate cleanly");

    assert_eq!(
        successors.len(),
        2,
        "expected exactly 2 successors (one per inner disjunct), got {}: {successors:#?}",
        successors.len()
    );

    // Every successor must carry the shared post-condition.
    for s in &successors {
        let ts = s
            .get("timestamp")
            .and_then(|v| v.as_int().ok())
            .expect("timestamp must be set in every successor");
        assert_eq!(
            ts, 1,
            "shared post-condition `timestamp' = timestamp + 1` was dropped from a successor"
        );
    }

    let mut xs: Vec<i64> = successors
        .iter()
        .map(|s| s.get("x").and_then(|v| v.as_int().ok()).unwrap_or(-1))
        .collect();
    xs.sort();
    let mut ys: Vec<i64> = successors
        .iter()
        .map(|s| s.get("y").and_then(|v| v.as_int().ok()).unwrap_or(-1))
        .collect();
    ys.sort();
    assert_eq!(
        xs,
        vec![0, 1],
        "expected x' values [0, 1] (one branch advances x), got {xs:?}"
    );
    assert_eq!(
        ys,
        vec![0, 1],
        "expected y' values [0, 1] (one branch advances y), got {ys:?}"
    );
}

#[test]
fn shared_post_condition_persists_through_inner_disjunction_at_initial_state() {
    // Per-state regression for the most common manifestation of the T1.5 bug:
    // the shared post-condition `timestamp' = timestamp + 1` was being
    // dropped from one of the inner-disjunction branches, leaving timestamp
    // unset (or stuttered) in some successors.
    let module = parse_tla_module_text(
        r#"---- MODULE T1_5_Shared ----
EXTENDS Naturals

VARIABLES x, y, timestamp

Init ==
    /\ x = 0
    /\ y = 0
    /\ timestamp = 0

Next ==
    /\ x + y < 15
    /\ \/ /\ x < 10
          /\ x' = x + 1
          /\ y' = y
       \/ /\ y < 10
          /\ y' = y + 1
          /\ x' = x
    /\ timestamp' = timestamp + 1
====
"#,
    )
    .expect("module parses");

    let next_def = module.definitions.get("Next").expect("Next defined");

    // Walk a few steps from (0, 0, 0) advancing along the x-axis branch and
    // confirm timestamp increments by one at each step in every successor.
    let mut frontier = vec![tla_state([
        ("x", TlaValue::Int(0)),
        ("y", TlaValue::Int(0)),
        ("timestamp", TlaValue::Int(0)),
    ])];

    for step in 1..=5 {
        let mut next_frontier = Vec::new();
        for state in &frontier {
            let successors = evaluate_next_states(&next_def.body, &module.definitions, state)
                .expect("step evaluates");
            for s in successors {
                let ts = s
                    .get("timestamp")
                    .and_then(|v| v.as_int().ok())
                    .expect("timestamp set in every successor");
                assert_eq!(
                    ts as i64, step,
                    "shared post-condition `timestamp' = timestamp + 1` was dropped \
                     at step {step}; got timestamp={ts} in {s:?}"
                );
                next_frontier.push(s);
            }
        }
        assert!(
            !next_frontier.is_empty(),
            "expected at least one successor at step {step}"
        );
        frontier = next_frontier;
    }
}
