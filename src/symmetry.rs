use crate::tla::{TlaState, TlaValue};
use crate::tla::hashed_arc::HashedArc;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

/// Symmetry specification for state space reduction
///
/// Symmetry reduction exploits the fact that many states are equivalent
/// under permutations of certain values. For example, if a system has
/// multiple identical processes, states that differ only by process IDs
/// are equivalent and only one representative needs to be explored.
///
/// IMPORTANT: Values can only be permuted within their own symmetry group.
/// For example, if Bots = {b1, b2} and Nodes = {n1, n2}, then b1 can be
/// swapped with b2, and n1 can be swapped with n2, but b1 CANNOT be
/// swapped with n1 - they are different types of model values.
#[derive(Debug, Clone)]
pub struct SymmetrySpec {
    /// The name of the symmetry set (e.g., "Symmetry")
    pub symmetric_set: String,
    /// Groups of symmetric values - values within each group can be permuted
    /// but values from different groups cannot be interchanged
    pub symmetry_groups: Option<Vec<HashSet<String>>>,
    /// Flat set of all symmetric values (for quick membership check)
    pub symmetric_values: Option<HashSet<String>>,
}

impl SymmetrySpec {
    pub fn new(symmetric_set: String) -> Self {
        Self {
            symmetric_set,
            symmetry_groups: None,
            symmetric_values: None,
        }
    }

    /// Initialize with separate symmetry groups
    /// Each group contains values that can be permuted among themselves
    pub fn initialize_with_groups(&mut self, groups: Vec<HashSet<String>>) {
        let all_values: HashSet<String> = groups.iter().flat_map(|g| g.iter().cloned()).collect();
        self.symmetric_values = Some(all_values);
        self.symmetry_groups = Some(groups);
    }

    /// Initialize the symmetric values from a TLA+ state (legacy single-group)
    ///
    /// This should be called once during model initialization to determine
    /// which values are in the symmetric set.
    pub fn initialize_from_config(&mut self, values: HashSet<String>) {
        self.symmetric_values = Some(values.clone());
        // Single group for backwards compatibility
        self.symmetry_groups = Some(vec![values]);
    }

    /// Check if a value is in the symmetric set
    pub fn is_symmetric_value(&self, value: &str) -> bool {
        self.symmetric_values
            .as_ref()
            .map(|vals| vals.contains(value))
            .unwrap_or(false)
    }

    /// Get the symmetry group that contains a given value
    pub fn get_group_for_value(&self, value: &str) -> Option<&HashSet<String>> {
        self.symmetry_groups
            .as_ref()
            .and_then(|groups| groups.iter().find(|group| group.contains(value)))
    }
}

/// Computes a canonical representative of a state's symmetry class
///
/// The canonical form is the lexicographically smallest permutation of
/// symmetric values in the state. This ensures that all symmetric states
/// map to the same canonical representative.
///
/// IMPORTANT: Permutations are applied independently to each symmetry group.
/// Values from different groups are never interchanged.
pub fn canonicalize_state<S>(state: &S, _symmetry: &SymmetrySpec) -> S
where
    S: Clone + Debug + Serialize + for<'de> Deserialize<'de>,
{
    // For generic types, we can't directly manipulate the structure
    // Fall back to identity for non-TlaState types
    state.clone()
}

/// Specialized canonicalization for TlaState that directly manipulates the structure.
///
/// Enumerates all permutations of each symmetric group (subject to
/// `MAX_GROUP_SIZE_FOR_FULL_ENUM`), applies every combination to the
/// state, and returns the lexicographically smallest candidate. Two
/// states in the same symmetry orbit therefore collapse to the same
/// canonical form, which is the contract the fingerprint store relies on.
///
/// The previous implementation just sorted the values that *appeared*
/// in the state and mapped them to the alphabetically-first labels.
/// That degenerates to identity whenever all symmetric values appear
/// (the common case for late-exploration states), producing no
/// reduction — which is why MCCheckpointCoord (3-element Node group)
/// previously explored ~6× more states than TLC.
pub fn canonicalize_tla_state(state: &TlaState, symmetry: &SymmetrySpec) -> TlaState {
    let groups = match &symmetry.symmetry_groups {
        Some(g) if !g.is_empty() => g,
        _ => return state.clone(),
    };

    // Safety valve: enumerating N! permutations gets expensive fast.
    // 5! = 120 is tolerable; 6! = 720 starts to bite; 7! = 5040 is too
    // much per-state work. Groups bigger than the cap fall back to
    // identity (no reduction), preserving correctness but losing the
    // win for those groups specifically. Any real TLA+ spec we've seen
    // uses 2–4 element symmetric groups.
    const MAX_GROUP_SIZE_FOR_FULL_ENUM: usize = 5;

    // For each group, pre-compute the sorted canonical order and the
    // set of value-permutations we'll try. A permutation here is a Vec
    // matching `sorted_canonical` positionally — element i of `perm`
    // is the value that should map *to* `sorted_canonical[i]`.
    let group_data: Vec<(Vec<String>, Vec<Vec<String>>)> = groups
        .iter()
        .map(|g| {
            let mut sorted: Vec<String> = g.iter().cloned().collect();
            sorted.sort();
            let perms = if sorted.len() <= MAX_GROUP_SIZE_FOR_FULL_ENUM {
                all_permutations(&sorted)
            } else {
                vec![sorted.clone()]
            };
            (sorted, perms)
        })
        .collect();

    // Compute the lex-min over the SEQ-NORMALIZED state, but return the winning
    // permutation applied to the ORIGINAL (un-normalized) state.
    //
    // Two representations of the same logical state (e.g. a log built as
    // `[i \in 1..n |-> e]` (Function) vs as a `Seq` via `Append`) must pick the
    // SAME canonical node permutation — otherwise the later fingerprint-time
    // Seq normalization (which makes them structurally equal) still can't dedup
    // them, because they'd carry different node permutations from this step.
    // Comparing on the normalized form makes the chosen permutation
    // representation-independent. We then apply that permutation to the original
    // state so invariant evaluation continues to see the original representation
    // (normalizing the *stored* state can expose ops that treat Seq vs Function
    // inconsistently and produce false violations).
    let norm_owned = crate::tla::value::normalize_state_if_changed(state);
    let state_norm: &TlaState = norm_owned.as_ref().unwrap_or(state);

    // Iterate over the cross-product of per-group permutations as a mixed-radix
    // counter, tracking the permutation whose normalized image is lex-min.
    let mut best_norm: Option<TlaState> = None;
    let mut best_perm: Option<HashMap<String, String>> = None;
    let mut indices = vec![0usize; group_data.len()];

    'outer: loop {
        let mut perm_map: HashMap<String, String> = HashMap::new();
        for (group_idx, perm_idx) in indices.iter().enumerate() {
            let (sorted_canonical, perms) = &group_data[group_idx];
            let permuted = &perms[*perm_idx];
            // `permuted[i]` is the source value that should become
            // `sorted_canonical[i]`. The HashMap is source -> canonical.
            for (i, canonical_label) in sorted_canonical.iter().enumerate() {
                perm_map.insert(permuted[i].clone(), canonical_label.clone());
            }
        }

        let candidate_norm = if perm_map.iter().all(|(k, v)| k == v) {
            state_norm.clone()
        } else {
            apply_permutation_to_state(state_norm, &perm_map)
        };

        if best_norm.as_ref().map_or(true, |b| &candidate_norm < b) {
            best_norm = Some(candidate_norm);
            best_perm = Some(perm_map);
        }

        // Increment indices (mixed-radix); stop when we overflow.
        let mut i = 0;
        loop {
            if i >= indices.len() {
                break 'outer;
            }
            indices[i] += 1;
            if indices[i] < group_data[i].1.len() {
                break;
            }
            indices[i] = 0;
            i += 1;
        }
    }

    match best_perm {
        Some(perm_map) if !perm_map.iter().all(|(k, v)| k == v) => {
            apply_permutation_to_state(state, &perm_map)
        }
        _ => state.clone(),
    }
}

/// All permutations of a slice. O(n!) — caller bounds n.
fn all_permutations(items: &[String]) -> Vec<Vec<String>> {
    let mut current = items.to_vec();
    let mut result = Vec::new();
    permute_helper(&mut current, 0, &mut result);
    result
}

fn permute_helper(arr: &mut Vec<String>, start: usize, result: &mut Vec<Vec<String>>) {
    if start >= arr.len() {
        result.push(arr.clone());
        return;
    }
    for i in start..arr.len() {
        arr.swap(start, i);
        permute_helper(arr, start + 1, result);
        arr.swap(start, i);
    }
}

/// Recursively collect all ModelValue strings that appear in a state
fn collect_model_values_in_state(
    state: &TlaState,
    groups: &[HashSet<String>],
    found: &mut [HashSet<String>],
) {
    for value in state.values() {
        collect_model_values_in_value(value, groups, found);
    }
}

fn collect_model_values_in_value(
    value: &TlaValue,
    groups: &[HashSet<String>],
    found: &mut [HashSet<String>],
) {
    match value {
        TlaValue::ModelValue(s) => {
            // Check which group this value belongs to
            for (group_idx, group) in groups.iter().enumerate() {
                if group.contains(s) {
                    found[group_idx].insert(s.clone());
                    break;
                }
            }
        }
        TlaValue::Set(items) => {
            for item in items.iter() {
                collect_model_values_in_value(item, groups, found);
            }
        }
        TlaValue::Seq(items) => {
            for item in items.iter() {
                collect_model_values_in_value(item, groups, found);
            }
        }
        TlaValue::Record(fields) => {
            for val in fields.values() {
                collect_model_values_in_value(val, groups, found);
            }
        }
        TlaValue::Function(map) => {
            for (k, v) in map.iter() {
                collect_model_values_in_value(k, groups, found);
                collect_model_values_in_value(v, groups, found);
            }
        }
        _ => {}
    }
}

/// Apply a permutation to all ModelValue instances in a state
fn apply_permutation_to_state(state: &TlaState, permutation: &HashMap<String, String>) -> TlaState {
    // A permutation renames symmetric ModelValues *inside* the values; the
    // variable names (schema) are unchanged. Map values while preserving the
    // schema Arc so we don't re-derive a schema or double-clone keys/values on
    // every canonicalization (the symmetry hot path).
    state.map_values(|v| apply_permutation_to_value(v, permutation))
}

fn apply_permutation_to_value(value: &TlaValue, permutation: &HashMap<String, String>) -> TlaValue {
    match value {
        TlaValue::ModelValue(s) => {
            if let Some(replacement) = permutation.get(s) {
                TlaValue::ModelValue(replacement.clone())
            } else {
                value.clone()
            }
        }
        TlaValue::Set(items) => {
            let new_items: BTreeSet<TlaValue> = items
                .iter()
                .map(|item| apply_permutation_to_value(item, permutation))
                .collect();
            TlaValue::Set(HashedArc::new(new_items))
        }
        TlaValue::Seq(items) => {
            let new_items: Vec<TlaValue> = items
                .iter()
                .map(|item| apply_permutation_to_value(item, permutation))
                .collect();
            TlaValue::Seq(HashedArc::new(new_items))
        }
        TlaValue::Record(fields) => {
            let new_fields: BTreeMap<String, TlaValue> = fields
                .iter()
                .map(|(k, v)| (k.clone(), apply_permutation_to_value(v, permutation)))
                .collect();
            TlaValue::Record(HashedArc::new(new_fields))
        }
        TlaValue::Function(map) => {
            let new_map: BTreeMap<TlaValue, TlaValue> = map
                .iter()
                .map(|(k, v)| {
                    (
                        apply_permutation_to_value(k, permutation),
                        apply_permutation_to_value(v, permutation),
                    )
                })
                .collect();
            TlaValue::Function(HashedArc::new(new_map))
        }
        _ => value.clone(),
    }
}

/// Symmetry-aware fingerprint computation
///
/// Instead of fingerprinting the raw state, we fingerprint its canonical
/// representative. This ensures that symmetric states get the same fingerprint.
pub fn symmetric_fingerprint<S>(state: &S, symmetry: Option<&SymmetrySpec>) -> u64
where
    S: Clone + Debug + Hash + Serialize + for<'de> Deserialize<'de>,
{
    use crate::model::fingerprint_hasher;
    use std::hash::Hasher;

    if let Some(spec) = symmetry {
        // Compute canonical form and fingerprint it
        let canonical = canonicalize_state(state, spec);
        let mut hasher = fingerprint_hasher();

        // Serialize to get deterministic hash
        if let Ok(bytes) = bincode::serialize(&canonical) {
            hasher.write(&bytes);
        }

        hasher.finish()
    } else {
        // No symmetry - use regular fingerprint
        let mut hasher = fingerprint_hasher();

        if let Ok(bytes) = bincode::serialize(state) {
            hasher.write(&bytes);
        }

        hasher.finish()
    }
}

/// Compute canonical permutation for a symmetry group
///
/// Maps the found values to canonical representatives. For example, if the
/// group is {bot1, bot2, bot3} and the state contains {bot3, bot1}, then
/// we map bot1 → bot1, bot3 → bot2 (using the first N sorted canonical values).
///
/// This ensures that ALL states using any 2 bots will canonicalize to use
/// bot1 and bot2, regardless of which specific bots they originally used.
pub fn compute_canonical_permutation(
    found_values: &[String],
    group: &HashSet<String>,
) -> HashMap<String, String> {
    if found_values.is_empty() {
        return HashMap::new();
    }

    // Sort the found values
    let mut sorted_found: Vec<String> = found_values.to_vec();
    sorted_found.sort();
    sorted_found.dedup();

    // Sort the full group to get canonical labels
    let mut canonical_labels: Vec<String> = group.iter().cloned().collect();
    canonical_labels.sort();

    // Map each found value to its canonical counterpart
    // found[i] → canonical[i]
    let mut permutation = HashMap::new();
    for (i, found_val) in sorted_found.iter().enumerate() {
        if i < canonical_labels.len() {
            permutation.insert(found_val.clone(), canonical_labels[i].clone());
        }
    }

    permutation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetry_spec_creation() {
        let spec = SymmetrySpec::new("Processes".to_string());
        assert_eq!(spec.symmetric_set, "Processes");
        assert!(spec.symmetric_values.is_none());
    }

    #[test]
    fn test_symmetry_initialization() {
        let mut spec = SymmetrySpec::new("Processes".to_string());
        let values = vec!["p1", "p2", "p3"]
            .into_iter()
            .map(String::from)
            .collect();

        spec.initialize_from_config(values);

        assert!(spec.is_symmetric_value("p1"));
        assert!(spec.is_symmetric_value("p2"));
        assert!(!spec.is_symmetric_value("p4"));
    }

    #[test]
    fn test_canonical_permutation() {
        let symmetric_values: HashSet<String> =
            vec!["a", "b", "c"].into_iter().map(String::from).collect();

        let state_values = vec!["c".to_string(), "a".to_string(), "b".to_string()];

        let perm = compute_canonical_permutation(&state_values, &symmetric_values);

        // Should have mappings for all symmetric values
        assert!(perm.contains_key("a"));
        assert!(perm.contains_key("b"));
        assert!(perm.contains_key("c"));
    }

    /// Regression test: states in the same symmetry orbit must produce
    /// the same canonical form. Before the lex-min-over-permutations
    /// fix, the previous algorithm degenerated to identity whenever
    /// all symmetric values appeared, so {x→n2, y→n1} and {x→n1, y→n2}
    /// were kept as separate states.
    #[test]
    fn canonicalize_collapses_orbit_when_all_symmetric_values_appear() {
        use std::sync::Arc;
        let mut spec = SymmetrySpec::new("NodeSymmetric".to_string());
        spec.initialize_with_groups(vec![
            ["n1", "n2"].iter().map(|s| s.to_string()).collect(),
        ]);

        let state_a: TlaState = [
            (Arc::<str>::from("x"), TlaValue::ModelValue("n1".to_string())),
            (Arc::<str>::from("y"), TlaValue::ModelValue("n2".to_string())),
        ]
        .into_iter()
        .collect();
        let state_b: TlaState = [
            (Arc::<str>::from("x"), TlaValue::ModelValue("n2".to_string())),
            (Arc::<str>::from("y"), TlaValue::ModelValue("n1".to_string())),
        ]
        .into_iter()
        .collect();

        let canon_a = canonicalize_tla_state(&state_a, &spec);
        let canon_b = canonicalize_tla_state(&state_b, &spec);
        assert_eq!(
            canon_a, canon_b,
            "states in the same orbit must have identical canonical forms"
        );
        // The lex-min representative is the one with x=n1, y=n2
        // (because x sorts before y; n1 < n2).
        assert_eq!(canon_a, state_a);
    }

    /// 3-element symmetry group (matches MCCheckpointCoord's Node = {n1,n2,n3}).
    /// All 6 permutations should collapse states in the same orbit.
    #[test]
    fn canonicalize_collapses_three_element_orbit() {
        use std::sync::Arc;
        let mut spec = SymmetrySpec::new("NodeSym".to_string());
        spec.initialize_with_groups(vec![
            ["n1", "n2", "n3"].iter().map(|s| s.to_string()).collect(),
        ]);

        let make_state = |a: &str, b: &str, c: &str| -> TlaState {
            [
                (Arc::<str>::from("a"), TlaValue::ModelValue(a.to_string())),
                (Arc::<str>::from("b"), TlaValue::ModelValue(b.to_string())),
                (Arc::<str>::from("c"), TlaValue::ModelValue(c.to_string())),
            ]
            .into_iter()
            .collect()
        };

        // All 6 permutations of (n1, n2, n3) across the 3 distinct variables.
        let orbit = [
            make_state("n1", "n2", "n3"),
            make_state("n1", "n3", "n2"),
            make_state("n2", "n1", "n3"),
            make_state("n2", "n3", "n1"),
            make_state("n3", "n1", "n2"),
            make_state("n3", "n2", "n1"),
        ];

        let canons: Vec<_> = orbit.iter().map(|s| canonicalize_tla_state(s, &spec)).collect();
        for c in &canons[1..] {
            assert_eq!(&canons[0], c, "all 6 orbit members must canonicalize identically");
        }
    }

    /// The crux of the full-scale dedup regression: two representations of the
    /// SAME logical state — a log built as a `Function` over `1..n` vs as a
    /// `Seq` — must produce the same dedup key under symmetry. The runtime key
    /// is `fingerprint(canonicalize(state))` and `fingerprint` normalizes
    /// `1..n` functions to `Seq`, so they dedup iff
    /// `normalize(canonicalize(S)) == normalize(canonicalize(T))`. The logs
    /// contain node values so symmetry actually permutes their contents (the
    /// case that only shows up at MaxLog>=2 with node-bearing logs).
    #[test]
    fn seq_and_function_log_states_share_dedup_key_under_symmetry() {
        use crate::tla::value::normalize_state_if_changed;
        use std::sync::Arc;

        let mut spec = SymmetrySpec::new("NodeSym".to_string());
        spec.initialize_with_groups(vec![
            ["n1", "n2", "n3"].iter().map(|s| s.to_string()).collect(),
        ]);
        let nv = |s: &str| TlaValue::ModelValue(s.to_string());

        // n1's log is the sequence <<n2, n3>>, expressed two ways.
        let log_func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), nv("n2")),
            (TlaValue::Int(2), nv("n3")),
        ])));
        let log_seq = TlaValue::Seq(HashedArc::new(vec![nv("n2"), nv("n3")]));

        // ReplicatedLog = [node |-> log]; only n1's log representation differs.
        let make = |n1_log: TlaValue| -> TlaState {
            let rlog = TlaValue::Function(HashedArc::new(BTreeMap::from([
                (nv("n1"), n1_log),
                (nv("n2"), TlaValue::Seq(HashedArc::new(vec![]))),
                (nv("n3"), TlaValue::Seq(HashedArc::new(vec![]))),
            ])));
            [
                (Arc::<str>::from("Leader"), nv("n1")),
                (Arc::<str>::from("ReplicatedLog"), rlog),
            ]
            .into_iter()
            .collect()
        };

        let dedup_key = |st: &TlaState| -> TlaState {
            let canon = canonicalize_tla_state(st, &spec);
            normalize_state_if_changed(&canon).unwrap_or(canon)
        };

        assert_eq!(
            dedup_key(&make(log_func)),
            dedup_key(&make(log_seq)),
            "Function-log and Seq-log representations of the same logical state \
             must produce the same dedup key under symmetry"
        );
    }

    #[test]
    fn test_apply_permutation() {
        let mut permutation = HashMap::new();
        permutation.insert("old_value".to_string(), "new_value".to_string());

        assert_eq!(
            apply_permutation_to_value(
                &TlaValue::ModelValue("old_value".to_string()),
                &permutation
            ),
            TlaValue::ModelValue("new_value".to_string())
        );
        assert_eq!(
            apply_permutation_to_value(
                &TlaValue::ModelValue("other_value".to_string()),
                &permutation
            ),
            TlaValue::ModelValue("other_value".to_string())
        );
    }
}
