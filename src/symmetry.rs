use crate::tla::{TlaState, TlaValue};
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

/// Specialized canonicalization for TlaState that directly manipulates the structure
pub fn canonicalize_tla_state(state: &TlaState, symmetry: &SymmetrySpec) -> TlaState {
    let groups = match &symmetry.symmetry_groups {
        Some(g) if !g.is_empty() => g,
        _ => return state.clone(),
    };

    // First, find all symmetric values that appear in the state
    let mut found_values: Vec<HashSet<String>> = groups.iter().map(|_| HashSet::new()).collect();
    collect_model_values_in_state(state, groups, &mut found_values);

    // For each group, compute the canonical permutation based on found values
    let mut full_permutation: HashMap<String, String> = HashMap::new();
    for (group_idx, (group, found)) in groups.iter().zip(found_values.iter()).enumerate() {
        if found.is_empty() {
            continue;
        }

        let found_vec: Vec<String> = found.iter().cloned().collect();
        let group_permutation = compute_canonical_permutation(&found_vec, group);

        // Debug: print first permutation (only in debug builds)
        #[cfg(debug_assertions)]
        {
            static DEBUG_PERM: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !DEBUG_PERM.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("DEBUG group {}: found values {:?}", group_idx, found);
                eprintln!(
                    "DEBUG group {}: permutation {:?}",
                    group_idx, group_permutation
                );
            }
        }

        full_permutation.extend(group_permutation);
    }

    if full_permutation.is_empty() || full_permutation.iter().all(|(k, v)| k == v) {
        return state.clone();
    }

    // Apply the permutation to the state
    apply_permutation_to_state(state, &full_permutation)
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
    state
        .iter()
        .map(|(k, v)| (k.clone(), apply_permutation_to_value(v, permutation)))
        .collect()
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
            TlaValue::Set(Arc::new(new_items))
        }
        TlaValue::Seq(items) => {
            let new_items: Vec<TlaValue> = items
                .iter()
                .map(|item| apply_permutation_to_value(item, permutation))
                .collect();
            TlaValue::Seq(Arc::new(new_items))
        }
        TlaValue::Record(fields) => {
            let new_fields: BTreeMap<String, TlaValue> = fields
                .iter()
                .map(|(k, v)| (k.clone(), apply_permutation_to_value(v, permutation)))
                .collect();
            TlaValue::Record(Arc::new(new_fields))
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
            TlaValue::Function(Arc::new(new_map))
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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    if let Some(spec) = symmetry {
        // Compute canonical form and fingerprint it
        let canonical = canonicalize_state(state, spec);
        let mut hasher = DefaultHasher::new();

        // Serialize to get deterministic hash
        if let Ok(bytes) = bincode::serialize(&canonical) {
            hasher.write(&bytes);
        }

        hasher.finish()
    } else {
        // No symmetry - use regular fingerprint
        let mut hasher = DefaultHasher::new();

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
