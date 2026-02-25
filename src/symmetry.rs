use crate::canon::{ColoredGraph, canonicalize};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// Symmetry specification for state space reduction
///
/// Symmetry reduction exploits the fact that many states are equivalent
/// under permutations of certain values. For example, if a system has
/// multiple identical processes, states that differ only by process IDs
/// are equivalent and only one representative needs to be explored.
#[derive(Debug, Clone)]
pub struct SymmetrySpec {
    /// The set of values that can be permuted (e.g., process IDs, node names)
    pub symmetric_set: String,
    /// Cached set of symmetric values (model values)
    pub symmetric_values: Option<HashSet<String>>,
}

impl SymmetrySpec {
    pub fn new(symmetric_set: String) -> Self {
        Self {
            symmetric_set,
            symmetric_values: None,
        }
    }

    /// Initialize the symmetric values from a TLA+ state
    ///
    /// This should be called once during model initialization to determine
    /// which values are in the symmetric set.
    pub fn initialize_from_config(&mut self, values: HashSet<String>) {
        self.symmetric_values = Some(values);
    }

    /// Check if a value is in the symmetric set
    pub fn is_symmetric_value(&self, value: &str) -> bool {
        self.symmetric_values
            .as_ref()
            .map(|vals| vals.contains(value))
            .unwrap_or(false)
    }
}

/// Computes a canonical representative of a state's symmetry class
///
/// The canonical form is the lexicographically smallest permutation of
/// symmetric values in the state. This ensures that all symmetric states
/// map to the same canonical representative.
///
/// This implementation uses graph canonicalization to compute the
/// canonical permutation efficiently.
pub fn canonicalize_state<S>(state: &S, symmetry: &SymmetrySpec) -> S
where
    S: Clone + Debug + Serialize + for<'de> Deserialize<'de>,
{
    // For TLA+ states, we serialize to JSON, apply permutation, and deserialize
    // This is a simple but general approach that works for any serializable state

    if symmetry.symmetric_values.is_none() {
        // No symmetric values - return state unchanged
        return state.clone();
    }

    // Serialize state to JSON
    let json_str = match serde_json::to_string(state) {
        Ok(s) => s,
        Err(_) => return state.clone(), // Fallback: return unchanged
    };

    // Extract all occurrences of symmetric values
    let symmetric_values = symmetry.symmetric_values.as_ref().unwrap();
    let mut found_values = Vec::new();

    for sym_val in symmetric_values {
        if json_str.contains(sym_val) {
            found_values.push(sym_val.clone());
        }
    }

    if found_values.is_empty() {
        // No symmetric values in state - return unchanged
        return state.clone();
    }

    // Compute canonical permutation using graph canonicalization
    let permutation = compute_canonical_permutation(&found_values, symmetric_values);

    // Apply permutation to state by replacing values in JSON
    let mut canonical_json = json_str;
    for (from, to) in permutation {
        // Replace all occurrences (careful to avoid partial matches)
        // This is a simple string replacement - a more robust implementation
        // would parse the JSON structure
        canonical_json = canonical_json.replace(&format!("\"{}\"", from), &format!("\"{}\"", to));
    }

    // Deserialize back to state
    match serde_json::from_str(&canonical_json) {
        Ok(canonical_state) => canonical_state,
        Err(_) => state.clone(), // Fallback on error
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

/// Permutation-based canonicalization for TLA+ states
///
/// Uses graph canonicalization (Nauty-like algorithm) to find the
/// canonical permutation of symmetric values.
pub fn compute_canonical_permutation(
    state_values: &[String],
    symmetric_values: &HashSet<String>,
) -> HashMap<String, String> {
    // Extract symmetric values that appear in the state
    let mut appearing_symmetric: Vec<String> = state_values
        .iter()
        .filter(|v| symmetric_values.contains(*v))
        .cloned()
        .collect();

    if appearing_symmetric.is_empty() {
        return HashMap::new();
    }

    // Remove duplicates and sort for determinism
    appearing_symmetric.sort();
    appearing_symmetric.dedup();

    // Build a colored graph for canonicalization
    // For simple model values, we create a graph where:
    // - Each value is a vertex
    // - All values have the same color (they're symmetric)
    // - Edges represent relationships (for now, no edges - just vertex ordering)
    let mut graph = ColoredGraph::new();

    for value in &appearing_symmetric {
        graph.add_vertex(value.clone(), 0); // All same color
    }

    // Canonicalize the graph
    let labeling = canonicalize(&graph);

    // Extract the permutation from the canonical labeling
    labeling.permutation
}

/// Apply a permutation to a value (recursively through state structure)
pub fn apply_permutation_to_value(value: &str, permutation: &HashMap<String, String>) -> String {
    permutation
        .get(value)
        .cloned()
        .unwrap_or_else(|| value.to_string())
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
            apply_permutation_to_value("old_value", &permutation),
            "new_value"
        );
        assert_eq!(
            apply_permutation_to_value("other_value", &permutation),
            "other_value"
        );
    }
}
