use proptest::prelude::*;
use std::collections::HashMap;
use tlaplusplus::storage::numa::{NumaDiagnostics, NumaTopology};

fn dual_socket_topology() -> NumaTopology {
    NumaTopology {
        node_count: 2,
        cpu_to_node: HashMap::new(),
        distances: vec![vec![10, 21], vec![21, 10]],
        cpus_per_node: vec![vec![0, 1], vec![2, 3]],
    }
}

#[test]
fn dedups_and_sorts_sampled_store_nodes() {
    let diag = NumaDiagnostics::from_memory_nodes(&dual_socket_topology(), &[0, 1], vec![1, 0, 1]);
    assert_eq!(diag.fingerprint_store_nodes, vec![0, 1]);
}

#[test]
fn ignores_out_of_range_store_nodes() {
    let diag = NumaDiagnostics::from_memory_nodes(&dual_socket_topology(), &[0, 1], vec![9, 1]);
    assert_eq!(diag.fingerprint_store_nodes, vec![1]);
}

#[test]
fn nearest_distance_uses_all_sampled_shards() {
    let diag = NumaDiagnostics::from_memory_nodes(&dual_socket_topology(), &[0, 1], vec![0, 1]);
    assert_eq!(diag.nearest_distance_to_store(0), Some(10));
    assert_eq!(diag.nearest_distance_to_store(1), Some(10));
}

#[test]
fn remote_distance_is_reported_when_only_one_node_has_shards() {
    let diag = NumaDiagnostics::from_memory_nodes(&dual_socket_topology(), &[0, 1], vec![0]);
    assert_eq!(diag.nearest_distance_to_store(0), Some(10));
    assert_eq!(diag.nearest_distance_to_store(1), Some(21));
}

proptest! {
    #[test]
    fn worker_assignments_are_grouped_by_node(assignments in prop::collection::vec(0usize..2, 0..64)) {
        let diag = NumaDiagnostics::from_memory_nodes(&dual_socket_topology(), &assignments, vec![0, 1]);
        let total_workers: usize = diag.workers_by_node.iter().map(Vec::len).sum();
        prop_assert_eq!(total_workers, assignments.len());

        for (worker_id, &node) in assignments.iter().enumerate() {
            prop_assert!(diag.workers_by_node[node].contains(&worker_id));
        }
    }
}
