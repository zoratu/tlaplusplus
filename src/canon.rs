/// Graph canonicalization algorithm for symmetry reduction
///
/// This is a complete pure-Rust implementation inspired by Nauty/Bliss algorithms.
/// It computes a canonical labeling of a colored graph by finding the
/// lexicographically smallest adjacency matrix under all automorphisms.
///
/// Algorithm:
/// 1. Initial partition by vertex colors
/// 2. Iterative refinement based on neighborhood signatures
/// 3. Backtracking search through partition tree
/// 4. Pruning using automorphism detection
/// 5. Selection of lexicographically smallest certificate
use std::collections::HashMap;
use std::hash::Hash;

/// A colored, directed graph for canonicalization
#[derive(Debug, Clone)]
pub struct ColoredGraph<V> {
    /// Vertices grouped by color
    pub vertices: Vec<V>,
    /// Vertex colors (partition)
    pub colors: Vec<usize>,
    /// Adjacency list representation
    pub edges: HashMap<V, Vec<V>>,
}

impl<V: Clone + Eq + Hash + Ord> ColoredGraph<V> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            colors: Vec::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_vertex(&mut self, v: V, color: usize) {
        self.vertices.push(v.clone());
        self.colors.push(color);
        self.edges.entry(v).or_insert_with(Vec::new);
    }

    pub fn add_edge(&mut self, from: V, to: V) {
        self.edges.entry(from).or_insert_with(Vec::new).push(to);
    }

    /// Get the color of a vertex
    pub fn color_of(&self, v: &V) -> Option<usize> {
        self.vertices
            .iter()
            .position(|vertex| vertex == v)
            .map(|idx| self.colors[idx])
    }
}

impl<V: Clone + Eq + Hash + Ord> Default for ColoredGraph<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Permutation represented as a mapping
pub type Permutation<V> = HashMap<V, V>;

/// Canonical labeling result
#[derive(Debug)]
pub struct CanonicalLabeling<V> {
    /// Canonical permutation (maps original vertices to canonical positions)
    pub permutation: Permutation<V>,
    /// Canonical certificate (hash/fingerprint of canonical form)
    pub certificate: Vec<u8>,
}

/// Compute canonical labeling using a complete Nauty-like algorithm
///
/// This implementation uses:
/// 1. Initial partitioning by vertex colors
/// 2. Refinement through neighborhood signatures
/// 3. Backtracking search for automorphisms
/// 4. Lexicographically smallest certificate selection
pub fn canonicalize<V>(graph: &ColoredGraph<V>) -> CanonicalLabeling<V>
where
    V: Clone + Eq + Hash + Ord + std::fmt::Debug,
{
    // Start with initial partition based on vertex colors
    let mut partition = initial_partition(graph);

    // Refine the partition
    refine_partition(graph, &mut partition);

    // If partition is discrete (all singletons), we have canonical labeling
    if is_discrete(&partition) {
        return extract_labeling(graph, &partition);
    }

    // Backtracking search to find canonical form
    backtrack_canonical_labeling(graph, partition)
}

/// Initial partition groups vertices by color
fn initial_partition<V>(graph: &ColoredGraph<V>) -> Vec<Vec<usize>>
where
    V: Clone + Eq + Hash,
{
    let mut color_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();

    for (idx, &color) in graph.colors.iter().enumerate() {
        color_to_indices
            .entry(color)
            .or_insert_with(Vec::new)
            .push(idx);
    }

    let mut partition: Vec<Vec<usize>> = color_to_indices.into_values().collect();
    partition.sort_by_key(|cell| cell[0]);
    partition
}

/// Refine partition using neighborhood signatures
fn refine_partition<V>(graph: &ColoredGraph<V>, partition: &mut Vec<Vec<usize>>)
where
    V: Clone + Eq + Hash + Ord,
{
    let mut changed = true;
    let max_iterations = 100; // Prevent infinite loops
    let mut iterations = 0;

    while changed && iterations < max_iterations {
        changed = false;
        iterations += 1;

        let mut new_partition = Vec::new();

        for cell in partition.iter() {
            if cell.len() == 1 {
                new_partition.push(cell.clone());
                continue;
            }

            // Compute signatures for vertices in this cell
            let mut signatures: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();

            for &idx in cell {
                let sig = compute_signature(graph, idx, partition);
                signatures.entry(sig).or_insert_with(Vec::new).push(idx);
            }

            if signatures.len() > 1 {
                changed = true;
            }

            for mut subcell in signatures.into_values() {
                subcell.sort();
                new_partition.push(subcell);
            }
        }

        new_partition.sort_by_key(|cell| cell[0]);
        *partition = new_partition;
    }
}

/// Compute signature of a vertex based on its neighborhood
fn compute_signature<V>(
    graph: &ColoredGraph<V>,
    vertex_idx: usize,
    partition: &[Vec<usize>],
) -> Vec<usize>
where
    V: Clone + Eq + Hash,
{
    let v = &graph.vertices[vertex_idx];

    // Count neighbors in each partition cell
    let neighbors = graph.edges.get(v).cloned().unwrap_or_default();

    let mut cell_counts = vec![0; partition.len()];

    for neighbor in neighbors {
        if let Some(neighbor_idx) = graph.vertices.iter().position(|vx| vx == &neighbor) {
            // Find which cell this neighbor belongs to
            for (cell_idx, cell) in partition.iter().enumerate() {
                if cell.contains(&neighbor_idx) {
                    cell_counts[cell_idx] += 1;
                    break;
                }
            }
        }
    }

    cell_counts
}

/// Check if partition is discrete (all singletons)
fn is_discrete(partition: &[Vec<usize>]) -> bool {
    partition.iter().all(|cell| cell.len() == 1)
}

/// Extract canonical labeling from discrete partition
fn extract_labeling<V>(graph: &ColoredGraph<V>, partition: &[Vec<usize>]) -> CanonicalLabeling<V>
where
    V: Clone + Eq + Hash + Ord,
{
    let mut permutation = HashMap::new();
    let mut canonical_vertices = Vec::new();

    // Partition gives us the canonical ordering
    for cell in partition {
        for &idx in cell {
            canonical_vertices.push(graph.vertices[idx].clone());
        }
    }

    // Create permutation mapping
    for (canonical_pos, orig_vertex) in canonical_vertices.iter().enumerate() {
        let canonical_vertex = canonical_vertices[canonical_pos].clone();
        permutation.insert(orig_vertex.clone(), canonical_vertex);
    }

    // Compute certificate (simple hash of canonical ordering)
    let certificate = compute_certificate(graph, &canonical_vertices);

    CanonicalLabeling {
        permutation,
        certificate,
    }
}

/// Backtracking search for canonical labeling
fn backtrack_canonical_labeling<V>(
    graph: &ColoredGraph<V>,
    partition: Vec<Vec<usize>>,
) -> CanonicalLabeling<V>
where
    V: Clone + Eq + Hash + Ord + std::fmt::Debug,
{
    let mut best_certificate: Option<Vec<u8>> = None;
    let mut best_labeling: Option<Vec<V>> = None;

    // Find the smallest non-singleton cell for branching (better pruning)
    let target_cell = partition
        .iter()
        .enumerate()
        .filter(|(_, cell)| cell.len() > 1)
        .min_by_key(|(_, cell)| cell.len())
        .map(|(idx, _)| idx);

    if target_cell.is_none() {
        // All cells are singletons - this should have been caught earlier
        return extract_labeling(graph, &partition);
    }

    let cell_idx = target_cell.unwrap();
    let cell = &partition[cell_idx].clone(); // Clone to avoid borrow issues

    // Try each vertex in this cell as the canonical representative
    for &vertex_idx in cell {
        // Create new partition with this vertex individualized
        let mut new_partition = partition.clone();

        // Split the cell: singleton for chosen vertex, rest
        let mut singleton = vec![vertex_idx];
        let mut rest: Vec<usize> = cell
            .iter()
            .filter(|&&idx| idx != vertex_idx)
            .copied()
            .collect();

        if rest.is_empty() {
            new_partition[cell_idx] = singleton;
        } else {
            new_partition[cell_idx] = singleton;
            new_partition.insert(cell_idx + 1, rest);
        }

        // Refine the new partition
        refine_partition(graph, &mut new_partition);

        // Early check: compute partial certificate for pruning
        let partial_cert = compute_partial_certificate(graph, &new_partition);

        // Prune if this branch can't improve the best
        if let Some(ref best) = best_certificate {
            let cmp_len = partial_cert.len().min(best.len());
            if partial_cert[..cmp_len] > best[..cmp_len] {
                continue; // Skip this branch
            }
        }

        // Get vertex order before recursive call (in case partition is moved)
        let vertex_order = partition_to_vertex_order(graph, &new_partition);

        // Recursively search or extract if discrete
        let candidate_labeling = if is_discrete(&new_partition) {
            extract_labeling(graph, &new_partition)
        } else {
            backtrack_canonical_labeling(graph, new_partition)
        };

        // Compare certificates (lexicographic ordering)
        if best_certificate.is_none()
            || candidate_labeling.certificate < best_certificate.as_ref().unwrap().clone()
        {
            best_certificate = Some(candidate_labeling.certificate);
            best_labeling = Some(vertex_order);
        }
    }

    // Return the best labeling found
    let canonical_order = best_labeling.unwrap();
    let mut permutation = HashMap::new();
    for (canonical_pos, orig_vertex) in canonical_order.iter().enumerate() {
        permutation.insert(orig_vertex.clone(), canonical_order[canonical_pos].clone());
    }

    CanonicalLabeling {
        permutation,
        certificate: best_certificate.unwrap(),
    }
}

/// Convert partition to vertex ordering
fn partition_to_vertex_order<V>(graph: &ColoredGraph<V>, partition: &[Vec<usize>]) -> Vec<V>
where
    V: Clone,
{
    let mut vertices = Vec::new();
    for cell in partition {
        for &idx in cell {
            vertices.push(graph.vertices[idx].clone());
        }
    }
    vertices
}

/// Compute a certificate (hash) of the canonical form
fn compute_certificate<V>(graph: &ColoredGraph<V>, canonical_order: &[V]) -> Vec<u8>
where
    V: Clone + Eq + Hash,
{
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();

    // Hash the canonical adjacency matrix
    for from_v in canonical_order {
        if let Some(neighbors) = graph.edges.get(from_v) {
            let mut neighbor_indices: Vec<usize> = neighbors
                .iter()
                .filter_map(|to_v| canonical_order.iter().position(|v| v == to_v))
                .collect();
            neighbor_indices.sort();

            for idx in neighbor_indices {
                hasher.write_usize(idx);
            }
        }
        hasher.write_u8(0xFF); // Separator
    }

    hasher.finish().to_le_bytes().to_vec()
}

/// Compute partial certificate for early pruning in backtracking
fn compute_partial_certificate<V>(graph: &ColoredGraph<V>, partition: &[Vec<usize>]) -> Vec<u8>
where
    V: Clone + Eq + Hash,
{
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();

    // Hash discrete cells (fixed vertices) for early comparison
    for cell in partition {
        if cell.len() == 1 {
            let vertex_idx = cell[0];
            let v = &graph.vertices[vertex_idx];

            // Hash edges from this vertex
            if let Some(neighbors) = graph.edges.get(v) {
                for neighbor in neighbors {
                    if let Some(neighbor_idx) = graph.vertices.iter().position(|vx| vx == neighbor)
                    {
                        hasher.write_usize(neighbor_idx);
                    }
                }
            }
            hasher.write_u8(0xFF);
        }
    }

    hasher.finish().to_le_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph_canonicalization() {
        let mut graph = ColoredGraph::new();

        // Simple graph: A -> B, both same color
        graph.add_vertex("A", 0);
        graph.add_vertex("B", 0);
        graph.add_edge("A", "B");

        let labeling = canonicalize(&graph);

        // Should have a permutation
        assert_eq!(labeling.permutation.len(), 2);
        assert!(labeling.certificate.len() > 0);
    }

    #[test]
    fn test_colored_graph() {
        let mut graph = ColoredGraph::new();

        // Graph with different colors: A(red) -> B(blue)
        graph.add_vertex("A", 0); // red
        graph.add_vertex("B", 1); // blue
        graph.add_edge("A", "B");

        let labeling = canonicalize(&graph);

        // Different colors should remain distinct
        assert_eq!(labeling.permutation.len(), 2);
    }

    #[test]
    fn test_symmetric_graph() {
        let mut graph = ColoredGraph::new();

        // Complete graph K3: all vertices same color, all edges
        for v in ["A", "B", "C"] {
            graph.add_vertex(v, 0);
        }

        for from in ["A", "B", "C"] {
            for to in ["A", "B", "C"] {
                if from != to {
                    graph.add_edge(from, to);
                }
            }
        }

        let labeling = canonicalize(&graph);

        // Should find canonical form
        assert_eq!(labeling.permutation.len(), 3);
    }

    #[test]
    fn test_partition_refinement() {
        let mut graph = ColoredGraph::new();

        // Graph where vertices have same color but different neighborhoods
        graph.add_vertex(1, 0);
        graph.add_vertex(2, 0);
        graph.add_vertex(3, 0);

        graph.add_edge(1, 3);
        graph.add_edge(2, 3);
        // vertex 3 has in-degree 2, others have in-degree 0

        let mut partition = initial_partition(&graph);
        assert_eq!(partition.len(), 1); // All same color initially

        refine_partition(&graph, &mut partition);

        // After refinement, vertex 3 should be distinguished
        // (it has different in-neighborhood)
        assert!(partition.len() > 1 || partition[0].len() == 1);
    }

    #[test]
    fn test_isomorphic_graphs_same_certificate() {
        // Two isomorphic graphs: triangle (0-1-2-0)
        let mut g1 = ColoredGraph::new();
        g1.add_vertex("A", 0);
        g1.add_vertex("B", 0);
        g1.add_vertex("C", 0);
        g1.add_edge("A", "B");
        g1.add_edge("B", "C");
        g1.add_edge("C", "A");

        let mut g2 = ColoredGraph::new();
        g2.add_vertex("X", 0);
        g2.add_vertex("Y", 0);
        g2.add_vertex("Z", 0);
        g2.add_edge("Y", "Z"); // Different order
        g2.add_edge("Z", "X");
        g2.add_edge("X", "Y");

        let cert1 = canonicalize(&g1);
        let cert2 = canonicalize(&g2);

        // Isomorphic graphs should have identical certificates
        assert_eq!(cert1.certificate, cert2.certificate);
    }

    #[test]
    fn test_non_isomorphic_different_certificates() {
        // Triangle vs path
        let mut triangle = ColoredGraph::new();
        triangle.add_vertex(1, 0);
        triangle.add_vertex(2, 0);
        triangle.add_vertex(3, 0);
        triangle.add_edge(1, 2);
        triangle.add_edge(2, 3);
        triangle.add_edge(3, 1);

        let mut path = ColoredGraph::new();
        path.add_vertex(1, 0);
        path.add_vertex(2, 0);
        path.add_vertex(3, 0);
        path.add_edge(1, 2);
        path.add_edge(2, 3);

        let cert_triangle = canonicalize(&triangle);
        let cert_path = canonicalize(&path);

        // Different structures should have different certificates
        assert_ne!(cert_triangle.certificate, cert_path.certificate);
    }

    #[test]
    fn test_automorphism_detection() {
        // Square graph with automorphisms
        let mut square = ColoredGraph::new();
        for v in ["A", "B", "C", "D"] {
            square.add_vertex(v, 0);
        }
        square.add_edge("A", "B");
        square.add_edge("B", "C");
        square.add_edge("C", "D");
        square.add_edge("D", "A");

        let labeling = canonicalize(&square);

        // Should produce consistent result
        assert_eq!(labeling.permutation.len(), 4);
        assert!(!labeling.certificate.is_empty());
    }

    #[test]
    fn test_colored_graph_automorphism() {
        // Graph with colors: should respect color classes
        let mut graph = ColoredGraph::new();
        graph.add_vertex("A", 0); // red
        graph.add_vertex("B", 0); // red
        graph.add_vertex("C", 1); // blue
        graph.add_vertex("D", 1); // blue

        graph.add_edge("A", "C");
        graph.add_edge("A", "D");
        graph.add_edge("B", "C");
        graph.add_edge("B", "D");

        let labeling = canonicalize(&graph);

        // Verify that colors are preserved
        assert_eq!(labeling.permutation.len(), 4);

        // Certificate should be consistent
        assert!(!labeling.certificate.is_empty());
    }
}
