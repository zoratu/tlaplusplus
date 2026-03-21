// Deterministic hash for consistent cross-node partition assignment

/// Consistent-hashing ring for fingerprint partition assignment.
///
/// Each physical node is mapped to a number of virtual nodes proportional
/// to its core count, spreading ownership across the hash ring for balanced
/// load even when nodes have heterogeneous capacity.
///
/// `home_node(fp)` determines which node owns a fingerprint, and therefore
/// which node is responsible for check-and-insert on that fingerprint.
pub struct PartitionRing {
    /// Sorted list of (hash_point, node_id) for binary-search lookup.
    virtual_nodes: Vec<(u64, u32)>,
    /// Number of virtual nodes to create per physical core.
    vnodes_per_core: u32,
}

impl PartitionRing {
    /// Create a new partition ring from a list of `(node_id, num_cores)` pairs.
    ///
    /// Each node gets `num_cores * vnodes_per_core` virtual nodes on the ring
    /// (default: 16 vnodes per core). The ring is sorted by hash point for
    /// efficient lookup via binary search.
    pub fn new(nodes: &[(u32, u32)]) -> Self {
        Self::with_vnodes_per_core(nodes, 16)
    }

    /// Create a ring with a custom number of virtual nodes per core.
    pub fn with_vnodes_per_core(nodes: &[(u32, u32)], vnodes_per_core: u32) -> Self {
        let mut virtual_nodes = Vec::new();
        for &(node_id, num_cores) in nodes {
            let count = num_cores * vnodes_per_core;
            for i in 0..count {
                let point = hash_vnode(node_id, i);
                virtual_nodes.push((point, node_id));
            }
        }
        virtual_nodes.sort_by_key(|&(point, _)| point);
        PartitionRing {
            virtual_nodes,
            vnodes_per_core,
        }
    }

    /// Determine which node owns the given fingerprint.
    ///
    /// Performs a binary search on the sorted ring to find the first virtual
    /// node whose hash point is >= the fingerprint, wrapping around if needed.
    ///
    /// # Panics
    ///
    /// Panics if the ring is empty (no nodes have been added).
    pub fn home_node(&self, fp: u64) -> u32 {
        assert!(!self.virtual_nodes.is_empty(), "partition ring is empty");
        // Hash the fingerprint for uniform distribution on the ring,
        // even if input fps are clustered (e.g., sequential integers)
        let hashed = splitmix64(fp);
        match self
            .virtual_nodes
            .binary_search_by_key(&hashed, |&(point, _)| point)
        {
            Ok(idx) => self.virtual_nodes[idx].1,
            Err(idx) => {
                if idx < self.virtual_nodes.len() {
                    self.virtual_nodes[idx].1
                } else {
                    // Wrap around to the first node on the ring.
                    self.virtual_nodes[0].1
                }
            }
        }
    }

    /// Add a node to the ring.
    pub fn add_node(&mut self, node_id: u32, num_cores: u32) {
        let count = num_cores * self.vnodes_per_core;
        for i in 0..count {
            let point = hash_vnode(node_id, i);
            let pos = self
                .virtual_nodes
                .binary_search_by_key(&point, |&(p, _)| p)
                .unwrap_or_else(|e| e);
            self.virtual_nodes.insert(pos, (point, node_id));
        }
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node_id: u32) {
        self.virtual_nodes.retain(|&(_, nid)| nid != node_id);
    }

    /// Number of virtual nodes currently on the ring.
    pub fn len(&self) -> usize {
        self.virtual_nodes.len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.virtual_nodes.is_empty()
    }
}

/// Hash a virtual node identifier to a point on the ring.
/// Deterministic hash for consistent cross-node partition assignment.
/// Uses splitmix64 finalizer — fast, deterministic, good distribution.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn hash_vnode(node_id: u32, vnode_index: u32) -> u64 {
    splitmix64((node_id as u64) << 32 | vnode_index as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_owns_everything() {
        let ring = PartitionRing::new(&[(0, 4)]);
        for fp in 0..1000u64 {
            assert_eq!(ring.home_node(fp), 0);
        }
    }

    #[test]
    fn two_equal_nodes_split_roughly_evenly() {
        let ring = PartitionRing::new(&[(0, 8), (1, 8)]);
        let mut counts = [0u32; 2];
        let n = 100_000u64;
        for fp in 0..n {
            let owner = ring.home_node(fp);
            counts[owner as usize] += 1;
        }
        // With equal core counts and 16 vnodes/core each, distribution should
        // be roughly 50/50. Allow 10% deviation.
        let ratio = counts[0] as f64 / n as f64;
        assert!(
            (0.40..=0.60).contains(&ratio),
            "node 0 got {:.1}% of fingerprints (expected ~50%)",
            ratio * 100.0
        );
    }

    #[test]
    fn heterogeneous_nodes_proportional() {
        // Node 0 has 4 cores, node 1 has 12 cores => node 1 should own ~3x more.
        let ring = PartitionRing::new(&[(0, 4), (1, 12)]);
        let mut counts = [0u32; 2];
        let n = 100_000u64;
        for fp in 0..n {
            let owner = ring.home_node(fp);
            counts[owner as usize] += 1;
        }
        let ratio_0 = counts[0] as f64 / n as f64;
        // Node 0 should own ~25% (4/16 of total cores).
        assert!(
            (0.15..=0.35).contains(&ratio_0),
            "node 0 got {:.1}% (expected ~25%)",
            ratio_0 * 100.0
        );
    }

    #[test]
    fn add_and_remove_node() {
        let mut ring = PartitionRing::new(&[(0, 4)]);
        assert_eq!(ring.home_node(42), 0);

        ring.add_node(1, 4);
        // After adding node 1, some fps should now route to node 1.
        let mut saw_node_1 = false;
        for fp in 0..1000u64 {
            if ring.home_node(fp) == 1 {
                saw_node_1 = true;
                break;
            }
        }
        assert!(saw_node_1, "node 1 should own at least some fingerprints");

        ring.remove_node(1);
        // After removing node 1, everything routes to node 0 again.
        for fp in 0..1000u64 {
            assert_eq!(ring.home_node(fp), 0);
        }
    }

    #[test]
    fn deterministic_assignment() {
        let ring = PartitionRing::new(&[(0, 4), (1, 4), (2, 4)]);
        let owner1 = ring.home_node(0xDEADBEEF);
        let owner2 = ring.home_node(0xDEADBEEF);
        assert_eq!(owner1, owner2, "same fp must always map to same node");
    }
}
