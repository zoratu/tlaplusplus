pub mod batch;
pub mod protocol;
pub mod ring;
pub mod transport;

use std::net::SocketAddr;

/// Configuration for a node in the distributed model-checking cluster.
///
/// Each node owns a partition of the fingerprint space determined by
/// `home_node(fp) = hash(fp) % num_nodes`. States that hash to remote
/// nodes are batched and sent over TCP; remote nodes check-and-insert
/// fingerprints and enqueue new states locally.
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    /// Unique identifier for this node in the cluster.
    pub node_id: u32,
    /// Address this node listens on for inbound cluster traffic.
    pub listen_addr: SocketAddr,
    /// Addresses of all peer nodes (excluding self).
    pub peers: Vec<SocketAddr>,
    /// Number of states to accumulate before sending a batch (default 512).
    pub batch_size: usize,
    /// Maximum milliseconds to wait before sending a partial batch (default 1).
    pub batch_timeout_ms: u64,
}

impl ClusterConfig {
    /// Number of nodes in the cluster (self + peers).
    pub fn num_nodes(&self) -> u32 {
        self.peers.len() as u32 + 1
    }
}
