pub mod bloom;
pub mod handler;
pub mod protocol;
pub mod transport;
pub mod work_stealer;

use std::net::SocketAddr;

/// Configuration for a node in the distributed model-checking cluster.
///
/// Each node runs a fully independent model checker with its own FP store
/// and work queue. The network is only used for work stealing (when a node's
/// queue empties) and bloom filter exchange (periodic dedup summaries).
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    /// Unique identifier for this node in the cluster.
    pub node_id: u32,
    /// Address this node listens on for inbound cluster traffic.
    pub listen_addr: SocketAddr,
    /// Addresses of all peer nodes (excluding self).
    pub peers: Vec<SocketAddr>,
}

impl ClusterConfig {
    /// Number of nodes in the cluster (self + peers).
    pub fn num_nodes(&self) -> u32 {
        self.peers.len() as u32 + 1
    }
}
