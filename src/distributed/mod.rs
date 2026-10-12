pub mod bloom;
pub mod handler;
pub mod protocol;
pub mod transport;
pub mod work_stealer;

use std::net::SocketAddr;

/// Environment variable carrying the cluster pre-shared authentication token.
///
/// Sourced from the environment (never a CLI flag) so the secret does not leak
/// via `ps`/argv or into tracked files. When set and non-empty, every peer must
/// present the same token during the connection handshake or its connection is
/// dropped before any message is processed.
pub const CLUSTER_TOKEN_ENV: &str = "TLAPLUSPLUS_CLUSTER_TOKEN";

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
    /// Optional pre-shared authentication token. When `Some(non-empty)`, the
    /// connection handshake rejects any peer that does not present the exact
    /// same token. `None` (or empty) means the legacy unauthenticated behavior
    /// — accept any peer that can reach the port — which is only safe on a
    /// loopback or otherwise-trusted network.
    ///
    /// The token authenticates peers but does **not** encrypt the channel
    /// (there is no TLS). It raises the bar from "anyone who can reach the
    /// port" to "anyone who knows the token"; for confidentiality or on-path
    /// protection, run the cluster over a private network, VPN, or SSH tunnel.
    pub auth_token: Option<String>,
}

impl ClusterConfig {
    /// Build a config, sourcing the auth token from [`CLUSTER_TOKEN_ENV`] and
    /// emitting a loud warning when the node binds a non-loopback address with
    /// no token configured. This is the constructor the CLI uses.
    pub fn with_env_auth(node_id: u32, listen_addr: SocketAddr, peers: Vec<SocketAddr>) -> Self {
        let auth_token = std::env::var(CLUSTER_TOKEN_ENV)
            .ok()
            .filter(|s| !s.is_empty());
        let cfg = Self {
            node_id,
            listen_addr,
            peers,
            auth_token,
        };
        cfg.warn_if_insecure();
        cfg
    }

    /// Print a prominent warning when the node exposes the unauthenticated
    /// work-stealing protocol to a non-loopback address. Binding `0.0.0.0` (or
    /// any routable address) with no token lets any host that can reach the
    /// port inject or steal states and fingerprints.
    pub fn warn_if_insecure(&self) {
        let ip = self.listen_addr.ip();
        let loopback = ip.is_loopback();
        if !loopback && self.auth_token.is_none() {
            eprintln!(
                "[cluster] WARNING: listening on {} with NO authentication token. \
                 Any host that can reach this port can inject or steal state. \
                 Set {} to a shared secret on every node, or bind a loopback/\
                 trusted-network address.",
                self.listen_addr, CLUSTER_TOKEN_ENV
            );
        }
    }

    /// Number of nodes in the cluster (self + peers).
    pub fn num_nodes(&self) -> u32 {
        self.peers.len() as u32 + 1
    }
}
