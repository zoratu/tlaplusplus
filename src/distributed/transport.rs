use anyhow::{Context, Result};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use super::ClusterConfig;
use super::protocol::{Message, decode_message, encode_message};

/// TCP transport layer for inter-node cluster communication.
///
/// Manages persistent outbound connections to each peer and accepts inbound
/// connections. Messages are framed with a 4-byte big-endian length prefix
/// and serialized via bincode.
pub struct ClusterTransport {
    config: ClusterConfig,
    /// Per-peer outbound TCP connections, keyed by node_id.
    /// Each value is an `Arc<tokio::sync::Mutex<TcpStream>>` so we can
    /// share the write half across tasks safely.
    connections: DashMap<u32, Arc<tokio::sync::Mutex<TcpStream>>>,
    /// Receiver end for inbound messages from all peers.
    inbound_rx: tokio::sync::Mutex<mpsc::Receiver<(u32, Message)>>,
    /// Sender end — cloned into each inbound-handler task.
    inbound_tx: mpsc::Sender<(u32, Message)>,
}

impl ClusterTransport {
    /// Create a new transport, bind the listen socket, and start accepting
    /// inbound connections.
    ///
    /// Does **not** connect to peers yet — call [`connect_to_peers`] after
    /// all nodes are listening.
    pub async fn new(config: ClusterConfig) -> Result<Arc<Self>> {
        let listener = TcpListener::bind(config.listen_addr)
            .await
            .with_context(|| {
                format!("failed to bind cluster listener on {}", config.listen_addr)
            })?;

        let (inbound_tx, inbound_rx) = mpsc::channel(16_384);

        let transport = Arc::new(ClusterTransport {
            config,
            connections: DashMap::new(),
            inbound_rx: tokio::sync::Mutex::new(inbound_rx),
            inbound_tx,
        });

        // Spawn acceptor task.
        let t = Arc::clone(&transport);
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let tx = t.inbound_tx.clone();
                        tokio::spawn(Self::handle_inbound(stream, tx));
                    }
                    Err(e) => {
                        eprintln!("[cluster] accept error: {e}");
                    }
                }
            }
        });

        Ok(transport)
    }

    /// Establish outbound TCP connections to all peers listed in the config.
    ///
    /// Peers are identified by their index in `config.peers` mapped to node IDs.
    /// Node IDs for peers are assigned as 0..num_peers, skipping `self.config.node_id`.
    pub async fn connect_to_peers(&self) -> Result<()> {
        for (idx, &peer_addr) in self.config.peers.iter().enumerate() {
            // Derive peer node_id: peers list is ordered, peer ids are the
            // sequential ids excluding our own node_id.
            let peer_id = if idx as u32 >= self.config.node_id {
                idx as u32 + 1
            } else {
                idx as u32
            };

            let stream = TcpStream::connect(peer_addr)
                .await
                .with_context(|| format!("failed to connect to peer {peer_id} at {peer_addr}"))?;
            stream.set_nodelay(true).ok();
            self.connections
                .insert(peer_id, Arc::new(tokio::sync::Mutex::new(stream)));
        }
        Ok(())
    }

    /// Send a message to a specific peer node.
    pub async fn send(&self, node_id: u32, msg: &Message) -> Result<()> {
        let conn = self
            .connections
            .get(&node_id)
            .ok_or_else(|| anyhow::anyhow!("no connection to node {node_id}"))?;
        let data = encode_message(msg)?;
        let mut stream = conn.lock().await;
        stream
            .write_all(&data)
            .await
            .with_context(|| format!("failed to send message to node {node_id}"))?;
        stream.flush().await?;
        Ok(())
    }

    /// Receive the next inbound message (blocking).
    ///
    /// Returns `None` if all senders have been dropped (transport shutting down).
    pub async fn recv(&self) -> Option<(u32, Message)> {
        let mut rx = self.inbound_rx.lock().await;
        rx.recv().await
    }

    /// Broadcast a message to all connected peers.
    pub async fn broadcast(&self, msg: &Message) -> Result<()> {
        let peer_ids: Vec<u32> = self.connections.iter().map(|r| *r.key()).collect();
        for peer_id in peer_ids {
            self.send(peer_id, msg).await?;
        }
        Ok(())
    }

    /// Register an already-connected stream for a specific node_id.
    ///
    /// Useful when a peer connects inbound and we want to reuse that
    /// connection for outbound traffic as well.
    pub fn register_connection(&self, node_id: u32, stream: TcpStream) {
        stream.set_nodelay(true).ok();
        self.connections
            .insert(node_id, Arc::new(tokio::sync::Mutex::new(stream)));
    }

    /// The node ID of this transport.
    pub fn node_id(&self) -> u32 {
        self.config.node_id
    }

    /// Handle a single inbound TCP connection: read framed messages and
    /// forward them to the inbound channel.
    async fn handle_inbound(mut stream: TcpStream, tx: mpsc::Sender<(u32, Message)>) {
        let mut len_buf = [0u8; 4];
        loop {
            // Read 4-byte length prefix.
            match stream.read_exact(&mut len_buf).await {
                Ok(_) => {}
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        eprintln!("[cluster] inbound read error: {e}");
                    }
                    return;
                }
            }
            let len = u32::from_be_bytes(len_buf) as usize;
            if len > 64 * 1024 * 1024 {
                eprintln!("[cluster] inbound message too large: {len} bytes, dropping connection");
                return;
            }

            let mut payload = vec![0u8; len];
            if let Err(e) = stream.read_exact(&mut payload).await {
                eprintln!("[cluster] inbound payload read error: {e}");
                return;
            }

            match decode_message(&payload) {
                Ok(msg) => {
                    // Extract source node_id from the message if possible,
                    // otherwise use u32::MAX as sentinel.
                    let from = match &msg {
                        Message::StealRequest { from_node, .. } => *from_node,
                        Message::StealResponse { .. } => u32::MAX,
                        Message::BloomExchange { from_node, .. } => *from_node,
                        Message::Join { node_id, .. } => *node_id,
                        Message::Leave { node_id } => *node_id,
                        Message::Heartbeat { node_id, .. } => *node_id,
                        Message::Stop { node_id, .. } => *node_id,
                        Message::TerminationToken { initiator, .. } => *initiator,
                    };
                    if tx.send((from, msg)).await.is_err() {
                        // Receiver dropped — transport is shutting down.
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("[cluster] failed to decode inbound message: {e}");
                }
            }
        }
    }
}
