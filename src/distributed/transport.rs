use anyhow::{Context, Result};
use dashmap::DashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use super::ClusterConfig;
use super::protocol::{Message, decode_message, encode_message};

/// Shorthand for the `Send`-able boxed future returned by [`Transport`] methods.
///
/// Returning a boxed future (rather than `async fn` in trait) keeps the trait
/// `dyn`-compatible without pulling in the `async-trait` crate.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Abstraction over the inter-node messaging substrate.
///
/// `ClusterTransport` is the production TCP implementation; tests use
/// [`MockTransport`] which routes messages through in-process channels.
///
/// Keeping the trait `dyn`-compatible (no generics, no `async fn`) lets the
/// handler tasks accept `Arc<dyn Transport>` so they can be exercised end-to-end
/// without standing up real listeners.
pub trait Transport: Send + Sync + 'static {
    /// Send a message to the named peer node.
    fn send<'a>(&'a self, node_id: u32, msg: &'a Message) -> BoxFuture<'a, Result<()>>;

    /// Receive the next inbound `(from_node, message)` pair, or `None` if the
    /// transport is shutting down.
    fn recv<'a>(&'a self) -> BoxFuture<'a, Option<(u32, Message)>>;

    /// Broadcast a message to all currently-connected peers.
    fn broadcast<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>>;

    /// This transport's node id.
    fn node_id(&self) -> u32;
}

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
                        // T10.2 phase-2 streaming-DFS variants. Source-node
                        // tagging matches the rest of the table: `from_node`
                        // for sender-initiated messages, `owner_node` for
                        // ack/response messages.
                        Message::PartitionEdge { from_node, .. } => *from_node,
                        Message::PartitionEdgeAck { from_node, .. } => *from_node,
                        Message::RedDfsProbe { from_node, .. } => *from_node,
                        Message::RedDfsResponse { from_node, .. } => *from_node,
                        Message::RequestStateBlob { from_node, .. } => *from_node,
                        Message::StateBlobResponse { from_node, .. } => *from_node,
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

impl Transport for ClusterTransport {
    fn send<'a>(&'a self, node_id: u32, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(ClusterTransport::send(self, node_id, msg))
    }

    fn recv<'a>(&'a self) -> BoxFuture<'a, Option<(u32, Message)>> {
        Box::pin(ClusterTransport::recv(self))
    }

    fn broadcast<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(ClusterTransport::broadcast(self, msg))
    }

    fn node_id(&self) -> u32 {
        ClusterTransport::node_id(self)
    }
}

// ---------------------------------------------------------------------------
// MockTransport: in-process [`Transport`] for unit-testing the handler.
// ---------------------------------------------------------------------------

/// Shared routing fabric for a set of [`MockTransport`] instances.
///
/// Holds one `tokio::mpsc::Sender` per node id. `MockTransport::send` routes
/// the message to the destination's sender; the destination's
/// `MockTransport::recv` drains its own receiver.
///
/// This intentionally mirrors the framing-free path of `ClusterTransport`
/// (skipping bincode/length-prefix for speed) so handler tests don't pay
/// serialization overhead.
#[derive(Clone, Default)]
pub struct MockNetwork {
    inner: Arc<dashmap::DashMap<u32, mpsc::Sender<(u32, Message)>>>,
}

impl MockNetwork {
    /// Create an empty network; nodes register themselves on construction.
    pub fn new() -> Self {
        Self::default()
    }
}

/// In-process `Transport` for unit tests.
///
/// Construct via [`MockTransport::new`] passing a shared [`MockNetwork`];
/// every `MockTransport` built from the same network can address every other.
pub struct MockTransport {
    node_id: u32,
    network: MockNetwork,
    /// This node's inbound mailbox.
    inbound_rx: tokio::sync::Mutex<mpsc::Receiver<(u32, Message)>>,
    /// Send hook: when set to true, all sends silently no-op.
    /// Useful for simulating a peer that has gone dark.
    pub drop_sends: std::sync::atomic::AtomicBool,
}

impl MockTransport {
    /// Register this node on `network` and return the transport.
    ///
    /// `channel_capacity` controls how many in-flight messages the inbound
    /// mailbox can buffer before backpressure kicks in (handler tests rarely
    /// queue more than a handful, so 1024 is plenty).
    pub fn new(node_id: u32, network: MockNetwork) -> Arc<Self> {
        Self::with_capacity(node_id, network, 1024)
    }

    /// Like [`MockTransport::new`] but with explicit mailbox capacity.
    pub fn with_capacity(node_id: u32, network: MockNetwork, capacity: usize) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(capacity);
        network.inner.insert(node_id, tx);
        Arc::new(MockTransport {
            node_id,
            network,
            inbound_rx: tokio::sync::Mutex::new(rx),
            drop_sends: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Returns true if the named peer has registered on the network.
    pub fn is_peer_known(&self, node_id: u32) -> bool {
        self.network.inner.contains_key(&node_id)
    }
}

impl Transport for MockTransport {
    fn send<'a>(&'a self, node_id: u32, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        let drop = self
            .drop_sends
            .load(std::sync::atomic::Ordering::Acquire);
        let from = self.node_id;
        // Capture the sender (clone) outside the future so the future itself
        // doesn't borrow the dashmap iterator.
        let dest = self.network.inner.get(&node_id).map(|s| s.clone());
        let cloned_msg = msg.clone();
        Box::pin(async move {
            if drop {
                // Caller asked us to silently swallow sends.
                return Ok(());
            }
            let Some(sender) = dest else {
                return Err(anyhow::anyhow!(
                    "mock transport: no peer registered with node_id {node_id}"
                ));
            };
            sender
                .send((from, cloned_msg))
                .await
                .map_err(|e| anyhow::anyhow!("mock transport send to {node_id} failed: {e}"))
        })
    }

    fn recv<'a>(&'a self) -> BoxFuture<'a, Option<(u32, Message)>> {
        Box::pin(async move {
            let mut rx = self.inbound_rx.lock().await;
            rx.recv().await
        })
    }

    fn broadcast<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        let drop = self
            .drop_sends
            .load(std::sync::atomic::Ordering::Acquire);
        let from = self.node_id;
        let peers: Vec<(u32, mpsc::Sender<(u32, Message)>)> = self
            .network
            .inner
            .iter()
            .filter(|r| *r.key() != from)
            .map(|r| (*r.key(), r.value().clone()))
            .collect();
        let cloned_msg = msg.clone();
        Box::pin(async move {
            if drop {
                return Ok(());
            }
            for (peer_id, sender) in peers {
                sender.send((from, cloned_msg.clone())).await.map_err(|e| {
                    anyhow::anyhow!("mock transport broadcast to {peer_id} failed: {e}")
                })?;
            }
            Ok(())
        })
    }

    fn node_id(&self) -> u32 {
        self.node_id
    }
}

#[cfg(test)]
mod mock_tests {
    use super::*;

    #[tokio::test]
    async fn mock_transport_routes_messages_between_nodes() {
        let net = MockNetwork::new();
        let a = MockTransport::new(0, net.clone());
        let b = MockTransport::new(1, net.clone());

        let msg = Message::StealRequest {
            from_node: 0,
            max_items: 8,
        };
        a.send(1, &msg).await.unwrap();

        let (from, got) = b.recv().await.unwrap();
        assert_eq!(from, 0);
        assert!(matches!(
            got,
            Message::StealRequest {
                from_node: 0,
                max_items: 8
            }
        ));
    }

    #[tokio::test]
    async fn mock_transport_send_to_unknown_peer_errors() {
        let net = MockNetwork::new();
        let a = MockTransport::new(0, net);
        let res = a
            .send(
                42,
                &Message::Heartbeat {
                    node_id: 0,
                    states_generated: 0,
                    states_distinct: 0,
                },
            )
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn mock_transport_broadcast_skips_self() {
        let net = MockNetwork::new();
        let a = MockTransport::new(0, net.clone());
        let b = MockTransport::new(1, net.clone());
        let c = MockTransport::new(2, net.clone());

        a.broadcast(&Message::Heartbeat {
            node_id: 0,
            states_generated: 1,
            states_distinct: 1,
        })
        .await
        .unwrap();

        // Both peers receive; a does not (broadcast skips self by node_id).
        let (_, _) = b.recv().await.unwrap();
        let (_, _) = c.recv().await.unwrap();
        // a's mailbox should have nothing pending. Use try_recv via a short
        // timeout so we don't wedge the test on a regression.
        let timeout = tokio::time::timeout(std::time::Duration::from_millis(50), a.recv()).await;
        assert!(timeout.is_err(), "self should not receive own broadcast");
    }

    #[tokio::test]
    async fn mock_transport_drop_sends_silently_swallows_traffic() {
        let net = MockNetwork::new();
        let a = MockTransport::new(0, net.clone());
        let b = MockTransport::new(1, net.clone());

        a.drop_sends
            .store(true, std::sync::atomic::Ordering::Release);

        a.send(
            1,
            &Message::Heartbeat {
                node_id: 0,
                states_generated: 0,
                states_distinct: 0,
            },
        )
        .await
        .unwrap();

        // b should not receive — the send was swallowed.
        let timeout = tokio::time::timeout(std::time::Duration::from_millis(50), b.recv()).await;
        assert!(timeout.is_err(), "drop_sends should swallow the message");
    }

    #[test]
    fn mock_transport_node_id_round_trip() {
        let net = MockNetwork::new();
        let t = MockTransport::new(7, net);
        assert_eq!(<MockTransport as Transport>::node_id(&*t), 7);
    }
}
