//! T6: end-to-end integration test for the cross-node work-stealing handshake.
//!
//! Spins up two `ClusterTransport`s on localhost, wires the inbound handler
//! and steal-trigger task on each side, and verifies:
//!
//! 1. The "victim" side (with states queued in its donate channel) responds
//!    to a `StealRequest` from the "thief" side with a non-empty
//!    `StealResponse`.
//! 2. The thief decrements its `pending_steal_requests` counter when the
//!    response arrives, so termination detection is unblocked.
//! 3. The donate channel is actually drained (states are transferred).
//! 4. A `StealRequest` to a victim with an empty backlog returns an empty
//!    response (no crash, no hang).
//!
//! Failure-mode coverage:
//! 5. Killing the victim's transport mid-steal causes the thief's
//!    pending-steal counter to roll back (peer marked down) within the
//!    timeout, so the thief can still terminate.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossbeam_channel::bounded;
use tlaplusplus::distributed::ClusterConfig;
use tlaplusplus::distributed::handler::{
    LocalPendingFn, StolenState, spawn_inbound_handler, spawn_steal_trigger_task,
};
use tlaplusplus::distributed::protocol::Message;
use tlaplusplus::distributed::transport::ClusterTransport;
use tlaplusplus::distributed::work_stealer::DistributedWorkStealer;

/// Build a 2-node cluster with both transports bound to OS-assigned ports
/// on 127.0.0.1, then cross-connect them.
async fn make_pair(
    rt_handle: tokio::runtime::Handle,
) -> (
    Arc<ClusterTransport>,
    Arc<ClusterTransport>,
    Arc<DistributedWorkStealer>,
    Arc<DistributedWorkStealer>,
) {
    // Bind two listeners first so we know their final addresses.
    use tokio::net::TcpListener;
    let l0 = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let l1 = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a0 = l0.local_addr().unwrap();
    let a1 = l1.local_addr().unwrap();
    drop(l0);
    drop(l1);

    let cfg0 = ClusterConfig {
        node_id: 0,
        listen_addr: a0,
        peers: vec![a1],
    };
    let cfg1 = ClusterConfig {
        node_id: 1,
        listen_addr: a1,
        peers: vec![a0],
    };

    let t0 = ClusterTransport::new(cfg0).await.unwrap();
    let t1 = ClusterTransport::new(cfg1).await.unwrap();
    // Give the acceptors a beat to come up.
    tokio::time::sleep(Duration::from_millis(50)).await;
    t0.connect_to_peers().await.unwrap();
    t1.connect_to_peers().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let s0 = Arc::new(DistributedWorkStealer::new(
        0,
        2,
        t0.clone(),
        rt_handle.clone(),
    ));
    let s1 = Arc::new(DistributedWorkStealer::new(1, 2, t1.clone(), rt_handle));
    (t0, t1, s0, s1)
}

#[test]
fn steal_handshake_transfers_states() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let handle = tokio::runtime::Handle::current();
        let (t0, t1, s0, s1) = make_pair(handle.clone()).await;

        // Channels for each node (thief = node 0, victim = node 1).
        let (stolen0_tx, stolen0_rx) = bounded::<StolenState>(1024);
        let (donate0_tx, donate0_rx) = bounded::<Vec<u8>>(1024);
        let (stolen1_tx, _stolen1_rx) = bounded::<StolenState>(1024);
        let (donate1_tx, donate1_rx) = bounded::<Vec<u8>>(1024);
        let _ = donate0_tx; // unused on thief side
        let _ = stolen1_tx.clone();

        // Pre-seed the victim's donate channel with 16 fake state blobs.
        for i in 0..16u8 {
            donate1_tx.send(vec![i, i, i, i]).unwrap();
        }

        // Pretend node 0 (thief) has zero queued and node 1 (victim) has plenty.
        let local_pending_thief: LocalPendingFn = Arc::new(|| 0u64);
        let local_pending_victim: LocalPendingFn = Arc::new(|| 1_000_000u64);

        let stop = Arc::new(AtomicBool::new(false));

        spawn_inbound_handler(
            &handle,
            t0.clone(),
            s0.clone(),
            stolen0_tx,
            donate0_rx,
            stop.clone(),
            local_pending_thief.clone(),
        );
        spawn_inbound_handler(
            &handle,
            t1.clone(),
            s1.clone(),
            stolen1_tx,
            donate1_rx,
            stop.clone(),
            local_pending_victim.clone(),
        );

        // Lower victim threshold so 16 queued states pass `can_donate`.
        s1.set_steal_victim_threshold(1);

        // Manually fire one steal from thief to victim.
        s0.begin_steal();
        let req = Message::StealRequest {
            from_node: 0,
            max_items: 8,
        };
        t0.send(1, &req).await.unwrap();

        // Wait up to 2s for the response to land.
        let mut received = 0usize;
        for _ in 0..200 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            while let Ok(_state) = stolen0_rx.try_recv() {
                received += 1;
            }
            if received >= 8 {
                break;
            }
        }

        assert_eq!(received, 8, "thief should have received 8 states");
        assert_eq!(
            s0.pending_steal_count(),
            0,
            "in-flight counter should be cleared on response"
        );
        assert!(
            s1.states_donated.load(Ordering::Relaxed) >= 8,
            "victim should have recorded 8 donated states"
        );
        // 8 states drained from the donation channel; 8 should remain.
        assert_eq!(donate1_tx.len(), 8);

        stop.store(true, Ordering::Release);
    });
}

#[test]
fn empty_victim_replies_empty_response() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let handle = tokio::runtime::Handle::current();
        let (t0, t1, s0, s1) = make_pair(handle.clone()).await;

        let (stolen0_tx, stolen0_rx) = bounded::<StolenState>(1024);
        let (_donate0_tx, donate0_rx) = bounded::<Vec<u8>>(1024);
        let (stolen1_tx, _stolen1_rx) = bounded::<StolenState>(1024);
        let (_donate1_tx, donate1_rx) = bounded::<Vec<u8>>(1024);

        let local_pending_thief: LocalPendingFn = Arc::new(|| 0u64);
        // Victim claims to have a tiny backlog — below the default 16K threshold.
        let local_pending_victim: LocalPendingFn = Arc::new(|| 100u64);

        let stop = Arc::new(AtomicBool::new(false));

        spawn_inbound_handler(
            &handle,
            t0.clone(),
            s0.clone(),
            stolen0_tx,
            donate0_rx,
            stop.clone(),
            local_pending_thief,
        );
        spawn_inbound_handler(
            &handle,
            t1.clone(),
            s1.clone(),
            stolen1_tx,
            donate1_rx,
            stop.clone(),
            local_pending_victim,
        );

        s0.begin_steal();
        t0.send(
            1,
            &Message::StealRequest {
                from_node: 0,
                max_items: 4096,
            },
        )
        .await
        .unwrap();

        // Wait up to 1s for the empty response.
        let mut got_response = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if s0.steal_responses_empty.load(Ordering::Relaxed) > 0 {
                got_response = true;
                break;
            }
        }

        assert!(got_response, "should observe an empty steal response");
        assert_eq!(
            s0.pending_steal_count(),
            0,
            "in-flight counter cleared on empty response too"
        );
        assert_eq!(stolen0_rx.len(), 0, "no states transferred");

        stop.store(true, Ordering::Release);
    });
}

#[test]
fn dead_peer_steal_times_out_and_marks_down() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let handle = tokio::runtime::Handle::current();
        let (t0, t1, s0, _s1) = make_pair(handle.clone()).await;

        let (stolen0_tx, _stolen0_rx) = bounded::<StolenState>(1024);
        let (_donate0_tx, donate0_rx) = bounded::<Vec<u8>>(1024);
        let local_pending_thief: LocalPendingFn = Arc::new(|| 0u64);
        let stop = Arc::new(AtomicBool::new(false));

        spawn_inbound_handler(
            &handle,
            t0.clone(),
            s0.clone(),
            stolen0_tx,
            donate0_rx,
            stop.clone(),
            local_pending_thief.clone(),
        );

        // Spawn the trigger task on the thief.
        s0.set_idle_before_steal(Duration::from_millis(10));
        s0.set_steal_victim_threshold(1);
        // Make the trigger fire by stamping last_local_work into the past.
        // Since we just constructed s0 it's already at "started_at" (0ns elapsed).
        spawn_steal_trigger_task(
            &handle,
            t0.clone(),
            s0.clone(),
            stop.clone(),
            local_pending_thief,
            50, // poll quickly
        );

        // Drop the victim's transport — TCP RST or just stop accepting.
        // ClusterTransport doesn't expose a clean shutdown, so we simulate
        // by simply not running a handler on t1 and dropping it. The send
        // from t0 should still succeed at the TCP layer (because t1's
        // background acceptor task is still alive holding the listener),
        // but the response will never come because no handler reads from
        // the inbound channel and replies. Our timeout path should fire
        // and mark peer 1 down.
        drop(t1);

        // Wait up to 4s (default timeout is 2s).
        let mut peer_down = false;
        for _ in 0..400 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if s0.is_peer_down(1) {
                peer_down = true;
                break;
            }
        }
        assert!(
            peer_down,
            "peer 1 should be marked down after steal timeout"
        );
        assert_eq!(
            s0.pending_steal_count(),
            0,
            "in-flight counter must roll back so termination detection isn't blocked"
        );

        stop.store(true, Ordering::Release);
    });
}
