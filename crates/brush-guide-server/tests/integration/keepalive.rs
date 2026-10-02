use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, encode_frame};
use brush_guide_server::server::{Timeouts, serve_with};
use futures_util::{SinkExt, StreamExt};
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::Message;

const TIMEOUTS: Timeouts = Timeouts {
    read_idle: Duration::from_millis(500),
    ping_every: Duration::from_millis(100),
};

async fn start(name: &str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let root = std::env::temp_dir().join(format!(
        "brush-guide-keepalive-{name}-{}",
        std::process::id()
    ));
    tokio::spawn(serve_with(
        listener,
        GuideConfig::default(),
        device,
        root,
        TIMEOUTS,
    ));
    format!("ws://{addr}")
}

fn hello(id: &str) -> Message {
    let hello = ClientHeader::Hello {
        session_id: id.into(),
        device_model: "test".into(),
        has_lidar: false,
    };
    Message::binary(encode_frame(&hello, &[]))
}

/// One server, two clients: a client that keeps reading answers the server's pings and stays connected well past
/// the idle timeout, while a client that stops polling (no pongs go out, like a vanished peer) is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keepalive_keeps_a_reading_client_and_drops_a_silent_one() {
    let url = start("keepalive").await;

    let (mut alive, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    alive.send(hello("alive")).await.unwrap();

    // Reading lets tungstenite answer the server's pings, so a silent but live client survives
    // well past the idle timeout.
    let start = Instant::now();
    let (mut pings, mut late_frames) = (0, 0);
    while start.elapsed() < TIMEOUTS.read_idle * 3 {
        match tokio::time::timeout(Duration::from_secs(2), alive.next()).await {
            Ok(Some(Ok(Message::Ping(_)))) => pings += 1,
            Ok(Some(Ok(Message::Binary(_)))) if start.elapsed() > TIMEOUTS.read_idle * 2 => {
                late_frames += 1;
            }
            Ok(Some(Ok(Message::Close(_))) | None) => {
                panic!("closed after {:?}", start.elapsed())
            }
            Ok(Some(Err(e))) => panic!("{e}"),
            _ => {}
        }
    }
    assert!(pings >= 3, "pings: {pings}");
    assert!(
        late_frames > 0,
        "no status frames after twice the idle timeout"
    );

    let (mut dead, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    dead.send(hello("dead")).await.unwrap();

    // Not polling the socket means no pongs go out, like a vanished peer.
    tokio::time::sleep(TIMEOUTS.read_idle * 2).await;

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(
            Instant::now() < deadline,
            "server did not close the connection"
        );
        let next = tokio::time::timeout(Duration::from_secs(5), dead.next())
            .await
            .expect("server did not close the connection");
        if matches!(next, Some(Ok(Message::Close(_)) | Err(_)) | None) {
            break;
        }
    }
}
