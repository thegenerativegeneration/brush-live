use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, encode_frame};
use brush_guide_server::server::{Timeouts, serve_with};
use futures_util::{SinkExt, StreamExt};
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::Message;

const TIMEOUTS: Timeouts = Timeouts {
    read_idle: Duration::from_secs(2),
    ping_every: Duration::from_millis(500),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_keeps_reading_stays_connected_without_keyframes() {
    let url = start("alive").await;
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    ws.send(hello("alive")).await.unwrap();

    // Reading lets tungstenite answer the server's pings, so a silent but live client survives
    // well past the idle timeout.
    let start = Instant::now();
    let (mut pings, mut late_frames) = (0, 0);
    while start.elapsed() < TIMEOUTS.read_idle * 3 {
        match tokio::time::timeout(Duration::from_secs(2), ws.next()).await {
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
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_stops_answering_is_dropped() {
    let url = start("dead").await;
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    ws.send(hello("dead")).await.unwrap();

    // Not polling the socket means no pongs go out, like a vanished peer.
    tokio::time::sleep(TIMEOUTS.read_idle * 2).await;

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(
            Instant::now() < deadline,
            "server did not close the connection"
        );
        let next = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("server did not close the connection");
        if matches!(next, Some(Ok(Message::Close(_)) | Err(_)) | None) {
            break;
        }
    }
}
