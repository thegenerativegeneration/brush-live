use brush_guide::config::GuideConfig;
use brush_guide::protocol::{
    ClientHeader, KeyframeHeader, ServerHeader, decode_frame, encode_frame,
};
use futures_util::{SinkExt, StreamExt};
use glam::{Mat4, Vec3};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

fn keyframe(id: u64, pos: Vec3) -> Vec<u8> {
    let (w, h) = (64u32, 48u32);
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x * 4) as u8, (y * 5) as u8, ((x ^ y) * 8) as u8])
    });
    let mut jpeg = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let pose = Mat4::look_at_rh(pos, Vec3::ZERO, Vec3::Y).inverse();
    let header = ClientHeader::Keyframe(KeyframeHeader {
        id,
        timestamp: 0.0,
        pose: pose.to_cols_array(),
        fx: 50.0,
        fy: 50.0,
        cx: 32.0,
        cy: 24.0,
        width: w,
        height: h,
        jpeg_len: jpeg.len() as u32,
        depth_size: None,
        depth_confidence: false,
        num_points: 0,
    });
    encode_frame(&header, &jpeg)
}

async fn next_frame<S>(ws: &mut S) -> (ServerHeader, Vec<u8>)
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(60), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Binary(b) = msg {
            let (header, payload) = decode_frame::<ServerHeader>(&b).unwrap();
            return (header, payload.to_vec());
        }
    }
}

async fn next_header<S>(ws: &mut S) -> ServerHeader
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    next_frame(ws).await.0
}

fn hello(session_id: &str) -> Message {
    let hello = ClientHeader::Hello {
        session_id: session_id.into(),
        device_model: "test".into(),
        has_lidar: false,
    };
    Message::binary(encode_frame(&hello, &[]))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_to_end() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let root = std::env::temp_dir().join(format!("brush-guide-ws-{}", std::process::id()));
    tokio::spawn(brush_guide_server::server::serve(
        listener,
        GuideConfig {
            mesh_enabled: true,
            ..GuideConfig::default()
        },
        device,
        root.clone(),
    ));

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
        .await
        .unwrap();
    // A session id that could escape the session root is refused; the connection stays open.
    ws.send(hello("../s1")).await.unwrap();
    match next_header(&mut ws).await {
        ServerHeader::Error { message } => assert_eq!(message, "invalid session id"),
        other => panic!("{other:?}"),
    }
    ws.send(hello("s1")).await.unwrap();

    // Malformed frame: error, connection survives.
    ws.send(Message::binary(vec![255, 255, 0, 0]))
        .await
        .unwrap();
    let mut saw_error = false;
    let mut acks = Vec::new();
    for (i, a) in [0.0f32, 1.0, 2.0].iter().enumerate() {
        ws.send(Message::binary(keyframe(
            i as u64,
            Vec3::new(2.0 * a.sin(), 0.0, 2.0 * a.cos()),
        )))
        .await
        .unwrap();
    }
    // Mesh bricks follow a score set with a version at least the score
    // set's; by the next score set the previous one's round has been sent.
    let (mut last_score, mut last_mesh) = (None, 0u64);
    let mut saw_mesh = false;
    while acks.len() < 3 || !saw_mesh || !saw_error {
        match next_header(&mut ws).await {
            ServerHeader::Ack { keyframe_id } => acks.push(keyframe_id),
            ServerHeader::Error { .. } => saw_error = true,
            ServerHeader::ScoreSet { version, .. } => {
                if let Some(prev) = last_score {
                    assert!(last_mesh >= prev, "round {prev} never sent as mesh bricks");
                }
                last_score = Some(version);
            }
            ServerHeader::MeshBricks { version, .. } => {
                let score = last_score.expect("mesh bricks after a score set");
                assert!(
                    version >= score,
                    "mesh bricks v{version} after score set v{score}"
                );
                last_mesh = version;
                saw_mesh = true;
            }
            _ => {}
        }
    }
    assert_eq!(acks, vec![0, 1, 2]);

    // Reconnect with the same session id and resend: acked, not duplicated.
    drop(ws);
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
        .await
        .unwrap();
    ws.send(hello("s1")).await.unwrap();
    ws.send(Message::binary(keyframe(2, Vec3::new(0.0, 0.0, 2.0))))
        .await
        .unwrap();
    // The new connection also gets the session's current mesh.
    let (mut acked, mut saw_mesh) = (false, false);
    while !acked || !saw_mesh {
        match next_header(&mut ws).await {
            ServerHeader::Ack { keyframe_id } => {
                assert_eq!(keyframe_id, 2);
                acked = true;
            }
            ServerHeader::MeshBricks { .. } => saw_mesh = true,
            ServerHeader::Error { message } => panic!("{message}"),
            _ => {}
        }
    }
    loop {
        if let ServerHeader::Status { num_keyframes, .. } = next_header(&mut ws).await {
            assert_eq!(num_keyframes, 3);
            break;
        }
    }

    ws.send(Message::binary(encode_frame(&ClientHeader::Finish, &[])))
        .await
        .unwrap();
    loop {
        if let (ServerHeader::Splat { ply_len }, payload) = next_frame(&mut ws).await {
            assert!(ply_len > 0);
            assert!(payload.is_empty(), "the PLY stays on the server");
            assert_eq!(
                std::fs::metadata(root.join("s1/splat.ply")).unwrap().len(),
                ply_len
            );
            break;
        }
    }

    // A different id on the same connection switches to a fresh session.
    ws.send(hello("s2")).await.unwrap();
    ws.send(Message::binary(keyframe(0, Vec3::new(0.0, 0.0, 2.0))))
        .await
        .unwrap();
    loop {
        match next_header(&mut ws).await {
            ServerHeader::Ack { keyframe_id } => {
                assert_eq!(keyframe_id, 0);
                break;
            }
            ServerHeader::Error { message } => panic!("{message}"),
            _ => {}
        }
    }
    loop {
        if let ServerHeader::Status { num_keyframes, .. } = next_header(&mut ws).await {
            assert!(
                num_keyframes <= 1,
                "status of the new session, got {num_keyframes}"
            );
            if num_keyframes == 1 {
                break;
            }
        }
    }
    std::fs::remove_dir_all(root).ok();
}
