use brush_guide::config::GuideConfig;
use brush_guide::protocol::{
    ClientHeader, KeyframeHeader, PoseUpdate, ServerHeader, StatusPayload, decode_frame, encode_frame,
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
        mono_depth_size: None,
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
        GuideConfig::default(),
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
    let mut saw_score = false;
    while acks.len() < 3 || !saw_score || !saw_error {
        match next_header(&mut ws).await {
            ServerHeader::Ack { keyframe_id } => acks.push(keyframe_id),
            ServerHeader::Error { .. } => saw_error = true,
            ServerHeader::ScoreSet { .. } => saw_score = true,
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
    let mut acked = false;
    while !acked {
        match next_header(&mut ws).await {
            ServerHeader::Ack { keyframe_id } => {
                assert_eq!(keyframe_id, 2);
                acked = true;
            }
            ServerHeader::Error { message } => panic!("{message}"),
            _ => {}
        }
    }
    loop {
        if let ServerHeader::Status(StatusPayload { num_keyframes, .. }) = next_header(&mut ws).await {
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
        if let ServerHeader::Status(StatusPayload { num_keyframes, .. }) = next_header(&mut ws).await {
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

/// A poses frame is applied without a reply; an invalid one gets an error and the connection stays usable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poses_are_applied_silently_and_bad_ones_rejected() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let root = std::env::temp_dir().join(format!("brush-guide-ws-poses-{}", std::process::id()));
    tokio::spawn(brush_guide_server::server::serve(
        listener,
        GuideConfig::default(),
        device,
        root.clone(),
    ));
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
        .await
        .unwrap();
    ws.send(hello("p1")).await.unwrap();
    ws.send(Message::binary(keyframe(0, Vec3::new(0.0, 0.0, 2.0))))
        .await
        .unwrap();
    let pose = Mat4::look_at_rh(Vec3::new(0.3, 0.0, 2.0), Vec3::ZERO, Vec3::Y)
        .inverse()
        .to_cols_array();
    // Not rigid (a scaled axis). A NaN cannot be sent: JSON writes it as null, a decode error.
    let mut bad = pose;
    bad[0] *= 2.0;
    for poses in [
        vec![PoseUpdate { id: 0, pose }],
        vec![PoseUpdate { id: 0, pose: bad }],
    ] {
        ws.send(Message::binary(encode_frame(
            &ClientHeader::Poses { poses },
            &[],
        )))
        .await
        .unwrap();
    }
    ws.send(Message::binary(keyframe(1, Vec3::new(1.0, 0.0, 2.0))))
        .await
        .unwrap();
    let mut errors = Vec::new();
    let mut acks = Vec::new();
    while acks.len() < 2 {
        match next_header(&mut ws).await {
            ServerHeader::Ack { keyframe_id } => acks.push(keyframe_id),
            ServerHeader::Error { message } => errors.push(message),
            _ => {}
        }
    }
    assert_eq!(acks, vec![0, 1]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].starts_with("poses: keyframe 0 pose: "),
        "{errors:?}"
    );
    std::fs::remove_dir_all(root).ok();
}

/// Train iterations of the next `n` status frames that report training, with when each arrived after `start`.
async fn training_status<S>(
    ws: &mut S,
    n: usize,
    start: tokio::time::Instant,
) -> Vec<(u64, Duration)>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut out = Vec::new();
    while out.len() < n {
        if let ServerHeader::Status(StatusPayload { train_iters, .. }) = next_header(ws).await
            && train_iters > 0
        {
            out.push((train_iters, start.elapsed()));
        }
    }
    out
}

/// With `status_every_step` a status frame follows every training step (lockstep replays gate on its
/// `train_iters`); without it status stays at one frame a second.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_every_step_pushes_the_step_count_promptly() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let root = std::env::temp_dir().join(format!("brush-guide-ws-steps-{}", std::process::id()));
    for every_step in [true, false] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let config = GuideConfig {
            status_every_step: every_step,
            ..GuideConfig::default()
        };
        tokio::spawn(brush_guide_server::server::serve(
            listener,
            config,
            device.clone(),
            root.clone(),
        ));
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
            .await
            .unwrap();
        ws.send(hello(if every_step { "steps-on" } else { "steps-off" }))
            .await
            .unwrap();
        ws.send(Message::binary(keyframe(0, Vec3::new(0.0, 0.0, 2.0))))
            .await
            .unwrap();
        let start = tokio::time::Instant::now();
        let got = training_status(&mut ws, 4, start).await;
        let iters: Vec<u64> = got.iter().map(|(i, _)| *i).collect();
        assert!(iters.windows(2).all(|w| w[0] <= w[1]), "{iters:?}");
        let span = got[3].1 - got[0].1;
        if every_step {
            assert!(
                span < Duration::from_millis(900),
                "4 frames over {span:?}: {got:?}"
            );
            assert!(
                iters[3] - iters[0] <= 6,
                "one frame per step or so: {iters:?}"
            );
        } else {
            assert!(
                span >= Duration::from_millis(2500),
                "4 frames over {span:?}: {got:?}"
            );
        }
    }
    std::fs::remove_dir_all(root).ok();
}
