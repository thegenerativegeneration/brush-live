use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, ServerHeader, decode_frame, encode_frame};
use brush_guide::session::GuideSession;
use burn::tensor::Device;
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio_tungstenite::tungstenite::Message;

type Sessions = Arc<Mutex<HashMap<String, GuideSession>>>;

pub async fn serve(listener: TcpListener, config: GuideConfig, device: Device, root: PathBuf) -> anyhow::Result<()> {
    let sessions: Sessions = Arc::default();
    loop {
        let (stream, peer) = listener.accept().await?;
        log::info!("connection from {peer}");
        let (config, device, root, sessions) = (config.clone(), device.clone(), root.clone(), sessions.clone());
        tokio::spawn(async move {
            if let Err(e) = handle(stream, config, device, root, sessions).await {
                log::warn!("connection {peer} ended: {e}");
            }
        });
    }
}

fn error_frame(message: impl Into<String>) -> Message {
    Message::binary(encode_frame(&ServerHeader::Error { message: message.into() }, &[]))
}

async fn handle(stream: TcpStream, config: GuideConfig, device: Device, root: PathBuf, sessions: Sessions) -> anyhow::Result<()> {
    let ws = tokio_tungstenite::accept_async(stream).await?;
    let (mut sink, mut source) = ws.split();

    let session = loop {
        let Some(msg) = source.next().await else { return Ok(()) };
        let Message::Binary(bytes) = msg? else { continue };
        match decode_frame::<ClientHeader>(&bytes) {
            Ok((ClientHeader::Hello { session_id, .. }, _)) => {
                let mut map = sessions.lock().await;
                // One live session at a time: a new id replaces older ones and frees their GPU memory.
                map.retain(|id, _| *id == session_id);
                // A panicked worker leaves a dead entry behind: only reuse it while alive,
                // otherwise start a fresh one rather than staying stuck until server restart.
                let alive = map.get(&session_id).is_some_and(GuideSession::is_alive);
                if !alive {
                    map.insert(
                        session_id.clone(),
                        GuideSession::start(config.clone(), device.clone(), root.join(&session_id)),
                    );
                }
                let s = map.get(&session_id).expect("just inserted or alive").clone();
                break s;
            }
            Ok(_) => sink.send(error_frame("expected hello")).await?,
            Err(e) => sink.send(error_frame(e.to_string())).await?,
        }
    };

    // All outgoing frames go through one channel so replies and pushes don't interleave badly.
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(64);
    let writer = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if sink.send(m).await.is_err() {
                break;
            }
        }
    });

    let push_tx = out_tx.clone();
    let mut scores = session.scores();
    let status = session.status();
    let pusher = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                changed = scores.changed() => {
                    if changed.is_err() { break; }
                    let frame = scores.borrow_and_update().as_ref().map(|s| s.to_frame());
                    if let Some(frame) = frame {
                        if push_tx.send(Message::binary(frame)).await.is_err() { break; }
                    }
                }
                _ = tick.tick() => {
                    let frame = status.borrow().to_frame();
                    if push_tx.send(Message::binary(frame)).await.is_err() { break; }
                }
            }
        }
    });

    while let Some(msg) = source.next().await {
        let Message::Binary(bytes) = msg? else { continue };
        let reply = match decode_frame::<ClientHeader>(&bytes) {
            Ok((ClientHeader::Keyframe(h), payload)) => {
                let id = h.id;
                match session.push_keyframe(h, payload.to_vec()).await {
                    Ok(()) => Message::binary(encode_frame(&ServerHeader::Ack { keyframe_id: id }, &[])),
                    Err(e) => error_frame(format!("keyframe {id}: {e}")),
                }
            }
            Ok((ClientHeader::Finish, _)) => match session.export_splat().await {
                Ok(ply) => Message::binary(encode_frame(&ServerHeader::Splat { ply_len: ply.len() as u64 }, &ply)),
                Err(e) => error_frame(e),
            },
            Ok((ClientHeader::Hello { .. }, _)) => error_frame("already greeted"),
            Err(e) => error_frame(e.to_string()),
        };
        if out_tx.send(reply).await.is_err() {
            break;
        }
    }
    pusher.abort();
    drop(out_tx);
    let _ = writer.await;
    Ok(())
}
