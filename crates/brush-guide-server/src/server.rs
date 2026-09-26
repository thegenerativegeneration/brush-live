use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, ServerHeader, decode_frame, encode_frame};
use brush_guide::session::{GuideSession, SessionError};
use burn::tensor::Device;
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

type Sessions = Arc<Mutex<HashMap<String, GuideSession>>>;

/// A connection that receives no frame (pings included) for this long is closed.
const READ_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct Shared {
    config: GuideConfig,
    device: Device,
    root: PathBuf,
    sessions: Sessions,
}

pub async fn serve(
    listener: TcpListener,
    config: GuideConfig,
    device: Device,
    root: PathBuf,
) -> anyhow::Result<()> {
    let shared = Shared {
        config,
        device,
        root,
        sessions: Arc::default(),
    };
    loop {
        let (stream, peer) = listener.accept().await?;
        log::info!("connection from {peer}");
        let shared = shared.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(stream, shared).await {
                log::warn!("connection {peer} ended: {e}");
            }
        });
    }
}

/// Session ids become directory names under the server root.
pub fn valid_session_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn error_frame(message: impl Into<String>) -> Message {
    Message::binary(encode_frame(
        &ServerHeader::Error {
            message: message.into(),
        },
        &[],
    ))
}

async fn get_or_start(shared: &Shared, session_id: &str) -> GuideSession {
    let mut map = shared.sessions.lock().await;
    // One live session at a time: a new id replaces older ones and frees their GPU memory.
    map.retain(|id, _| id == session_id);
    // A panicked worker leaves a dead entry behind: only reuse it while alive,
    // otherwise start a fresh one rather than staying stuck until server restart.
    let alive = map.get(session_id).is_some_and(GuideSession::is_alive);
    if !alive {
        map.insert(
            session_id.to_owned(),
            GuideSession::start(
                shared.config.clone(),
                shared.device.clone(),
                shared.root.join(session_id),
            ),
        );
    }
    map.get(session_id).expect("just inserted or alive").clone()
}

/// Forwards the session's score sets as they change, and its status every second.
fn spawn_pusher(session: &GuideSession, out: mpsc::Sender<Message>) -> JoinHandle<()> {
    let mut scores = session.scores();
    let status = session.status();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                changed = scores.changed() => {
                    if changed.is_err() { break; }
                    let frame = scores.borrow_and_update().as_ref().map(|s| s.to_frame());
                    if let Some(frame) = frame
                        && out.send(Message::binary(frame)).await.is_err()
                    {
                        break;
                    }
                }
                _ = tick.tick() => {
                    let frame = status.borrow().to_frame();
                    if out.send(Message::binary(frame)).await.is_err() { break; }
                }
            }
        }
    })
}

struct Greeted {
    id: String,
    session: GuideSession,
    pusher: JoinHandle<()>,
}

/// What to do after handling one client frame.
enum Next {
    Reply(Message),
    Nothing,
    /// Send this, then close: the session's worker is gone.
    Close(Message),
}

fn session_failure(context: &str, e: &SessionError) -> Next {
    let frame = error_frame(format!("{context}: {e}"));
    if matches!(e, SessionError::Stopped) {
        Next::Close(frame)
    } else {
        Next::Reply(frame)
    }
}

async fn handle(stream: TcpStream, shared: Shared) -> anyhow::Result<()> {
    let ws = tokio_tungstenite::accept_async(stream).await?;
    let (mut sink, mut source) = ws.split();

    // All outgoing frames go through one channel so replies and pushes don't interleave badly.
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(64);
    let writer = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if sink.send(m).await.is_err() {
                break;
            }
        }
    });

    let mut greeted: Option<Greeted> = None;
    let result = loop {
        let msg = match tokio::time::timeout(READ_IDLE_TIMEOUT, source.next()).await {
            Err(_) => {
                log::info!("closing idle connection");
                let _ = out_tx.send(Message::Close(None)).await;
                break Ok(());
            }
            Ok(None) => break Ok(()),
            Ok(Some(Err(e))) => break Err(e.into()),
            Ok(Some(Ok(msg))) => msg,
        };
        let Message::Binary(bytes) = msg else {
            continue;
        };
        let next = match (decode_frame::<ClientHeader>(&bytes), &greeted) {
            (Err(e), _) => Next::Reply(error_frame(e.to_string())),
            (Ok((ClientHeader::Hello { session_id, .. }, _)), _)
                if !valid_session_id(&session_id) =>
            {
                Next::Reply(error_frame("invalid session id"))
            }
            (Ok((ClientHeader::Hello { session_id, .. }, _)), Some(g)) if g.id == session_id => {
                Next::Reply(error_frame("already greeted"))
            }
            (Ok((ClientHeader::Hello { session_id, .. }, _)), _) => {
                // A new id switches this connection to that session (e.g. relocalisation reset).
                if let Some(old) = greeted.take() {
                    old.pusher.abort();
                }
                let session = get_or_start(&shared, &session_id).await;
                let pusher = spawn_pusher(&session, out_tx.clone());
                greeted = Some(Greeted {
                    id: session_id,
                    session,
                    pusher,
                });
                Next::Nothing
            }
            (Ok(_), None) => Next::Reply(error_frame("expected hello")),
            (Ok((ClientHeader::Keyframe(h), payload)), Some(g)) => {
                let id = h.id;
                match g.session.push_keyframe(h, payload.to_vec()).await {
                    Ok(()) => Next::Reply(Message::binary(encode_frame(
                        &ServerHeader::Ack { keyframe_id: id },
                        &[],
                    ))),
                    Err(e) => session_failure(&format!("keyframe {id}"), &e),
                }
            }
            (Ok((ClientHeader::Finish, _)), Some(g)) => {
                let path = shared.root.join(&g.id).join("splat.ply");
                match g.session.finish(&path).await {
                    Ok(ply_len) => {
                        log::info!("session {} finished: {}", g.id, path.display());
                        Next::Reply(Message::binary(encode_frame(
                            &ServerHeader::Splat { ply_len },
                            &[],
                        )))
                    }
                    Err(e) => session_failure("finish", &e),
                }
            }
        };
        match next {
            Next::Nothing => {}
            Next::Reply(m) => {
                if out_tx.send(m).await.is_err() {
                    break Ok(());
                }
            }
            Next::Close(m) => {
                let _ = out_tx.send(m).await;
                let _ = out_tx.send(Message::Close(None)).await;
                break Ok(());
            }
        }
    };
    if let Some(g) = greeted {
        g.pusher.abort();
    }
    drop(out_tx);
    let _ = writer.await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_safe_directory_names() {
        for ok in ["s1", "replay-123", "A_b-9", &"x".repeat(64)] {
            assert!(valid_session_id(ok), "{ok}");
        }
        for bad in ["", "../s1", "a/b", ".", "..", "a b", "é", &"x".repeat(65)] {
            assert!(!valid_session_id(bad), "{bad}");
        }
    }
}
