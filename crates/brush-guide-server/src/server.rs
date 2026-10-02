use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, ServerHeader, decode_frame, encode_frame};
use brush_guide::session::{GuideSession, SessionError, forward_frames};
use brush_guide::warmup::Warmup;
use burn::tensor::Device;
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc, watch};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

type Sessions = Arc<Mutex<HashMap<String, GuideSession>>>;

/// How long a connection may stay silent, and how often the server pings it. A client that is
/// alive answers pings (WebSocket libraries do this on their own), so only dead peers time out,
/// not a phone that is standing still and sending no keyframes.
#[derive(Clone, Copy, Debug)]
pub struct Timeouts {
    pub read_idle: Duration,
    pub ping_every: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            read_idle: Duration::from_secs(60),
            ping_every: Duration::from_secs(20),
        }
    }
}

#[derive(Clone)]
struct Shared {
    timeouts: Timeouts,
    config: GuideConfig,
    device: Device,
    root: PathBuf,
    sessions: Sessions,
    /// Sessions take no commands until this turns true (GPU warm-up).
    ready: Option<watch::Receiver<bool>>,
}

/// Serves on `listener`. With `config.warmup`, a GPU warm-up starts at once
/// on its own thread; connections are accepted meanwhile and their sessions
/// queue keyframes until it is done.
pub async fn serve(
    listener: TcpListener,
    config: GuideConfig,
    device: Device,
    root: PathBuf,
) -> anyhow::Result<()> {
    let warmup = config
        .warmup
        .then(|| Warmup::spawn(config.clone(), device.clone()));
    let ready = warmup.as_ref().map(Warmup::ready);
    serve_inner(listener, config, device, root, Timeouts::default(), ready).await
}

/// Serves without a warm-up.
pub async fn serve_with(
    listener: TcpListener,
    config: GuideConfig,
    device: Device,
    root: PathBuf,
    timeouts: Timeouts,
) -> anyhow::Result<()> {
    serve_inner(listener, config, device, root, timeouts, None).await
}

async fn serve_inner(
    listener: TcpListener,
    config: GuideConfig,
    device: Device,
    root: PathBuf,
    timeouts: Timeouts,
    ready: Option<watch::Receiver<bool>>,
) -> anyhow::Result<()> {
    let shared = Shared {
        timeouts,
        config,
        device,
        root,
        sessions: Arc::default(),
        ready,
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
            GuideSession::start_when(
                shared.config.clone(),
                shared.device.clone(),
                shared.root.join(session_id),
                shared.ready.clone(),
            ),
        );
    }
    map.get(session_id).expect("just inserted or alive").clone()
}

fn spawn_pusher(session: &GuideSession, out: mpsc::Sender<Message>) -> JoinHandle<()> {
    let session = session.clone();
    tokio::spawn(async move {
        forward_frames(&session, |frame| {
            let out = out.clone();
            async move { out.send(Message::binary(frame)).await.is_ok() }
        })
        .await;
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
    let ping_tx = out_tx.clone();
    let ping_every = shared.timeouts.ping_every;
    let pinger = tokio::spawn(async move {
        let mut tick = tokio::time::interval(ping_every);
        tick.tick().await;
        loop {
            tick.tick().await;
            if ping_tx
                .send(Message::Ping(Vec::new().into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    let mut greeted: Option<Greeted> = None;
    let result = loop {
        let msg = match tokio::time::timeout(shared.timeouts.read_idle, source.next()).await {
            Err(_) => {
                log::info!(
                    "closing connection: no frames or pongs within {:?}",
                    shared.timeouts.read_idle
                );
                // The peer is presumably gone, so don't wait on a writer that may be stuck on it.
                let _ = out_tx.try_send(Message::Close(None));
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
    pinger.abort();
    drop(out_tx);
    // Let queued frames (e.g. a final error) go out, but don't hang on a dead peer.
    let abort = writer.abort_handle();
    if tokio::time::timeout(Duration::from_secs(2), writer)
        .await
        .is_err()
    {
        abort.abort();
    }
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
