//! The frames a session sends to its client, in the order the server sends
//! them: each score set, and a status frame once a second (after every
//! training step with `status_every_step`).

use super::GuideSession;
use std::future::Future;
use std::time::Duration;

/// Forwards `session`'s outgoing frames to `send` until it returns `false`
/// or the session ends.
pub async fn forward_frames<F, Fut>(session: &GuideSession, mut send: F)
where
    F: FnMut(Vec<u8>) -> Fut,
    Fut: Future<Output = bool>,
{
    let mut scores = session.scores();
    let mut status = session.status();
    let every_step = session.status_every_step();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            changed = status.changed(), if every_step => {
                if changed.is_err() { break; }
                let frame = status.borrow_and_update().to_frame();
                if !send(frame).await { break; }
            }
            changed = scores.changed() => {
                if changed.is_err() { break; }
                let frame = scores.borrow_and_update().as_ref().map(|s| s.to_frame());
                let Some(frame) = frame else { continue };
                if !send(frame).await { break; }
            }
            _ = tick.tick() => {
                let frame = status.borrow().to_frame();
                if !send(frame).await { break; }
            }
        }
    }
}
