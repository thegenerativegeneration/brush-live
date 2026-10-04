//! Process-wide pause of `bge_warm_up` runs. Each paused engine holds the
//! pause; while any does, pre-warms park between splat counts, where the
//! previous count's last readback has synced the GPU.

use brush_guide::warmup::Warmup;
use std::sync::{LazyLock, Mutex, PoisonError};
use tokio::sync::watch;

struct State {
    /// Engines currently paused.
    holders: u32,
    /// `parked` and `ready` of each running pre-warm, by registration id.
    running: Vec<(u64, watch::Receiver<bool>, watch::Receiver<bool>)>,
    next_id: u64,
}

static STATE: Mutex<State> = Mutex::new(State {
    holders: 0,
    running: Vec::new(),
    next_id: 0,
});
static PAUSE: LazyLock<watch::Sender<bool>> = LazyLock::new(|| watch::Sender::new(false));

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The flag a pre-warm passes to [`Warmup::spawn_pausable`].
pub(crate) fn flag() -> watch::Receiver<bool> {
    PAUSE.subscribe()
}

/// One more engine paused: pre-warms stop at their next safe point.
pub(crate) fn hold() {
    let mut s = state();
    s.holders += 1;
    PAUSE.send_replace(true);
}

/// One engine fewer paused; pre-warms go on once none is.
pub(crate) fn release() {
    let mut s = state();
    s.holders = s.holders.saturating_sub(1);
    if s.holders == 0 {
        PAUSE.send_replace(false);
    }
}

/// Keeps a running pre-warm visible to [`settle`] until dropped.
pub(crate) struct Registration(u64);

pub(crate) fn register(warmup: &Warmup) -> Registration {
    let mut s = state();
    let id = s.next_id;
    s.next_id += 1;
    s.running.push((id, warmup.parked(), warmup.ready()));
    Registration(id)
}

impl Drop for Registration {
    fn drop(&mut self) {
        state().running.retain(|(id, ..)| *id != self.0);
    }
}

/// Returns once every running pre-warm is parked, done or gone.
pub(crate) async fn settle() {
    let running: Vec<_> = state()
        .running
        .iter()
        .map(|(_, parked, ready)| (parked.clone(), ready.clone()))
        .collect();
    for (parked, ready) in running {
        parked_or_done(parked, ready).await;
    }
}

/// Waits until a warm-up is parked (false) or done or gone (true). Checks
/// `ready` first: after the warm-up ends, `parked` is closed and reads as
/// false.
pub(crate) async fn parked_or_done(
    mut parked: watch::Receiver<bool>,
    mut ready: watch::Receiver<bool>,
) -> bool {
    tokio::select! {
        biased;
        _ = ready.wait_for(|r| *r) => true,
        Ok(_) = parked.wait_for(|p| *p) => false,
    }
}
