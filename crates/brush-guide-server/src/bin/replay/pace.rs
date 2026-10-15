//! When the replay sends each keyframe: on a wall-clock schedule (the default), or in lockstep with the server's
//! training steps, so the steps per keyframe stay the same however busy the machine is.

use std::time::Duration;
use tokio::sync::{mpsc, watch};

/// How the replay paces its keyframes.
pub(crate) enum Pace {
    /// Keyframe k at [`crate::order::send_time`]`(k, rate)` after `started`.
    Wall {
        started: tokio::time::Instant,
        rate: f32,
    },
    /// Keyframe k once the server has trained [`step_target`]`(k, …)` steps.
    Lockstep(Lockstep),
}

impl Pace {
    /// Waits until the `k`-th keyframe (0-based, in send order) may go out.
    pub(crate) async fn before_send(&mut self, k: usize) -> anyhow::Result<()> {
        match self {
            Pace::Wall { started, rate } => {
                tokio::time::sleep_until(*started + crate::order::send_time(k, *rate)).await;
                Ok(())
            }
            Pace::Lockstep(lock) => lock.before_send(k).await.map(drop),
        }
    }
}

/// Training steps the server must have done before the `k`-th keyframe goes out: `k · per_keyframe`, rounded.
/// Absolute, so a keyframe that went out late does not shift the later ones.
pub(crate) fn step_target(k: usize, per_keyframe: f32) -> u64 {
    (k as f64 * f64::from(per_keyframe)).round() as u64
}

/// Lockstep pacing on the server's step count (`train_iters` of its status frames, sent after every step with
/// `status_every_step`).
pub(crate) struct Lockstep {
    steps: watch::Receiver<u64>,
    per_keyframe: f32,
    /// How long the step count may stand still before the wait fails (the server stopped training).
    stall: Duration,
    /// The server's step count when each keyframe went out, in send order.
    sent_at: Vec<u64>,
}

/// Steps between consecutive keyframes of a lockstep replay.
#[derive(Debug)]
pub(crate) struct Summary {
    pub(crate) keyframes: usize,
    pub(crate) mean: f64,
    pub(crate) min: u64,
    pub(crate) max: u64,
    /// Keyframes that went out at least one keyframe's worth of steps past their target: the replay, not the
    /// server, was the bottleneck there.
    pub(crate) late: usize,
}

impl Lockstep {
    pub(crate) fn new(steps: watch::Receiver<u64>, per_keyframe: f32, stall: Duration) -> Self {
        Self::with_sent_at(steps, per_keyframe, stall, Vec::new())
    }

    /// Like [`Self::new`] with the step counts of keyframes already sent.
    pub(crate) fn with_sent_at(
        steps: watch::Receiver<u64>,
        per_keyframe: f32,
        stall: Duration,
        sent_at: Vec<u64>,
    ) -> Self {
        Self {
            steps,
            per_keyframe,
            stall,
            sent_at,
        }
    }

    pub(crate) fn per_keyframe(&self) -> f32 {
        self.per_keyframe
    }

    #[cfg(test)]
    pub(crate) fn sent_at(&self) -> &[u64] {
        &self.sent_at
    }

    /// Waits for the `k`-th keyframe's [`step_target`], records the step count it goes out at and returns it.
    pub(crate) async fn before_send(&mut self, k: usize) -> anyhow::Result<u64> {
        let at = self.wait_for(step_target(k, self.per_keyframe)).await?;
        self.sent_at.push(at);
        Ok(at)
    }

    /// Waits until the server reports at least `target` steps and returns its count. Fails when the count does
    /// not move for `stall`, or when the status source ends first.
    pub(crate) async fn wait_for(&mut self, target: u64) -> anyhow::Result<u64> {
        loop {
            let now = *self.steps.borrow_and_update();
            if now >= target {
                return Ok(now);
            }
            match tokio::time::timeout(self.stall, self.steps.changed()).await {
                Ok(Ok(())) => {}
                Ok(Err(_closed)) => {
                    anyhow::bail!("status ended at {now} steps, waiting for {target}")
                }
                Err(_elapsed) => anyhow::bail!(
                    "no training step for {:?} at {now} steps, waiting for {target}: did the server stop training?",
                    self.stall
                ),
            }
        }
    }

    pub(crate) fn summary(&self) -> Summary {
        let gaps: Vec<u64> = self
            .sent_at
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect();
        let late = self
            .sent_at
            .iter()
            .enumerate()
            .filter(|&(k, &at)| {
                at as f64 - step_target(k, self.per_keyframe) as f64 >= f64::from(self.per_keyframe)
            })
            .count();
        Summary {
            keyframes: self.sent_at.len(),
            mean: if gaps.is_empty() {
                0.0
            } else {
                gaps.iter().sum::<u64>() as f64 / gaps.len() as f64
            },
            min: gaps.iter().copied().min().unwrap_or(0),
            max: gaps.iter().copied().max().unwrap_or(0),
            late,
        }
    }
}

/// The items of a channel in order, each handed out once its [`Pace`] lets it go.
pub(crate) struct Paced<T> {
    items: mpsc::Receiver<T>,
    pace: Pace,
    /// Items handed out so far.
    sent: usize,
}

impl<T> Paced<T> {
    pub(crate) fn new(items: mpsc::Receiver<T>, pace: Pace) -> Self {
        Self {
            items,
            pace,
            sent: 0,
        }
    }

    /// The next item once the pace lets it go (it is taken from the channel first, so preparing it overlaps the
    /// wait); `None` when the channel is done.
    pub(crate) async fn next(&mut self) -> Option<anyhow::Result<T>> {
        let item = self.items.recv().await?;
        if let Err(e) = self.pace.before_send(self.sent).await {
            return Some(Err(e));
        }
        self.sent += 1;
        Some(Ok(item))
    }

    pub(crate) fn sent(&self) -> usize {
        self.sent
    }

    pub(crate) fn into_pace(self) -> Pace {
        self.pace
    }
}

#[cfg(test)]
#[path = "pace_tests.rs"]
mod tests;
