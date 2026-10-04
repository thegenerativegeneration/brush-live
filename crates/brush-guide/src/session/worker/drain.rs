//! How many queued commands the worker takes between two training steps.
//! Ingest (JPEG decode, a quarter-size render with an alpha readback for
//! seeding, the splat append) runs between steps, so with one command per
//! step it falls behind whenever steps get slow.

use super::super::Command;
use std::time::Duration;
use tokio::sync::mpsc;
use web_time::Instant;

/// Commands handled per loop iteration at most.
const MAX_DRAIN: u32 = 16;
/// Bounds of the drain's time budget, which is the last training step's
/// duration: ingest gets at most about as much wall time as training while
/// a backlog lasts, whatever the step rate.
const MIN_BUDGET: Duration = Duration::from_millis(100);
const MAX_BUDGET: Duration = Duration::from_secs(1);

/// How many queued commands one loop iteration takes: up to [`MAX_DRAIN`],
/// and none once its budget has passed since the first was taken.
pub(super) struct Drain {
    start: Instant,
    budget: Duration,
    count: u32,
}

impl Drain {
    pub(super) fn new(start: Instant, last_step: Duration) -> Self {
        Self {
            start,
            budget: last_step.clamp(MIN_BUDGET, MAX_BUDGET),
            count: 0,
        }
    }

    pub(super) fn handled(&mut self) {
        self.count += 1;
    }

    pub(super) fn more(&self, now: Instant) -> bool {
        self.count < MAX_DRAIN && now.duration_since(self.start) < self.budget
    }
}

/// The next queued command without waiting; `Err` once the session is gone.
pub(super) fn try_take(
    rx: &mut mpsc::Receiver<Command>,
) -> Result<Option<Command>, mpsc::error::TryRecvError> {
    match rx.try_recv() {
        Ok(c) => Ok(Some(c)),
        Err(mpsc::error::TryRecvError::Empty) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_commands_until_the_cap() {
        let t0 = Instant::now();
        let mut d = Drain::new(t0, Duration::ZERO);
        assert!(d.more(t0), "nothing handled yet");
        for _ in 0..MAX_DRAIN - 1 {
            d.handled();
            assert!(d.more(t0));
        }
        d.handled();
        assert!(!d.more(t0), "cap reached");
    }

    #[test]
    fn budget_follows_the_last_step_within_bounds() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut d = Drain::new(t0, ms(10));
        d.handled();
        assert!(d.more(t0 + ms(99)), "fast steps: the minimum budget");
        assert!(!d.more(t0 + MIN_BUDGET));
        let d = Drain::new(t0, ms(600));
        assert!(d.more(t0 + ms(599)));
        assert!(!d.more(t0 + ms(600)));
        let d = Drain::new(t0, Duration::from_secs(5));
        assert!(!d.more(t0 + MAX_BUDGET), "slow steps: the maximum budget");
    }
}
