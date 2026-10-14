//! When held-out evals run: every N received keyframes, or every few seconds; extras at checkpoints.

/// Seconds before a checkpoint whose eval scored nothing is tried again.
const CHECKPOINT_RETRY_S: f64 = 1.0;

/// Decides when an eval is due and whether it carries the extras (LPIPS, seam split, saved renders).
pub(super) struct EvalSchedule {
    every_keyframes: u32,
    interval_s: f64,
    /// Next received-keyframe count that makes an eval due (keyframe mode).
    next: u32,
    /// Checkpoints not yet consumed by an eval that scored views, ascending.
    extras_at: Vec<u32>,
    last_eval_s: Option<f64>,
    /// When a pending checkpoint may be tried again after an eval that scored nothing.
    retry_at_s: Option<f64>,
}

impl EvalSchedule {
    /// `every_keyframes` 0 evals every `interval_s` seconds instead.
    pub(super) fn new(every_keyframes: u32, interval_s: f32, mut extras_at: Vec<u32>) -> Self {
        extras_at.sort_unstable();
        Self {
            every_keyframes,
            interval_s: f64::from(interval_s),
            next: every_keyframes,
            extras_at,
            last_eval_s: None,
            retry_at_s: None,
        }
    }

    /// `Some(extras)` when an eval is due at `received` keyframes and `now_s` seconds. Changes no state: the caller
    /// reports the eval with [`Self::record`].
    pub(super) fn due(&self, received: u32, now_s: f64) -> Option<bool> {
        let checkpoint = self.checkpoint_pending(received) && self.retry_at_s.is_none_or(|t| now_s >= t);
        let periodic = if self.every_keyframes > 0 {
            received >= self.next
        } else {
            self.last_eval_s.is_none_or(|last| now_s - last >= self.interval_s)
        };
        (checkpoint || periodic).then_some(checkpoint)
    }

    /// Marks the eval `due` asked for as done. The periodic schedule moves on either way; checkpoints up to
    /// `received` are consumed only by an eval with `extras` that `scored` views (no splats yet, or no view
    /// rendering, keeps them pending, tried again after [`CHECKPOINT_RETRY_S`]).
    pub(super) fn record(&mut self, received: u32, now_s: f64, extras: bool, scored: bool) {
        if extras && scored {
            self.extras_at.retain(|&c| c > received);
            self.retry_at_s = None;
        } else if extras && self.checkpoint_pending(received) {
            self.retry_at_s = Some(now_s + CHECKPOINT_RETRY_S);
        }
        if self.every_keyframes > 0 && received >= self.next {
            self.next = (received / self.every_keyframes + 1) * self.every_keyframes;
        }
        self.last_eval_s = Some(now_s);
    }

    fn checkpoint_pending(&self, received: u32) -> bool {
        self.extras_at.first().is_some_and(|&c| c <= received)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `due`, and when an eval is due, `record` it as one that scored views.
    fn step(s: &mut EvalSchedule, received: u32, now_s: f64) -> Option<bool> {
        let due = s.due(received, now_s);
        if let Some(extras) = due {
            s.record(received, now_s, extras, true);
        }
        due
    }

    #[test]
    fn keyframe_mode_evals_every_n_received_and_extras_at_checkpoints() {
        let mut s = EvalSchedule::new(50, 10.0, vec![117, 413]);
        assert_eq!(step(&mut s, 49, 0.0), None);
        assert_eq!(step(&mut s, 50, 0.0), Some(false));
        assert_eq!(step(&mut s, 51, 0.0), None, "once per multiple");
        assert_eq!(step(&mut s, 118, 0.0), Some(true), "first eval past checkpoint 117 carries the extras");
        assert_eq!(step(&mut s, 150, 0.0), Some(false));
        // A backlog jumps past several multiples and a checkpoint at once: one eval, with extras.
        assert_eq!(step(&mut s, 420, 0.0), Some(true));
        assert_eq!(step(&mut s, 449, 0.0), None);
        assert_eq!(step(&mut s, 450, 0.0), Some(false));
    }

    #[test]
    fn time_mode_keeps_todays_interval() {
        let mut s = EvalSchedule::new(0, 30.0, vec![]);
        assert_eq!(step(&mut s, 1, 0.0), Some(false));
        assert_eq!(step(&mut s, 5, 29.0), None);
        assert_eq!(step(&mut s, 9, 30.0), Some(false));
    }

    #[test]
    fn a_checkpoint_stays_pending_until_an_eval_scores_views() {
        let mut s = EvalSchedule::new(50, 10.0, vec![21]);
        assert_eq!(s.due(21, 0.0), Some(true));
        assert_eq!(s.due(21, 0.0), Some(true), "due changes no state");
        // No splats yet, or no view rendered: nothing scored.
        s.record(21, 0.0, true, false);
        assert_eq!(s.due(22, 0.5), None, "retried no sooner than a second later");
        assert_eq!(s.due(23, 1.0), Some(true), "still pending");
        s.record(23, 1.0, true, true);
        assert_eq!(s.due(24, 1.0), None, "consumed by the eval that scored");
        // A skipped periodic eval is not retried: the next one is at the next multiple.
        assert_eq!(s.due(50, 2.0), Some(false));
        s.record(50, 2.0, false, false);
        assert_eq!(s.due(51, 9.0), None);
        assert_eq!(s.due(100, 9.0), Some(false));
    }

    #[test]
    fn a_periodic_eval_inside_the_retry_window_leaves_the_checkpoint_pending() {
        let mut s = EvalSchedule::new(50, 10.0, vec![48]);
        assert_eq!(s.due(48, 0.0), Some(true));
        s.record(48, 0.0, true, false);
        // The periodic eval at 50 falls inside the retry window: no extras, and it scores.
        assert_eq!(s.due(50, 0.5), Some(false));
        s.record(50, 0.5, false, true);
        assert_eq!(s.due(51, 1.0), Some(true), "the checkpoint is still pending after the window");
    }
}
