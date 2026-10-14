//! When held-out evals run: every N received keyframes, or every few seconds; extras at checkpoints.

/// Decides when an eval is due and whether it carries the extras (LPIPS, seam split, saved renders).
pub(super) struct EvalSchedule {
    every_keyframes: u32,
    interval_s: f64,
    /// Next received-keyframe count that makes an eval due (keyframe mode).
    next: u32,
    /// Checkpoints not yet consumed, ascending.
    extras_at: Vec<u32>,
    last_eval_s: Option<f64>,
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
        }
    }

    /// `Some(extras)` when an eval is due at `received` keyframes and `now_s` seconds; marks it done.
    pub(super) fn due(&mut self, received: u32, now_s: f64) -> Option<bool> {
        let checkpoint = self.extras_at.first().is_some_and(|&c| c <= received);
        let periodic = if self.every_keyframes > 0 {
            received >= self.next
        } else {
            self.last_eval_s.is_none_or(|last| now_s - last >= self.interval_s)
        };
        if !checkpoint && !periodic {
            return None;
        }
        if checkpoint {
            self.extras_at.retain(|&c| c > received);
        }
        if self.every_keyframes > 0 && received >= self.next {
            self.next = (received / self.every_keyframes + 1) * self.every_keyframes;
        }
        self.last_eval_s = Some(now_s);
        Some(checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframe_mode_evals_every_n_received_and_extras_at_checkpoints() {
        let mut s = EvalSchedule::new(50, 10.0, vec![117, 413]);
        assert_eq!(s.due(49, 0.0), None);
        assert_eq!(s.due(50, 0.0), Some(false));
        assert_eq!(s.due(51, 0.0), None, "once per multiple");
        assert_eq!(s.due(118, 0.0), Some(true), "first eval past checkpoint 117 carries the extras");
        assert_eq!(s.due(150, 0.0), Some(false));
        // A backlog jumps past several multiples and a checkpoint at once: one eval, with extras.
        assert_eq!(s.due(420, 0.0), Some(true));
        assert_eq!(s.due(449, 0.0), None);
        assert_eq!(s.due(450, 0.0), Some(false));
    }

    #[test]
    fn time_mode_keeps_todays_interval() {
        let mut s = EvalSchedule::new(0, 30.0, vec![]);
        assert_eq!(s.due(1, 0.0), Some(false));
        assert_eq!(s.due(5, 29.0), None);
        assert_eq!(s.due(9, 30.0), Some(false));
    }
}
