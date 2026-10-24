use crate::protocol::StepPhases;
use brush_train::profile::StepProfile;

/// With debug timing on, the step profile is logged every this many steps.
pub const PROFILE_LOG_EVERY: u32 = 100;

/// Seconds behind [`StepPhases`] that are not refine.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct StepSeconds {
    batch: f64,
    forward: f64,
    loss: f64,
    backward: f64,
    optimizer: f64,
    stats: f64,
    pub(super) area: f64,
    pub(super) hold: f64,
}

impl StepSeconds {
    pub(super) fn add_profile(&mut self, p: &StepProfile, batch_s: f64) {
        self.batch += batch_s;
        self.forward += p.forward_s;
        self.loss += p.loss_s;
        self.backward += p.backward_s;
        self.optimizer += p.optimizer_s;
        self.stats += p.stats_noise_s;
    }

    /// Cumulative phase times in whole milliseconds (rounded down), with refine from `refine_total_s`.
    pub(super) fn phases(&self, refine_total_s: f64) -> StepPhases {
        let ms = |s: f64| (s * 1e3) as u64;
        StepPhases {
            batch_ms: ms(self.batch),
            forward_ms: ms(self.forward),
            loss_ms: ms(self.loss),
            backward_ms: ms(self.backward),
            optimizer_ms: ms(self.optimizer),
            stats_ms: ms(self.stats),
            refine_ms: ms(refine_total_s),
            area_ms: ms(self.area),
            hold_ms: ms(self.hold),
        }
    }
}

/// Logs the mean per-step phase times of `p` (taken since the last log) at debug level.
pub(super) fn log_step_profile(p: &StepProfile, batch_s: f64, num_splats: u32) {
    let ms = |s: f64| s * 1e3 / f64::from(p.steps);
    log::debug!(
        target: crate::timing::TARGET,
        "step profile over {} steps at {} splats: batch {:.2} ms, forward {:.2}, loss {:.2}, backward {:.2}, optimizer {:.2}, stats+noise {:.2} ms",
        p.steps,
        num_splats,
        ms(batch_s),
        ms(p.forward_s),
        ms(p.loss_s),
        ms(p.backward_s),
        ms(p.optimizer_s),
        ms(p.stats_noise_s)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_rounds_seconds_down_to_whole_ms() {
        let s = StepSeconds {
            batch: 0.0019,
            forward: 1.25,
            area: 2.0,
            hold: 0.0005,
            ..StepSeconds::default()
        };
        let p = s.phases(0.0101);
        assert_eq!(p.batch_ms, 1);
        assert_eq!(p.forward_ms, 1250);
        assert_eq!(p.area_ms, 2000);
        assert_eq!(p.hold_ms, 0);
        assert_eq!(p.refine_ms, 10);
        assert_eq!(p.loss_ms + p.backward_ms + p.optimizer_ms + p.stats_ms, 0);
    }
}
