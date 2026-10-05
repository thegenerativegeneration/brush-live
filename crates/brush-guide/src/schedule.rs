//! When the worker runs a voxel round, and how fast training steps.
//!
//! The voxel round (cells from the splat parameters) has a [`Cadence`]: a
//! minimum start-to-start interval and a share of wall time.

/// Start-to-start spacing of one kind of round: a round starts no sooner than
/// `min_interval_s` after the previous one started, and no sooner than its
/// duration divided by `budget`, so rounds take at most `budget` of wall time.
#[derive(Clone, Debug)]
pub struct Cadence {
    budget: f64,
    min_interval_s: f64,
    /// (start, duration) of the last round.
    last: Option<(f64, f64)>,
}

impl Cadence {
    pub fn new(budget: f32, min_interval_s: f32) -> Self {
        Self {
            budget: f64::from(budget.clamp(0.01, 1.0)),
            min_interval_s: f64::from(min_interval_s.max(0.0)),
            last: None,
        }
    }

    /// Start-to-start interval after the last round; 0 before the first.
    pub fn interval(&self) -> f64 {
        self.last
            .map_or(0.0, |(_, d)| self.min_interval_s.max(d / self.budget))
    }

    /// Earliest start of the next round; −∞ before the first.
    pub fn next_due(&self) -> f64 {
        self.last
            .map_or(f64::NEG_INFINITY, |(start, _)| start + self.interval())
    }

    pub fn due(&self, now_s: f64) -> bool {
        now_s >= self.next_due()
    }

    /// Records a round that started at `start_s`. A start less than half an
    /// interval after it was due counts as on time, so the time the worker
    /// needs to notice a round is due (a training step) does not add up
    /// into the cadence.
    pub fn record(&mut self, start_s: f64, duration_s: f64) {
        let late = start_s - self.next_due();
        let start = if self.last.is_some() && late > 0.0 && late < 0.5 * self.interval() {
            start_s - late
        } else {
            start_s
        };
        self.last = Some((start, duration_s.max(0.0)));
    }
}

/// Caps training at `max_per_s` iterations per second (0: uncapped) by
/// spacing steps at least `1 / max_per_s` apart. Time spent elsewhere is not
/// banked, so a step after a long round runs at once but no burst follows.
pub struct IterThrottle {
    min_gap_s: Option<f64>,
    last_step_s: Option<f64>,
}

impl IterThrottle {
    pub fn new(max_per_s: f32) -> Self {
        Self {
            min_gap_s: (max_per_s > 0.0).then(|| 1.0 / f64::from(max_per_s)),
            last_step_s: None,
        }
    }

    /// Seconds to wait before the next step may run, if any.
    pub fn wait_s(&self, now_s: f64) -> Option<f64> {
        let wait = self.last_step_s? + self.min_gap_s? - now_s;
        (wait > 0.0).then_some(wait)
    }

    pub fn record_step(&mut self, now_s: f64) {
        self.last_step_s = Some(now_s);
    }
}

#[cfg(test)]
mod tests;
