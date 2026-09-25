/// Spaces scoring rounds so they take at most `budget` of wall time, and never
/// run closer together than `min_interval_s`.
pub struct ScoreScheduler {
    budget: f32,
    min_interval_s: f32,
    /// (end, duration) of the last round.
    last: Option<(f64, f64)>,
}

impl ScoreScheduler {
    pub fn new(budget: f32, min_interval_s: f32) -> Self {
        Self {
            budget,
            min_interval_s,
            last: None,
        }
    }

    pub fn due(&self, now_s: f64) -> bool {
        let Some((end, duration)) = self.last else {
            return true;
        };
        let budget = self.budget.clamp(0.01, 1.0) as f64;
        let wait = (self.min_interval_s as f64).max(duration * (1.0 - budget) / budget);
        now_s - end >= wait
    }

    pub fn record(&mut self, end_s: f64, duration_s: f64) {
        self.last = Some((end_s, duration_s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn due_immediately_then_respects_min_interval() {
        let mut s = ScoreScheduler::new(0.25, 2.0);
        assert!(s.due(0.0));
        s.record(1.0, 0.1);
        assert!(!s.due(2.5));
        assert!(s.due(3.0));
    }

    #[test]
    fn slow_rounds_stretch_the_interval_to_the_budget() {
        let mut s = ScoreScheduler::new(0.25, 2.0);
        s.record(10.0, 4.0); // 4 s of scoring at 25 % budget -> wait 12 s
        assert!(!s.due(21.9));
        assert!(s.due(22.0));
    }
}
