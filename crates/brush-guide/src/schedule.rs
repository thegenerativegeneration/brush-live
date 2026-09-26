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

/// Newest views always scored in a round, when the capture exceeds the limit.
pub const SCORE_RECENT_VIEWS: usize = 40;

/// Indices (ascending) of at most `max_views` of `num_views` views to score:
/// the newest `SCORE_RECENT_VIEWS` plus one seeded random pick from each of
/// `max_views - SCORE_RECENT_VIEWS` equal strata over the older views, so a
/// round's cost stays bounded as the capture grows.
pub fn select_score_views(num_views: usize, max_views: usize, seed: u64) -> Vec<usize> {
    use rand::{RngExt as _, SeedableRng};
    let max_views = max_views.max(1);
    if num_views <= max_views {
        return (0..num_views).collect();
    }
    let recent = SCORE_RECENT_VIEWS.min(max_views);
    let older = num_views - recent;
    let strata = max_views - recent;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..strata)
        .map(|j| rng.random_range(j * older / strata..(j + 1) * older / strata))
        .chain(older..num_views)
        .collect()
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

    #[test]
    fn short_captures_score_every_view() {
        assert_eq!(select_score_views(30, 120, 7), (0..30).collect::<Vec<_>>());
        assert_eq!(
            select_score_views(120, 120, 7),
            (0..120).collect::<Vec<_>>()
        );
    }

    #[test]
    fn long_captures_keep_newest_and_stratify_the_rest() {
        let picked = select_score_views(500, 120, 7);
        assert_eq!(picked.len(), 120);
        assert!(picked.windows(2).all(|w| w[0] < w[1]), "sorted, unique");
        assert!(
            (460..500).all(|i| picked.contains(&i)),
            "newest 40 included"
        );
        // 80 strata over the 460 older views: one pick in each.
        let older = &picked[..80];
        for (j, &i) in older.iter().enumerate() {
            assert!((j * 460 / 80..(j + 1) * 460 / 80).contains(&i), "{j}: {i}");
        }
        assert_eq!(picked, select_score_views(500, 120, 7));
        assert_ne!(picked, select_score_views(500, 120, 8));
    }

    #[test]
    fn tiny_limit_keeps_only_newest() {
        assert_eq!(
            select_score_views(100, 10, 0),
            (90..100).collect::<Vec<_>>()
        );
    }
}
