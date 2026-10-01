//! When the worker runs its two kinds of round, and which views a Fisher
//! pass scores.
//!
//! The voxel round (cells from the splat parameters, TSDF fusion, meshing)
//! and the Fisher pass (render + backward per view) each have a
//! [`Cadence`]: a minimum start-to-start interval and a share of wall time.

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

    pub fn has_run(&self) -> bool {
        self.last.is_some()
    }

    /// Duration of the last round; 0 before the first.
    pub fn last_duration(&self) -> f64 {
        self.last.map_or(0.0, |(_, d)| d)
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

/// What the worker should run now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    Voxel,
    /// A Fisher pass over at most this many views.
    Fisher(usize),
}

/// Predicted cost of a Fisher pass, from the last one.
#[derive(Clone, Copy, Debug)]
pub struct FisherCost {
    pub per_view_s: f64,
    /// Readbacks and CPU work that do not grow with the view count.
    pub fixed_s: f64,
}

/// Keeps time between the voxel rounds clear: a due Fisher pass waits for a
/// gap before the next voxel round that holds nearly as many views as the
/// gap right after a voxel round (at least `min_views`, at most
/// `max_views`), then takes as many views as fit there.
/// A pass that has waited a whole extra interval runs anyway, with
/// `min_views`, so uncertainty keeps updating on a machine too slow for both.
/// The first pass of a session runs before the first voxel round, so the
/// first score set already carries uncertainty.
pub struct RoundScheduler {
    pub voxel: Cadence,
    pub fisher: Cadence,
    max_views: usize,
    min_views: usize,
}

/// Slack left before a voxel round is due when fitting a Fisher pass.
const FIT_MARGIN_S: f64 = 0.1;
/// A Fisher pass waits for a gap holding at least the views that fit in
/// this share of the time between two voxel rounds.
const SLOT_SHARE: f64 = 0.75;

impl RoundScheduler {
    pub fn new(voxel: Cadence, fisher: Cadence, max_views: usize, min_views: usize) -> Self {
        let max_views = max_views.max(1);
        Self {
            voxel,
            fisher,
            max_views,
            min_views: min_views.clamp(1, max_views),
        }
    }

    pub fn next(&self, now_s: f64, cost: Option<FisherCost>) -> Option<Round> {
        if !self.fisher.has_run() {
            return Some(Round::Fisher(self.max_views));
        }
        if self.voxel.due(now_s) {
            return Some(Round::Voxel);
        }
        if !self.fisher.due(now_s) {
            return None;
        }
        let Some(cost) = cost else {
            return Some(Round::Fisher(self.max_views));
        };
        let fit = |gap: f64| -> usize {
            let room = gap - FIT_MARGIN_S - cost.fixed_s;
            if room <= 0.0 {
                0
            } else if cost.per_view_s <= 0.0 {
                self.max_views
            } else {
                (room / cost.per_view_s).floor().min(self.max_views as f64) as usize
            }
        };
        // The best slot is right after a voxel round; noticing it takes a
        // training step, hence the slack.
        let slot = self.voxel.interval() - self.voxel.last_duration();
        let target = fit(SLOT_SHARE * slot).clamp(self.min_views, self.max_views);
        let now = fit(self.voxel.next_due() - now_s);
        if now >= target {
            return Some(Round::Fisher(now));
        }
        let starving = now_s >= self.fisher.next_due() + self.fisher.interval();
        starving.then_some(Round::Fisher(self.min_views))
    }
}

/// Which views a Fisher pass of `max_views` views scores: the newest
/// `recent = max_views / 3` every pass, plus one view from each of
/// `max_views − recent` equal strata over the older views. Within a stratum
/// the pick rotates with the pass number from a seeded offset, so consecutive
/// passes take different views and `len` consecutive passes over an unchanged
/// stratum of `len` views visit each view once.
#[derive(Clone, Copy, Debug)]
pub struct ViewSample {
    pub max_views: usize,
    pub recent: usize,
}

impl ViewSample {
    pub fn new(max_views: usize) -> Self {
        let max_views = max_views.max(1);
        Self {
            max_views,
            recent: (max_views / 3).max(1),
        }
    }

    fn strata(&self, num_views: usize) -> (usize, usize) {
        let older = num_views.saturating_sub(self.recent);
        (older, self.max_views - self.recent)
    }

    /// Indices (ascending) of the views pass `pass` scores out of `num_views`.
    pub fn select(&self, num_views: usize, pass: u64, seed: u64) -> Vec<usize> {
        if num_views <= self.max_views {
            return (0..num_views).collect();
        }
        let (older, strata) = self.strata(num_views);
        (0..strata)
            .filter_map(|j| {
                let (a, b) = (j * older / strata, (j + 1) * older / strata);
                let len = (b - a) as u64;
                (len > 0).then(|| {
                    let offset = splitmix64(seed ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    a + (offset.wrapping_add(pass) % len) as usize
                })
            })
            .chain(older..num_views)
            .collect()
    }

    /// Horvitz–Thompson weight of view `index` when picked by [`Self::select`]:
    /// 1 for a view scored in every pass, otherwise the size of its stratum.
    /// Over the seeded offset the pick is uniform within its stratum, so the
    /// view's chance of being scored in a pass is 1 / size and the weighted
    /// sums of one pass estimate sums over every view without bias.
    pub fn weight(&self, index: usize, num_views: usize) -> f32 {
        let (older, strata) = self.strata(num_views);
        if num_views <= self.max_views || index >= older {
            return 1.0;
        }
        if strata == 0 {
            return 0.0;
        }
        // Stratum j covers j·older/strata .. (j+1)·older/strata.
        let j = ((index + 1) * strata).div_ceil(older) - 1;
        ((j + 1) * older / strata - j * older / strata) as f32
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests;
