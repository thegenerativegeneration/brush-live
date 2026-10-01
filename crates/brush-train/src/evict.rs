//! Eviction keeps an open-ended (live) training run within `max_splats`.
//!
//! When refine wants to grow or split past the budget, or new keyframe seeds
//! were dropped for lack of room, the least important splats in the whole
//! model are pruned down to `max_splats · (1 − headroom)`. Splitting and
//! growth may then fill up to `max_splats · (1 − headroom / 2)`; the last
//! `headroom / 2` stays free for seeding new keyframes.
//!
//! Importance is supplied from outside (`set_importance`) per splat and is
//! carried through append, prune and split like the optimizer state. Splats
//! that were never scored, are younger than `min_age` refines, or lie in the
//! protected view cone are never evicted. Within one eviction no 1 m cell
//! loses more than `max_cell_fraction` of its splats, so a spatially
//! correlated low score cannot wipe a region in one go, and a new importance
//! set must arrive between two evictions so each one ranks fresh scores.

use crate::config::TrainConfig;
use crate::lod::top_k_indices;
use crate::stats::RefineRecord;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct EvictConfig {
    /// Fraction of `max_splats` freed by one eviction.
    pub headroom: f32,
    /// Refines a splat must survive before it can be evicted.
    pub min_age: u32,
    /// Most of a 1 m cell's splats one eviction may take.
    pub max_cell_fraction: f32,
}

/// Edge of the cells `EvictConfig::max_cell_fraction` applies to, in metres.
pub const EVICT_CELL_M: f32 = 1.0;

/// A view cone (apex, unit axis, cos of the half angle) whose splats are
/// never evicted, e.g. the newest keyframe's camera.
#[derive(Clone, Copy, Debug)]
pub struct ProtectCone {
    pub position: glam::Vec3,
    pub forward: glam::Vec3,
    pub cos_half_fov: f32,
}

/// The count eviction prunes down to.
pub fn evict_target(max_splats: u32, headroom: f32) -> u32 {
    let free = (max_splats as f64 * f64::from(headroom.clamp(0.0, 1.0))).round() as u32;
    max_splats.saturating_sub(free)
}

/// The count splitting and growth may fill up to while eviction is on.
pub fn growth_limit(max_splats: u32, headroom: f32) -> u32 {
    let reserve = (max_splats as f64 * f64::from(headroom.clamp(0.0, 1.0)) / 2.0).round() as u32;
    max_splats.saturating_sub(reserve)
}

/// Splats to evict at a refine with `current` splats, when split and growth
/// want `demand` more and `seed_shortfall` seeds were dropped since the last
/// refine. Zero unless growth is blocked.
pub fn evict_count(
    current: u32,
    demand: u32,
    seed_shortfall: u32,
    max_splats: u32,
    headroom: f32,
) -> u32 {
    let blocked = (demand > 0
        && current.saturating_add(demand) > growth_limit(max_splats, headroom))
        || seed_shortfall > 0;
    if blocked {
        current.saturating_sub(evict_target(max_splats, headroom))
    } else {
        0
    }
}

/// Per-splat age (refines survived) and importance, row-aligned with the splats.
pub(crate) struct SplatLife {
    pub age: Tensor<1>,
    /// `+inf` until scored.
    pub importance: Tensor<1>,
}

impl SplatLife {
    pub(crate) fn new(n: usize, device: &Device) -> Self {
        Self {
            age: Tensor::zeros([n], device),
            importance: Tensor::full([n], f32::INFINITY, device),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.age.dims()[0]
    }

    pub(crate) fn pad(&mut self, n: usize) {
        let device = self.age.device();
        self.age = Tensor::cat(vec![self.age.clone(), Tensor::zeros([n], &device)], 0);
        self.importance = Tensor::cat(
            vec![
                self.importance.clone(),
                Tensor::full([n], f32::INFINITY, &device),
            ],
            0,
        );
    }

    pub(crate) fn keep(&mut self, indices: Tensor<1, Int>) {
        self.age = self.age.clone().select(0, indices.clone());
        self.importance = self.importance.clone().select(0, indices);
    }

    /// Appends one child per parent index: age 0, the parent's importance.
    pub(crate) fn split(&mut self, parents: Tensor<1, Int>) {
        let n = parents.dims()[0];
        let device = self.age.device();
        let inherited = self.importance.clone().select(0, parents);
        self.age = Tensor::cat(vec![self.age.clone(), Tensor::zeros([n], &device)], 0);
        self.importance = Tensor::cat(vec![self.importance.clone(), inherited], 0);
    }

    /// Takes new scores; NaN keeps a splat's previous score, or 0 if it was
    /// never scored (+inf).
    pub(crate) fn set_importance(&mut self, values: &[f32]) {
        let device = self.importance.device();
        let new: Tensor<1> =
            Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), &device);
        let old = self.importance.clone();
        let fallback = old.clone().mask_fill(old.is_inf(), 0.0);
        self.importance = new.clone().mask_where(new.is_nan(), fallback);
    }

    pub(crate) fn tick(&mut self) {
        self.age = self.age.clone() + 1.0;
    }
}

/// Eviction settings plus the state the trainer carries between refines.
pub(crate) struct Eviction {
    pub config: EvictConfig,
    pub life: Option<SplatLife>,
    pub protect: Option<ProtectCone>,
    pub seed_shortfall: u32,
    /// An importance set arrived since the last eviction.
    pub fresh: bool,
}

impl Eviction {
    pub(crate) fn new(config: EvictConfig) -> Self {
        Self {
            config,
            life: None,
            protect: None,
            seed_shortfall: 0,
            fresh: false,
        }
    }
}

async fn count_true(mask: Tensor<1, Bool>) -> u32 {
    mask.int()
        .sum()
        .into_scalar_async::<i32>()
        .await
        .expect("mask count readback") as u32
}

/// How many splats this refine's force-split and growth would add if the
/// budget allowed it, with `num_dead` dead splats about to be replaced.
pub(crate) async fn growth_demand(
    refiner: &RefineRecord,
    config: &TrainConfig,
    iter: u32,
    num_dead: u32,
) -> u32 {
    let mut demand = 0;
    if config.split_at_screen_size > 0.0 {
        demand += count_true(refiner.above_screen_size(config.split_at_screen_size)).await;
    }
    if iter >= config.growth_start_iter && iter < config.growth_stop_iter {
        let above = count_true(refiner.above_threshold(config.growth_grad_threshold)).await;
        let grow = (above as f32 * config.growth_select_fraction).round() as u32;
        demand += grow.saturating_sub(num_dead);
    }
    demand
}

/// True for splats whose mean lies inside `cone`.
fn in_cone(means: Tensor<2>, cone: ProtectCone) -> Tensor<1, Bool> {
    let device = means.device();
    let n = means.dims()[0];
    let p = cone.position;
    let f = cone.forward.normalize_or_zero();
    let apex = Tensor::<1>::from_floats([p.x, p.y, p.z], &device).reshape([1, 3]);
    let axis = Tensor::<1>::from_floats([f.x, f.y, f.z], &device).reshape([3, 1]);
    let d = means - apex;
    let len = d.clone().powi_scalar(2).sum_dim(1).sqrt().reshape([n]);
    let along = d.matmul(axis).reshape([n]);
    along.greater_equal(len.mul_scalar(cone.cos_half_fov))
}

/// Picks up to `count` splats to evict, lowest importance first, among
/// those that are scored, at least `min_age` refines old, not in `exclude`
/// and not in `protect`, taking at most `max_cell_fraction` of any 1 m cell.
/// Returns the eviction mask and how many it holds.
pub(crate) async fn select_evictions(
    life: &SplatLife,
    means: Tensor<2>,
    exclude: Tensor<1, Bool>,
    protect: Option<ProtectCone>,
    config: &EvictConfig,
    count: u32,
) -> (Tensor<1, Bool>, u32) {
    let n = life.len();
    let device = life.age.device();
    let mut eligible = life
        .age
        .clone()
        .greater_equal_elem(config.min_age as f32)
        .bool_and(life.importance.clone().is_finite())
        .bool_and(exclude.bool_not());
    if let Some(cone) = protect {
        eligible = eligible.bool_and(in_cone(means.clone(), cone).bool_not());
    }
    let ranked = life
        .importance
        .clone()
        .mask_fill(eligible.bool_not(), f32::INFINITY);
    let scores = read_f32(ranked).await;
    let means = read_f32(means).await;
    let picked = pick_evictions(&scores, &means, count as usize, config.max_cell_fraction);
    let mut evict = vec![false; n];
    for &i in &picked {
        evict[i] = true;
    }
    let mask = Tensor::<1, Bool>::from_data(TensorData::new(evict, [n]), &device);
    (mask, picked.len() as u32)
}

async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("evict readback")
        .try_into_vec::<f32>()
        .expect("f32 readback")
}

/// Up to `count` indices in ascending `scores` order, skipping infinite
/// (ineligible) ones and those whose [`EVICT_CELL_M`] cell (by the flat xyz
/// `means`) already gave `max_cell_fraction` of its splats, rounded down.
pub(crate) fn pick_evictions(
    scores: &[f32],
    means: &[f32],
    count: usize,
    max_cell_fraction: f32,
) -> Vec<usize> {
    let cell = |i: usize| {
        let m = glam::Vec3::from_slice(&means[i * 3..i * 3 + 3]);
        (m / EVICT_CELL_M).floor().as_ivec3()
    };
    let mut quota: hashbrown::HashMap<glam::IVec3, usize> = hashbrown::HashMap::new();
    for i in 0..scores.len() {
        *quota.entry(cell(i)).or_default() += 1;
    }
    let fraction = f64::from(max_cell_fraction.clamp(0.0, 1.0));
    for q in quota.values_mut() {
        *q = (*q as f64 * fraction + 1e-6).floor() as usize;
    }
    let negated: Vec<f32> = scores.iter().map(|s| -s).collect();
    let mut picked = Vec::with_capacity(count);
    for i in top_k_indices(&negated, scores.len()) {
        if picked.len() == count || !scores[i].is_finite() {
            break;
        }
        let q = quota.get_mut(&cell(i)).expect("every splat has a cell");
        if *q > 0 {
            *q -= 1;
            picked.push(i);
        }
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_lowest_within_the_cell_quota() {
        // Cell (0,0,0): 10 splats with the lowest scores 0..10. Cell
        // (1,0,0): 10 splats with scores 100..110. Two ineligible (+inf).
        let mut scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
        scores.extend((0..10).map(|i| 100.0 + i as f32));
        scores.extend([f32::INFINITY; 2]);
        let mut means = Vec::new();
        for i in 0..22 {
            let x = if (10..20).contains(&i) { 1.5 } else { 0.5 };
            means.extend([x, 0.5, 0.5]);
        }
        // Without a cap the six lowest all come from the first cell.
        let mut all = pick_evictions(&scores, &means, 6, 1.0);
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);
        // At 30 % the first cell (12 splats incl. the 2 ineligible) gives 3,
        // the rest comes from the second cell's lowest.
        let mut capped = pick_evictions(&scores, &means, 6, 0.3);
        capped.sort_unstable();
        assert_eq!(capped, vec![0, 1, 2, 10, 11, 12]);
        // Ineligible splats are never picked, even if the count is not met.
        let few = pick_evictions(&scores, &means, 50, 1.0);
        assert_eq!(few.len(), 20);
        assert!(few.iter().all(|&i| i < 20));
    }

    #[test]
    fn target_and_limit_split_the_headroom() {
        assert_eq!(evict_target(100_000, 0.1), 90_000);
        assert_eq!(growth_limit(100_000, 0.1), 95_000);
        assert_eq!(evict_target(300_000, 0.1), 270_000);
        assert_eq!(growth_limit(300_000, 0.1), 285_000);
        assert_eq!(evict_target(15, 0.1), 13);
        assert_eq!(growth_limit(15, 0.1), 14);
    }

    #[test]
    fn evicts_to_target_only_when_blocked() {
        // Room left for the demand: nothing.
        assert_eq!(evict_count(90_000, 5_000, 0, 100_000, 0.1), 0);
        // Demand crosses the growth limit: down to the target.
        assert_eq!(evict_count(94_000, 2_000, 0, 100_000, 0.1), 4_000);
        assert_eq!(evict_count(100_000, 1, 0, 100_000, 0.1), 10_000);
        // Dropped seeds trigger it too, even without growth demand.
        assert_eq!(evict_count(99_000, 0, 10, 100_000, 0.1), 9_000);
        // Over the growth limit but nothing wants to grow: nothing.
        assert_eq!(evict_count(100_000, 0, 0, 100_000, 0.1), 0);
        // Already below the target: nothing to evict.
        assert_eq!(evict_count(80_000, 50_000, 7, 100_000, 0.1), 0);
        // Above the budget (e.g. loaded over cap) without overflow.
        assert_eq!(evict_count(120_000, u32::MAX, 0, 100_000, 0.1), 30_000);
    }
}
