//! Eviction keeps an open-ended (live) training run within `max_splats`.
//!
//! Eviction only makes room for newly observed areas. A 1 m cell counts as
//! newly observed for `recent_refines` refines after a keyframe seeded at
//! least [`MIN_CELL_SEEDS`] splats in it, i.e. the model was still
//! transparent there. Split and growth candidates inside those cells that
//! the budget blocks add up as a backlog; at a refine with fresh scores the
//! least important splats in the whole model are pruned, as many as that
//! backlog but at most down to `max_splats · (1 − headroom)`, and that
//! refine splits and grows only inside the recent cells. Dropped keyframe
//! seeds evict down to that target directly. Once no keyframe has seeded a
//! cell for `recent_refines` refines nothing is evicted, so the model
//! settles when capture stops or only revisits covered areas. Splitting and
//! growth may fill up to `max_splats · (1 − headroom / 2)`; the last
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
    /// Refines a seeded cell stays newly observed for.
    pub recent_refines: u32,
}

/// Seeds one keyframe must place in a cell to mark it newly observed, so a
/// few seeds in a small hole of a covered area do not count.
pub const MIN_CELL_SEEDS: usize = 8;

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

/// The [`EVICT_CELL_M`] cell containing `p`.
pub fn cell_of(p: glam::Vec3) -> glam::IVec3 {
    (p / EVICT_CELL_M).floor().as_ivec3()
}

/// Cells newly observed by recent keyframes (see the module docs).
#[derive(Default)]
pub struct RecentCells {
    /// Cell → refine count when a keyframe last seeded it.
    stamps: hashbrown::HashMap<glam::IVec3, u32>,
    refines: u32,
    last_seeded: Option<u32>,
    /// Blocked split/growth demand inside recent cells since the last eviction.
    pub backlog: u32,
}

impl RecentCells {
    /// Records one keyframe's seed positions (flat xyz, before the budget
    /// drops any).
    pub fn note_keyframe(&mut self, seed_means: &[f32]) {
        let mut counts: hashbrown::HashMap<glam::IVec3, usize> = hashbrown::HashMap::new();
        for p in seed_means.as_chunks::<3>().0 {
            *counts.entry(cell_of(glam::Vec3::from_array(*p))).or_default() += 1;
        }
        for (cell, n) in counts {
            if n >= MIN_CELL_SEEDS {
                self.stamps.insert(cell, self.refines);
                self.last_seeded = Some(self.refines);
            }
        }
    }

    /// Advances to the next refine.
    pub fn tick(&mut self) {
        self.refines += 1;
    }

    /// Whether any cell was seeded within the last `window` refines.
    pub fn active(&self, window: u32) -> bool {
        self.last_seeded
            .is_some_and(|r| self.refines - r < window)
    }

    /// Whether `p` lies in a cell seeded within the last `window` refines.
    pub fn contains(&self, p: glam::Vec3, window: u32) -> bool {
        self.stamps
            .get(&cell_of(p))
            .is_some_and(|&r| self.refines - r < window)
    }

    /// Per splat (flat xyz `means`), whether it lies in a recent cell.
    pub fn mask(&self, means: &[f32], window: u32) -> Vec<bool> {
        means
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| self.contains(glam::Vec3::from_array(*p), window))
            .collect()
    }
}

/// Split and growth demand inside recent cells: the force-split candidates
/// plus `growth_select_fraction` of the high-gradient candidates, the same
/// ratio refine applies to the whole model.
pub fn recent_demand(
    in_recent: &[bool],
    oversized: &[bool],
    above_threshold: &[bool],
    growth_select_fraction: f32,
) -> u32 {
    let count = |mask: &[bool]| {
        in_recent
            .iter()
            .zip(mask)
            .filter(|&(&r, &m)| r && m)
            .count()
    };
    let grow = (count(above_threshold) as f32 * growth_select_fraction).round() as u32;
    count(oversized) as u32 + grow
}

/// Splats to evict at a refine with `current` splats, when split and growth
/// inside recent cells want `recent_demand` more and `seed_shortfall` seeds
/// were dropped since the last refine. Evicts what the free room up to the
/// growth limit cannot hold, at most down to the target; dropped seeds
/// evict down to the target.
pub fn evict_count(
    current: u32,
    recent_demand: u32,
    seed_shortfall: u32,
    max_splats: u32,
    headroom: f32,
) -> u32 {
    let cap = current.saturating_sub(evict_target(max_splats, headroom));
    if seed_shortfall > 0 {
        return cap;
    }
    let free = growth_limit(max_splats, headroom).saturating_sub(current);
    recent_demand.saturating_sub(free).min(cap)
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
    pub recent: RecentCells,
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
            recent: RecentCells::default(),
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
    means_cpu: &[f32],
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
    let picked = pick_evictions(&scores, means_cpu, count as usize, config.max_cell_fraction);
    let mut evict = vec![false; n];
    for &i in &picked {
        evict[i] = true;
    }
    let mask = Tensor::<1, Bool>::from_data(TensorData::new(evict, [n]), &device);
    (mask, picked.len() as u32)
}

pub(crate) async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("evict readback")
        .try_into_vec::<f32>()
        .expect("f32 readback")
}

pub(crate) async fn read_bool(t: Tensor<1, Bool>) -> Vec<bool> {
    t.int()
        .into_data_async()
        .await
        .expect("evict readback")
        .convert::<i32>()
        .try_into_vec::<i32>()
        .expect("mask readback")
        .into_iter()
        .map(|v| v != 0)
        .collect()
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
    let cell = |i: usize| cell_of(glam::Vec3::from_slice(&means[i * 3..i * 3 + 3]));
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
#[path = "evict_tests.rs"]
mod tests;
