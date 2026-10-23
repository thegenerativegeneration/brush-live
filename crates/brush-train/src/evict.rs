//! Eviction keeps an open-ended (live) training run within `max_splats`.
//!
//! Eviction only makes room for newly observed areas. A 1 m cell counts as
//! newly observed for `recent_refines` refines after a keyframe seeded at
//! least [`MIN_CELL_SEEDS`] splats in it, i.e. the model was still
//! transparent there. Split and growth candidates inside those cells that
//! the budget blocks add up as a backlog; at a refine the
//! least important splats in the whole model are pruned, as many as that
//! backlog but at most down to `max_splats · (1 − headroom)`, and that
//! refine splits and grows only inside the recent cells. Dropped keyframe
//! seeds evict down to that target directly. Once no keyframe has seeded a
//! cell for `recent_refines` refines nothing is evicted, so the model
//! settles when capture stops or only revisits covered areas. Splitting and
//! growth may fill up to `max_splats · (1 − headroom / 2)`; the last
//! `headroom / 2` stays free for seeding new keyframes.
//!
//! Importance is the refine record's mean Speedy-Splat score per seeing view;
//! a splat no view saw keeps its value. It is per splat and carried through
//! append, prune and split like the optimizer state. Splats that were never
//! scored, are younger than `min_age` refines, or lie in the protected view
//! cone are never evicted. Within one eviction no 1 m cell loses more than
//! `max_cell_fraction` of its splats, so a spatially correlated low score
//! cannot wipe a region in one go.

use crate::budget::GroupLimit;
use crate::config::TrainConfig;
use crate::lod::top_k_indices;
use crate::stats::{RefineRecord, trained_only};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};

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
#[cfg_attr(test, derive(Clone))]
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

/// Per-splat age (refines survived), importance and whether a step ever trained it, row-aligned with the splats.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct SplatLife {
    pub age: Tensor<1>,
    /// `+inf` until scored.
    pub importance: Tensor<1>,
    /// 1 once a step updated the row (it was not frozen), else 0; rows appended back from the host take the flag they
    /// carry (trained without one), split children take their parent's.
    pub trained: Tensor<1>,
}

impl SplatLife {
    pub(crate) fn new(n: usize, device: &Device) -> Self {
        Self {
            age: Tensor::zeros([n], device),
            importance: Tensor::full([n], f32::INFINITY, device),
            trained: Tensor::zeros([n], device),
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
        self.trained = Tensor::cat(vec![self.trained.clone(), Tensor::zeros([n], &device)], 0);
    }

    pub(crate) fn keep(&mut self, indices: Tensor<1, Int>) {
        self.age = self.age.clone().select(0, indices.clone());
        self.importance = self.importance.clone().select(0, indices.clone());
        self.trained = self.trained.clone().select(0, indices);
    }

    /// Sets the age, importance and trained flag of the last `age.len()` rows (they come back from the host); empty
    /// `trained` marks them all trained.
    pub(crate) fn set_tail(&mut self, age: &[f32], importance: &[f32], trained: &[bool]) {
        let (k, len) = (age.len(), self.len());
        assert_eq!(importance.len(), k, "one importance per age");
        assert!(trained.is_empty() || trained.len() == k, "one trained flag per age, or none");
        assert!(k <= len, "{k} tail rows for {len} rows");
        if k == 0 {
            return;
        }
        let device = self.age.device();
        let tail = |v: &[f32]| Tensor::<1>::from_data(TensorData::new(v.to_vec(), [k]), &device);
        self.age = self.age.clone().slice_assign(s![len - k..len], tail(age));
        self.importance = self
            .importance
            .clone()
            .slice_assign(s![len - k..len], tail(importance));
        let flags = match trained {
            [] => Tensor::ones([k], &device),
            flags => tail(&flags.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect::<Vec<_>>()),
        };
        self.trained = self.trained.clone().slice_assign(s![len - k..len], flags);
    }

    /// Appends one child per parent index: age 0, the parent's importance and trained flag.
    pub(crate) fn split(&mut self, parents: Tensor<1, Int>) {
        let n = parents.dims()[0];
        let device = self.age.device();
        let trained = self.trained.clone().select(0, parents.clone());
        self.trained = Tensor::cat(vec![self.trained.clone(), trained], 0);
        let inherited = self.importance.clone().select(0, parents);
        self.age = Tensor::cat(vec![self.age.clone(), Tensor::zeros([n], &device)], 0);
        self.importance = Tensor::cat(vec![self.importance.clone(), inherited], 0);
    }

    /// Mean importance per seeing view over the refine window; splats no view saw keep theirs.
    pub(crate) fn update_importance(&mut self, sum: Tensor<1>, views: Tensor<1>) {
        let seen = views.clone().greater_elem(0.0);
        let mean = sum.div(views.clamp_min(1.0));
        self.importance = self.importance.clone().mask_where(seen, mean);
    }

    /// Marks the rows a step updated as trained: every row, or the rows not set in `frozen`.
    pub(crate) fn mark_trained(&mut self, frozen: Option<&Tensor<1, Bool>>) {
        self.trained = match frozen {
            Some(f) => self.trained.clone().mask_fill(f.clone().bool_not(), 1.0),
            None => self.trained.clone().ones_like(),
        };
    }

    pub(crate) fn tick(&mut self) {
        self.age = self.age.clone() + 1.0;
    }
}

/// Eviction settings plus the state the trainer carries between refines.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct Eviction {
    pub config: EvictConfig,
    pub protect: Option<ProtectCone>,
    pub seed_shortfall: u32,
    pub recent: RecentCells,
}

impl Eviction {
    pub(crate) fn new(config: EvictConfig) -> Self {
        Self {
            config,
            protect: None,
            seed_shortfall: 0,
            recent: RecentCells::default(),
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
/// budget allowed it, with `num_dead` dead splats about to be replaced. Rows flagged in `untrained` (no step
/// trained them since the last refine) neither split nor grow, so they add none.
pub(crate) async fn growth_demand(
    refiner: &RefineRecord,
    config: &TrainConfig,
    iter: u32,
    num_dead: u32,
    untrained: Option<&Tensor<1, Bool>>,
) -> u32 {
    let mut demand = 0;
    if config.split_at_screen_size > 0.0 {
        let oversized = refiner.above_screen_size(config.split_at_screen_size);
        demand += count_true(trained_only(oversized, untrained)).await;
    }
    if iter >= config.growth_start_iter && iter < config.growth_stop_iter {
        let above = refiner.above_threshold(config.growth_grad_threshold);
        let above = count_true(trained_only(above, untrained)).await;
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
/// Rows flagged in `untrained_held` (held, never trained) rank at importance 0, scored or not. With `limit` (each
/// row's group and the rows each group may give), a row goes only while its group may give one, unless it is flagged
/// in `untrained_held`. The rows `prior` (taken earlier in this refine) count against their cell's and group's share.
/// Returns the eviction mask and the rows it holds, in pick order.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn select_evictions(
    life: &SplatLife,
    means: Tensor<2>,
    means_cpu: &[f32],
    exclude: Tensor<1, Bool>,
    protect: Option<ProtectCone>,
    config: &EvictConfig,
    count: u32,
    untrained_held: Option<Tensor<1, Bool>>,
    limit: Option<(&[u32], Vec<u32>)>,
    prior: &[usize],
) -> (Tensor<1, Bool>, Vec<usize>) {
    let n = life.len();
    let device = life.age.device();
    let importance = match &untrained_held {
        Some(u) => life.importance.clone().mask_fill(u.clone(), 0.0),
        None => life.importance.clone(),
    };
    let mut eligible = life
        .age
        .clone()
        .greater_equal_elem(config.min_age as f32)
        .bool_and(importance.clone().is_finite())
        .bool_and(exclude.bool_not());
    if let Some(cone) = protect {
        eligible = eligible.bool_and(in_cone(means.clone(), cone).bool_not());
    }
    let ranked = importance.mask_fill(eligible.bool_not(), f32::INFINITY);
    let scores = read_f32(ranked).await;
    let limit = match limit {
        Some((group, quota)) => {
            let free = match untrained_held {
                Some(u) => read_bool(u).await,
                None => Vec::new(),
            };
            Some(GroupLimit { group, quota, free })
        }
        None => None,
    };
    let picked = pick_evictions_within(&scores, means_cpu, count as usize, config.max_cell_fraction, limit, prior);
    let mut evict = vec![false; n];
    for &i in &picked {
        evict[i] = true;
    }
    let mask = Tensor::<1, Bool>::from_data(TensorData::new(evict, [n]), &device);
    (mask, picked)
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
#[cfg(test)]
pub(crate) fn pick_evictions(
    scores: &[f32],
    means: &[f32],
    count: usize,
    max_cell_fraction: f32,
) -> Vec<usize> {
    pick_evictions_within(scores, means, count, max_cell_fraction, None, &[])
}

/// [`pick_evictions`] that also skips the rows `limit` does not admit ([`GroupLimit`]), continuing an eviction that
/// already took the rows `prior` in this refine: their cells and groups have that much less left to give.
pub(crate) fn pick_evictions_within(
    scores: &[f32],
    means: &[f32],
    count: usize,
    max_cell_fraction: f32,
    mut limit: Option<GroupLimit>,
    prior: &[usize],
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
    for &i in prior {
        if let Some(q) = quota.get_mut(&cell(i)) {
            *q = q.saturating_sub(1);
        }
        if let Some(l) = limit.as_mut() {
            l.take(i);
        }
    }
    let negated: Vec<f32> = scores.iter().map(|s| -s).collect();
    let mut picked = Vec::with_capacity(count);
    for i in top_k_indices(&negated, scores.len()) {
        if picked.len() == count || !scores[i].is_finite() {
            break;
        }
        if limit.as_ref().is_some_and(|l| !l.admits(i)) {
            continue;
        }
        let q = quota.get_mut(&cell(i)).expect("every splat has a cell");
        if *q > 0 {
            *q -= 1;
            if let Some(l) = limit.as_mut() {
                l.take(i);
            }
            picked.push(i);
        }
    }
    picked
}

#[cfg(test)]
#[path = "evict_tests.rs"]
mod tests;
