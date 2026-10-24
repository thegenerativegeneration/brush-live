use super::{EvictConfig, ProtectCone, SplatLife, cell_of};
use crate::budget::GroupLimit;
use crate::config::TrainConfig;
use crate::lod::top_k_indices;
use crate::stats::{RefineRecord, trained_only};
use burn::tensor::{Bool, Tensor, TensorData};

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

/// What [`select_evictions`] picks from, beyond the life state, config and count.
pub(crate) struct PickInputs<'a> {
    /// Every splat's mean, on the device (for the protected cone).
    pub means: Tensor<2>,
    /// The same means, flat xyz on the host (for the cell quotas).
    pub means_cpu: &'a [f32],
    /// Rows that are never picked.
    pub exclude: Tensor<1, Bool>,
    /// Splats inside this cone are never picked.
    pub protect: Option<ProtectCone>,
    /// Rows flagged here (held, never trained) rank at importance 0, scored or not, and bypass `limit`.
    pub untrained_held: Option<Tensor<1, Bool>>,
    /// Each row's group and the rows each group may give: a row goes only while its group may give one.
    pub limit: Option<(&'a [u32], Vec<u32>)>,
    /// The rows taken earlier in this refine; they count against their cell's and group's share.
    pub prior: &'a [usize],
}

/// Picks up to `count` splats to evict, lowest importance first, among
/// those that are scored, at least `min_age` refines old, not excluded and
/// not protected, taking at most `max_cell_fraction` of any 1 m cell.
/// Returns the eviction mask and the rows it holds, in pick order.
pub(crate) async fn select_evictions(
    life: &SplatLife,
    config: &EvictConfig,
    count: u32,
    inputs: PickInputs<'_>,
) -> (Tensor<1, Bool>, Vec<usize>) {
    let PickInputs {
        means,
        means_cpu,
        exclude,
        protect,
        untrained_held,
        limit,
        prior,
    } = inputs;
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
/// (ineligible) ones and those whose [`super::EVICT_CELL_M`] cell (by the flat xyz
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
