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

mod life;
mod picker;
mod recent;

#[cfg(test)]
pub(crate) use picker::{pick_evictions, pick_evictions_within};
pub(crate) use life::SplatLife;
pub(crate) use picker::{PickInputs, growth_demand, read_bool, read_f32, select_evictions};
pub use recent::{RecentCells, recent_demand};

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

#[cfg(test)]
mod tests;
