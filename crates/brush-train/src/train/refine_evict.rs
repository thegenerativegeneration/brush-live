//! Eviction at a refine: the budget eviction (seed shortfall, blocked growth while keyframes seed new cells) and the
//! eviction of a group over its cap under a row budget ([`crate::budget`]).

use super::{SplatTrainer, radius};
use crate::budget::{EvictionRelax, RowBudget};
use crate::cone::ViewCone;
use crate::config::TrainConfig;
use crate::evict::{
    EvictConfig, Eviction, PickInputs, SplatLife, growth_demand, read_bool, read_f32, recent_demand,
    select_evictions,
};
use crate::stats::{RefineRecord, trained_only};
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Bool, Tensor, TensorData};

impl SplatTrainer {
    /// Adds this refine's evictions to the `dead` prune mask; rows set in `held` are no victims unless no step ever
    /// trained them, and with a `budget` that limits eviction a row goes only within its group's quota. Rows flagged
    /// in `full` (their group is at its cap) add no growth demand. Returns the mask and the rows evicted, ascending.
    pub(super) async fn add_evictions(
        &mut self,
        iter: u32,
        splats: &Splats,
        refiner: &RefineRecord,
        dead: Tensor<1, Bool>,
        held: Option<Tensor<1, Bool>>,
        budget: Option<&RowBudget>,
        full: Option<Tensor<1, Bool>>,
    ) -> (Tensor<1, Bool>, Vec<usize>) {
        self.tally.eviction_fell_back = false;
        let Some(ev) = self.evict.as_mut() else {
            return (dead, Vec::new());
        };
        let Some(life) = self.rows.life.as_mut() else {
            return (dead, Vec::new());
        };
        life.tick();
        let active = ev.recent.active(ev.config.recent_refines);
        if !active {
            ev.recent.backlog = 0;
            if ev.seed_shortfall == 0 {
                return (dead, Vec::new());
            }
        }
        let current = splats.num_splats();
        let means = if active {
            add_recent_demand(ev, &self.config, iter, splats, refiner, &dead, full).await
        } else {
            None
        };
        let shortfall = std::mem::take(&mut ev.seed_shortfall);
        let backlog = ev.recent.backlog;
        let want = crate::evict::evict_count(current, backlog, shortfall, self.config.max_splats, ev.config.headroom);
        if want == 0 {
            return (dead, Vec::new());
        }
        let means = match means {
            Some(m) => m,
            None => read_f32(splats.means()).await,
        };
        // Held rows no step ever trained (raw seeds on ground outside the active tiles) do not block the budget.
        let untrained_held = held.as_ref().map(|h| h.clone().bool_and(life.trained.clone().lower_elem(0.5)));
        let (held, untrained_held) = (held.as_ref(), untrained_held.as_ref());
        let pick = Pick {
            life,
            splats,
            protect: ev.protect,
            config: &ev.config,
            means: &means,
            held,
            untrained_held,
        };
        let (mask, mut picked) = pick.victims(dead.clone(), budget, want, &[]).await;
        let (mask, fell_back) = pick.widen(&widening_tiers(budget), &dead, want, mask, &mut picked).await;
        picked.sort_unstable();
        let count = picked.len();
        if count > 0 {
            ev.recent.backlog = 0;
        }
        log::info!(
            "evict: {count} of {want} wanted (recent backlog {backlog}, seed shortfall {shortfall}, {current} splats)"
        );
        self.tally.eviction_fell_back = fell_back;
        (dead.bool_or(mask), picked)
    }

    /// Adds to the `dead` prune mask, per group of `budget` over its `max_rows`, the least important rows not yet in
    /// it (by eviction importance) down to that count. Needs eviction state; without it nothing is added. Returns the
    /// mask and the rows added, ascending.
    pub(super) async fn evict_over_cap(
        &self,
        dead: Tensor<1, Bool>,
        budget: &RowBudget,
    ) -> (Tensor<1, Bool>, Vec<usize>) {
        let Some(life) = self.rows.life.as_ref().filter(|_| self.evict.is_some()) else {
            return (dead, Vec::new());
        };
        if !budget.any_over_cap() {
            return (dead, Vec::new());
        }
        let n = budget.group.len();
        let scores = read_f32(life.importance.clone()).await;
        let dead_rows = read_bool(dead.clone()).await;
        let picked = budget.over_cap(&scores, &dead_rows);
        if picked.is_empty() {
            return (dead, picked);
        }
        let mut flags = vec![false; n];
        for &i in &picked {
            flags[i] = true;
        }
        let mask = Tensor::<1, Bool>::from_data(TensorData::new(flags, [n]), &dead.device());
        log::info!("evict: {} over their group's cap", picked.len());
        (dead.bool_or(mask), picked)
    }

    /// Hands the refine window's importance sums to the eviction life, when eviction tracks one.
    pub(super) fn note_refine_importance(&mut self, refiner: &RefineRecord) {
        if self.evict.is_some()
            && let Some(life) = self.rows.life.as_mut()
        {
            life.update_importance(
                refiner.importance_sum.clone(),
                refiner.importance_views.clone(),
            );
        }
    }

    /// Adds this refine's evictions to the `prune` mask: the budget eviction ([`Self::add_evictions`]) and, under a
    /// row budget that fits `splats`, the eviction of groups over their cap. Takes the row budget and records the
    /// per-group eviction counts in `budget_evicted`.
    pub(super) async fn evict_for_refine(
        &mut self,
        iter: u32,
        splats: &Splats,
        refiner: &RefineRecord,
        prune: Tensor<1, Bool>,
        held: Option<Tensor<1, Bool>>,
    ) -> RefineEvictions {
        let n = splats.num_splats() as usize;
        let budget = self.rows.budget.take().filter(|b| Self::budget_fits(b, Some(n)));
        let full = budget
            .as_ref()
            .and_then(RowBudget::no_growth)
            .map(|f| Tensor::<1, Bool>::from_data(TensorData::new(f, [n]), &prune.device()));
        let (mut prune, mut evicted) = self
            .add_evictions(iter, splats, refiner, prune, held, budget.as_ref(), full.clone())
            .await;
        let for_total = evicted.len();
        if let Some(budget) = &budget {
            let (mask, over) = self.evict_over_cap(prune.clone(), budget).await;
            evicted.extend(over);
            self.tally.budget_evicted = budget.count_by_group(&evicted);
            prune = mask;
        } else {
            self.tally.budget_evicted.clear();
        }
        RefineEvictions {
            prune,
            full,
            num_evicted: evicted.len() as u32,
            for_total,
        }
    }

    /// After an eviction for the total budget (`evicted_for_total > 0`), the rows of `splats` in newly observed cells:
    /// the freed room goes to them only (rows a group gave for its own cap free no room for others). `None` otherwise
    /// or without eviction state. Ticks the recent-cell window.
    pub(super) async fn recent_growth_mask(
        &mut self,
        splats: &Splats,
        evicted_for_total: usize,
    ) -> Option<Vec<bool>> {
        let e = self.evict.as_mut()?;
        let mask = if evicted_for_total > 0 {
            let means = read_f32(splats.means()).await;
            Some(e.recent.mask(&means, e.config.recent_refines))
        } else {
            None
        };
        e.recent.tick();
        mask
    }
}

/// The evictions of one refine, from [`SplatTrainer::evict_for_refine`].
pub(super) struct RefineEvictions {
    /// The prune mask with the evicted rows added.
    pub prune: Tensor<1, Bool>,
    /// Rows of a group at its cap: no growth from them.
    pub full: Option<Tensor<1, Bool>>,
    /// Rows evicted in total.
    pub num_evicted: u32,
    /// Rows evicted for the total budget (not for a group's cap).
    pub for_total: usize,
}

/// Adds the growth demand of recently observed cells to `ev`'s backlog when this refine's growth would pass the
/// growth limit, and clears the backlog when it would not. Returns the means it read for that, if any.
async fn add_recent_demand(
    ev: &mut Eviction,
    config: &TrainConfig,
    iter: u32,
    splats: &Splats,
    refiner: &RefineRecord,
    dead: &Tensor<1, Bool>,
    full: Option<Tensor<1, Bool>>,
) -> Option<Vec<f32>> {
    let num_dead = dead
        .clone()
        .int()
        .sum()
        .into_scalar_async::<i32>()
        .await
        .expect("dead count readback") as u32;
    let untrained = radius::either(refiner.untrained(), full);
    let demand = growth_demand(refiner, config, iter, num_dead, untrained.as_ref()).await;
    let limit = crate::evict::growth_limit(config.max_splats, ev.config.headroom);
    if splats.num_splats().saturating_add(demand) <= limit {
        ev.recent.backlog = 0;
        return None;
    }
    let m = read_f32(splats.means()).await;
    let in_recent = ev.recent.mask(&m, ev.config.recent_refines);
    let oversized = if config.split_at_screen_size > 0.0 {
        read_bool(trained_only(
            refiner.above_screen_size(config.split_at_screen_size),
            untrained.as_ref(),
        ))
        .await
    } else {
        vec![]
    };
    let growing = iter >= config.growth_start_iter && iter < config.growth_stop_iter;
    let above = if growing {
        read_bool(trained_only(
            refiner.above_threshold(config.growth_grad_threshold),
            untrained.as_ref(),
        ))
        .await
    } else {
        vec![]
    };
    let recent = recent_demand(&in_recent, &oversized, &above, config.growth_select_fraction);
    ev.recent.backlog = ev.recent.backlog.saturating_add(recent);
    Some(m)
}

/// The budgets a limited eviction short of its want widens to, in order: none unless `budget` limits eviction and
/// has a fallback; the fallback; and after a fallback that opens groups, no limit.
pub(super) fn widening_tiers(budget: Option<&RowBudget>) -> Vec<RowBudget> {
    let Some(b) = budget.filter(|b| b.limit_eviction) else {
        return Vec::new();
    };
    let Some(fallback) = b.fallback.clone() else {
        return Vec::new();
    };
    let opens_groups = matches!(fallback, EvictionRelax::Groups(_));
    let mut wide = b.clone();
    wide.relax(fallback);
    if !opens_groups {
        return vec![wide];
    }
    let mut all = wide.clone();
    all.relax(EvictionRelax::Unlimited);
    vec![wide, all]
}

/// What one eviction pick shares between the limited pass and the widened ones.
struct Pick<'a> {
    life: &'a SplatLife,
    splats: &'a Splats,
    protect: Option<ViewCone>,
    config: &'a EvictConfig,
    means: &'a [f32],
    held: Option<&'a Tensor<1, Bool>>,
    untrained_held: Option<&'a Tensor<1, Bool>>,
}

impl Pick<'_> {
    /// Up to `want` victims not in `dead`: held rows only when no step trained them, or when `budget` lets held rows go
    /// ([`RowBudget::evict_held`]); within `budget`'s quotas when it limits eviction, less what the rows `prior` (taken
    /// by an earlier pick of this refine) used of their cells' and groups' shares.
    async fn victims(
        &self,
        dead: Tensor<1, Bool>,
        budget: Option<&RowBudget>,
        want: u32,
        prior: &[usize],
    ) -> (Tensor<1, Bool>, Vec<usize>) {
        let evict_held = budget.is_some_and(|b| b.limit_eviction && b.evict_held);
        let exclude = match (self.held, self.untrained_held) {
            (Some(held), Some(u)) if !evict_held => dead.bool_or(held.clone().bool_and(u.clone().bool_not())),
            _ => dead,
        };
        let quota = budget.and_then(|b| b.quota());
        let limit = quota.map(|q| (budget.expect("a quota comes with a budget").group.as_slice(), q));
        select_evictions(
            self.life,
            self.config,
            want,
            PickInputs {
                means: self.splats.means(),
                means_cpu: self.means,
                exclude,
                protect: self.protect,
                untrained_held: self.untrained_held.cloned(),
                limit,
                prior,
            },
        )
        .await
    }

    /// Short of `want` under the fair shares: widens to each of `tiers` in turn until `picked` holds `want`, each pick
    /// continuing the last (their cells and groups keep what they gave), and adds the victims to `mask` and `picked`.
    /// Returns the mask and whether a wider tier found any.
    async fn widen(
        &self,
        tiers: &[RowBudget],
        dead: &Tensor<1, Bool>,
        want: u32,
        mut mask: Tensor<1, Bool>,
        picked: &mut Vec<usize>,
    ) -> (Tensor<1, Bool>, bool) {
        let limited = picked.len();
        for wide in tiers {
            if picked.len() >= want as usize {
                break;
            }
            let rest = want - picked.len() as u32;
            let (more_mask, more) = self
                .victims(dead.clone().bool_or(mask.clone()), Some(wide), rest, picked)
                .await;
            mask = mask.bool_or(more_mask);
            picked.extend(more);
        }
        if picked.len() <= limited {
            return (mask, false);
        }
        log::info!("evict: limited eviction found {limited} of {want}; widened, {} more", picked.len() - limited);
        (mask, true)
    }
}
