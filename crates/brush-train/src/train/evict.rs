//! Eviction at a refine: the budget eviction (seed shortfall, blocked growth while keyframes seed new cells) and the
//! eviction of a group over its cap under a row budget ([`crate::budget`]).

use super::{SplatTrainer, radius};
use crate::budget::{EvictionRelax, RowBudget};
use crate::evict::{growth_demand, read_bool, read_f32, recent_demand, select_evictions};
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
        self.eviction_fell_back = false;
        let mut fell_back = false;
        let Some(ev) = self.evict.as_mut() else {
            return (dead, Vec::new());
        };
        let Some(life) = ev.life.as_mut() else {
            return (dead, Vec::new());
        };
        life.tick();
        let window = ev.config.recent_refines;
        let active = ev.recent.active(window);
        if !active {
            ev.recent.backlog = 0;
            if ev.seed_shortfall == 0 {
                return (dead, Vec::new());
            }
        }
        let current = splats.num_splats();
        let mut means = None;
        if active {
            let num_dead = dead
                .clone()
                .int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("dead count readback") as u32;
            let untrained = radius::either(refiner.untrained(), full);
            let demand =
                growth_demand(refiner, &self.config, iter, num_dead, untrained.as_ref()).await;
            let limit = crate::evict::growth_limit(self.config.max_splats, ev.config.headroom);
            if current.saturating_add(demand) > limit {
                let m = read_f32(splats.means()).await;
                let in_recent = ev.recent.mask(&m, window);
                let oversized = if self.config.split_at_screen_size > 0.0 {
                    read_bool(trained_only(
                        refiner.above_screen_size(self.config.split_at_screen_size),
                        untrained.as_ref(),
                    ))
                    .await
                } else {
                    vec![]
                };
                let growing =
                    iter >= self.config.growth_start_iter && iter < self.config.growth_stop_iter;
                let above = if growing {
                    read_bool(trained_only(
                        refiner.above_threshold(self.config.growth_grad_threshold),
                        untrained.as_ref(),
                    ))
                    .await
                } else {
                    vec![]
                };
                let recent = recent_demand(
                    &in_recent,
                    &oversized,
                    &above,
                    self.config.growth_select_fraction,
                );
                ev.recent.backlog = ev.recent.backlog.saturating_add(recent);
                means = Some(m);
            } else {
                ev.recent.backlog = 0;
            }
        }
        let shortfall = std::mem::take(&mut ev.seed_shortfall);
        let backlog = ev.recent.backlog;
        let want = crate::evict::evict_count(
            current,
            backlog,
            shortfall,
            self.config.max_splats,
            ev.config.headroom,
        );
        if want == 0 {
            return (dead, Vec::new());
        }
        let means = match means {
            Some(m) => m,
            None => read_f32(splats.means()).await,
        };
        // Held rows no step ever trained (raw seeds on ground outside the active tiles) do not block the budget.
        let untrained_held = held.as_ref().map(|h| h.clone().bool_and(life.trained.clone().lower_elem(0.5)));
        let pick = Pick { means: &means, held: held.as_ref(), untrained_held: untrained_held.as_ref() };
        let (mut mask, mut picked) = pick.victims(life, splats, (ev.protect, &ev.config), dead.clone(), budget, want, &[]).await;
        // Short of `want` under the fair shares: widen to the fallback, and from opened groups on to no limit, each
        // pick continuing the last (their cells and groups keep what they gave).
        let mut tiers = Vec::new();
        if let Some(b) = budget.filter(|b| b.limit_eviction) {
            if let Some(fallback) = b.fallback.clone() {
                let opens_groups = matches!(fallback, EvictionRelax::Groups(_));
                let mut wide = b.clone();
                wide.relax(fallback);
                if opens_groups {
                    let mut all = wide.clone();
                    all.relax(EvictionRelax::Unlimited);
                    tiers.push(wide);
                    tiers.push(all);
                } else {
                    tiers.push(wide);
                }
            }
        }
        let limited = picked.len();
        for wide in &tiers {
            if picked.len() >= want as usize {
                break;
            }
            let rest = want - picked.len() as u32;
            let (more_mask, more) = pick
                .victims(life, splats, (ev.protect, &ev.config), dead.clone().bool_or(mask.clone()), Some(wide), rest, &picked)
                .await;
            mask = mask.bool_or(more_mask);
            picked.extend(more);
        }
        if picked.len() > limited {
            log::info!("evict: limited eviction found {limited} of {want}; widened, {} more", picked.len() - limited);
            fell_back = true;
        }
        picked.sort_unstable();
        let count = picked.len();
        if count > 0 {
            ev.recent.backlog = 0;
        }
        log::info!(
            "evict: {count} of {want} wanted (recent backlog {backlog}, seed shortfall {shortfall}, {current} splats)"
        );
        self.eviction_fell_back = fell_back;
        (dead.bool_or(mask), picked)
    }

    /// Adds to the `dead` prune mask, per group of `budget` over its `max_rows`, the least important rows not yet in
    /// it (by eviction importance) down to that count. Needs eviction state; without it nothing is added. Returns the
    /// mask and the rows added, ascending.
    pub(super) async fn evict_over_cap(&self, dead: Tensor<1, Bool>, budget: &RowBudget) -> (Tensor<1, Bool>, Vec<usize>) {
        let Some(life) = self.evict.as_ref().and_then(|e| e.life.as_ref()) else {
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
}

/// What one eviction pick shares between the limited pass and the widened one.
struct Pick<'a> {
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
        life: &crate::evict::SplatLife,
        splats: &Splats,
        (protect, config): (Option<crate::evict::ProtectCone>, &crate::evict::EvictConfig),
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
            life,
            splats.means(),
            self.means,
            exclude,
            protect,
            config,
            want,
            self.untrained_held.cloned(),
            limit,
            prior,
        )
        .await
    }
}
