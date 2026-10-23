//! The training radius in the trainer, and rows the caller holds: which splats a step leaves alone, and how refine
//! keeps the rows no step trained since the last refine as they were (raw params and Mip-Splatting floor).

use brush_render::gaussian_splats::{Splats, fold_min_scale};
use burn::module::Param;
use burn::tensor::{Bool, Device, Tensor, activation::sigmoid, s};

use super::{MIN_SCALE_FREEZE_FRAC, SplatTrainer};
use crate::evict::SplatLife;
use crate::stats::RefineRecord;

/// `[N]` flags of the `means` (`[N, 3]`, inner backend) farther than `radius` from `center`.
pub(super) fn beyond_radius(means: Tensor<2>, center: glam::Vec3, radius: f32) -> Tensor<1, Bool> {
    let n = means.dims()[0];
    let c =
        Tensor::<1>::from_floats([center.x, center.y, center.z], &means.device()).reshape([1, 3]);
    let diff = means - c;
    diff.clone()
        .mul(diff)
        .sum_dim(1)
        .reshape([n])
        .greater_elem(radius * radius)
}

/// Rows either mask flags; `None` when neither is set.
pub(super) fn either(a: Option<Tensor<1, Bool>>, b: Option<Tensor<1, Bool>>) -> Option<Tensor<1, Bool>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.bool_or(b)),
        (a, b) => a.or(b),
    }
}

/// `mask` with `n` unset rows appended.
pub(super) fn pad_unset(mask: Tensor<1, Bool>, n: usize) -> Tensor<1, Bool> {
    let device = mask.device();
    Tensor::cat(vec![mask, Tensor::<1>::zeros([n], &device).greater_elem(0.0)], 0)
}

/// What refine works on when untrained rows keep their floor: the splats without a floor attached, the trained rows'
/// floor baked into their params and the untrained rows' params raw.
pub(super) struct PartialBake {
    pub splats: Splats,
    /// Opacity and world scales of every row with its floor folded in, as a full bake gives (for the prune checks).
    pub opacities: Tensor<1>,
    pub scales: Tensor<2>,
    /// The floor every row carried into the refine.
    pub floor: Tensor<1>,
}

/// Bakes `splats`' floor into the rows not flagged in `untrained`; those keep their raw params. Hands `splats` back
/// unchanged when no floor is attached.
pub(super) fn bake_trained(
    mut splats: Splats,
    untrained: &Tensor<1, Bool>,
) -> Result<PartialBake, Splats> {
    let Some(floor) = splats.min_scale.take() else {
        return Err(splats);
    };
    let (raw_t, raw_o) = (splats.transforms.val(), splats.raw_opacities.val());
    let (baked_t, baked_o) = fold_min_scale(raw_t.clone(), raw_o.clone(), floor.clone());
    let n = raw_t.dims()[0];
    let rows = untrained.clone().reshape([n, 1]).expand([n, 10]);
    let transforms = baked_t.clone().mask_where(rows, raw_t);
    let raw_opac = baked_o.clone().mask_where(untrained.clone(), raw_o);
    splats.transforms =
        Param::initialized(splats.transforms.id, transforms.detach()).set_require_grad(true);
    splats.raw_opacities =
        Param::initialized(splats.raw_opacities.id, raw_opac.detach()).set_require_grad(true);
    Ok(PartialBake {
        splats,
        opacities: sigmoid(baked_o),
        scales: baked_t.slice(s![.., 7..10]).exp(),
        floor,
    })
}

/// `fresh` (the refine's new floor, rows after the split children appended) with the rows set in `untrained`
/// (`kept`'s rows, the rows before the children) taken from `kept`; `fresh` itself without a kept floor.
pub(super) fn keep_floor(fresh: Tensor<1>, kept: Option<(Tensor<1>, Tensor<1, Bool>)>) -> Tensor<1> {
    let Some((kept, untrained)) = kept else {
        return fresh;
    };
    let extra = fresh.dims()[0] - kept.dims()[0];
    let device = fresh.device();
    let kept = Tensor::cat(vec![kept, Tensor::zeros([extra], &device)], 0);
    let untrained = Tensor::cat(
        vec![
            untrained,
            Tensor::<1>::zeros([extra], &device).greater_elem(0.0),
        ],
        0,
    );
    fresh.mask_where(untrained, kept)
}

impl SplatTrainer {
    /// Bakes the current 3D-filter floor into `splats` for refine. While the floor still refreshes and rows no step
    /// trained since the last refine exist (training radius on or rows held), those rows stay raw and keep their floor
    /// through the refine, so a frozen splat does not get its floor folded in again at every refine; the kept floor
    /// goes to `refiner.kept_floor`, and the opacities and scales of every row with its floor folded in come back for
    /// the prune checks. Otherwise every row is baked and `None` comes back.
    pub(super) fn bake_for_refine(
        &self,
        refiner: &mut RefineRecord,
        splats: Splats,
        progress: f32,
    ) -> (Splats, Option<(Tensor<1>, Tensor<2>)>) {
        let refresh_floor = progress < MIN_SCALE_FREEZE_FRAC
            && self.config.min_scale_factor > 0.0
            && !self.view_cams.is_empty();
        match refiner.untrained().filter(|_| refresh_floor) {
            Some(untrained) => match bake_trained(splats, &untrained) {
                Ok(bake) => {
                    refiner.kept_floor = Some(bake.floor);
                    (bake.splats, Some((bake.opacities, bake.scales)))
                }
                Err(splats) => (splats, None),
            },
            None => (splats.bake_min_scale(), None),
        }
    }

    /// The rows this step leaves alone: splats beyond the training radius plus the rows the caller holds. Records the
    /// radius part in `last_frozen` (`None` with the radius off) and drops held rows that no longer fit `rows`.
    pub(super) fn frozen_for_step(
        &mut self,
        means: Tensor<2>,
        camera_pos: glam::Vec3,
        rows: usize,
    ) -> Option<Tensor<1, Bool>> {
        let beyond = (self.config.train_radius_m > 0.0)
            .then(|| beyond_radius(means, camera_pos, self.config.train_radius_m));
        self.last_frozen = beyond.clone();
        if let Some(held) = &self.frozen_rows
            && !Self::held_fits(held.dims()[0], Some(rows))
        {
            self.frozen_rows = None;
        }
        either(beyond, self.frozen_rows.clone())
    }

    /// Marks the rows `frozen` does not hold as trained in the refine record and the eviction life. Call after the
    /// record gathered this step's stats.
    pub(super) fn mark_trained(
        &mut self,
        frozen: Option<&Tensor<1, Bool>>,
        visible: &Tensor<1>,
        rows: u32,
        device: &Device,
    ) {
        if let Some(record) = self.refine_record.as_mut() {
            match frozen {
                Some(frozen) => record.mark_trained(frozen.clone().bool_not().float()),
                // Rows held earlier in the window train again.
                None if record.trained.is_some() => record.mark_trained(visible.clone().ones_like()),
                None => {}
            }
        }
        if let Some(e) = self.evict.as_mut() {
            e.life
                .get_or_insert_with(|| SplatLife::new(rows as usize, device))
                .mark_trained(frozen);
        }
    }
}
