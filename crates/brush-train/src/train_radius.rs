//! The training radius in the trainer: which splats a step leaves alone, and how refine keeps the rows no step
//! trained since the last refine as they were (raw params and Mip-Splatting floor).

use brush_render::gaussian_splats::{Splats, fold_min_scale};
use burn::module::Param;
use burn::tensor::{Bool, Tensor, activation::sigmoid, s};

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
/// (`kept`'s rows, the rows before the children) taken from `kept`.
pub(super) fn keep_floor(
    fresh: Tensor<1>,
    kept: Tensor<1>,
    untrained: Tensor<1, Bool>,
) -> Tensor<1> {
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
