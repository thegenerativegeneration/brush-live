//! Refine bookkeeping beyond upstream's prune checks: Adam steps the optimizer clamped, and splats whose colour
//! diverged.

use brush_render::gaussian_splats::Splats;
use burn::tensor::{Bool, Tensor, s};

use super::{MAX_SH0, SplatOptim, SplatTrainer};

impl SplatTrainer {
    /// Adds the Adam steps `optim` clamped since the last refine to `adam_clamped` and resets their counts.
    pub(super) async fn count_clamped(&mut self, optim: &mut SplatOptim) {
        let clamped = [
            optim.transforms.take_clamped(),
            optim.sh_coeffs.take_clamped(),
            optim.opacities.take_clamped(),
        ]
        .into_iter()
        .flatten()
        .reduce(|a, b| a + b);
        if let Some(clamped) = clamped {
            let n = clamped
                .into_scalar_async::<i64>()
                .await
                .expect("Failed to count clamped optimizer steps");
            self.adam_clamped += n.max(0) as u64;
        }
    }

    /// The rows of `splats` whose DC colour coefficient exceeds [`MAX_SH0`] in any channel, leaving out the rows
    /// already flagged in `non_finite`. Adds their count to `sh_pruned`.
    pub(super) async fn diverged_colours(
        &mut self,
        splats: &Splats,
        non_finite: Tensor<1, Bool>,
    ) -> Tensor<1, Bool> {
        let n = splats.num_splats() as usize;
        let diverged = splats
            .sh_coeffs
            .val()
            .slice(s![.., 0..1, ..])
            .reshape([n, 3])
            .abs()
            .greater_elem(MAX_SH0)
            .any_dim(1)
            .squeeze_dim(1)
            .bool_and(non_finite.bool_not());
        let count = diverged
            .clone()
            .int()
            .sum()
            .into_scalar_async::<i32>()
            .await
            .expect("Failed to count diverged colours") as u32;
        self.sh_pruned += u64::from(count);
        diverged
    }
}
