use burn::{
    prelude::Int,
    tensor::{Bool, Device, Tensor},
};
use tracing::trace_span;

pub(crate) struct RefineRecord {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: Tensor<1>,
    pub vis_weight: Tensor<1>,
    pub max_screen_size: Tensor<1>,
    // Eviction importance: summed per-step score and the number of steps with a
    // score above zero, per splat.
    pub importance_sum: Tensor<1>,
    pub importance_views: Tensor<1>,
}

impl RefineRecord {
    pub(crate) fn new(num_points: u32, device: &Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<1>::zeros([num_points as usize], device),
            vis_weight: Tensor::<1>::zeros([num_points as usize], device),
            max_screen_size: Tensor::<1>::zeros([num_points as usize], device),
            importance_sum: Tensor::<1>::zeros([num_points as usize], device),
            importance_views: Tensor::<1>::zeros([num_points as usize], device),
        }
    }

    pub(crate) fn above_threshold(&self, threshold: f32) -> Tensor<1, Bool> {
        self.refine_weight_norm
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    /// Visible splats whose max 2D screen-space extent (as a fraction of the
    /// image dim) exceeds `threshold` — i.e. the "too big on screen" outliers.
    pub(crate) fn above_screen_size(&self, threshold: f32) -> Tensor<1, Bool> {
        self.max_screen_size
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    pub(crate) fn gather_stats(
        &mut self,
        refine_weight: Tensor<1>,
        importance: Tensor<1>,
        visible: Tensor<1>,
        screen_radius: Tensor<1>,
    ) {
        let _span = trace_span!("Gather stats").entered();
        self.refine_weight_norm = refine_weight.max_pair(self.refine_weight_norm.clone());
        self.importance_views =
            self.importance_views.clone() + importance.clone().greater_elem(0.0).float();
        self.importance_sum = self.importance_sum.clone() + importance;
        self.vis_weight = self.vis_weight.clone() + visible;
        self.max_screen_size = screen_radius.max_pair(self.max_screen_size.clone());
    }

    pub(crate) fn vis_mask(&self) -> Tensor<1, Bool> {
        self.vis_weight.clone().greater_elem(0.0)
    }

    pub(crate) fn keep(self, indices: Tensor<1, Int>) -> Self {
        Self {
            refine_weight_norm: self.refine_weight_norm.select(0, indices.clone()),
            vis_weight: self.vis_weight.clone().select(0, indices.clone()),
            max_screen_size: self.max_screen_size.select(0, indices.clone()),
            importance_sum: self.importance_sum.select(0, indices.clone()),
            importance_views: self.importance_views.select(0, indices),
        }
    }

    /// Zero-pad the per-splat stats with `n` fresh (never-observed) entries,
    /// for splats appended after this record was created.
    pub(crate) fn pad(self, n: usize) -> Self {
        let device = self.vis_weight.device();
        let pad = |t: Tensor<1>| Tensor::cat(vec![t, Tensor::zeros([n], &device)], 0);
        Self {
            refine_weight_norm: pad(self.refine_weight_norm),
            vis_weight: pad(self.vis_weight),
            max_screen_size: pad(self.max_screen_size),
            importance_sum: pad(self.importance_sum),
            importance_views: pad(self.importance_views),
        }
    }
}
