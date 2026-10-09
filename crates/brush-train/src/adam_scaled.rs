use burn::tensor::{ElementConversion, Int, Tensor};

/// Largest |m̂ / (√v̂ + ε)| one step may take, before the learning rate. Consistent moments stay below
/// `(1 - β1) / √(1 - β2) · √k ≈ 3.2 · √k` for a second moment reduced over `k` elements (about 11 for the 12 SH
/// coefficients of degree 1; healthy replays measure at most 12.5 without bias correction). A larger value means the
/// moments are inconsistent, e.g. an aborted GPU command buffer left `v` unwritten (zero): dividing by ε = 1e-15
/// would then move the parameter by ~1e12 · m̂ per step.
pub(crate) const MAX_NORMALIZED_STEP: f32 = 32.0;

/// Adam with per-parameter second-moment reduction (via [`AdamState::reduce_moment_2`])
/// and per-component learning-rate scaling (via [`AdamState::scaling`]).
///
/// Hand-rolled rather than built on burn's `Adam`, which has neither the
/// per-component LR scaling nor the second-moment reduction below. The
/// trainer also edits momentum tensors in place during refine (split/prune)
/// and swaps the transforms LR scaling every step; burn's low-level
/// `Optimizer` trait does now hand back live `Tensor` state rather than only
/// serialized records, so that part is no longer a blocker — the two missing
/// behaviours are what keep this separate.
#[derive(Clone)]
pub(crate) struct AdamScaled {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Per-parameter momentum state. When `reduce_moment_2` is set on the owning
/// [`AdamState`], `moment_2` has size 1 in trailing dims; [`AdamState::map_momentum`]
/// callers must stay shape-agnostic along those.
#[derive(Clone)]
pub(crate) struct MomentumState<const D: usize> {
    pub moment_1: Tensor<D>,
    pub moment_2: Tensor<D>,
    pub time: usize,
}

/// Per-parameter optimizer state.
#[derive(Clone)]
pub(crate) struct AdamState<const D: usize> {
    pub momentum: Option<MomentumState<D>>,
    /// Per-component learning rate scaling (e.g. different LR for means vs
    /// rotations vs scales within the transforms tensor).
    pub scaling: Option<Tensor<D>>,
    /// When true, the second moment is reduced to a scalar per row. Set by the
    /// caller when initializing state for parameters where per-element variance
    /// is not needed.
    pub reduce_moment_2: bool,
    /// Elements whose step hit [`MAX_NORMALIZED_STEP`] since the last [`Self::take_clamped`], summed on the device so
    /// counting needs no readback per step.
    pub clamped: Option<Tensor<1, Int>>,
}

impl<const D: usize> AdamState<D> {
    pub fn new(scaling: Option<Tensor<D>>, reduce_moment_2: bool) -> Self {
        Self {
            momentum: None,
            scaling,
            reduce_moment_2,
            clamped: None,
        }
    }

    /// The clamped-element count since the last call (a `[1]` tensor), `None` if no step ran since.
    pub fn take_clamped(&mut self) -> Option<Tensor<1, Int>> {
        self.clamped.take()
    }

    /// Apply `map_fn` to both momentum tensors (no-op before the first step).
    /// `map_fn` must be shape-agnostic along trailing dims since `moment_2`
    /// may have size-1 trailing dims under `reduce_moment_2`.
    pub fn map_momentum(&mut self, map_fn: impl Fn(Tensor<D>) -> Tensor<D>) {
        self.momentum = self.momentum.take().map(|mut moment| {
            moment.moment_1 = map_fn(moment.moment_1);
            moment.moment_2 = map_fn(moment.moment_2);
            moment
        });
    }
}

impl AdamScaled {
    pub fn new(epsilon: f32) -> Self {
        Self {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon,
        }
    }

    /// One Adam step for a single parameter. `tensor` and `grad` live on the
    /// inner (non-autodiff) backend; `state` is updated in place.
    ///
    /// `grad_sq_mean` supplies the reduced second moment when the caller can
    /// produce it more cheaply than squaring `grad`. It must equal what
    /// `mean_trailing_dims(grad * grad)` would give, and is only read when
    /// [`AdamState::reduce_moment_2`] is set.
    ///
    /// The normalized step `m̂ / (√v̂ + ε)` is clamped to ±[`MAX_NORMALIZED_STEP`] and the clamped elements are
    /// counted in [`AdamState::clamped`].
    pub fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        state: &mut AdamState<D>,
    ) -> Tensor<D> {
        let (grad, momentum) = self.transform(
            grad,
            grad_sq_mean,
            state.momentum.take(),
            state.reduce_moment_2,
        );
        state.momentum = Some(momentum);
        let over = grad
            .clone()
            .abs()
            .greater_elem(MAX_NORMALIZED_STEP)
            .int()
            .sum();
        state.clamped = Some(match state.clamped.take() {
            Some(count) => count + over,
            None => over,
        });
        let grad = grad.clamp(-MAX_NORMALIZED_STEP, MAX_NORMALIZED_STEP);

        let delta = if let Some(scale) = &state.scaling {
            grad * (scale.clone() * lr).unsqueeze()
        } else {
            grad * lr
        };
        tensor - delta
    }

    fn transform<const D: usize>(
        &self,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        momentum_state: Option<MomentumState<D>>,
        reduce_moment_2: bool,
    ) -> (Tensor<D>, MomentumState<D>) {
        let grad_sq_for_moment = if reduce_moment_2 && D > 1 {
            grad_sq_mean.unwrap_or_else(|| mean_trailing_dims(grad.clone().powi_scalar(2)))
        } else {
            grad.clone().powi_scalar(2)
        };

        let state = if let Some(mut state) = momentum_state {
            let factor = 1.0 - self.beta_1;
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor));

            let factor = 1.0 - self.beta_2;
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad_sq_for_moment.mul_scalar(factor));

            state.time += 1;
            state
        } else {
            let factor = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor);

            let factor = 1.0 - self.beta_2;
            let moment_2 = grad_sq_for_moment.mul_scalar(factor);

            MomentumState {
                moment_1,
                moment_2,
                time: 1,
            }
        };

        let time = (state.time as i32).elem();
        let moment_1_corrected = state
            .moment_1
            .clone()
            .mul_scalar(1f32 / (1f32 - self.beta_1.powi(time)));
        let moment_2_corrected = state
            .moment_2
            .clone()
            .mul_scalar(1f32 / (1f32 - self.beta_2.powi(time)));
        // moment_2_corrected broadcasts when it has reduced trailing dims
        let grad = moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        (grad, state)
    }
}

/// Reduce to a single mean per row by averaging across all trailing dims (1..D).
/// Result has size 1 in each trailing dim so it broadcasts back to the full shape.
fn mean_trailing_dims<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    debug_assert!(D > 1, "mean_trailing_dims requires D > 1");
    let shape = t.dims();
    let trailing_count: usize = shape[1..].iter().product();

    // Reduce over the trailing dims directly; a flatten + reshape would end
    // burn's fusion block.
    let dims: Vec<usize> = (1..D).collect();
    t.sum_dims(&dims) / trailing_count as f32
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use burn::tensor::Device;

    /// A first moment with a zero second moment (an output an aborted GPU command buffer left unwritten) and no
    /// gradient: the step stays within `MAX_NORMALIZED_STEP · lr` instead of dividing by the epsilon.
    #[tokio::test]
    async fn zero_second_moment_step_is_bounded() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let adam = AdamScaled::new(1e-15);
        let mut state = AdamState::<2>::new(None, true);
        state.momentum = Some(MomentumState {
            moment_1: Tensor::full([4, 3], 1e-3, &device),
            moment_2: Tensor::zeros([4, 1], &device),
            time: 100,
        });
        let lr = 2e-3;
        let param = Tensor::<2>::zeros([4, 3], &device);
        let grad = Tensor::<2>::zeros([4, 3], &device);
        let stepped = adam.step(lr, param, &grad, None, &mut state);
        let moved = stepped
            .abs()
            .max()
            .into_data_async()
            .await
            .expect("readback")
            .try_into_vec::<f32>()
            .expect("f32")[0];
        let bound = MAX_NORMALIZED_STEP * lr as f32;
        assert!(moved <= bound * 1.001, "moved {moved}, bound {bound}");
    }
}
