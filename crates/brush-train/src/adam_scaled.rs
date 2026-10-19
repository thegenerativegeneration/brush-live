use burn::tensor::{Bool, ElementConversion, Int, Tensor};

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

    /// Appends `n` rows to both moments: `tail`'s (first, second moment; a second moment with size-1 trailing dims
    /// broadcasts to this state's), or zeros. No-op before the first step, which starts every row's moments at zero.
    pub fn append_rows(&mut self, n: usize, tail: Option<(Tensor<D>, Tensor<D>)>) {
        let Some(m) = self.momentum.as_mut() else {
            return;
        };
        let rows = |old: &Tensor<D>| {
            let mut dims = old.dims();
            dims[0] = n;
            dims
        };
        let (dims_1, dims_2) = (rows(&m.moment_1), rows(&m.moment_2));
        let device = m.moment_1.device();
        let (tail_1, tail_2) = match tail {
            Some((t1, t2)) => (t1, t2.expand(dims_2)),
            None => (Tensor::zeros(dims_1, &device), Tensor::zeros(dims_2, &device)),
        };
        m.moment_1 = Tensor::cat(vec![m.moment_1.clone(), tail_1], 0);
        m.moment_2 = Tensor::cat(vec![m.moment_2.clone(), tail_2], 0);
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

/// Adam's first-moment decay per step.
pub const BETA_1: f32 = 0.9;

impl AdamScaled {
    pub fn new(epsilon: f32) -> Self {
        Self {
            beta_1: BETA_1,
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
    ///
    /// Rows set in `frozen` (`[N]`, one per first-dim row) get no update: their values and both moments keep their
    /// bits (moments start at zero for rows frozen on the first step), as a visibility-masked sparse Adam leaves the
    /// rows it skips. The step count is shared and still advances.
    pub fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        state: &mut AdamState<D>,
        frozen: Option<&Tensor<1, Bool>>,
    ) -> Tensor<D> {
        let old = frozen.map(|_| state.momentum.clone());
        let (grad, mut momentum) = self.transform(
            grad,
            grad_sq_mean,
            state.momentum.take(),
            state.reduce_moment_2,
        );
        let grad = if let (Some(frozen), Some(old)) = (frozen, old) {
            let (m1, m2) = match old {
                Some(old) => (old.moment_1, old.moment_2),
                None => (
                    momentum.moment_1.zeros_like(),
                    momentum.moment_2.zeros_like(),
                ),
            };
            momentum.moment_1 = keep_rows(momentum.moment_1, m1, frozen);
            momentum.moment_2 = keep_rows(momentum.moment_2, m2, frozen);
            let mask = row_mask(frozen, grad.dims());
            grad.mask_fill(mask, 0.0)
        } else {
            grad
        };
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
        match frozen {
            Some(frozen) => keep_rows(tensor.clone() - delta, tensor, frozen),
            None => tensor - delta,
        }
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

/// `[N]` row flags broadcast to `dims` (`[N, ..]`).
fn row_mask<const D: usize>(rows: &Tensor<1, Bool>, dims: [usize; D]) -> Tensor<D, Bool> {
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    rows.clone().reshape(shape).expand(dims)
}

/// `new` with the rows set in `rows` taken from `old`.
fn keep_rows<const D: usize>(new: Tensor<D>, old: Tensor<D>, rows: &Tensor<1, Bool>) -> Tensor<D> {
    let mask = row_mask(rows, new.dims());
    new.mask_where(mask, old)
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
    use burn::tensor::{Device, TensorData};

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
        let stepped = adam.step(lr, param, &grad, None, &mut state, None);
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

    /// Task 17b repro of a burn fusion bug: tensors reshaped and then row-selected in one transaction, as the
    /// trainer's row read first registered the moments, read the reduced second moment back shifted by one row, the
    /// last value taken from another tensor. Fails as of 2026-10-16 (alone, in a fresh process); one reduce and select
    /// on its own reads back correctly. The row read now selects before it reshapes.
    #[tokio::test]
    #[ignore = "fails: burn fusion row shift (repro)"]
    async fn a_reduced_second_moment_selects_its_own_rows() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let adam = AdamScaled::new(1e-15);
        let mut state = AdamState::<3>::new(None, true);
        let n: usize = 75;
        let grad = |k: f32| {
            Tensor::<1>::from_floats((0..n * 12).map(|i| ((i * 31) % 17) as f32 * k - 0.5 * k).collect::<Vec<_>>().as_slice(), &device)
                .reshape([n, 4, 3])
        };
        let mut param = Tensor::<3>::zeros([n, 4, 3], &device);
        for k in [1e-3, 2e-3] {
            param = adam.step(1e-2, param, &grad(k), None, &mut state, None);
        }
        let m = state.momentum.expect("stepped");
        let full: Vec<f32> = m.moment_2.clone().into_data_async().await.unwrap().try_to_vec().unwrap();
        let idx: Vec<i32> = (0..n as i32).filter(|i| i % 3 == 0).collect();
        let at = Tensor::<1, Int>::from_data(TensorData::new(idx.clone(), [idx.len()]), &device);
        // As the trainer's row read registered them: reshaped, then selected, in one transaction.
        let tx = burn::tensor::Transaction::default()
            .register(param.flatten::<2>(1, 2).select(0, at.clone()))
            .register(m.moment_1.flatten::<2>(1, 2).select(0, at.clone()))
            .register(m.moment_2.flatten::<2>(1, 2).mean_dim(1).select(0, at.clone()))
            .register(full_like_rows(n, &device).unsqueeze_dim::<2>(1).select(0, at));
        let picked: Vec<f32> = tx.execute_async().await.unwrap().remove(2).try_to_vec().unwrap();
        let want: Vec<f32> = idx.iter().map(|&i| full[i as usize]).collect();
        assert_eq!(picked, want);
    }

    fn full_like_rows(n: usize, device: &Device) -> Tensor<1> {
        Tensor::<1>::from_floats((0..n).map(|i| i as f32).collect::<Vec<_>>().as_slice(), device)
    }

    async fn bits<const D: usize>(t: Tensor<D>) -> Vec<u32> {
        t.into_data_async()
            .await
            .expect("readback")
            .try_into_vec::<f32>()
            .expect("f32")
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    /// One reduced-moment step of a `[4, 3]` parameter from nonzero moments, with `frozen` rows.
    async fn step_with(frozen: Option<[bool; 4]>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let adam = AdamScaled::new(1e-15);
        let mut state = AdamState::<2>::new(None, true);
        let ramp = |k: f32| {
            Tensor::<1>::from_floats(
                std::array::from_fn::<f32, 12, _>(|i| (i as f32 - 5.5) * k),
                &device,
            )
            .reshape([4, 3])
        };
        state.momentum = Some(MomentumState {
            moment_1: ramp(1e-4),
            moment_2: Tensor::from_floats([[1e-6], [2e-6], [3e-6], [4e-6]], &device),
            time: 10,
        });
        let frozen = frozen.map(|f| Tensor::<1, Bool>::from_data(TensorData::from(f), &device));
        let stepped = adam.step(1e-2, ramp(1.0), &ramp(-3e-3), None, &mut state, frozen.as_ref());
        let m = state.momentum.expect("momentum");
        (bits(stepped).await, bits(m.moment_1).await, bits(m.moment_2).await)
    }

    /// Whether `a` and `b` are the same floats up to one unit in the last place: the masked step fuses into other
    /// kernels than the unmasked one, which may contract a multiply-add differently.
    fn within_ulp(a: &[u32], b: &[u32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.abs_diff(*y) <= 1)
    }

    /// An empty mask steps as no mask.
    #[tokio::test]
    async fn frozen_rows_none_set_is_the_unmasked_step() {
        let (a, b) = (step_with(None).await, step_with(Some([false; 4])).await);
        assert!(within_ulp(&a.0, &b.0) && within_ulp(&a.1, &b.1) && within_ulp(&a.2, &b.2), "{a:?} vs {b:?}");
    }

    /// Frozen rows keep their values and moments; the others step as without a mask.
    #[tokio::test]
    async fn frozen_rows_keep_values_and_moments() {
        let (p0, m10, m20) = step_with(None).await;
        let (p, m1, m2) = step_with(Some([false, true, false, true])).await;
        let ramp = |k: f32| (0..12).map(|i| ((i as f32 - 5.5) * k).to_bits()).collect::<Vec<_>>();
        let before = (ramp(1.0), ramp(1e-4), [1e-6f32, 2e-6, 3e-6, 4e-6].map(f32::to_bits).to_vec());
        for row in 0..4 {
            let (r3, r1) = (row * 3..row * 3 + 3, row..row + 1);
            if row % 2 == 1 {
                assert_eq!(p[r3.clone()], before.0[r3.clone()], "row {row} value");
                assert_eq!(m1[r3.clone()], before.1[r3], "row {row} m1");
                assert_eq!(m2[r1.clone()], before.2[r1], "row {row} m2");
            } else {
                assert_ne!(p[r3.clone()], before.0[r3.clone()], "row {row} did not step");
                assert!(within_ulp(&p[r3.clone()], &p0[r3.clone()]), "row {row} value");
                assert!(within_ulp(&m1[r3.clone()], &m10[r3]), "row {row} m1");
                assert!(within_ulp(&m2[r1.clone()], &m20[r1]), "row {row} m2");
            }
        }
    }
}
