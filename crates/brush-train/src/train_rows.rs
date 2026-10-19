//! Row surgery on the trainer: read splat rows to the host, take them out of training and append them back, with
//! the optimizer moments, refine record, life state, origin tags and Mip floor kept row-aligned the way prune and
//! append keep them. Rows read to the host carry their Adam moments and trained flags, so rows appended back resume
//! where they stopped. Adam's bias correction uses the trainer's one step count for every row; a row appended back
//! with its moments is corrected like a row that stayed and was held still, which is what it would have been.

use super::{SplatTrainer, map_splats_and_opt};
use crate::adam_scaled::AdamState;
use crate::evict::SplatLife;
use crate::origin::slot_of;
use crate::rows::{RowMoments, SplatRows};
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Device, Int, Tensor, TensorData, Transaction, kind::Basic};

/// Adam moments (first, second) per parameter group for rows being appended, on the optimizer's device.
pub(super) struct MomentTails {
    pub transforms: (Tensor<2>, Tensor<2>),
    pub sh_coeffs: (Tensor<3>, Tensor<3>),
    pub opacities: (Tensor<1>, Tensor<1>),
}

impl MomentTails {
    /// `m`'s `n` rows with `coeffs` SH coefficients per channel on `device`; the SH second moment is `[n, 1, 1]`.
    fn new(m: &RowMoments, n: usize, coeffs: usize, device: &Device) -> Self {
        let t2 = |v: &[f32]| Tensor::<2>::from_data(TensorData::new(v.to_vec(), [n, 10]), device);
        let t1 = |v: &[f32]| Tensor::<1>::from_data(TensorData::new(v.to_vec(), [n]), device);
        Self {
            transforms: (t2(&m.transforms_m1), t2(&m.transforms_m2)),
            sh_coeffs: (
                Tensor::from_data(TensorData::new(m.sh_m1.clone(), [n, coeffs, 3]), device),
                Tensor::from_data(TensorData::new(m.sh_m2.clone(), [n, 1, 1]), device),
            ),
            opacities: (t1(&m.opacity_m1), t1(&m.opacity_m2)),
        }
    }
}

/// `t`'s rows at `rows`, all of them when `None`. Pick before reshaping inside one transaction (burn bug, issue #78).
fn pick<const D: usize, K: Basic>(t: Tensor<D, K>, rows: Option<&Tensor<1, Int>>) -> Tensor<D, K> {
    match rows {
        Some(r) => t.select(0, r.clone()),
        None => t,
    }
}

impl SplatTrainer {
    /// Every row of `splats` (as between steps) on the host, with its Mip floor, life state (trained flag included),
    /// origin tag and Adam moments. Moments are `None` before the first step, trained flags empty without eviction.
    pub async fn read_rows(&self, splats: &Splats) -> SplatRows {
        self.read_rows_at(splats, None, true).await
    }

    /// [`Self::read_rows`] without the Adam moments, for exports that need the model only.
    pub async fn read_model_rows(&self, splats: &Splats) -> SplatRows {
        self.read_rows_at(splats, None, false).await
    }

    /// The rows of `splats` at `rows` (all when `None`), in that order, read back in one transaction; with their Adam
    /// moments when `with_moments` and every parameter group has them. The moments are picked before they are
    /// reshaped: reducing the SH second moment and then picking rows read back the rows one below the picked ones
    /// (burn bug, issue #78).
    async fn read_rows_at(&self, splats: &Splats, rows: Option<&[usize]>, with_moments: bool) -> SplatRows {
        let n = splats.num_splats() as usize;
        let indices = rows.map(|r| index_tensor(r, &splats.device().inner()));
        let at = indices.as_ref();
        let count = rows.map_or(n, <[usize]>::len);
        let life = self.evict.as_ref().and_then(|e| e.life.as_ref());
        if let Some(life) = life {
            assert_eq!(life.len(), n, "life out of step with the splats");
        }
        if let Some(origin) = &self.origin {
            assert_eq!(origin.len(), n, "origin tags out of step with the splats");
        }

        let mut tx = Transaction::default()
            .register(pick(splats.transforms.val().inner(), at))
            .register(pick(splats.sh_coeffs.val().inner(), at))
            .register(pick(splats.raw_opacities.val().inner(), at));
        if let Some(floor) = &splats.min_scale {
            tx = tx.register(pick(floor.clone(), at));
        }
        if let Some(life) = life {
            tx = tx
                .register(pick(life.age.clone(), at))
                .register(pick(life.importance.clone(), at))
                .register(pick(life.trained.clone(), at));
        }
        if let Some(o) = self.optim.as_ref().filter(|_| with_moments) {
            let groups = [
                ("transforms", moment_rows(&o.transforms)),
                ("SH", moment_rows(&o.sh_coeffs)),
                ("opacity", moment_rows(&o.opacities)),
            ];
            for (name, rows) in groups {
                if let Some((m1, m2)) = rows {
                    assert!(m1 == n && m2 == n, "{name} moments ({m1}, {m2} rows) out of step with {n} splats");
                }
            }
        }
        let moments = self.optim.as_ref().filter(|_| with_moments).and_then(|o| {
            Some([
                pick(o.transforms.momentum.as_ref()?.moment_1.clone(), at),
                pick(o.transforms.momentum.as_ref()?.moment_2.clone(), at),
                pick(o.sh_coeffs.momentum.as_ref()?.moment_1.clone(), at).flatten(1, 2),
                per_row(pick(o.sh_coeffs.momentum.as_ref()?.moment_2.clone(), at)),
                pick(o.opacities.momentum.as_ref()?.moment_1.clone(), at).unsqueeze_dim(1),
                pick(o.opacities.momentum.as_ref()?.moment_2.clone(), at).unsqueeze_dim(1),
            ])
        });
        if let Some(m) = &moments {
            for t in m {
                tx = tx.register(t.clone());
            }
        }
        if let Some(origin) = &self.origin {
            tx = tx.register(pick(origin.tags(), at));
        }
        let mut data = tx
            .execute_async()
            .await
            .expect("splat rows readback")
            .into_iter();
        let origin = match &self.origin {
            Some(_) => data
                .next_back()
                .expect("origin tags readback")
                .convert::<i32>()
                .try_into_vec::<i32>()
                .expect("int origin tags")
                .into_iter()
                .map(slot_of)
                .collect(),
            None => vec![None; count],
        };
        let mut next = || {
            data.next()
                .expect("one buffer per registered tensor")
                .convert::<f32>()
                .try_into_vec::<f32>()
                .expect("f32 rows")
        };

        let transforms = next();
        let sh_coeffs = next();
        let raw_opacities = next();
        let min_scale = if splats.min_scale.is_some() {
            next()
        } else {
            Vec::new()
        };
        let (age, importance, trained) = if life.is_some() {
            let (age, importance) = (next(), next());
            (age, importance, next().into_iter().map(|t| t > 0.5).collect())
        } else {
            (vec![0.0; count], vec![f32::INFINITY; count], Vec::new())
        };
        let moments = moments.map(|_| RowMoments {
            transforms_m1: next(),
            transforms_m2: next(),
            sh_m1: next(),
            sh_m2: next(),
            opacity_m1: next(),
            opacity_m2: next(),
        });

        let mut means = Vec::with_capacity(count * 3);
        let mut rotations = Vec::with_capacity(count * 4);
        let mut log_scales = Vec::with_capacity(count * 3);
        for row in transforms.chunks_exact(10) {
            means.extend_from_slice(&row[0..3]);
            rotations.extend_from_slice(&row[3..7]);
            log_scales.extend_from_slice(&row[7..10]);
        }
        SplatRows {
            means,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
            min_scale,
            origin,
            age,
            importance,
            sh_coeffs_per_channel: splats.sh_coeffs.dims()[1] as u32,
            trained,
            moments,
        }
    }

    /// Takes the rows flagged in `take` (one flag per splat) out of `splats` (as between steps) and hands them back
    /// on the host as [`Self::read_rows`] reads them. The remaining rows keep their parameters, Adam moments, refine
    /// record, life state, origin tags, Mip floor, last frozen flag, held flag ([`SplatTrainer::set_frozen_rows`])
    /// and group ([`SplatTrainer::set_row_budget`]), in order. When no row or every row is flagged, `splats` come back
    /// unchanged with `None`.
    pub async fn take_rows(
        &mut self,
        splats: Splats,
        take: &[bool],
    ) -> (Splats, Option<SplatRows>) {
        let n = splats.num_splats() as usize;
        assert_eq!(take.len(), n, "one take flag per splat");
        let (taken, kept): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| take[i]);
        if taken.is_empty() || kept.is_empty() {
            return (splats, None);
        }
        let rows = self.read_rows_at(&splats, Some(&taken), true).await;

        let keep = index_tensor(&kept, &splats.device());
        let keep_inner = keep.clone().without_autodiff();
        let mut splats = match self.optim.as_mut() {
            Some(optim) => map_splats_and_opt(
                splats,
                optim,
                |x| x.select(0, keep.clone()),
                |x| x.select(0, keep.clone()),
                |x| x.select(0, keep.clone()),
                |x| x.select(0, keep_inner.clone()),
                |x| x.select(0, keep_inner.clone()),
                |x| x.select(0, keep_inner.clone()),
            ),
            None => {
                let mut s = splats;
                s.transforms = s.transforms.map(|x| x.select(0, keep.clone()));
                s.sh_coeffs = s.sh_coeffs.map(|x| x.select(0, keep.clone()));
                s.raw_opacities = s.raw_opacities.map(|x| x.select(0, keep.clone()));
                s
            }
        };
        splats.min_scale = splats.min_scale.map(|f| f.select(0, keep_inner.clone()));
        if let Some(life) = self.evict.as_mut().and_then(|e| e.life.as_mut()) {
            life.keep(keep_inner.clone());
        }
        if let Some(origin) = self.origin.as_mut() {
            origin.keep(keep_inner.clone());
        }
        self.last_frozen = self
            .last_frozen
            .take()
            .map(|f| f.select(0, keep_inner.clone()));
        self.frozen_rows = self
            .frozen_rows
            .take()
            .map(|f| f.select(0, keep_inner.clone()));
        if let Some(budget) = self.row_budget.as_mut() {
            budget.remove(take);
        }
        self.refine_record = self.refine_record.take().map(|r| r.keep(keep_inner));
        self.check_origin_rows(&splats);
        (splats, Some(rows))
    }

    /// Appends `rows` to `splats` (as between steps) with their origin tags, age, importance, trained flags (trained
    /// without flags) and Adam moments (zero without moments, or before the first step, which starts every row's
    /// moments at zero). The refine record counts them as never observed (and untrained with the training radius on).
    /// When `splats` carry a Mip floor the rows keep theirs (a zero one without), so they come back as they were
    /// taken; otherwise their floor is folded into their params and their moments stay as they are. Panics when a
    /// column of `rows` is out of length ([`SplatRows::check`]) or their SH degree differs from the splats'.
    pub fn append_rows(&mut self, splats: Splats, rows: &SplatRows) -> Splats {
        if let Err(e) = rows.check() {
            panic!("{e}");
        }
        if rows.is_empty() {
            return splats;
        }
        assert_eq!(
            rows.sh_coeffs_per_channel as usize,
            splats.sh_coeffs.dims()[1],
            "restored rows must match the live SH degree"
        );
        let device = splats.device().inner();
        // Life starts at the first step; rows appended before it carry state the step would otherwise reset.
        if let Some(e) = self.evict.as_mut() {
            e.life
                .get_or_insert_with(|| SplatLife::new(splats.num_splats() as usize, &device));
        }
        let new = rows.to_splats(&device);
        let coeffs = rows.sh_coeffs_per_channel as usize;
        let moments = rows.moments.as_ref().map(|m| MomentTails::new(m, rows.len(), coeffs, &device));
        let splats = if splats.min_scale.is_some() {
            let mut new = new;
            let floor = new.min_scale.take();
            self.append_raw(splats, new, floor, None, moments)
        } else {
            self.append_raw(splats, new.bake_min_scale(), None, None, moments)
        };
        if let Some(origin) = self.origin.as_mut() {
            origin.set_tail(&rows.origin);
        }
        if let Some(life) = self.evict.as_mut().and_then(|e| e.life.as_mut()) {
            life.set_tail(&rows.age, &rows.importance, &rows.trained);
        }
        splats
    }

    /// Per splat, the training steps since the last refine in which it contributed to the render (the refine
    /// record's `importance_views`); `None` before the first step after a refine.
    pub fn seen_since_refine(&self) -> Option<Tensor<1>> {
        self.refine_record
            .as_ref()
            .map(|r| r.importance_views.clone())
    }
}

/// The row counts of `state`'s first and second moment; `None` before the first step.
fn moment_rows<const D: usize>(state: &AdamState<D>) -> Option<(usize, usize)> {
    state.momentum.as_ref().map(|m| (m.moment_1.dims()[0], m.moment_2.dims()[0]))
}

/// The reduced SH second moment `[N, 1, 1]` as `[N, 1]`.
fn per_row(t: Tensor<3>) -> Tensor<2> {
    let [n, a, b] = t.dims();
    debug_assert_eq!((a, b), (1, 1), "the SH second moment is reduced per row");
    t.reshape([n, 1])
}

fn index_tensor(rows: &[usize], device: &burn::tensor::Device) -> Tensor<1, Int> {
    let data: Vec<i32> = rows.iter().map(|&i| i as i32).collect();
    Tensor::from_data(TensorData::new(data, [rows.len()]), device)
}

#[cfg(all(test, not(target_family = "wasm")))]
#[path = "train_rows_tests.rs"]
mod tests;

#[cfg(all(test, not(target_family = "wasm")))]
#[path = "train_moments_tests.rs"]
mod moments_tests;
