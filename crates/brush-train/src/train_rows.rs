//! Row surgery on the trainer: read splat rows to the host, take them out of training and append them back, with
//! the optimizer moments, refine record, life state, origin tags and Mip floor kept row-aligned the way prune and
//! append keep them.

use super::{SplatTrainer, map_splats_and_opt};
use crate::evict::SplatLife;
use crate::rows::SplatRows;
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Int, Tensor, TensorData, Transaction};

/// `t`'s rows at `rows`, all of them when `None`.
fn pick<const D: usize>(t: Tensor<D>, rows: Option<&Tensor<1, Int>>) -> Tensor<D> {
    match rows {
        Some(r) => t.select(0, r.clone()),
        None => t,
    }
}

impl SplatTrainer {
    /// Every row of `splats` (as between steps) on the host, with its Mip floor, life state and origin tag.
    pub async fn read_rows(&self, splats: &Splats) -> SplatRows {
        self.read_rows_at(splats, None).await
    }

    /// The rows of `splats` at `rows` (all when `None`), in that order, read back in one transaction.
    async fn read_rows_at(&self, splats: &Splats, rows: Option<&[usize]>) -> SplatRows {
        let n = splats.num_splats() as usize;
        let indices = rows.map(|r| index_tensor(r, &splats.device().inner()));
        let at = indices.as_ref();
        let count = rows.map_or(n, <[usize]>::len);
        let life = self.evict.as_ref().and_then(|e| e.life.as_ref());
        if let Some(life) = life {
            assert_eq!(life.len(), n, "life out of step with the splats");
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
                .register(pick(life.importance.clone(), at));
        }
        let mut data = tx
            .execute_async()
            .await
            .expect("splat rows readback")
            .into_iter()
            .map(|d| d.convert::<f32>().try_into_vec::<f32>().expect("f32 rows"));
        let mut next = || data.next().expect("one buffer per registered tensor");

        let transforms = next();
        let sh_coeffs = next();
        let raw_opacities = next();
        let min_scale = if splats.min_scale.is_some() {
            next()
        } else {
            Vec::new()
        };
        let (age, importance) = if life.is_some() {
            (next(), next())
        } else {
            (vec![0.0; count], vec![f32::INFINITY; count])
        };
        let origin = match &self.origin {
            Some(o) => {
                let all = o.slots().await;
                match rows {
                    Some(r) => r.iter().map(|&i| all[i]).collect(),
                    None => all,
                }
            }
            None => vec![None; count],
        };

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
        }
    }

    /// Takes the rows flagged in `take` (one flag per splat) out of `splats` (as between steps) and hands them back
    /// on the host. The remaining rows keep their parameters, Adam moments, refine record, life state, origin tags,
    /// Mip floor and last frozen flag, in order. When no row or every row is flagged, `splats` come back unchanged
    /// with `None`.
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
        let rows = self.read_rows_at(&splats, Some(&taken)).await;

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
        self.refine_record = self.refine_record.take().map(|r| r.keep(keep_inner));
        self.check_origin_rows(&splats);
        (splats, Some(rows))
    }

    /// Appends `rows` to `splats` (as between steps) with their origin tags, age and importance. Their Adam moments
    /// start at zero and the refine record counts them as never observed (and untrained with the training radius on).
    /// When `splats` carry a Mip floor the rows keep theirs (a zero one without), so they come back as they were
    /// taken; otherwise their floor is folded into their params. Panics when the rows' SH degree differs from the
    /// splats'.
    pub fn append_rows(&mut self, splats: Splats, rows: &SplatRows) -> Splats {
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
        let splats = if splats.min_scale.is_some() {
            let mut new = new;
            let floor = new.min_scale.take();
            self.append_raw(splats, new, floor, None)
        } else {
            self.append_raw(splats, new.bake_min_scale(), None, None)
        };
        if let Some(origin) = self.origin.as_mut() {
            origin.set_tail(&rows.origin);
        }
        if let Some(life) = self.evict.as_mut().and_then(|e| e.life.as_mut()) {
            life.set_tail(&rows.age, &rows.importance);
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

fn index_tensor(rows: &[usize], device: &burn::tensor::Device) -> Tensor<1, Int> {
    let data: Vec<i32> = rows.iter().map(|&i| i as i32).collect();
    Tensor::from_data(TensorData::new(data, [rows.len()]), device)
}

#[cfg(all(test, not(target_family = "wasm")))]
#[path = "train_rows_tests.rs"]
mod tests;
