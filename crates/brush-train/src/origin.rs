//! Which training view's seeds created each splat, row-aligned with the splats (a split child inherits its
//! parent's), so a correction of that view's pose can move the splats anchored to it: the rigid part of
//! Splat-SLAM's map deformation (Sandström et al., arXiv 2405.16544, §3.2).

use crate::quat_vec::quaternion_vec_multiply;
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Device, Int, Tensor, TensorData, s};
use glam::Affine3A;

/// Per splat: the seeding view's slot + 1; 0 for splats appended without one, which never move.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct SplatOrigin {
    rows: Tensor<1, Int>,
    /// At least the largest tag in `rows` (keep and split never raise it), so the move's lookup table covers every
    /// tag without a readback.
    max_tag: usize,
}

impl SplatOrigin {
    pub(crate) fn new(n: usize, slot: u32, device: &Device) -> Self {
        Self {
            rows: Self::filled(n, Some(slot), device),
            max_tag: tag(Some(slot)),
        }
    }

    fn filled(n: usize, slot: Option<u32>, device: &Device) -> Tensor<1, Int> {
        let value = tag(slot) as i32;
        Tensor::from_data(TensorData::new(vec![value; n], [n]), device)
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.dims()[0]
    }

    /// The per-row tags on the device (see [`slot_of`]).
    pub(crate) fn tags(&self) -> Tensor<1, Int> {
        self.rows.clone()
    }

    /// Each splat's seeding view slot, `None` without one; reads the tags back from the device.
    pub(crate) async fn slots(&self) -> Vec<Option<u32>> {
        let tags = self
            .rows
            .clone()
            .into_data_async()
            .await
            .expect("origin tags readback")
            .convert::<i32>()
            .try_into_vec::<i32>()
            .expect("int origin tags");
        tags.into_iter().map(slot_of).collect()
    }

    /// `n` appended splats seeded by view `slot` (`None`: no origin).
    pub(crate) fn pad(&mut self, n: usize, slot: Option<u32>) {
        let new = Self::filled(n, slot, &self.rows.device());
        self.rows = Tensor::cat(vec![self.rows.clone(), new], 0);
        self.max_tag = self.max_tag.max(tag(slot));
    }

    pub(crate) fn keep(&mut self, indices: Tensor<1, Int>) {
        self.rows = self.rows.clone().select(0, indices);
    }

    /// Sets the tags of the last `slots.len()` rows to `slots` (`None`: no origin).
    pub(crate) fn set_tail(&mut self, slots: &[Option<u32>]) {
        let (k, len) = (slots.len(), self.len());
        assert!(k <= len, "{k} tail tags for {len} rows");
        if k == 0 {
            return;
        }
        let tags: Vec<i32> = slots.iter().map(|&s| tag(s) as i32).collect();
        let tail = Tensor::from_data(TensorData::new(tags, [k]), &self.rows.device());
        self.rows = self.rows.clone().slice_assign(s![len - k..len], tail);
        let max_new = slots.iter().map(|&s| tag(s)).max().unwrap_or(0);
        self.max_tag = self.max_tag.max(max_new);
    }

    /// Appends one child per parent index, with the parent's origin.
    pub(crate) fn split(&mut self, parents: Tensor<1, Int>) {
        let inherited = self.rows.clone().select(0, parents);
        self.rows = Tensor::cat(vec![self.rows.clone(), inherited], 0);
    }

    /// Moves every splat seeded by a view in `moves` by that view's rigid correction: mean `Δ·μ`, rotation
    /// `q_Δ ⊗ q`. Scales, colours, opacities, the parameter ids and the Mip floor are kept; SH bands above 0 are
    /// not rotated (corrections turned by at most about 6° in the 2026-10 device captures; Brush has no SH rotation)
    /// and optimizer moments stay as they are (the first moment decays within tens of steps; the second holds
    /// per-component magnitudes a small turn barely changes).
    pub(crate) fn move_splats(&self, mut splats: Splats, moves: &[(u32, Affine3A)]) -> Splats {
        assert_eq!(
            self.len(),
            splats.num_splats() as usize,
            "origin tags out of step with the splats"
        );
        let Some(max_moved) = moves.iter().map(|(slot, _)| tag(Some(*slot))).max() else {
            return splats;
        };
        // One row per tag that is present or moved; a view newer than every moved one keeps the identity.
        let rows = self.max_tag.max(max_moved) + 1;
        // Row r holds [qw, qx, qy, qz, tx, ty, tz] for tag r; identity for tag 0 and views that did not move.
        let mut table = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].repeat(rows);
        for (slot, m) in moves {
            let (_, q, t) = m.to_scale_rotation_translation();
            let r = tag(Some(*slot)) * 7;
            table[r..r + 7].copy_from_slice(&[q.w, q.x, q.y, q.z, t.x, t.y, t.z]);
        }
        let device = self.rows.device();
        let per_splat = Tensor::<2>::from_data(TensorData::new(table, [rows, 7]), &device)
            .select(0, self.rows.clone());
        let dq = per_splat.clone().slice(s![.., 0..4]);
        let dt = per_splat.slice(s![.., 4..7]);
        splats.transforms = splats.transforms.map(|t| {
            let means = quaternion_vec_multiply(dq.clone(), t.clone().slice(s![.., 0..3])) + dt;
            // q_Δ is a unit quaternion, so the product keeps |q|; stored rotations are unnormalised anyway (the
            // render kernels normalise them).
            let rots = quat_mul(dq, t.clone().slice(s![.., 3..7]));
            Tensor::cat(vec![means, rots, t.slice(s![.., 7..10])], 1)
        });
        splats
    }
}

/// The tag of a splat seeded by view `slot`: `slot + 1`, 0 without one.
fn tag(slot: Option<u32>) -> usize {
    slot.map_or(0, |s| s as usize + 1)
}

/// The view slot tag `t` stands for; `None` for 0.
pub(crate) fn slot_of(t: i32) -> Option<u32> {
    (t > 0).then(|| t as u32 - 1)
}

/// Hamilton product `a ⊗ b` of `[N, 4]` quaternions stored `[w, x, y, z]`.
fn quat_mul(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let c = |t: &Tensor<2>, k: usize| t.clone().slice(s![.., k..k + 1]);
    let (aw, ax, ay, az) = (c(&a, 0), c(&a, 1), c(&a, 2), c(&a, 3));
    let (bw, bx, by, bz) = (c(&b, 0), c(&b, 1), c(&b, 2), c(&b, 3));
    let w = aw.clone() * bw.clone()
        - ax.clone() * bx.clone()
        - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x = aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
        - az.clone() * by.clone();
    let y = aw.clone() * by.clone() - ax.clone() * bz.clone()
        + ay.clone() * bw.clone()
        + az.clone() * bx.clone();
    let z = aw * bz + ax * by - ay * bx + az * bw;
    Tensor::cat(vec![w, x, y, z], 1)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
    use glam::{Quat, Vec3};

    async fn ints(t: Tensor<1, Int>) -> Vec<i32> {
        t.into_data_async()
            .await
            .expect("readback")
            .convert::<i32>()
            .try_into_vec::<i32>()
            .expect("ints")
    }

    async fn floats(t: Tensor<2>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("readback")
            .try_to_vec::<f32>()
            .expect("floats")
    }

    fn three_splats(rotations: Vec<f32>, device: &Device) -> Splats {
        Splats::from_raw(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            rotations,
            vec![-3.0; 9],
            vec![0.5; 9],
            vec![inverse_sigmoid(0.5); 3],
            SplatRenderMode::Default,
            device,
        )
    }

    /// Tags follow append, prune (keep) and split like the splats' rows.
    #[tokio::test]
    async fn tags_follow_append_keep_and_split() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut o = SplatOrigin::new(2, 0, &device);
        o.pad(2, Some(3));
        o.pad(1, None);
        assert_eq!(ints(o.rows.clone()).await, vec![1, 1, 4, 4, 0]);
        o.keep(Tensor::from_data(
            TensorData::new(vec![0i32, 2, 4], [3]),
            &device,
        ));
        assert_eq!(ints(o.rows.clone()).await, vec![1, 4, 0]);
        o.split(Tensor::from_data(TensorData::new(vec![1i32], [1]), &device));
        assert_eq!(ints(o.rows.clone()).await, vec![1, 4, 0, 4]);
        assert_eq!(o.len(), 4);
    }

    /// Only splats seeded by a moved view move: mean by the rigid transform, rotation composed in front.
    #[tokio::test]
    async fn only_splats_of_moved_views_move() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        // The moved splat starts tilted about x, so composing the turn in front differs from composing it behind.
        let tilt = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let mut rotations = [1.0, 0.0, 0.0, 0.0].repeat(3);
        rotations[4..8].copy_from_slice(&[tilt.w, tilt.x, tilt.y, tilt.z]);
        let splats = three_splats(rotations, &device);
        let mut o = SplatOrigin::new(1, 0, &device);
        o.pad(1, Some(2));
        o.pad(1, None);
        let turn = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let moved = o.move_splats(
            splats,
            &[(
                2,
                glam::Affine3A::from_rotation_translation(turn, Vec3::new(0.0, 0.5, 0.0)),
            )],
        );
        let means = floats(moved.means()).await;
        assert!(
            (means[0] - 1.0).abs() < 1e-5 && means[1].abs() < 1e-5,
            "slot 0 stays: {means:?}"
        );
        // (1, 0, 0) turned 90° about y is (0, 0, -1), then lifted by 0.5.
        let m1 = Vec3::new(means[3], means[4], means[5]);
        assert!((m1 - Vec3::new(0.0, 0.5, -1.0)).length() < 1e-5, "{m1}");
        assert!((means[7] - 2.0).abs() < 1e-5, "no origin stays: {means:?}");
        let rots = floats(moved.rotations()).await;
        let q1 = Quat::from_xyzw(rots[5], rots[6], rots[7], rots[4]);
        assert!(
            (turn * tilt).dot(tilt * turn).abs() < 0.9,
            "the test needs non-commuting rotations"
        );
        assert!(q1.dot(turn * tilt).abs() > 1.0 - 1e-5, "{q1}");
        assert_eq!(&rots[0..4], &[1.0, 0.0, 0.0, 0.0]);
    }

    /// A view newer than every moved view has a tag above the moved slots; its splats stay where they are.
    #[tokio::test]
    async fn splats_of_a_newer_unmoved_view_stay() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let splats = three_splats([1.0, 0.0, 0.0, 0.0].repeat(3), &device);
        let mut o = SplatOrigin::new(1, 0, &device);
        o.pad(2, Some(5));
        let shift = glam::Affine3A::from_translation(Vec3::new(0.0, 0.0, 0.25));
        let moved = o.move_splats(splats, &[(0, shift)]);
        let means = floats(moved.means()).await;
        assert_eq!(
            means,
            vec![1.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            "only view 0's splat moves"
        );
        let rots = floats(moved.rotations()).await;
        assert_eq!(rots, [1.0, 0.0, 0.0, 0.0].repeat(3));
    }
}
