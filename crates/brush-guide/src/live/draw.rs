//! Which view a training step trains on: the recent window, all views, or (with `train_views_radius_m`) the views
//! near the walker, and the cache that serves those near views.

use brush_dataset::scene::{SceneBatch, SceneView, view_to_packed_data};
use burn::tensor::TensorData;
use glam::Vec3;
use rand::RngExt as _;
use std::collections::HashMap;
use std::ops::Range;

/// Where one training step's batch comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Draw {
    /// A view added since the recent loader was built (index into the views), loaded directly.
    Pending(usize),
    RecentLoader,
    AllLoader,
    /// A view within `train_views_radius_m` of the newest view (index into the views), from the [`NearCache`].
    Near(usize),
}

/// Picks one step's source. With probability `recent_fraction` a recent-window draw (a pending view or the recent
/// loader, see [`pick_pending`]); otherwise all views (`near` `None`: the radius is off) or a uniform pick among
/// `near`, falling back to the recent-window draw when `near` is empty. With `near` `None` this draws the same
/// random numbers as before the near option existed.
pub(super) fn choose_draw(
    rng: &mut impl rand::Rng,
    recent_fraction: f32,
    recent_len: usize,
    pending: Range<usize>,
    near: Option<&[usize]>,
) -> Draw {
    let use_recent = rng.random::<f32>() < recent_fraction;
    if !use_recent {
        match near {
            None => return Draw::AllLoader,
            Some(near) if !near.is_empty() => {
                return Draw::Near(near[rng.random_range(0..near.len())]);
            }
            Some(_) => {}
        }
    }
    match pick_pending(rng, recent_len, pending) {
        Some(i) => Draw::Pending(i),
        None => Draw::RecentLoader,
    }
}

/// Indices of the cameras whose centre lies within `radius` (3D, inclusive) of the last camera's centre, ascending.
pub(super) fn near_views(
    positions: impl DoubleEndedIterator<Item = Vec3> + Clone,
    radius: f32,
) -> Vec<usize> {
    let Some(centre) = positions.clone().next_back() else {
        return Vec::new();
    };
    positions
        .enumerate()
        .filter(|(_, p)| p.distance(centre) <= radius)
        .map(|(i, _)| i)
        .collect()
}

/// For a recent-window draw: one of the `pending` views (added since the
/// recent loader was built) to train on directly, or `None` to draw from the
/// loader's `loader_len` views. Every view gets the same share.
pub(super) fn pick_pending(
    rng: &mut impl rand::Rng,
    loader_len: usize,
    pending: Range<usize>,
) -> Option<usize> {
    if pending.is_empty() {
        return None;
    }
    let r = rng.random_range(0..loader_len + pending.len());
    (r >= loader_len).then(|| pending.start + r - loader_len)
}

/// A training batch for one view, decoded and packed as `SceneLoader` does.
pub(super) async fn load_batch(view: &SceneView) -> SceneBatch {
    let raw = view.image.load().await.expect("keyframe image decodes");
    let (img_packed, has_alpha) = view_to_packed_data(raw, view.image.alpha_mode());
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: view.image.alpha_mode(),
        camera: view.camera,
    }
}

/// Packed images of near views, by view index, within a byte budget. Entries hold no camera: a hit takes the
/// view's current camera, so pose updates need no invalidation.
pub(super) struct NearCache {
    batches: HashMap<usize, SceneBatch>,
    bytes: u64,
    budget: u64,
}

impl NearCache {
    pub(super) fn new(budget: u64) -> Self {
        Self {
            batches: HashMap::new(),
            bytes: 0,
            budget,
        }
    }

    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The batch for view `index` with its current camera. A miss decodes the view and caches it if it fits the
    /// budget after dropping the views outside `near` (ascending).
    pub(super) async fn batch(
        &mut self,
        index: usize,
        view: &SceneView,
        near: &[usize],
    ) -> SceneBatch {
        if let Some(cached) = self.batches.get(&index) {
            return SceneBatch {
                camera: view.camera,
                ..cached.clone()
            };
        }
        let mut batch = load_batch(view).await;
        let size = batch.packed_bytes();
        if self.bytes + size > self.budget {
            self.batches.retain(|i, b| {
                let keep = near.binary_search(i).is_ok();
                if !keep {
                    self.bytes -= b.packed_bytes();
                }
                keep
            });
        }
        if self.bytes + size <= self.budget {
            batch.img_packed = share_packed(batch.img_packed);
            self.bytes += size;
            self.batches.insert(index, batch.clone());
        }
        batch
    }

    #[cfg(test)]
    pub(super) fn cached_views(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.batches.keys().copied().collect();
        v.sort_unstable();
        v
    }
}

/// Moves the packed pixels behind a refcount, so a cache hit hands out a view of the same allocation (as
/// `SceneLoader` does).
fn share_packed(data: TensorData) -> TensorData {
    TensorData::from_bytes(data.bytes.shared(), data.shape, data.dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn no_pending_views_leave_the_rng_untouched() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut fresh = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..10 {
            assert!(pick_pending(&mut rng, 20, 20..20).is_none());
        }
        assert_eq!(rng.random::<u64>(), fresh.random::<u64>());
    }

    #[test]
    fn pending_views_after_the_window_slid_map_to_their_indices() {
        // 35 views pooled at the last rebuild (the loader holds the last 20), 2 new since.
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let picked: std::collections::HashSet<usize> = (0..2000)
            .filter_map(|_| pick_pending(&mut rng, 20, 35..37))
            .collect();
        assert_eq!(picked, [35, 36].into_iter().collect());
    }

    #[test]
    fn pending_views_get_a_loader_views_share() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let draws = 22_000;
        let mut hits = [0usize; 2];
        for _ in 0..draws {
            if let Some(i) = pick_pending(&mut rng, 20, 20..22) {
                hits[i - 20] += 1;
            }
        }
        // Each of the 22 views should get ~1/22 of the draws (1000).
        for h in hits {
            assert!((850..1150).contains(&h), "pending view drawn {h} times");
        }
    }

    #[test]
    fn no_pending_views_always_uses_the_loader() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!((0..1000).all(|_| pick_pending(&mut rng, 20, 20..20).is_none()));
    }

    /// The draw `train_step` made before the near option: recent with `recent_fraction`, else the all loader.
    fn draw_before_near(
        rng: &mut rand::rngs::StdRng,
        recent_len: usize,
        pending: Range<usize>,
    ) -> Draw {
        let use_recent = rng.random::<f32>() < 0.7;
        match use_recent
            .then(|| pick_pending(rng, recent_len, pending))
            .flatten()
        {
            Some(i) => Draw::Pending(i),
            None if use_recent => Draw::RecentLoader,
            None => Draw::AllLoader,
        }
    }

    #[test]
    fn radius_off_draws_the_same_sequence_as_before() {
        let mut old = rand::rngs::StdRng::seed_from_u64(11);
        let mut new = rand::rngs::StdRng::seed_from_u64(11);
        for step in 0..5000 {
            let pending = if step % 3 == 0 { 20..20 } else { 20..22 };
            let want = draw_before_near(&mut old, 20, pending.clone());
            assert_eq!(
                choose_draw(&mut new, 0.7, 20, pending, None),
                want,
                "step {step}"
            );
        }
    }

    #[test]
    fn radius_on_draws_only_near_views_besides_the_recent_window() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let near = [3usize, 5, 8];
        let draws = 30_000;
        let mut hits: HashMap<usize, usize> = HashMap::new();
        for _ in 0..draws {
            match choose_draw(&mut rng, 0.7, 20, 20..21, Some(&near)) {
                Draw::Near(i) => *hits.entry(i).or_default() += 1,
                Draw::Pending(i) => assert_eq!(i, 20),
                Draw::RecentLoader => {}
                Draw::AllLoader => panic!("all-views draw with the radius on"),
            }
        }
        assert_eq!(
            hits.keys()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from(near)
        );
        // 30 % of the draws, a third of them per near view: ~3000 each.
        for (i, h) in hits {
            assert!((2700..3300).contains(&h), "near view {i} drawn {h} times");
        }
    }

    #[test]
    fn no_near_views_fall_back_to_the_recent_draw() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let mut pending = 0;
        for _ in 0..5000 {
            match choose_draw(&mut rng, 0.7, 20, 20..22, Some(&[])) {
                Draw::Pending(i) => {
                    assert!((20..22).contains(&i));
                    pending += 1;
                }
                Draw::RecentLoader => {}
                d => panic!("{d:?} without near views"),
            }
        }
        // Every draw is a recent-window draw, 2 of its 22 views pending: ~455.
        assert!((350..560).contains(&pending), "{pending} pending draws");
    }

    #[test]
    fn near_views_lie_within_the_radius_of_the_last_camera() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 4.0),
            Vec3::new(9.0, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        // Distances to the last: 5, 4.47, 4, 5, 0.
        assert_eq!(near_views(positions.into_iter(), 4.5), vec![1, 2, 4]);
        assert_eq!(near_views(positions.into_iter(), 5.0), vec![0, 1, 2, 3, 4]);
        assert!(near_views(std::iter::empty::<Vec3>(), 5.0).is_empty());
    }
}
