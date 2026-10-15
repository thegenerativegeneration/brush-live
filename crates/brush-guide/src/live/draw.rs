//! Which view a training step trains on: a recent-window draw (the recent loader or a view added since it was
//! built) or the all-views loader.

use brush_dataset::scene::{SceneBatch, SceneView, view_to_packed_data};
use rand::RngExt as _;

/// Where one training step's batch comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Draw {
    /// A view added since the recent loader was built (index into the views), loaded directly.
    Pending(usize),
    RecentLoader,
    AllLoader,
}

/// Picks one step's source: with probability `recent_fraction` a recent-window draw (a `pending` view or the
/// recent loader's `recent_len` views, see [`pick_pending`]), otherwise the all-views loader.
pub(super) fn choose_draw(
    rng: &mut impl rand::Rng,
    recent_fraction: f32,
    recent_len: usize,
    pending: &[usize],
) -> Draw {
    let use_recent = rng.random::<f32>() < recent_fraction;
    if !use_recent {
        return Draw::AllLoader;
    }
    match pick_pending(rng, recent_len, pending) {
        Some(i) => Draw::Pending(i),
        None => Draw::RecentLoader,
    }
}

/// For a recent-window draw: one of the `pending` views (indices into the views, added since the recent loader was
/// built) to train on directly, or `None` to draw from the loader's `loader_len` views. Every view gets the same
/// share.
pub(super) fn pick_pending(
    rng: &mut impl rand::Rng,
    loader_len: usize,
    pending: &[usize],
) -> Option<usize> {
    if pending.is_empty() {
        return None;
    }
    let r = rng.random_range(0..loader_len + pending.len());
    (r >= loader_len).then(|| pending[r - loader_len])
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn no_pending_views_leave_the_rng_untouched() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut fresh = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..10 {
            assert!(pick_pending(&mut rng, 20, &[]).is_none());
        }
        assert_eq!(rng.random::<u64>(), fresh.random::<u64>());
    }

    #[test]
    fn pending_views_after_the_window_slid_map_to_their_indices() {
        // 35 views pooled at the last rebuild (the loader holds the last 20), 2 new since.
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let picked: std::collections::HashSet<usize> = (0..2000)
            .filter_map(|_| pick_pending(&mut rng, 20, &[35, 36]))
            .collect();
        assert_eq!(picked, [35, 36].into_iter().collect());
    }

    #[test]
    fn pending_views_get_a_loader_views_share() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let draws = 22_000;
        let mut hits = [0usize; 2];
        for _ in 0..draws {
            if let Some(i) = pick_pending(&mut rng, 20, &[20, 21]) {
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
        assert!((0..1000).all(|_| pick_pending(&mut rng, 20, &[]).is_none()));
    }

    /// `pick_pending` as it was when pending views were always a contiguous range.
    fn pick_pending_from_range(
        rng: &mut rand::rngs::StdRng,
        loader_len: usize,
        pending: std::ops::Range<usize>,
    ) -> Option<usize> {
        if pending.is_empty() {
            return None;
        }
        let r = rng.random_range(0..loader_len + pending.len());
        (r >= loader_len).then(|| pending.start + r - loader_len)
    }

    /// The draw `train_step` made before area training: recent with `recent_fraction`, else the all loader.
    fn draw_before_area(
        rng: &mut rand::rngs::StdRng,
        recent_len: usize,
        pending: std::ops::Range<usize>,
    ) -> Draw {
        let use_recent = rng.random::<f32>() < 0.7;
        match use_recent
            .then(|| pick_pending_from_range(rng, recent_len, pending))
            .flatten()
        {
            Some(i) => Draw::Pending(i),
            None if use_recent => Draw::RecentLoader,
            None => Draw::AllLoader,
        }
    }

    /// With area training off the pending views are the views after the recent loader's, as before, and the draws
    /// consume the same random numbers.
    #[test]
    fn area_off_draws_the_same_sequence_as_before() {
        let mut old = rand::rngs::StdRng::seed_from_u64(11);
        let mut new = rand::rngs::StdRng::seed_from_u64(11);
        for step in 0..5000 {
            let pending = if step % 3 == 0 { 20..20 } else { 20..22 };
            let want = draw_before_area(&mut old, 20, pending.clone());
            let pending: Vec<usize> = pending.collect();
            assert_eq!(choose_draw(&mut new, 0.7, 20, &pending), want, "step {step}");
        }
    }

    /// Pending views need not be contiguous: area training leaves out those outside the area.
    #[test]
    fn pending_draws_pick_only_the_given_views() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let picked: std::collections::BTreeSet<usize> = (0..3000)
            .filter_map(|_| match choose_draw(&mut rng, 0.7, 4, &[21, 25]) {
                Draw::Pending(i) => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(picked, [21, 25].into_iter().collect());
    }
}
