//! The active set of area training with `area_freeze` inactive: the tiles the newest keyframe looks at, ranked by how
//! much of its image they cover, and the rules that follow from it for training views and seeds.

use super::tiles::{TileKey, TileRegistry};
use super::views::{camera_sees_box, projected_area};
use brush_render::camera::Camera;
use glam::Vec3;
use std::collections::BTreeMap;

/// Boxes (min, max) of the tiles that may become active: every resident tile of the registry over its footprint and
/// height range, and every tile of `fresh` (heights of rows seeded since the registry last read the splats back), its
/// range joined with the registry's. Parked tiles and tiles the walker at `centre` has left
/// ([`TileRegistry::is_left`]) are left out: they take no seeds and stay as they were.
pub fn candidate_boxes(
    registry: &TileRegistry,
    fresh: &BTreeMap<TileKey, (f32, f32)>,
    centre: Vec3,
) -> BTreeMap<TileKey, (Vec3, Vec3)> {
    let size = registry.params().tile_size;
    let mut ranges: BTreeMap<TileKey, (f32, f32)> = BTreeMap::new();
    for key in registry.resident_keys() {
        if let Some(t) = registry.get(key) {
            ranges.insert(key, (t.y_min, t.y_max));
        }
    }
    for (&key, &(lo, hi)) in fresh {
        if registry.get(key).is_some_and(|t| !t.resident) {
            continue;
        }
        let r = ranges.entry(key).or_insert((lo, hi));
        *r = (r.0.min(lo), r.1.max(hi));
    }
    ranges
        .into_iter()
        .filter(|&(key, _)| !registry.is_left(key, centre))
        .map(|(key, (lo, hi))| {
            let min = key.min_xz(size);
            (key, (Vec3::new(min.x, lo, min.y), Vec3::new(min.x + size, hi, min.y + size)))
        })
        .collect()
}

/// The active tiles for `camera`: the tile under it first (whether it holds rows or not), then the `boxes` it sees
/// up to `view_m` deep ([`camera_sees_box`]) by the share of its image they cover there ([`projected_area`]), largest
/// first (ties by key), at most `max` tiles in all (0 counts as 1).
pub fn rank_active(
    camera: &Camera,
    boxes: &BTreeMap<TileKey, (Vec3, Vec3)>,
    view_m: f32,
    max: u32,
    tile_size: f32,
) -> Vec<TileKey> {
    let underfoot = TileKey::of(camera.position, tile_size);
    let mut seen: Vec<(f32, TileKey)> = boxes
        .iter()
        .filter(|&(&key, &(lo, hi))| key != underfoot && camera_sees_box(camera, lo, hi, view_m))
        .map(|(&key, &(lo, hi))| (projected_area(camera, lo, hi, view_m), key))
        .filter(|&(area, _)| area > 0.0)
        .collect();
    seen.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    std::iter::once(underfoot)
        .chain(seen.into_iter().map(|(_, key)| key))
        .take(max.max(1) as usize)
        .collect()
}

/// Indices of the `cameras` that see one of `boxes` up to `far` deep ([`camera_sees_box`]), ascending.
pub fn views_seeing(cameras: &[Camera], boxes: &[(Vec3, Vec3)], far: f32) -> Vec<usize> {
    cameras
        .iter()
        .enumerate()
        .filter(|(_, c)| boxes.iter().any(|&(lo, hi)| camera_sees_box(c, lo, hi, far)))
        .map(|(i, _)| i)
        .collect()
}

/// What [`keep_active_seeds`] did with the seeds it looked at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeedSort {
    /// Seeds dropped for landing in a tile that holds rows but is not active.
    pub dropped: u32,
    /// Seeds kept outside the active tiles, on ground no tile holds rows on yet.
    pub new_ground: u32,
}

/// Clears `keep` for every seed (flat xyz `seed_means`) still kept that lands in a tile with rows, resident in the
/// registry or seeded since its last readback (`fresh`), that is not among `active`. Seeds on ground no tile holds
/// yet stay: a tile becomes a candidate only once it holds rows, so without them the walker's way ahead would never
/// get rows to rank.
pub fn keep_active_seeds(
    registry: &TileRegistry,
    fresh: &BTreeMap<TileKey, (f32, f32)>,
    active: &[TileKey],
    seed_means: &[f32],
    keep: &mut [bool],
) -> SeedSort {
    let size = registry.params().tile_size;
    let mut sort = SeedSort::default();
    for (s, k) in seed_means.chunks_exact(3).zip(keep.iter_mut()) {
        let key = TileKey::of(Vec3::new(s[0], s[1], s[2]), size);
        if !*k || active.contains(&key) {
            continue;
        }
        if registry.get(key).is_some_and(|t| t.resident) || fresh.contains_key(&key) {
            *k = false;
            sort.dropped += 1;
        } else {
            sort.new_ground += 1;
        }
    }
    sort
}

/// Per tile of the flat xyz `means`, the lowest and highest y.
pub fn seed_heights(means: &[f32], tile_size: f32) -> BTreeMap<TileKey, (f32, f32)> {
    let mut h: BTreeMap<TileKey, (f32, f32)> = BTreeMap::new();
    for p in means.chunks_exact(3) {
        let r = h.entry(TileKey::of(Vec3::new(p[0], p[1], p[2]), tile_size)).or_insert((p[1], p[1]));
        *r = (r.0.min(p[1]), r.1.max(p[1]));
    }
    h
}

#[cfg(test)]
#[path = "active_tests.rs"]
mod tests;
