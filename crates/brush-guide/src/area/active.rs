//! The active set of area training with `area_freeze` inactive: the tiles around the newest keyframe's camera and the
//! tiles it looks at, ranked by how much of its image they cover, and the rules that follow from it for training
//! views and seeds.

use super::tiles::{TileKey, TileRegistry};
use super::views::{camera_sees_box, projected_area};
use brush_render::camera::Camera;
use glam::Vec3;
use std::collections::BTreeMap;

/// Tiles whose footprint lies within this many metres of the camera are active and outside the ranking, so a room at
/// a corner of tiles trains whole.
pub const NEAR_TILE_M: f32 = 2.5;

/// An active tile stays active while the camera sees its box up to this many metres beyond `area_view_m`.
pub const KEEP_MARGIN_M: f32 = 2.0;

/// A tile in view replaces an active one only when it covers this many times its image share.
pub const CHALLENGE_RATIO: f32 = 1.3;

/// Share of the image below which a kept tile still ranks (its box passed the conservative frustum test).
const KEPT_FLOOR: f32 = 1e-6;

/// Rows of a tile and the band of their heights: the 5th to the 95th percentile, so a few stray rows do not stretch
/// the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heights {
    pub lo: f32,
    pub hi: f32,
    pub rows: u32,
}

impl Heights {
    /// Both sets of rows together: the band that holds both bands, and the rows summed.
    pub fn widen(self, other: Heights) -> Heights {
        Heights { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi), rows: self.rows + other.rows }
    }
}

/// Widens `into`'s entries by `add`'s ([`Heights::widen`]); tiles only in `add` are copied.
pub fn widen_all(into: &mut BTreeMap<TileKey, Heights>, add: &BTreeMap<TileKey, Heights>) {
    for (&key, &h) in add {
        into.entry(key).and_modify(|e| *e = e.widen(h)).or_insert(h);
    }
}

/// Per tile of the flat xyz `means`: the 5th and 95th percentile of y (nearest rank) and the row count.
pub fn tile_heights(means: &[f32], tile_size: f32) -> BTreeMap<TileKey, Heights> {
    let mut ys: BTreeMap<TileKey, Vec<f32>> = BTreeMap::new();
    for p in means.chunks_exact(3) {
        ys.entry(TileKey::of(Vec3::new(p[0], p[1], p[2]), tile_size)).or_default().push(p[1]);
    }
    ys.into_iter()
        .map(|(key, mut y)| {
            let n = y.len();
            let rank = |p: f32| ((p * (n - 1) as f32).round() as usize).min(n - 1);
            let lo = *y.select_nth_unstable_by(rank(0.05), f32::total_cmp).1;
            let hi = *y.select_nth_unstable_by(rank(0.95), f32::total_cmp).1;
            (key, Heights { lo, hi, rows: n as u32 })
        })
        .collect()
}

/// A tile that may become active: its box (footprint by height band) and its rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub min: Vec3,
    pub max: Vec3,
    pub rows: u32,
}

/// The tiles that may become active: every resident tile of the registry and every tile of `heights` (bands of
/// resident rows and of rows seeded since), a resident tile without a band taking the registry's height range and
/// count. Parked tiles and tiles the walker at `centre` has left ([`TileRegistry::is_left`]) are left out: they take
/// no seeds and stay as they were.
pub fn candidate_boxes(
    registry: &TileRegistry,
    heights: &BTreeMap<TileKey, Heights>,
    centre: Vec3,
) -> BTreeMap<TileKey, Candidate> {
    let size = registry.params().tile_size;
    let mut bands = heights.clone();
    for key in registry.resident_keys() {
        if let (Some(t), false) = (registry.get(key), bands.contains_key(&key)) {
            bands.insert(key, Heights { lo: t.y_min, hi: t.y_max, rows: t.splats });
        }
    }
    bands
        .into_iter()
        .filter(|&(key, _)| registry.get(key).is_none_or(|t| t.resident) && !registry.is_left(key, centre))
        .map(|(key, h)| {
            let min = key.min_xz(size);
            let (lo, hi) = (Vec3::new(min.x, h.lo, min.y), Vec3::new(min.x + size, h.hi, min.y + size));
            (key, Candidate { min: lo, max: hi, rows: h.rows })
        })
        .collect()
}

/// The knobs of [`rank_active`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveRule {
    /// How deep the camera looks for tiles, metres.
    pub view_m: f32,
    /// Ranked tiles at most, besides the near ones.
    pub max: u32,
    /// Rows a ranked tile needs at least.
    pub min_rows: u32,
    pub tile_size: f32,
}

/// The tiles whose footprint lies within [`NEAR_TILE_M`] of `p` horizontally: the one under it first, then by
/// distance (ties by key).
pub fn near_tiles(p: Vec3, tile_size: f32) -> Vec<TileKey> {
    if !p.is_finite() {
        return Vec::new();
    }
    let cell = |c: f32| (c / tile_size).floor() as i32;
    let span = |c: f32| cell(c - NEAR_TILE_M)..=cell(c + NEAR_TILE_M);
    let mut near: Vec<(f32, TileKey)> = span(p.x)
        .flat_map(|x| span(p.z).map(move |z| TileKey { x, z }))
        .map(|k| (k.footprint_distance(p, tile_size), k))
        .filter(|&(d, _)| d <= NEAR_TILE_M)
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    near.into_iter().map(|(_, k)| k).collect()
}

/// The active tiles for `camera`: the near tiles ([`near_tiles`], rows or not), then at most `rule.max` candidates
/// with `rule.min_rows` rows or more, by the share of its image their box covers up to `rule.view_m` deep
/// ([`projected_area`]). Hysteresis keeps the set from flapping: a candidate among `incumbents` (the last active set)
/// stays in the running while the camera sees its box up to `view_m` + [`KEEP_MARGIN_M`] ([`camera_sees_box`]) and
/// ranks by its share there times [`CHALLENGE_RATIO`], so another tile replaces it only by covering that much more.
pub fn rank_active(
    camera: &Camera,
    candidates: &BTreeMap<TileKey, Candidate>,
    rule: ActiveRule,
    incumbents: &[TileKey],
) -> Vec<TileKey> {
    let near = near_tiles(camera.position, rule.tile_size);
    let keep_m = rule.view_m + KEEP_MARGIN_M;
    // (score, incumbent, key)
    let mut ranked: Vec<(f32, bool, TileKey)> = candidates
        .iter()
        .filter(|&(key, c)| !near.contains(key) && c.rows >= rule.min_rows)
        .filter_map(|(&key, c)| {
            if incumbents.contains(&key) {
                camera_sees_box(camera, c.min, c.max, keep_m).then(|| {
                    let area = projected_area(camera, c.min, c.max, keep_m).max(KEPT_FLOOR);
                    (area * CHALLENGE_RATIO, true, key)
                })
            } else {
                let seen = camera_sees_box(camera, c.min, c.max, rule.view_m);
                let area = if seen { projected_area(camera, c.min, c.max, rule.view_m) } else { 0.0 };
                (area > 0.0).then_some((area, false, key))
            }
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    near.into_iter().chain(ranked.into_iter().take(rule.max as usize).map(|(.., key)| key)).collect()
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
    fresh: &BTreeMap<TileKey, Heights>,
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

#[cfg(test)]
#[path = "active_tests.rs"]
mod tests;
