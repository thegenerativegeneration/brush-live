//! Tile keys and the tile registry: which tiles are resident, which are parked, and the rules that move them.

use crate::config::GuideConfig;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A square column of ground in x and z, `tile_size` metres on a side; height is unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileKey {
    pub x: i32,
    pub z: i32,
}

impl TileKey {
    /// The tile of a point: floor of x and z over the tile size. NaN maps to tile (0, 0) and infinities saturate.
    pub fn of(p: Vec3, tile_size: f32) -> Self {
        Self { x: (p.x / tile_size).floor() as i32, z: (p.z / tile_size).floor() as i32 }
    }

    /// The tile's corner with the smallest x and z.
    pub fn min_xz(self, tile_size: f32) -> Vec2 {
        Vec2::new(self.x as f32, self.z as f32) * tile_size
    }

    /// Horizontal distance from `centre` to the tile's square; 0 inside.
    pub fn footprint_distance(self, centre: Vec3, tile_size: f32) -> f32 {
        let lo = self.min_xz(tile_size);
        let c = Vec2::new(centre.x, centre.z);
        (c.clamp(lo, lo + Vec2::splat(tile_size)) - c).length()
    }

    /// File-name stem, e.g. `-1_3`.
    pub fn stem(self) -> String {
        format!("{}_{}", self.x, self.z)
    }
}

/// The area-training knobs the registry needs, taken from [`GuideConfig`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaParams {
    pub tile_size: f32,
    pub radius: f32,
    pub hysteresis: f32,
    pub seen_window: u32,
    pub min_resident_keyframes: u32,
    pub far_seed: f32,
}

impl AreaParams {
    /// `None` when area training is off (`area_radius_m <= 0`).
    pub fn from_config(c: &GuideConfig) -> Option<AreaParams> {
        (c.area_radius_m > 0.0).then_some(AreaParams {
            tile_size: c.area_tile_size_m,
            radius: c.area_radius_m,
            hysteresis: c.area_hysteresis_m,
            seen_window: c.area_seen_window_steps,
            min_resident_keyframes: c.area_min_resident_keyframes,
            far_seed: c.area_far_seed_m,
        })
    }
}

/// What the registry knows about one tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TileInfo {
    /// In the live engine (true) or parked on disk (false).
    pub resident: bool,
    pub splats: u32,
    pub y_min: f32,
    pub y_max: f32,
    /// Training step at which a view last drew a splat of this tile.
    pub last_seen_step: u32,
    /// Keyframe count at the last restore, if the tile was ever restored.
    pub restored_at_keyframe: Option<u32>,
    pub parked_at_step: u32,
    /// Counts the parkings of this tile.
    pub version: u32,
}

/// Parked tiles to restore from a seed batch, and which seeds survive.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeedTriage {
    /// Parked tiles within `radius + far_seed` that hold at least `min_seeds` seeds.
    pub restore: Vec<TileKey>,
    /// Per seed: false for every seed that lies in a parked tile (far, sparse, or about to be restored and reseeded).
    pub keep: Vec<bool>,
}

/// How to make room for a restore.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoomPlan {
    /// Resident tiles to park before restoring, farthest first.
    pub park_first: Vec<TileKey>,
    /// Whether the restore fits once `park_first` is parked.
    pub restore: bool,
    /// Splats still missing when every candidate is not enough (the candidates are parked anyway).
    pub shortfall: u32,
}

/// Tile counts and splat counts by state; sight tiles are resident tiles beyond the radius, kept by visibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AreaSummary {
    pub resident_tiles: u32,
    pub sight_tiles: u32,
    pub parked_tiles: u32,
    pub resident_splats: u32,
    pub sight_splats: u32,
    pub parked_splats: u32,
}

/// Which tiles are resident or parked, and the pure rules that decide parking and restoring.
pub struct TileRegistry {
    params: AreaParams,
    tiles: BTreeMap<TileKey, TileInfo>,
}

impl TileRegistry {
    pub fn new(params: AreaParams) -> Self {
        Self { params, tiles: BTreeMap::new() }
    }

    pub fn params(&self) -> &AreaParams {
        &self.params
    }

    pub fn get(&self, key: TileKey) -> Option<&TileInfo> {
        self.tiles.get(&key)
    }

    /// Rebuilds every resident tile from a readback of all resident splats (flat xyz `means`, `seen` per row or
    /// empty for none): counts, height range, and `last_seen_step = step` for tiles with a seen row or new tiles.
    /// Resident tiles without rows are dropped. Returns parked tiles that hold resident rows; those stay parked here.
    pub fn note_resident(&mut self, means: &[f32], seen: &[bool], step: u32) -> Vec<TileKey> {
        let mut found: BTreeMap<TileKey, (u32, f32, f32, bool)> = BTreeMap::new();
        for (i, p) in means.chunks_exact(3).enumerate() {
            let p = Vec3::new(p[0], p[1], p[2]);
            let e = found
                .entry(TileKey::of(p, self.params.tile_size))
                .or_insert((0, f32::INFINITY, f32::NEG_INFINITY, false));
            e.0 += 1;
            e.1 = e.1.min(p.y);
            e.2 = e.2.max(p.y);
            e.3 |= seen.get(i).copied().unwrap_or(false);
        }
        self.tiles.retain(|k, t| !t.resident || found.contains_key(k));
        let mut conflicts = Vec::new();
        for (key, (n, lo, hi, was_seen)) in found {
            match self.tiles.get_mut(&key) {
                Some(t) if !t.resident => conflicts.push(key),
                Some(t) => {
                    (t.splats, t.y_min, t.y_max) = (n, lo, hi);
                    if was_seen {
                        t.last_seen_step = step;
                    }
                }
                None => {
                    self.tiles.insert(
                        key,
                        TileInfo {
                            resident: true,
                            splats: n,
                            y_min: lo,
                            y_max: hi,
                            last_seen_step: step,
                            restored_at_keyframe: None,
                            parked_at_step: 0,
                            version: 0,
                        },
                    );
                }
            }
        }
        conflicts
    }

    /// Resident tiles to park: beyond radius + hysteresis, unseen for more than the window, and not restored within
    /// the last `min_resident_keyframes` keyframes.
    pub fn to_park(&self, centre: Vec3, step: u32, keyframes: u32) -> Vec<TileKey> {
        let p = &self.params;
        self.tiles
            .iter()
            .filter(|(k, t)| {
                t.resident
                    && k.footprint_distance(centre, p.tile_size) > p.radius + p.hysteresis
                    && step.saturating_sub(t.last_seen_step) > p.seen_window
                    && t.restored_at_keyframe.is_none_or(|r| keyframes.saturating_sub(r) >= p.min_resident_keyframes)
            })
            .map(|(k, _)| *k)
            .collect()
    }

    /// Parked tiles whose footprint lies within the radius of `centre`.
    pub fn parked_within_radius(&self, centre: Vec3) -> Vec<TileKey> {
        self.tiles
            .iter()
            .filter(|(k, t)| !t.resident && k.footprint_distance(centre, self.params.tile_size) <= self.params.radius)
            .map(|(k, _)| *k)
            .collect()
    }

    /// Sorts fresh seeds (flat xyz) against the parked tiles; see [`SeedTriage`].
    pub fn triage_seeds(&self, seed_means: &[f32], centre: Vec3, min_seeds: usize) -> SeedTriage {
        let p = &self.params;
        let mut counts: BTreeMap<TileKey, usize> = BTreeMap::new();
        let mut keep = Vec::with_capacity(seed_means.len() / 3);
        for s in seed_means.chunks_exact(3) {
            let key = TileKey::of(Vec3::new(s[0], s[1], s[2]), p.tile_size);
            let parked = self.tiles.get(&key).is_some_and(|t| !t.resident);
            if parked {
                *counts.entry(key).or_default() += 1;
            }
            keep.push(!parked);
        }
        let restore = counts
            .into_iter()
            .filter(|(k, n)| *n >= min_seeds && k.footprint_distance(centre, p.tile_size) <= p.radius + p.far_seed)
            .map(|(k, _)| k)
            .collect();
        SeedTriage { restore, keep }
    }

    /// Plans the parking that makes room for restoring `restore_splats` splats next to `resident` splats under a
    /// budget of `max_splats`. Candidates are resident tiles beyond the radius and unseen within the window,
    /// farthest first; tiles restored fewer than `min_resident_keyframes` keyframes before `keyframes` are skipped.
    pub fn plan_room(
        &self,
        restore_splats: u32,
        resident: u32,
        max_splats: u32,
        centre: Vec3,
        step: u32,
        keyframes: u32,
    ) -> RoomPlan {
        let p = &self.params;
        let room = max_splats.saturating_sub(resident);
        if room >= restore_splats {
            return RoomPlan { park_first: Vec::new(), restore: true, shortfall: 0 };
        }
        let mut candidates: Vec<(f32, TileKey, u32)> = self
            .tiles
            .iter()
            .filter(|(_, t)| {
                t.resident
                    && step.saturating_sub(t.last_seen_step) > p.seen_window
                    && t.restored_at_keyframe.is_none_or(|r| keyframes.saturating_sub(r) >= p.min_resident_keyframes)
            })
            .map(|(k, t)| (k.footprint_distance(centre, p.tile_size), *k, t.splats))
            .filter(|(d, ..)| *d > p.radius)
            .collect();
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut park_first = Vec::new();
        let mut freed = 0u32;
        for (_, key, splats) in candidates {
            if room + freed >= restore_splats {
                break;
            }
            park_first.push(key);
            freed += splats;
        }
        let have = room + freed;
        RoomPlan { park_first, restore: have >= restore_splats, shortfall: restore_splats.saturating_sub(have) }
    }

    /// Records that a tile now lives on disk with `splats` splats. `key` must come from the registry.
    pub fn mark_parked(&mut self, key: TileKey, splats: u32, step: u32) {
        let t = self.tiles.get_mut(&key);
        debug_assert!(t.is_some(), "mark_parked: unknown tile {key:?}");
        if let Some(t) = t {
            t.resident = false;
            t.splats = splats;
            t.parked_at_step = step;
            t.version += 1;
        }
    }

    /// Records that a parked tile is back in the engine at keyframe count `keyframes`. `key` must come from the
    /// registry.
    pub fn mark_restored(&mut self, key: TileKey, keyframes: u32, step: u32) {
        let t = self.tiles.get_mut(&key);
        debug_assert!(t.is_some(), "mark_restored: unknown tile {key:?}");
        if let Some(t) = t {
            t.resident = true;
            t.restored_at_keyframe = Some(keyframes);
            t.last_seen_step = step;
        }
    }

    /// Records that rows were merged into parked tile `key`'s file, which now holds `splats` rows (a new version).
    /// It stays parked from the same step. `key` must be a parked tile of the registry.
    pub fn note_merged(&mut self, key: TileKey, splats: u32) {
        let t = self.tiles.get_mut(&key);
        debug_assert!(t.as_ref().is_some_and(|t| !t.resident), "note_merged: no parked tile {key:?}");
        if let Some(t) = t {
            t.splats = splats;
            t.version += 1;
        }
    }

    pub fn drop_tile(&mut self, key: TileKey) {
        self.tiles.remove(&key);
    }

    pub fn parked_keys(&self) -> Vec<TileKey> {
        self.keys_where(false)
    }

    pub fn resident_keys(&self) -> Vec<TileKey> {
        self.keys_where(true)
    }

    fn keys_where(&self, resident: bool) -> Vec<TileKey> {
        self.tiles.iter().filter(|(_, t)| t.resident == resident).map(|(k, _)| *k).collect()
    }

    /// The tile's bounding box (min, max) over its footprint and recorded height range.
    pub fn tile_box(&self, key: TileKey) -> Option<(Vec3, Vec3)> {
        let t = self.tiles.get(&key)?;
        let lo = key.min_xz(self.params.tile_size);
        let s = self.params.tile_size;
        Some((Vec3::new(lo.x, t.y_min, lo.y), Vec3::new(lo.x + s, t.y_max, lo.y + s)))
    }

    /// Tile and splat counts by state around `centre`.
    pub fn summary(&self, centre: Vec3) -> AreaSummary {
        let mut s = AreaSummary::default();
        for (k, t) in &self.tiles {
            if !t.resident {
                s.parked_tiles += 1;
                s.parked_splats += t.splats;
                continue;
            }
            s.resident_tiles += 1;
            s.resident_splats += t.splats;
            if k.footprint_distance(centre, self.params.tile_size) > self.params.radius {
                s.sight_tiles += 1;
                s.sight_splats += t.splats;
            }
        }
        s
    }

    /// `[x, z, state]` per tile: 0 trained (resident within the radius), 1 sight (resident beyond it), 2 parked.
    pub fn tile_states(&self, centre: Vec3) -> Vec<[i32; 3]> {
        self.tiles
            .iter()
            .map(|(k, t)| {
                let state = if !t.resident {
                    2
                } else if k.footprint_distance(centre, self.params.tile_size) > self.params.radius {
                    1
                } else {
                    0
                };
                [k.x, k.z, state]
            })
            .collect()
    }
}

/// Row indices of the flat xyz `means` grouped by tile.
pub fn group_rows(means: &[f32], tile_size: f32) -> BTreeMap<TileKey, Vec<usize>> {
    let mut g: BTreeMap<TileKey, Vec<usize>> = BTreeMap::new();
    for (i, p) in means.chunks_exact(3).enumerate() {
        g.entry(TileKey::of(Vec3::new(p[0], p[1], p[2]), tile_size)).or_default().push(i);
    }
    g
}

/// Per row of the flat xyz `means`: whether it lies in one of `keys`.
pub fn rows_in(keys: &[TileKey], means: &[f32], tile_size: f32) -> Vec<bool> {
    let set: BTreeSet<TileKey> = keys.iter().copied().collect();
    means
        .chunks_exact(3)
        .map(|p| set.contains(&TileKey::of(Vec3::new(p[0], p[1], p[2]), tile_size)))
        .collect()
}

#[cfg(test)]
#[path = "tiles_tests.rs"]
mod tests;
