//! The per-tile splat budget (`area_tile_max_splats`) and room at the total cap: host-side row counts per tile, which
//! seeds a full tile turns away, which tiles park for room after the left ones, and the row budget the trainer applies
//! at a refine ([`brush_train::budget`]).

use super::active::Heights;
use super::tiles::{TileKey, TileRegistry, pick_until};
use brush_train::budget::{GroupBudget, NO_GROUP, RowBudget};
use glam::Vec3;
use std::collections::BTreeMap;

/// Rows per tile as the host knows them: the resident tiles' rows at the last readback (the registry) plus the rows
/// seeded since (`fresh`).
pub fn tile_counts(registry: &TileRegistry, fresh: &BTreeMap<TileKey, Heights>) -> BTreeMap<TileKey, u32> {
    let mut counts: BTreeMap<TileKey, u32> = registry
        .resident_keys()
        .into_iter()
        .filter_map(|k| registry.get(k).map(|t| (k, t.splats)))
        .collect();
    for (&key, h) in fresh {
        *counts.entry(key).or_default() += h.rows;
    }
    counts
}

/// `[x, z, splats]` per tile of the registry, resident (rows at the last readback) or parked (rows in its file).
pub fn tile_splats(registry: &TileRegistry) -> Vec<[i32; 3]> {
    let mut keys = registry.resident_keys();
    keys.extend(registry.parked_keys());
    keys.sort();
    keys.into_iter()
        .filter_map(|k| registry.get(k).map(|t| [k.x, k.z, i32::try_from(t.splats).unwrap_or(i32::MAX)]))
        .collect()
}

/// Clears `keep` for the seeds (flat xyz) still kept that a tile has no room for under `cap` rows, given its rows in
/// `counts`: a tile at or over the cap takes none, a tile with room for fewer than land in it takes that many, evenly
/// spread over its seeds in order. Returns how many it cleared. A cap of 0 clears none.
pub fn keep_under_tile_cap(
    counts: &BTreeMap<TileKey, u32>,
    cap: u32,
    seed_means: &[f32],
    tile_size: f32,
    keep: &mut [bool],
) -> u32 {
    if cap == 0 {
        return 0;
    }
    let mut landing: BTreeMap<TileKey, Vec<usize>> = BTreeMap::new();
    for (i, s) in seed_means.chunks_exact(3).enumerate() {
        if keep.get(i).copied().unwrap_or(false) {
            landing.entry(TileKey::of(Vec3::new(s[0], s[1], s[2]), tile_size)).or_default().push(i);
        }
    }
    let mut cleared = 0;
    for (key, seeds) in landing {
        let n = seeds.len();
        let room = cap.saturating_sub(counts.get(&key).copied().unwrap_or(0)) as usize;
        if room >= n {
            continue;
        }
        // Seed j stays when the running share of the room steps up at it: exactly `room` of the `n`, evenly spaced.
        for (j, &i) in seeds.iter().enumerate() {
            if (j + 1) * room / n == j * room / n {
                keep[i] = false;
                cleared += 1;
            }
        }
    }
    cleared
}

/// Resident tiles to park for room once the left ones are not enough: tiles that are not among `keep` (the active and
/// near tiles), lie beyond radius + max(hysteresis, far seed) of `centre` (nearer, a distance or seed restore would
/// bring them straight back) and are not `seen` by the newest camera, farthest first, until they hold `need` splats.
/// Tiles restored fewer than `min_resident_keyframes` keyframes before `keyframes` are skipped, and so is a tile whose
/// parking would leave none of the `resident` splats.
pub fn far_inactive_to_park(
    registry: &TileRegistry,
    need: u32,
    resident: u32,
    centre: Vec3,
    keyframes: u32,
    keep: &[TileKey],
    seen: impl Fn(TileKey) -> bool,
) -> Vec<TileKey> {
    let p = registry.params();
    let beyond = p.radius + p.hysteresis.max(p.far_seed);
    let mut candidates: Vec<(f32, TileKey, u32)> = registry
        .resident_keys()
        .into_iter()
        .filter(|k| !keep.contains(k))
        .filter_map(|k| {
            let t = registry.get(k)?;
            let d = k.footprint_distance(centre, p.tile_size);
            let settled =
                t.restored_at_keyframe.is_none_or(|r| keyframes.saturating_sub(r) >= p.min_resident_keyframes);
            (d > beyond && settled && !seen(k)).then_some((d, k, t.splats))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    pick_until(candidates.into_iter().map(|(_, k, n)| (k, n)), need, resident).0
}

/// Who an eviction may take from: the active tiles, each down to its fair share of `max_splats`.
#[derive(Clone, Copy, Debug)]
pub struct FairShare<'a> {
    pub active: &'a [TileKey],
    pub max_splats: u32,
}

/// The trainer's row budget for the rows of a readback (flat xyz `means`), one group per tile (the tiles returned in
/// group order): a tile holding `tile_cap` rows or more is full and keeps at most `tile_cap` (0: no cap). With `fair`
/// a budget eviction takes only from active tiles, each at most its rows above `max_splats` / active tiles. `None`
/// when neither rule applies (no cap and no fair share), so the trainer refines as without area training.
pub fn row_budget(
    means: &[f32],
    tile_size: f32,
    tile_cap: u32,
    fair: Option<FairShare>,
) -> Option<(Vec<TileKey>, RowBudget)> {
    if tile_cap == 0 && fair.is_none() {
        return None;
    }
    let mut index: BTreeMap<TileKey, u32> = BTreeMap::new();
    let mut keys = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let group: Vec<u32> = means
        .chunks_exact(3)
        .map(|p| {
            let key = TileKey::of(Vec3::new(p[0], p[1], p[2]), tile_size);
            let g = *index.entry(key).or_insert_with(|| {
                keys.push(key);
                counts.push(0);
                (keys.len() - 1) as u32
            });
            counts[g as usize] += 1;
            g
        })
        .collect();
    let share = fair.map(|f| f.max_splats / (f.active.len().max(1) as u32));
    let groups = keys
        .iter()
        .zip(&counts)
        .map(|(key, &n)| GroupBudget {
            max_rows: if tile_cap == 0 { u32::MAX } else { tile_cap },
            full: tile_cap > 0 && n >= tile_cap,
            evictable: match (fair, share) {
                (Some(f), Some(share)) if f.active.contains(key) => n.saturating_sub(share),
                _ => 0,
            },
        })
        .collect();
    debug_assert!(group.iter().all(|&g| g != NO_GROUP));
    Some((keys, RowBudget { group, groups, limit_eviction: fair.is_some(), evict_held: false }))
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
