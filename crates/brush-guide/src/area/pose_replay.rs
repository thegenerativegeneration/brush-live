//! Pose-correction replay for parked tiles: a parked tile's splats move the way resident splats moved while it was
//! parked (a loop closure, say). Resident splats follow each correction of their seeding view
//! (`LiveModel::update_poses` → `SplatTrainer::move_by_origin`); a parked tile records each seeding view's pose at
//! park time and, on restore, moves by the one rigid step from that pose to the view's current pose. Rigid steps
//! compose, so several corrections while parked land where resident splats landed, each applied once.

use super::park_file::SlotPose;
use brush_train::rows::SplatRows;
use glam::{Affine3A, Quat, Vec3};
use std::collections::HashMap;

/// A camera pose: position and rotation, as `Camera::local_to_world` reads them.
pub type CameraPose = (Vec3, Quat);

/// The share of a tile's rows [`move_spread`] may leave out: its smallest view groups, together at most this share,
/// do not decide whether the tile is dropped.
pub const MIN_SLOT_SHARE: f32 = 0.05;

/// Per recorded slot with a current pose: the world transform taking the camera at park time onto the current
/// camera, as `LiveModel::update_poses` builds for resident splats (`new.local_to_world() * old.world_to_local()`).
/// A slot whose current pose equals its recorded one exactly yields no move, so its rows come back bit for bit:
/// `update_poses` stores a received pose exactly and leaves the camera untouched for a resent one, so any
/// difference, however small, is a correction the resident splats followed.
pub fn slot_moves(
    recorded: &[SlotPose],
    current: &dyn Fn(u32) -> Option<CameraPose>,
) -> Vec<(u32, Affine3A)> {
    recorded
        .iter()
        .filter_map(|r| {
            let (p, q) = current(r.slot)?;
            if (p, q) == (r.position, r.rotation) {
                return None;
            }
            let old = Affine3A::from_rotation_translation(r.rotation, r.position);
            Some((
                r.slot,
                Affine3A::from_rotation_translation(q, p) * old.inverse(),
            ))
        })
        .collect()
}

/// Per moved slot, the move split into rotation and translation, as `SplatOrigin::move_splats` splits it.
fn parts(moves: &[(u32, Affine3A)]) -> HashMap<u32, (Quat, Vec3)> {
    moves
        .iter()
        .map(|(slot, m)| {
            let (_, q, t) = m.to_scale_rotation_translation();
            (*slot, (q, t))
        })
        .collect()
}

/// How far apart the tile's views carry it; a tile beyond `area_restore_inconsistency_m` is dropped, since its views
/// disagree about where it is. The spread is the tile's centroid moved by each view group (one per moved slot; rows
/// without an origin or whose slot has no move form one unmoved group), the largest distance between two moved
/// centroids, 0 with fewer than two groups. The smallest groups are left out while together they hold at most
/// [`MIN_SLOT_SHARE`] of the rows, so a few stray rows do not decide but many small groups that disagree still do.
pub fn move_spread(rows: &SplatRows, moves: &[(u32, Affine3A)]) -> f32 {
    let n = rows.len();
    if n == 0 {
        return 0.0;
    }
    let sum = (0..n).fold(glam::DVec3::ZERO, |acc, i| acc + rows.mean(i).as_dvec3());
    let centroid = (sum / n as f64).as_vec3();
    let parts = parts(moves);
    let mut counts: HashMap<Option<u32>, usize> = HashMap::new();
    for o in &rows.origin {
        *counts
            .entry(o.filter(|s| parts.contains_key(s)))
            .or_default() += 1;
    }
    let mut groups: Vec<(usize, Option<u32>)> =
        counts.into_iter().map(|(slot, c)| (c, slot)).collect();
    groups.sort_unstable();
    let may_drop = MIN_SLOT_SHARE * n as f32;
    let mut dropped = 0usize;
    let kept = groups.iter().skip_while(|&&(count, _)| {
        let drop = (dropped + count) as f32 <= may_drop;
        if drop {
            dropped += count;
        }
        drop
    });
    let centres: Vec<Vec3> = kept
        .map(|&(_, slot)| match slot.and_then(|s| parts.get(&s)) {
            Some(&(q, t)) => q * centroid + t,
            None => centroid,
        })
        .collect();
    let mut spread = 0.0f32;
    for (i, a) in centres.iter().enumerate() {
        for b in &centres[i + 1..] {
            spread = spread.max(a.distance(*b));
        }
    }
    spread
}

/// Hamilton product `a ⊗ b` of quaternions stored `[w, x, y, z]`, unnormalised `b` kept at its length.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Moves every row whose origin slot has a move as `SplatOrigin::move_splats` does on the GPU: mean `q_Δ·μ + t_Δ`,
/// rotation `q_Δ ⊗ q` (stored w x y z, its length kept). Scales, colours, opacities, the floor and the eviction
/// state stay; SH bands above 0 are not rotated (corrections turn by at most about 6° on the device; Brush has no SH
/// rotation). Rows without an origin or a move stay.
pub fn apply_moves(rows: &mut SplatRows, moves: &[(u32, Affine3A)]) {
    let parts = parts(moves);
    if parts.is_empty() {
        return;
    }
    for i in 0..rows.len() {
        let Some(&(q, t)) = rows.origin[i].and_then(|s| parts.get(&s)) else {
            continue;
        };
        let mean = q * rows.mean(i) + t;
        rows.means[i * 3..i * 3 + 3].copy_from_slice(&mean.to_array());
        let r = &mut rows.rotations[i * 4..i * 4 + 4];
        let moved = quat_mul([q.w, q.x, q.y, q.z], [r[0], r[1], r[2], r[3]]);
        r.copy_from_slice(&moved);
    }
}

#[cfg(test)]
#[path = "pose_replay_tests.rs"]
mod tests;
