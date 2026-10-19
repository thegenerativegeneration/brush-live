//! Views bucketed by pose for the sight rule (`live::area_sight`): 0.5 m position cells × 8 yaw × 3 pitch bins. One
//! view per bucket represents it in the frustum tests and the score round's vote.

use brush_render::camera::Camera;
use glam::{IVec3, Vec3};
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// Edge of a position cell, metres.
pub const CELL_M: f32 = 0.5;
/// Bins of the viewing direction about the vertical (y up) and of its elevation.
pub const YAW_BINS: f32 = 8.0;
pub const PITCH_BINS: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Bucket {
    cell: IVec3,
    yaw: i32,
    pitch: i32,
}

/// The bucket of a camera's current pose; `None` for a position or direction that is not finite.
fn bucket_of(c: &Camera) -> Option<Bucket> {
    let forward = (c.rotation * Vec3::Z).normalize();
    if !c.position.is_finite() || !forward.is_finite() {
        return None;
    }
    // Bins centred on the axes, so a camera looking straight along one does not straddle two bins.
    let yaw = forward.x.atan2(forward.z) + PI + PI / YAW_BINS;
    let pitch = forward.y.clamp(-1.0, 1.0).asin() + FRAC_PI_2;
    Some(Bucket {
        cell: (c.position / CELL_M).floor().as_ivec3(),
        yaw: ((yaw / TAU * YAW_BINS) as i32).rem_euclid(YAW_BINS as i32),
        pitch: ((pitch / PI * PITCH_BINS) as i32).min(PITCH_BINS as i32 - 1),
    })
}

/// Per view of `cameras` (focal lengths in px in `focals`), its bucket's representative: the view with the largest
/// focal, ties to the lowest index. A view whose pose is not finite represents itself.
pub fn representatives(cameras: &[Camera], focals: &[f32]) -> Vec<usize> {
    let buckets: Vec<Option<Bucket>> = cameras.iter().map(bucket_of).collect();
    let mut best: HashMap<Bucket, usize> = HashMap::new();
    for (i, b) in buckets.iter().enumerate() {
        if let Some(b) = b {
            best.entry(*b)
                .and_modify(|r| {
                    if focals[i] > focals[*r] {
                        *r = i
                    }
                })
                .or_insert(i);
        }
    }
    buckets.iter().enumerate().map(|(i, b)| b.map_or(i, |b| best[&b])).collect()
}

#[cfg(test)]
#[path = "buckets_tests.rs"]
mod tests;
