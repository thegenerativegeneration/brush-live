//! When two camera poses count as the same.

use glam::{Quat, Vec3};

/// Below both a pose counts as unchanged (a resent correction).
const SAME_POSITION_M: f32 = 1e-5;
const SAME_ROTATION_DOT: f32 = 1.0 - 1e-6;

/// Whether two camera poses (position, rotation) count as the same: a resent correction moves nothing.
pub(crate) fn same_pose(a: (Vec3, Quat), b: (Vec3, Quat)) -> bool {
    a.0.distance(b.0) < SAME_POSITION_M && a.1.dot(b.1).abs() > SAME_ROTATION_DOT
}
