//! Colour fused alongside the distance, as in `Open3D`'s coloured TSDF
//! integration (`ScalableTSDFVolume` with `TSDFVolumeColorType::RGB8`):
//! each observation's colour enters a running average with the same weight
//! as its distance, here including the drop-off weight behind the surface.
//!
//! Colours are linear RGB stored as f16, 6 bytes per voxel (48 KB per
//! brick besides 72 KB of distance, weight and status), allocated on a
//! brick's first coloured observation. At the weight cap an f16 average
//! still moves for differences down to about 0.005 (half a step of 8-bit
//! sRGB near white); 8-bit channels would halve the memory but stall the
//! average for differences below about 10/255.

use glam::{IVec3, Vec3};
use half::f16;

use crate::geometry::colour::linear_to_srgb8;

use super::changes::MeshedState;
use super::{BRICK_VOXELS, Brick, BrickKey, Tsdf, VOXEL, split_voxel, voxel_index};

/// Mean |Δ sRGB| per channel (0–255) from which `colour_changed` reports
/// a brick.
const COLOUR_CHANGE: f32 = 8.0;

/// Linear RGB per voxel of a brick; NaN until the voxel's first coloured
/// observation.
pub(super) struct VoxelColours(Vec<[f16; 3]>);

impl VoxelColours {
    pub(super) fn new() -> Self {
        Self(vec![[f16::NAN; 3]; BRICK_VOXELS])
    }

    /// Averages `rgb` with weight `w_obs` into voxel `i`, whose colour
    /// weighs `w_old`; a voxel without colour takes `rgb`.
    pub(super) fn fuse(&mut self, i: usize, rgb: [f32; 3], w_old: f32, w_obs: f32) {
        let new = match self.get(i) {
            Some(old) => {
                let w = w_old + w_obs;
                std::array::from_fn(|c| (w_old * old[c] + w_obs * rgb[c]) / w)
            }
            None => rgb,
        };
        self.0[i] = new.map(f16::from_f32);
    }

    pub(super) fn get(&self, i: usize) -> Option<[f32; 3]> {
        let c = self.0[i];
        (!c[0].is_nan()).then(|| c.map(f16::to_f32))
    }
}

impl Tsdf {
    /// Linear RGB at `world`, trilinear between the voxel centres around
    /// it that are observed and coloured, their weights renormalised;
    /// `None` if none of the eight is.
    pub fn rgb(&self, world: Vec3) -> Option<[f32; 3]> {
        let g = world / VOXEL - 0.5;
        let base = g.floor();
        let f = g - base;
        let base = base.as_ivec3();
        let (mut sum, mut total) = (Vec3::ZERO, 0.0);
        for corner in 0..8 {
            let o = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let Some(c) = self.voxel_colour(base + o) else {
                continue;
            };
            let o = o.as_vec3();
            let k = (Vec3::ONE - o) * (Vec3::ONE - f) + o * f;
            let k = k.x * k.y * k.z;
            sum += k * Vec3::from(c);
            total += k;
        }
        (total > 1e-6).then(|| (sum / total).to_array())
    }

    /// Colour of global voxel `g`, if it is observed and coloured.
    pub(super) fn voxel_colour(&self, g: IVec3) -> Option<[f32; 3]> {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get(&key)?;
        let i = voxel_index(local);
        if !brick.observed[i] {
            return None;
        }
        brick.colour.as_ref()?.get(i)
    }
}

/// 8-bit sRGB of every observed, coloured voxel of `brick`, for comparing
/// later colours against; empty if the brick has no colour. Change
/// detection needs no more than the 8 bits sent.
pub(super) fn colour_snapshot(brick: &Brick) -> Vec<Option<[u8; 3]>> {
    let Some(colours) = &brick.colour else {
        return Vec::new();
    };
    (0..BRICK_VOXELS)
        .map(|i| {
            brick.observed[i]
                .then(|| colours.get(i).map(linear_to_srgb8))
                .flatten()
        })
        .collect()
}

/// Whether `brick`'s near-surface voxels (observed and |sdf| < `TRUNC`
/// now, or when meshed), coloured then and now, changed by a mean
/// |Δ sRGB| of `COLOUR_CHANGE` or more per channel since `snapshot`.
fn colour_stale(brick: &Brick, meshed: &MeshedState) -> bool {
    let (Some(colours), false) = (&brick.colour, meshed.colour.is_empty()) else {
        return false;
    };
    let (mut sum, mut channels) = (0u32, 0u32);
    for i in 0..BRICK_VOXELS {
        let near_now = brick.observed[i] && brick.tsdf[i].abs() < 1.0;
        if !near_now && !meshed.near_surface(i) {
            continue;
        }
        let now = brick.observed[i].then(|| colours.get(i)).flatten();
        let (Some(then), Some(now)) = (meshed.colour[i], now) else {
            continue;
        };
        let now = linear_to_srgb8(now);
        sum += (0..3)
            .map(|c| u32::from(then[c].abs_diff(now[c])))
            .sum::<u32>();
        channels += 3;
    }
    channels > 0 && sum as f32 / channels as f32 >= COLOUR_CHANGE
}

impl Tsdf {
    /// Bricks meshed before whose colour changed by a mean |Δ sRGB| of
    /// `COLOUR_CHANGE` (8 of 255) or more per channel since
    /// `mark_meshed`, over their near-surface voxels (observed, |sdf| <
    /// `TRUNC` now or then) coloured both times; in key order. A brick
    /// can be in both this and `changed`.
    pub fn colour_changed(&self) -> Vec<BrickKey> {
        let mut keys: Vec<BrickKey> = self
            .bricks
            .iter()
            .filter(|(_, b)| b.meshed.as_ref().is_some_and(|m| colour_stale(b, m)))
            .map(|(&k, _)| k)
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Sets the colour of every voxel of existing bricks `keys` to `rgb`.
    #[cfg(test)]
    pub(crate) fn fill_colour(&mut self, keys: &[BrickKey], rgb: [f32; 3]) {
        for key in keys {
            let brick = self.bricks.get_mut(key).expect("allocated brick");
            let colours = brick.colour.get_or_insert_with(VoxelColours::new);
            for i in 0..BRICK_VOXELS {
                colours.0[i] = rgb.map(f16::from_f32);
            }
        }
    }
}
