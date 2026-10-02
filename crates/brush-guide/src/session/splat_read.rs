//! Splat parameters read back from the GPU and turned into the per-Gaussian
//! inputs of the voxel aggregation.

use crate::scores::voxel::{GaussianScore, SplatGeom, ViewCone};
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::Splats;
use burn::tensor::Tensor;
use glam::{Quat, Vec3};

pub(super) async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("splat readback")
        .try_to_vec::<f32>()
        .expect("f32 splat data")
}

/// Splat parameters read back once per round, flat per splat.
pub(super) struct SplatRead {
    pub(super) means: Vec<f32>,
    pub(super) opac: Vec<f32>,
    pub(super) rots: Vec<f32>,
    pub(super) scales: Vec<f32>,
}

impl SplatRead {
    pub(super) async fn new(splats: &Splats) -> Self {
        Self {
            means: read_f32(splats.means()).await,
            opac: read_f32(splats.opacities()).await,
            rots: read_f32(splats.rotations()).await,
            scales: read_f32(splats.scales()).await,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.opac.len()
    }

    pub(super) fn geom(&self, i: usize) -> SplatGeom {
        let (m, r, s) = (&self.means, &self.rots, &self.scales);
        SplatGeom {
            pos: Vec3::new(m[i * 3], m[i * 3 + 1], m[i * 3 + 2]),
            opacity: self.opac[i],
            axis: shortest_axis(&r[i * 4..i * 4 + 4], &s[i * 3..i * 3 + 3]),
            flatness: flatness(&s[i * 3..i * 3 + 3]),
        }
    }

    pub(super) fn geoms(&self) -> Vec<SplatGeom> {
        (0..self.len()).map(|i| self.geom(i)).collect()
    }

    /// Per-Gaussian inputs of a Fisher pass's aggregation.
    pub(super) fn scores(&self, coverage: &[f32], fisher_pos: &[[f32; 9]]) -> Vec<GaussianScore> {
        (0..self.len())
            .map(|i| {
                let g = self.geom(i);
                GaussianScore {
                    pos: g.pos,
                    opacity: g.opacity,
                    coverage: coverage[i],
                    fisher_pos: fisher_pos[i],
                    axis: g.axis,
                    flatness: g.flatness,
                }
            })
            .collect()
    }
}

pub(super) fn view_cones(views: &[SceneView]) -> Vec<ViewCone> {
    views
        .iter()
        .map(|v| {
            let c = &v.camera;
            ViewCone {
                position: c.position,
                forward: c.rotation * Vec3::Z,
                cos_half_fov: (0.5 * c.fov_x.max(c.fov_y) as f32).cos(),
            }
        })
        .collect()
}

/// World direction of the Gaussian's shortest scale axis; zero for a
/// degenerate rotation.
pub(super) fn shortest_axis(r: &[f32], s: &[f32]) -> Vec3 {
    // Brush stores rotations as [w, x, y, z].
    let q = Quat::from_xyzw(r[1], r[2], r[3], r[0]);
    if !q.is_finite() || q.length_squared() == 0.0 {
        return Vec3::ZERO;
    }
    let k = if s[0] <= s[1] && s[0] <= s[2] {
        0
    } else if s[1] <= s[2] {
        1
    } else {
        2
    };
    q.normalize() * Vec3::AXES[k]
}

/// `1 − s_min / s_mid` of the Gaussian's scales: 0 when round (or
/// degenerate), towards 1 for a flat disc.
pub(super) fn flatness(s: &[f32]) -> f32 {
    let mut v = [s[0], s[1], s[2]];
    v.sort_by(f32::total_cmp);
    if !(v[0].is_finite() && v[1].is_finite()) || v[1] <= 0.0 {
        return 0.0;
    }
    (1.0 - v[0] / v[1]).clamp(0.0, 1.0)
}
