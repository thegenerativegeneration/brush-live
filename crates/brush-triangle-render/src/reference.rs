use crate::TriangleSeed;
use brush_recon::{Capabilities, ReconError, RenderRequest};
use glam::{UVec2, Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ReferenceConfig {
    pub sigma: f32,
    pub weight_floor: f32,
    pub near_m: f32,
}
impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            sigma: 1.,
            weight_floor: 0.,
            near_m: 0.05,
        }
    }
}

impl ReferenceConfig {
    pub fn validate(&self) -> Result<(), ReconError> {
        if !self.sigma.is_finite()
            || self.sigma <= 0.
            || !self.near_m.is_finite()
            || self.near_m <= 0.
            || !self.weight_floor.is_finite()
            || !(0.0..=1.0).contains(&self.weight_floor)
        {
            return Err(ReconError::input(
                "sigma/near must be finite and positive; weight floor must be 0..1",
            ));
        }
        Ok(())
    }
}

pub struct ReferenceImage {
    pub size: UVec2,
    pub rgb: Vec<[f32; 3]>,
    pub alpha: Vec<f32>,
    pub depth_m: Vec<f32>,
    pub normals: Vec<[f32; 3]>,
    pub visible_faces: usize,
    pub culled_faces: usize,
}

pub(crate) struct ProjectedFace {
    pub id: usize,
    pub points: [Vec2; 3],
    pub inverse_z: [f32; 3],
    pub colors: [Vec3; 3],
    pub normal: Vec3,
    pub centroid_z: f32,
    pub weight: f32,
    pub area_twice: f32,
    pub inradius: f32,
    pub edge_lengths: [f32; 3],
    pub min: UVec2,
    pub max: UVec2,
}

pub(crate) fn edge(a: Vec2, b: Vec2, p: Vec2) -> f32 {
    (b - a).perp_dot(p - a)
}

pub(crate) fn sigmoid(v: f32) -> f32 {
    1. / (1. + (-v).exp())
}

fn project_face(
    seed: &TriangleSeed,
    colors: &[[f32; 3]],
    weights: &[f32],
    id: usize,
    request: &RenderRequest,
    cfg: &ReferenceConfig,
) -> Option<ProjectedFace> {
    let ids = seed.faces[id];
    if ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0] {
        return None;
    }
    let world = ids.map(|i| Vec3::from(seed.positions[i as usize]));
    let cross = (world[1] - world[0]).cross(world[2] - world[0]);
    let area = cross.length() * 0.5;
    if !area.is_finite() || area < 1e-8 {
        return None;
    }
    let local = world.map(|p| request.camera.world_to_local().transform_point3(p));
    if local.iter().any(|p| !p.is_finite() || p.z <= cfg.near_m) {
        return None;
    }
    let inverse_z = local.map(|p| p.z.recip());
    let focal = request.camera.focal(request.size);
    let center = request.camera.center(request.size);
    let points = std::array::from_fn(|v| local[v].truncate() * inverse_z[v] * focal + center);
    if points.iter().any(|p: &Vec2| !p.is_finite()) {
        return None;
    }
    let area_twice = edge(points[0], points[1], points[2]);
    if area_twice.abs() * 0.5 < 1e-6 {
        return None;
    }
    let edge_lengths = std::array::from_fn(|i| (points[(i + 1) % 3] - points[i]).length());
    let inradius = area_twice.abs() / edge_lengths.iter().sum::<f32>();
    if !inradius.is_finite() || inradius <= 0. {
        return None;
    }
    let min = points
        .iter()
        .fold(Vec2::splat(f32::INFINITY), |a, b| a.min(*b))
        .floor()
        .max(Vec2::ZERO)
        .min(request.size.as_vec2())
        .as_uvec2();
    let max = points
        .iter()
        .fold(Vec2::splat(f32::NEG_INFINITY), |a, b| a.max(*b))
        .ceil()
        .max(Vec2::ZERO)
        .min(request.size.as_vec2())
        .as_uvec2();
    if min.x >= max.x || min.y >= max.y {
        return None;
    }
    let mut normal = cross / (2. * area);
    let centroid = (world[0] + world[1] + world[2]) / 3.;
    if normal.dot(request.camera.position - centroid) < 0. {
        normal = -normal;
    }
    let weight = ids
        .into_iter()
        .map(|i| cfg.weight_floor + (1. - cfg.weight_floor) * sigmoid(weights[i as usize]))
        .fold(f32::INFINITY, f32::min);
    Some(ProjectedFace {
        id,
        points,
        inverse_z,
        colors: ids.map(|i| Vec3::from(colors[i as usize].map(sigmoid))),
        normal,
        centroid_z: (local[0].z + local[1].z + local[2].z) / 3.,
        weight,
        area_twice,
        inradius,
        edge_lengths,
        min,
        max,
    })
}

pub(crate) fn project_reference(
    seed: &TriangleSeed,
    colors: &[[f32; 3]],
    weights: &[f32],
    request: &RenderRequest,
    cfg: &ReferenceConfig,
) -> Result<Vec<ProjectedFace>, ReconError> {
    seed.validate()?;
    cfg.validate()?;
    request.validate(Capabilities {
        depth: true,
        normals: true,
    })?;
    if colors.len() != seed.positions.len()
        || weights.len() != seed.positions.len()
        || colors
            .iter()
            .flatten()
            .chain(weights.iter())
            .any(|v| !v.is_finite())
    {
        return Err(ReconError::input(
            "triangle logit shapes or values are invalid",
        ));
    }
    let mut projected: Vec<_> = (0..seed.faces.len())
        .filter_map(|id| project_face(seed, colors, weights, id, request, cfg))
        .collect();
    projected.sort_by(|a, b| a.centroid_z.total_cmp(&b.centroid_z).then(a.id.cmp(&b.id)));
    Ok(projected)
}

pub fn render_reference(
    seed: &TriangleSeed,
    rgb_logits: &[[f32; 3]],
    weight_logits: &[f32],
    request: &RenderRequest,
    cfg: &ReferenceConfig,
) -> Result<ReferenceImage, ReconError> {
    let projected = project_reference(seed, rgb_logits, weight_logits, request, cfg)?;
    let n = brush_recon::frame::pixel_count(request.size)?;
    let mut rgb = vec![Vec3::ZERO; n];
    let mut alpha = vec![0.; n];
    let mut transmittance = vec![1.; n];
    let mut depth_m = vec![0.; n];
    let mut normals = vec![Vec3::ZERO; n];
    for face in &projected {
        for y in face.min.y..face.max.y {
            for x in face.min.x..face.max.x {
                let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let e: [f32; 3] =
                    std::array::from_fn(|v| edge(face.points[v], face.points[(v + 1) % 3], point));
                let distance = (0..3)
                    .map(|v| e[v] * face.area_twice.signum() / face.edge_lengths[v])
                    .fold(f32::INFINITY, f32::min);
                if distance <= 0. {
                    continue;
                }
                let a = (face.weight * (distance / face.inradius).powf(cfg.sigma)).min(0.99);
                let perspective: [f32; 3] =
                    std::array::from_fn(|v| e[(v + 1) % 3] / face.area_twice * face.inverse_z[v]);
                let denominator = perspective.iter().sum::<f32>();
                let color = (0..3)
                    .map(|v| face.colors[v] * perspective[v])
                    .sum::<Vec3>()
                    / denominator;
                let index = (y * request.size.x + x) as usize;
                let blend = transmittance[index] * a;
                rgb[index] += color * blend;
                depth_m[index] += denominator.recip() * blend;
                normals[index] += face.normal * blend;
                transmittance[index] *= 1. - a;
            }
        }
    }
    for i in 0..n {
        rgb[i] += request.background * transmittance[i];
        alpha[i] = 1. - transmittance[i];
        if alpha[i] > 1e-6 {
            depth_m[i] /= alpha[i];
            normals[i] /= alpha[i];
        } else {
            depth_m[i] = 0.;
            normals[i] = Vec3::ZERO;
        }
    }
    Ok(ReferenceImage {
        size: request.size,
        rgb: rgb.into_iter().map(|v| v.to_array()).collect(),
        alpha,
        depth_m,
        normals: normals.into_iter().map(|v| v.to_array()).collect(),
        visible_faces: projected.len(),
        culled_faces: seed.faces.len() - projected.len(),
    })
}
