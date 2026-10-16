use brush_recon::{ReconError, ReconFrame};
use brush_triangle_render::TriangleSeed;
use glam::{UVec2, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedConfig {
    pub anchor_ids: Vec<u64>,
    pub stride: usize,
    pub max_faces: usize,
    pub min_confidence: u8,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
    pub max_relative_depth_jump: f32,
    pub min_area_m2: f32,
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            anchor_ids: vec![174, 184, 194, 204],
            stride: 4,
            max_faces: 20_000,
            min_confidence: 2,
            min_depth_m: 0.1,
            max_depth_m: 8.0,
            max_relative_depth_jump: 0.05,
            min_area_m2: 1e-8,
        }
    }
}

pub fn seed_from_training(
    train: &[Arc<ReconFrame>],
    cfg: &SeedConfig,
) -> Result<TriangleSeed, ReconError> {
    if cfg.anchor_ids.is_empty()
        || cfg.stride == 0
        || cfg.max_faces == 0
        || cfg.min_confidence > 2
        || !cfg.min_depth_m.is_finite()
        || cfg.min_depth_m <= 0.
        || !cfg.max_depth_m.is_finite()
        || cfg.max_depth_m < cfg.min_depth_m
        || !cfg.max_relative_depth_jump.is_finite()
        || cfg.max_relative_depth_jump < 0.
        || !cfg.min_area_m2.is_finite()
        || cfg.min_area_m2 < 0.
        || cfg.anchor_ids.iter().collect::<BTreeSet<_>>().len() != cfg.anchor_ids.len()
    {
        return Err(ReconError::input("invalid depth seeding configuration"));
    }
    let mut seed = TriangleSeed {
        positions: vec![],
        faces: vec![],
        colors: vec![],
    };
    let mut groups = vec![];
    let mut ids = BTreeSet::new();
    for f in train {
        f.validate()?;
        if !ids.insert(f.id) {
            return Err(ReconError::input("duplicate training ID"));
        }
    }
    for id in &cfg.anchor_ids {
        let frame = train
            .iter()
            .find(|f| f.id == *id)
            .ok_or_else(|| ReconError::input(format!("training anchor {id} is absent")))?;
        let depth = frame
            .lidar
            .as_ref()
            .ok_or_else(|| ReconError::input(format!("training anchor {id} has no LiDAR")))?;
        let xs: Vec<_> = (0..depth.size.x as usize).step_by(cfg.stride).collect();
        let ys: Vec<_> = (0..depth.size.y as usize).step_by(cfg.stride).collect();
        let mut grid = vec![None; xs.len() * ys.len()];
        let focal = frame.camera.focal(depth.size);
        let center = frame.camera.center(depth.size);
        for (gy, &y) in ys.iter().enumerate() {
            for (gx, &x) in xs.iter().enumerate() {
                let uv = Vec2::new(
                    (x as f32 + 0.5) / depth.size.x as f32,
                    (y as f32 + 0.5) / depth.size.y as f32,
                );
                let Some(z) = depth
                    .sample_uv(uv, cfg.min_confidence)
                    .filter(|z| *z >= cfg.min_depth_m && *z <= cfg.max_depth_m)
                else {
                    continue;
                };
                let xy = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - center) / focal * z;
                let world = frame
                    .camera
                    .local_to_world()
                    .transform_point3(Vec3::new(xy.x, xy.y, z));
                let index = u32::try_from(seed.positions.len())
                    .map_err(|_| ReconError::input("too many seed vertices"))?;
                seed.positions.push(world.to_array());
                let size = UVec2::new(frame.rgb.width(), frame.rgb.height());
                let pixel = frame.rgb.get_pixel(
                    ((uv.x * size.x as f32) as u32).min(size.x - 1),
                    ((uv.y * size.y as f32) as u32).min(size.y - 1),
                );
                seed.colors.push(pixel.0.map(|v| v as f32 / 255.));
                grid[gy * xs.len() + gx] = Some(index);
            }
        }
        let mut faces = vec![];
        for y in 0..ys.len().saturating_sub(1) {
            for x in 0..xs.len().saturating_sub(1) {
                let tl = y * xs.len() + x;
                let tr = tl + 1;
                let bl = tl + xs.len();
                let br = bl + 1;
                for corners in [[tl, tr, bl], [tr, br, bl]] {
                    let Some(face) = corners
                        .map(|i| grid[i])
                        .into_iter()
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    let face = [face[0], face[1], face[2]];
                    let points = face.map(|i| Vec3::from(seed.positions[i as usize]));
                    let z = points.map(|p| frame.camera.world_to_local().transform_point3(p).z);
                    let lo = z.into_iter().fold(f32::INFINITY, f32::min);
                    let hi = z.into_iter().fold(f32::NEG_INFINITY, f32::max);
                    if hi - lo > cfg.max_relative_depth_jump * lo {
                        continue;
                    }
                    let area = (points[1] - points[0])
                        .cross(points[2] - points[0])
                        .length()
                        * 0.5;
                    if !area.is_finite() || area < cfg.min_area_m2 || area == 0. {
                        continue;
                    }
                    faces.push(face);
                }
            }
        }
        groups.push(faces.into_iter());
    }
    while seed.faces.len() < cfg.max_faces {
        let before = seed.faces.len();
        for group in &mut groups {
            if let Some(face) = group.next() {
                seed.faces.push(face);
            }
            if seed.faces.len() == cfg.max_faces {
                break;
            }
        }
        if seed.faces.len() == before {
            break;
        }
    }
    if seed.faces.is_empty() {
        return Err(ReconError::input(
            "training depth produced no admissible triangles",
        ));
    }
    let referenced: BTreeSet<_> = seed.faces.iter().flatten().copied().collect();
    let mut remap = vec![0_u32; seed.positions.len()];
    let mut positions = vec![];
    let mut colors = vec![];
    for old in referenced {
        remap[old as usize] = positions.len() as u32;
        positions.push(seed.positions[old as usize]);
        colors.push(seed.colors[old as usize]);
    }
    for face in &mut seed.faces {
        *face = face.map(|i| remap[i as usize]);
    }
    seed.positions = positions;
    seed.colors = colors;
    seed.validate()?;
    Ok(seed)
}
