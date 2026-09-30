use brush_render::camera::Camera;
use glam::{Affine3A, UVec2, Vec2, Vec3};

/// Pinhole projection matching the splat rasteriser: `u = fx·x/z + cx`,
/// pixel `(i, j)` covering `[i, i+1) × [j, j+1)` (its centre at `i + 0.5`).
pub(super) struct Projection {
    size: UVec2,
    focal: Vec2,
    centre: Vec2,
    world_to_local: Affine3A,
}

impl Projection {
    pub(super) fn new(camera: &Camera, size: UVec2) -> Self {
        Self {
            size,
            focal: camera.focal(size),
            centre: camera.center(size),
            world_to_local: camera.world_to_local(),
        }
    }

    /// Camera-space point and pixel index of world point `p`, if it lies in
    /// front of the camera and inside the image.
    pub(super) fn project(&self, p: Vec3) -> Option<(Vec3, usize)> {
        let local = self.world_to_local.transform_point3(p);
        if local.z <= 0.0 {
            return None;
        }
        let u = self.focal.x * local.x / local.z + self.centre.x;
        let v = self.focal.y * local.y / local.z + self.centre.y;
        if !(u >= 0.0 && v >= 0.0 && u < self.size.x as f32 && v < self.size.y as f32) {
            return None;
        }
        Some((local, v as usize * self.size.x as usize + u as usize))
    }

    /// Camera-space point of pixel `i`'s centre at depth `d`.
    pub(super) fn unproject(&self, i: usize, d: f32) -> Vec3 {
        let (x, y) = (i as u32 % self.size.x, i as u32 / self.size.x);
        Vec3::new(
            (x as f32 + 0.5 - self.centre.x) / self.focal.x * d,
            (y as f32 + 0.5 - self.centre.y) / self.focal.y * d,
            d,
        )
    }

    /// Whether a sphere may intersect the view frustum up to depth `far`.
    pub(super) fn sees_sphere(&self, centre: Vec3, radius: f32, far: f32) -> bool {
        let c = self.world_to_local.transform_point3(centre);
        if c.z < -radius || c.z > far + radius {
            return false;
        }
        // Side planes through the optical centre and the image borders, as
        // `x - k·z = 0` with the inside on the side of the principal axis.
        let size = self.size.as_vec2();
        let sides = [
            (c.x, -self.centre.x / self.focal.x, 1.0),
            (c.x, (size.x - self.centre.x) / self.focal.x, -1.0),
            (c.y, -self.centre.y / self.focal.y, 1.0),
            (c.y, (size.y - self.centre.y) / self.focal.y, -1.0),
        ];
        sides.iter().all(|&(a, k, sign)| {
            let dist = sign * (a - k * c.z) / (1.0 + k * k).sqrt();
            dist >= -radius
        })
    }
}
