//! Seam-band PSNR for held-out evals: pixels whose surface lies near a tile border are scored apart from the rest.

use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::{TextureMode, render_splats};
use burn::tensor::Tensor;
use glam::{UVec2, Vec2, Vec3};

/// Depth that codes to a colour of 1.
const DEPTH_FAR_M: f32 = 200.0;
/// Zeroth-order SH constant: colour = 0.5 + SH_C0 * dc.
const SH_C0: f32 = 0.282_094_8;

/// Horizontal distance (metres) from `p` to the nearest border of the `tile_size` grid in x and z.
pub(super) fn border_distance(p: Vec3, tile_size: f32) -> f32 {
    let d = |v: f32| {
        let r = v.rem_euclid(tile_size);
        r.min(tile_size - r)
    };
    d(p.x).min(d(p.z))
}

/// Per pixel: `Some(true)` if its surface is within `band_m` of a tile border, `Some(false)` if not, `None`
/// without a surface.
pub(super) fn border_band(points: &[Option<Vec3>], tile_size: f32, band_m: f32) -> Vec<Option<bool>> {
    points
        .iter()
        .map(|p| p.map(|p| border_distance(p, tile_size) < band_m))
        .collect()
}

/// PSNR over the band pixels and over the other labelled pixels; `render` and `gt` are row-major RGB.
/// A group without pixels gives `None`.
pub(super) fn split_psnr(render: &[f32], gt: &[f32], band: &[Option<bool>]) -> (Option<f32>, Option<f32>) {
    let mut sum = [0.0f64; 2];
    let mut count = [0usize; 2];
    for (i, label) in band.iter().enumerate() {
        let Some(in_band) = label else { continue };
        let g = usize::from(!in_band);
        for c in 0..3 {
            let d = f64::from(render[i * 3 + c] - gt[i * 3 + c]);
            sum[g] += d * d;
        }
        count[g] += 1;
    }
    let psnr = |g: usize| {
        (count[g] > 0).then(|| {
            let mse = (sum[g] / (count[g] * 3) as f64).max(1e-10);
            (10.0 * (1.0 / mse).log10()) as f32
        })
    };
    (psnr(0), psnr(1))
}

/// World position of each pixel's surface (row-major, `size`), from a render that codes depth as colour;
/// `None` where the accumulated opacity is below 0.5.
pub(super) async fn surface_points(splats: &Splats, camera: &Camera, size: UVec2) -> Vec<Option<Vec3>> {
    let device = splats.device();
    let n = splats.num_splats() as usize;
    let fwd = camera.rotation * Vec3::Z;
    let forward = Tensor::<1>::from_floats([fwd.x, fwd.y, fwd.z], &device).reshape([3, 1]);
    let offset = camera.position.dot(fwd);
    let z = splats.means().matmul(forward).sub_scalar(offset).reshape([n]);
    let dc = z.div_scalar(DEPTH_FAR_M).sub_scalar(0.5).div_scalar(SH_C0).reshape([n, 1, 1]);
    let dc = Tensor::cat(vec![dc.clone(), dc.clone(), dc], 2);
    let mut coded = Splats::from_tensor_data(
        splats.means(),
        splats.rotations(),
        splats.log_scales(),
        dc,
        splats.raw_opacities.val(),
        SplatRenderMode::Default,
    );
    if let Some(f) = &splats.min_scale {
        coded = coded.with_min_scale(f.clone());
    }
    let (img, _) = render_splats(coded, camera, size, Vec3::ZERO, None, TextureMode::Float).await;
    let Ok(Ok(data)) = img.into_data_async().await.map(|d| d.try_to_vec::<f32>()) else {
        return vec![None; (size.x * size.y) as usize];
    };
    let (f, c) = (camera.focal(size), camera.center(size));
    let to_world = camera.local_to_world();
    (0..(size.x * size.y) as usize)
        .map(|i| {
            let (r, a) = (data[i * 4], data[i * 4 + 3]);
            if a < 0.5 {
                return None;
            }
            let depth = r / a * DEPTH_FAR_M;
            let px = Vec2::new((i as u32 % size.x) as f32 + 0.5, (i as u32 / size.x) as f32 + 0.5);
            let local = Vec3::new((px.x - c.x) / f.x * depth, (px.y - c.y) / f.y * depth, depth);
            Some(to_world.transform_point3(local))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::gaussian_splats::inverse_sigmoid;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::Quat;

    #[test]
    fn border_distance_is_horizontal_and_handles_negative_coordinates() {
        assert!((border_distance(Vec3::new(0.3, 50.0, 5.0), 10.0) - 0.3).abs() < 1e-6);
        assert!((border_distance(Vec3::new(-0.2, 0.0, 4.0), 10.0) - 0.2).abs() < 1e-6);
        assert!((border_distance(Vec3::new(-9.6, 0.0, -15.0), 10.0) - 0.4).abs() < 1e-5);
        assert!((border_distance(Vec3::new(5.0, 0.0, 5.0), 10.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn band_labels_surface_pixels_only() {
        let pts = [Some(Vec3::new(0.2, 0.0, 5.0)), Some(Vec3::new(5.0, 0.0, 5.0)), None];
        assert_eq!(border_band(&pts, 10.0, 0.5), vec![Some(true), Some(false), None]);
    }

    #[test]
    fn split_psnr_scores_each_group_and_skips_empty_ones() {
        // Two pixels: the band pixel is off by 0.1 in every channel, the interior pixel is exact.
        let render = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2];
        let gt = [0.6, 0.6, 0.6, 0.2, 0.2, 0.2];
        let (band, interior) = split_psnr(&render, &gt, &[Some(true), Some(false)]);
        assert!((band.unwrap() - 20.0).abs() < 1e-3, "{band:?}");
        assert!(interior.unwrap() > 60.0, "exact pixels give a capped, high PSNR: {interior:?}");
        assert_eq!(split_psnr(&render, &gt, &[None, None]), (None, None));
    }

    /// A wall of opaque splats at z = 3 in front of the camera comes back at depth 3.
    #[tokio::test]
    async fn surface_points_find_a_wall_at_its_depth() {
        let device = brush_cube::test_helpers::test_device().await.into();
        let n = 40;
        let means: Vec<f32> = (0..n * n)
            .flat_map(|i| [((i % n) as f32 - 20.0) * 0.1, ((i / n) as f32 - 20.0) * 0.1, 3.0])
            .collect();
        let splats = Splats::from_raw(
            means,
            [1.0, 0.0, 0.0, 0.0].repeat(n * n),
            vec![-2.5; n * n * 3],
            vec![0.5; n * n * 3],
            vec![inverse_sigmoid(0.99); n * n],
            SplatRenderMode::Default,
            &device,
        );
        let fov = 60f64.to_radians();
        let camera = Camera::new(Vec3::ZERO, Quat::IDENTITY, fov, fov, glam::vec2(0.5, 0.5), CameraModel::Pinhole);
        let pts = surface_points(&splats, &camera, UVec2::new(32, 32)).await;
        let centre = pts[16 * 32 + 16].expect("the wall covers the centre pixel");
        assert!((centre.z - 3.0).abs() < 0.05, "{centre}");
    }
}
