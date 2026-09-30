//! Expected depth from the splat, after gsplat's `render_mode="ED"`: each
//! Gaussian is coloured with its camera-space depth in R and a constant 1 in
//! G on a black background, so R = Σ wᵢzᵢ and G = Σ wᵢ and depth = R / G.
//! The float render path clamps colours to ±100 only, so depth in metres
//! passes through unscaled. The splat's own colour is rendered alongside
//! (`render_colour`) at the same size.

use brush_render::camera::Camera;
use brush_render::gaussian_splats::{Splats, TextureMode, render_splats};
use brush_render::shaders::SH_C0;
use burn::Tensor;
use burn::module::{Param, ParamId};
use burn::tensor::s;
use glam::Vec3;

use super::colour::srgb_to_linear;

/// Pixels whose accumulated opacity is below this have no depth.
const MIN_ALPHA: f32 = 0.5;

/// Expected depth (metres along the camera's forward axis) and accumulated opacity per pixel, row-major.
pub struct DepthImage {
    pub width: u32,
    pub height: u32,
    pub depth: Vec<f32>,
    pub alpha: Vec<f32>,
}

/// Renders the expected depth of `splats` seen from `camera`. Pixels with
/// `alpha < 0.5` get `depth = NaN`.
pub async fn render_expected_depth(
    splats: &Splats,
    camera: &Camera,
    size: glam::UVec2,
) -> DepthImage {
    let means = splats.means().detach();
    let device = means.device();
    let n = means.dims()[0];

    let forward = camera.rotation * Vec3::Z;
    let forward = Tensor::<2>::from_floats([[forward.x], [forward.y], [forward.z]], &device);
    let position = Tensor::<2>::from_floats([camera.position.to_array()], &device);
    let z = means.sub(position).matmul(forward); // [N,1]

    // DC coefficient for colour c is (c − 0.5) / C0; the kernel adds 0.5 back.
    let colour = Tensor::cat(vec![z.clone(), z.ones_like(), z.zeros_like()], 1); // [N,3]
    let sh = colour.sub_scalar(0.5).div_scalar(SH_C0).reshape([n, 1, 3]);

    let mut coloured = splats.clone();
    coloured.sh_coeffs = Param::initialized(ParamId::new(), sh);

    let (img, _) =
        render_splats(coloured, camera, size, Vec3::ZERO, None, TextureMode::Float).await;
    let rg = img
        .slice(s![.., .., 0..2])
        .into_data_async()
        .await
        .expect("depth readback")
        .try_to_vec::<f32>()
        .expect("f32 depth");

    let (depth, alpha) = rg
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[weighted, alpha]| {
            let depth = if alpha >= MIN_ALPHA {
                weighted / alpha
            } else {
                f32::NAN
            };
            (depth, alpha)
        })
        .unzip();

    DepthImage {
        width: size.x,
        height: size.y,
        depth,
        alpha,
    }
}

/// Colour of the splat per pixel as linear RGB in `[0, 1]`, and
/// accumulated opacity, row-major.
pub struct ColourImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
    pub alpha: Vec<f32>,
}

/// Renders the colour of `splats` seen from `camera` with their full
/// spherical harmonics on a black background, divided by the accumulated
/// opacity and converted from sRGB to linear. Pixels with `alpha < 0.5`,
/// which have no depth either, get `rgb = NaN`.
pub async fn render_colour(splats: &Splats, camera: &Camera, size: glam::UVec2) -> ColourImage {
    let (img, _) = render_splats(
        splats.clone(),
        camera,
        size,
        Vec3::ZERO,
        None,
        TextureMode::Float,
    )
    .await;
    let rgba = img
        .into_data_async()
        .await
        .expect("colour readback")
        .try_to_vec::<f32>()
        .expect("f32 colour");

    let (rgb, alpha) = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[r, g, b, alpha]| {
            let rgb = if alpha >= MIN_ALPHA {
                [r, g, b].map(|c| srgb_to_linear((c / alpha).clamp(0.0, 1.0)))
            } else {
                [f32::NAN; 3]
            };
            (rgb, alpha)
        })
        .unzip();

    ColourImage {
        width: size.x,
        height: size.y,
        rgb,
        alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
    use brush_render::kernels::camera_model::CameraModel;
    use burn::tensor::Device;
    use glam::{Quat, UVec2, Vec3, vec2};

    const SIZE: UVec2 = UVec2::new(64, 64);

    async fn device() -> Device {
        Device::from(brush_cube::test_helpers::test_device().await)
    }

    /// Camera at the origin looking along +Z, 60° fov.
    fn camera() -> Camera {
        let fov = 60f64.to_radians();
        Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            fov,
            fov,
            vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// Flat grid of near-opaque, thin Gaussians in the plane `z`, spanning
    /// `x ∈ [x0, x1]`, `y ∈ [-half, half]` at `spacing`. Colour is irrelevant
    /// to the depth render, so every splat is mid-grey.
    fn layer(z: f32, x0: f32, x1: f32, half: f32, spacing: f32) -> Vec<[f32; 3]> {
        let nx = ((x1 - x0) / spacing).round() as i32;
        let ny = (2.0 * half / spacing).round() as i32;
        let mut means = Vec::new();
        for i in 0..=nx {
            for j in 0..=ny {
                means.push([x0 + i as f32 * spacing, -half + j as f32 * spacing, z]);
            }
        }
        means
    }

    fn splats(means: &[[f32; 3]], spacing: f32, device: &Device) -> Splats {
        splats_with_opacity(means, spacing, 0.99, device)
    }

    fn splats_with_opacity(
        means: &[[f32; 3]],
        spacing: f32,
        opacity: f32,
        device: &Device,
    ) -> Splats {
        painted(means, spacing, opacity, [0.5; 3], device)
    }

    /// Like `splats_with_opacity`, every splat of sRGB colour `srgb`.
    fn painted(
        means: &[[f32; 3]],
        spacing: f32,
        opacity: f32,
        srgb: [f32; 3],
        device: &Device,
    ) -> Splats {
        let n = means.len();
        let (s_xy, s_z) = (spacing.ln(), 0.002f32.ln());
        Splats::from_raw(
            means.iter().flatten().copied().collect(),
            [1.0, 0.0, 0.0, 0.0].repeat(n),
            [s_xy, s_xy, s_z].repeat(n),
            srgb.map(|c| (c - 0.5) / SH_C0).repeat(n),
            vec![inverse_sigmoid(opacity); n],
            SplatRenderMode::Default,
            device,
        )
    }

    /// Pixels in the central 50 % of the image (both axes).
    fn central(img: &DepthImage) -> impl Iterator<Item = usize> + '_ {
        central_of(img.width, img.height)
    }

    fn central_of(w: u32, h: u32) -> impl Iterator<Item = usize> {
        (h / 4..3 * h / 4).flat_map(move |y| (w / 4..3 * w / 4).map(move |x| (y * w + x) as usize))
    }

    fn assert_depth(img: &DepthImage, expected: f32) {
        for i in central(img) {
            let (d, a) = (img.depth[i], img.alpha[i]);
            assert!(a > 0.9, "pixel {i}: alpha {a}");
            assert!(
                (d - expected).abs() < 0.02,
                "pixel {i}: depth {d}, want {expected}"
            );
        }
    }

    #[tokio::test]
    async fn flat_layer_renders_its_distance() {
        let device = device().await;
        let s = splats(&layer(2.0, -1.6, 1.6, 1.6, 0.05), 0.05, &device);
        let img = render_expected_depth(&s, &camera(), SIZE).await;
        assert_eq!((img.width, img.height), (SIZE.x, SIZE.y));
        assert_eq!(img.depth.len(), (SIZE.x * SIZE.y) as usize);
        assert_depth(&img, 2.0);
    }

    /// Depth above 1 must survive the float render path un-clamped.
    #[tokio::test]
    async fn far_layer_is_not_clamped() {
        let device = device().await;
        let s = splats(&layer(4.0, -3.0, 3.0, 3.0, 0.08), 0.08, &device);
        let img = render_expected_depth(&s, &camera(), SIZE).await;
        assert_depth(&img, 4.0);
    }

    #[tokio::test]
    async fn empty_region_is_nan_with_low_alpha() {
        let device = device().await;
        // Layer covers only the left part of the view (+x is image right).
        let s = splats(&layer(2.0, -1.6, -0.6, 1.6, 0.05), 0.05, &device);
        let img = render_expected_depth(&s, &camera(), SIZE).await;
        let w = img.width;
        let mut checked = 0;
        for y in 0..img.height {
            for x in 3 * w / 4..w {
                let i = (y * w + x) as usize;
                assert!(
                    img.depth[i].is_nan(),
                    "pixel ({x},{y}): depth {}",
                    img.depth[i]
                );
                assert!(
                    img.alpha[i] < 0.5,
                    "pixel ({x},{y}): alpha {}",
                    img.alpha[i]
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
        // The covered side still has depth.
        let i = (img.height / 2 * w + w / 8) as usize;
        assert!(
            (img.depth[i] - 2.0).abs() < 0.02,
            "covered pixel: depth {}",
            img.depth[i]
        );
    }

    #[tokio::test]
    async fn front_surface_dominates() {
        let device = device().await;
        let mut means = layer(2.0, -1.6, 1.6, 1.6, 0.05);
        means.extend(layer(4.0, -3.2, 3.2, 3.2, 0.05));
        let s = splats(&means, 0.05, &device);
        let img = render_expected_depth(&s, &camera(), SIZE).await;
        assert_depth(&img, 2.0);
    }

    /// Depth is normalised by accumulated opacity, so a translucent layer
    /// still reads its true distance.
    #[tokio::test]
    async fn translucent_layer_is_normalised() {
        let device = device().await;
        let s = splats_with_opacity(&layer(2.0, -1.6, 1.6, 1.6, 0.1), 0.05, 0.4, &device);
        let img = render_expected_depth(&s, &camera(), SIZE).await;
        let mut translucent = 0;
        for i in central(&img) {
            let (d, a) = (img.depth[i], img.alpha[i]);
            if a < 0.9 {
                translucent += 1;
            }
            if a >= 0.5 {
                assert!((d - 2.0).abs() < 0.02, "pixel {i}: depth {d}, alpha {a}");
            } else {
                assert!(d.is_nan(), "pixel {i}: depth {d}, alpha {a}");
            }
        }
        assert!(translucent > 0, "layer should be translucent somewhere");
    }

    /// Depth is measured along the camera's own forward axis from its position.
    #[tokio::test]
    async fn posed_camera_measures_along_its_forward_axis() {
        let device = device().await;
        // Layer in the plane x = 3, facing a camera at x = 1 looking along +X.
        let means: Vec<[f32; 3]> = layer(0.0, -1.6, 1.6, 1.6, 0.05)
            .into_iter()
            .map(|[a, b, _]| [3.0, 0.5 + b, a])
            .collect();
        let n = means.len();
        let s_xy = 0.05f32.ln();
        let s = Splats::from_raw(
            means.iter().flatten().copied().collect(),
            [1.0, 0.0, 0.0, 0.0].repeat(n),
            [0.002f32.ln(), s_xy, s_xy].repeat(n),
            vec![0.0; n * 3],
            vec![inverse_sigmoid(0.99); n],
            SplatRenderMode::Default,
            &device,
        );
        let mut cam = camera();
        cam.position = Vec3::new(1.0, 0.5, 0.0);
        cam.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        assert!((cam.rotation * Vec3::Z - Vec3::X).length() < 1e-6);
        let img = render_expected_depth(&s, &cam, SIZE).await;
        assert_depth(&img, 2.0);
    }

    /// sRGB (0.8, 0.5, 0.2) renders as its linear value, at a translucent
    /// layer too.
    #[tokio::test]
    async fn colour_is_linear_and_normalised_by_opacity() {
        let device = device().await;
        let srgb = [0.8, 0.5, 0.2];
        let linear = [0.603_827, 0.214_041, 0.033_105];
        for opacity in [0.99, 0.4] {
            let means = layer(2.0, -1.6, 1.6, 1.6, 0.1);
            let s = painted(&means, 0.05, opacity, srgb, &device);
            let img = render_colour(&s, &camera(), SIZE).await;
            assert_eq!(img.rgb.len(), (SIZE.x * SIZE.y) as usize);
            let mut checked = 0;
            for i in central_of(img.width, img.height) {
                let (rgb, a) = (img.rgb[i], img.alpha[i]);
                if a < 0.5 {
                    assert!(rgb.iter().all(|c| c.is_nan()), "pixel {i}: {rgb:?}");
                    continue;
                }
                for (c, want) in rgb.iter().zip(linear) {
                    assert!((c - want).abs() < 0.01, "pixel {i}: {rgb:?}, alpha {a}");
                }
                checked += 1;
            }
            assert!(checked > 0, "opacity {opacity}");
        }
    }

    /// A red splat plane, rendered and fused with its colour, meshes to
    /// vertices within 5/255 of red.
    #[tokio::test]
    async fn red_plane_meshes_red() {
        use crate::geometry::mesh::mesh_brick;
        use crate::geometry::tsdf::Tsdf;

        let device = device().await;
        let s = painted(
            &layer(2.5, -2.0, 2.0, 2.0, 0.05),
            0.05,
            0.99,
            [1.0, 0.0, 0.0],
            &device,
        );
        let cam = camera();
        let depth = render_expected_depth(&s, &cam, SIZE).await;
        let colour = render_colour(&s, &cam, SIZE).await;
        let mut tsdf = Tsdf::new();
        tsdf.integrate(&depth, Some(&colour), &cam);
        let meshes: Vec<_> = tsdf
            .take_changed()
            .into_iter()
            .filter_map(|k| mesh_brick(&tsdf, k))
            .collect();
        let colours: Vec<[u8; 3]> = meshes.iter().flat_map(|m| m.colours.clone()).collect();
        let vertices: usize = meshes.iter().map(|m| m.positions.len()).sum();
        assert!(vertices > 100 && colours.len() == vertices);
        for c in colours {
            assert!(c[0] >= 250 && c[1] <= 5 && c[2] <= 5, "vertex colour {c:?}");
        }
    }
}
