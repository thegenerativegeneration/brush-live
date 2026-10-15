//! Tests of `eval.rs`.

use super::*;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;

const SIZE: u32 = 48;
const SKY: [u8; 3] = [90, 160, 230];

fn camera() -> Camera {
    let fov = 60f64.to_radians();
    Camera::new(Vec3::ZERO, glam::Quat::IDENTITY, fov, fov, glam::vec2(0.5, 0.5), CameraModel::Pinhole)
}

/// One opaque splat behind the camera: the render is empty.
fn splats_out_of_view(device: &Device) -> Splats {
    Splats::from_raw(
        vec![0.0, 0.0, -5.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![-3.0; 3],
        vec![0.5; 3],
        vec![inverse_sigmoid(0.99)],
        SplatRenderMode::Default,
        device,
    )
}

/// A grey opaque splat ahead of the camera covering part of the view.
fn splat_ahead(device: &Device) -> Splats {
    Splats::from_raw(
        vec![0.0, 0.0, 2.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![-1.5; 3],
        vec![0.3; 3],
        vec![inverse_sigmoid(0.99)],
        SplatRenderMode::Default,
        device,
    )
}

fn sky_image() -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(SIZE, SIZE, image::Rgb(SKY)))
}

fn sky_globe(device: &Device) -> ShBackground {
    let c = Vec3::from_array(SKY.map(|v| f32::from(v) / 255.0));
    ShBackground::constant(&device.clone().autodiff(), c)
}

async fn scalar(t: Tensor<1>) -> f32 {
    t.into_scalar_async::<f32>().await.expect("scalar")
}

async fn floats(t: Tensor<3>) -> Vec<f32> {
    t.into_data_async().await.expect("readback").try_into_vec::<f32>().expect("f32")
}

#[tokio::test]
async fn an_empty_render_over_a_sky_globe_scores_the_sky() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let globe = sky_globe(&device);
    let mode = AlphaMode::Transparent;
    let over = eval_stats_over(splats_out_of_view(&device), &camera(), sky_image(), mode, Some(&globe), &device)
        .await
        .unwrap();
    let psnr = scalar(over.psnr).await;
    assert!(psnr > 40.0, "globe composited: psnr {psnr}");
    let black = eval_stats(splats_out_of_view(&device), &camera(), sky_image(), mode, &device).await.unwrap();
    let psnr_black = scalar(black.psnr).await;
    assert!(psnr_black < 10.0, "black background: psnr {psnr_black}");
}

#[tokio::test]
async fn the_composite_keeps_the_splat_and_fills_only_what_it_leaves_open() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let globe = sky_globe(&device);
    let mode = AlphaMode::Transparent;
    let over = eval_stats_over(splat_ahead(&device), &camera(), sky_image(), mode, Some(&globe), &device)
        .await
        .unwrap();
    let black = eval_stats(splat_ahead(&device), &camera(), sky_image(), mode, &device).await.unwrap();
    let (over, black) = (floats(over.rendered).await, floats(black.rendered).await);
    let px = |img: &[f32], x: u32, y: u32| {
        let i = ((y * SIZE + x) * 3) as usize;
        [img[i], img[i + 1], img[i + 2]]
    };
    let sky = SKY.map(|v| f32::from(v) / 255.0);
    let corner = px(&over, 0, 0);
    for c in 0..3 {
        assert!((corner[c] - sky[c]).abs() < 1.5 / 255.0, "corner {corner:?} vs sky {sky:?}");
    }
    assert_eq!(px(&black, 0, 0), [0.0; 3], "black corner");
    // Where the splat covers the view: its own colour plus the sky its alpha leaves open.
    let res = glam::uvec2(SIZE, SIZE);
    let (img, _) = render_splats(splat_ahead(&device), &camera(), res, Vec3::ZERO, None, TextureMode::Float).await;
    let alpha = floats(img.slice(s![.., .., 3..4])).await;
    let i = (SIZE / 2 * SIZE + SIZE / 2) as usize;
    assert!(alpha[i] > 0.9, "the splat covers the centre: alpha {}", alpha[i]);
    let (centre, centre_black) = (px(&over, SIZE / 2, SIZE / 2), px(&black, SIZE / 2, SIZE / 2));
    for c in 0..3 {
        let expected = centre_black[c] + (1.0 - alpha[i]) * sky[c];
        assert!((centre[c] - expected).abs() < 1.5 / 255.0, "centre {centre:?}, expected channel {c} {expected}");
    }
}

#[tokio::test]
async fn without_a_globe_the_render_is_the_black_background_render() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let mode = AlphaMode::Transparent;
    let sample = eval_stats_over(splat_ahead(&device), &camera(), sky_image(), mode, None, &device).await.unwrap();
    let res = glam::uvec2(SIZE, SIZE);
    let (img, _) = render_splats(splat_ahead(&device), &camera(), res, Vec3::ZERO, None, TextureMode::Float).await;
    let reference = (img.slice(s![.., .., 0..3]) * 255.0).round() / 255.0;
    assert_eq!(floats(sample.rendered).await, floats(reference).await);
}

/// The globe as the eval composites it, at the training size (500 x 375, which the block grid doesn't divide) and
/// an off-centre principal point: the corner pixels match the globe evaluated per pixel along `pixel_dirs`, within
/// the block upsample's tolerance and the 8-bit roundtrip.
#[tokio::test]
async fn the_eval_background_matches_the_globe_per_pixel_at_the_corners() {
    use crate::sh_background::{pixel_dirs, sh_basis};
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let globe = crate::sh_background::tests::varied_globe(&device.clone().autodiff());
    let rotation = glam::Quat::from_axis_angle(Vec3::new(0.2, 1.0, 0.1).normalize(), 0.8);
    let cam = Camera::new(Vec3::new(0.3, -0.2, 0.1), rotation, 1.3, 1.0, glam::vec2(0.42, 0.57), CameraModel::Pinhole);
    let size = glam::uvec2(500, 375);
    let behind = cam.position - rotation * Vec3::Z * 5.0;
    let splats = Splats::from_raw(vec![behind.x, behind.y, behind.z], vec![1.0, 0.0, 0.0, 0.0], vec![-3.0; 3],
        vec![0.5; 3], vec![inverse_sigmoid(0.99)], SplatRenderMode::Default, &device);
    let gt = DynamicImage::ImageRgb8(image::RgbImage::new(size.x, size.y));
    let sample = eval_stats_over(splats, &cam, gt, AlphaMode::Transparent, Some(&globe), &device).await.unwrap();
    let rendered = floats(sample.rendered).await;
    let dirs = pixel_dirs(&cam, size);
    for (x, y) in [(0, 0), (size.x - 1, size.y - 1)] {
        let i = (y * size.x + x) as usize;
        let dir = Tensor::<1>::from_floats(&dirs[i * 3..i * 3 + 3], &globe.coeffs().device()).reshape([1, 3]);
        let exact: Vec<f32> =
            globe.image(sh_basis(dir)).into_data_async().await.unwrap().try_into_vec::<f32>().unwrap();
        for c in 0..3 {
            let got = rendered[i * 3 + c];
            assert!((got - exact[c]).abs() < 4.5 / 255.0, "pixel ({x}, {y}) channel {c}: {got} vs {}", exact[c]);
        }
    }
}
