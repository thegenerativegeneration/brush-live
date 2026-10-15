//! Tests of `holdout.rs`.

use super::*;
use brush_dataset::load_image::LoadImage;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_vfs::BrushVfs;
use std::path::PathBuf;
use std::sync::Arc;

/// A view over a 2x2 JPEG written to `dir`.
fn view(dir: &Path, id: u64) -> SceneView {
    let rel = PathBuf::from(format!("{id}.jpg"));
    image::RgbImage::new(2, 2).save(dir.join(&rel)).unwrap();
    let vfs = Arc::new(BrushVfs::from_directory_files(dir, vec![rel.clone()]));
    SceneView {
        image: LoadImage::new(vfs, rel, None, 2, None, false),
        camera: Camera::default(),
    }
}

#[test]
fn every_nth_new_keyframe_is_held_out_with_non_sequential_ids() {
    let dir = std::env::temp_dir().join(format!("holdout-unit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = Holdout::new(3, 30.0);
    let mut held = Vec::new();
    for id in [10u64, 11, 40, 41, 77, 90, 91, 92, 99] {
        if h.next_is_held() {
            h.add_held(id, Camera::default(), view(&dir, id), &dir);
            held.push(id);
        } else {
            h.note_trained();
        }
    }
    assert_eq!(held, vec![40, 90, 99]);
    assert!(h.contains(90) && !h.contains(91));
    let json = std::fs::read_to_string(dir.join("holdout.json")).unwrap();
    assert_eq!(json, "[40,90,99]");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn next_is_held_changes_no_state() {
    let h = Holdout::new(2, 30.0);
    assert!(!h.next_is_held() && !h.next_is_held());
    let mut h = h;
    h.note_trained();
    assert!(h.next_is_held() && h.next_is_held());
}

#[test]
fn nothing_is_held_out_after_max_received() {
    let config = GuideConfig { holdout_every: 2, holdout_max_received: 5, ..GuideConfig::default() };
    let mut h = Holdout::from_config(&config);
    let mut held = Vec::new();
    for i in 1..=10 {
        held.extend(h.next_is_held().then_some(i));
        h.note_trained();
    }
    assert_eq!(held, vec![2, 4], "received counts 6, 8, 10 come after the limit");
}

#[test]
fn extras_follow_the_config_and_the_seam_waits_for_its_count_or_finish() {
    let config = GuideConfig {
        eval_lpips: true,
        eval_seam_band_m: 0.5,
        eval_seam_from: 214,
        eval_extras_views: vec![9, 99],
        eval_save_views: vec![9],
        ..GuideConfig::default()
    };
    let dir = Path::new("/session");
    let at = |extras, received, finish| EvalExtras::for_eval(&config, extras, received, finish, dir, 7);
    let checkpoint = at(true, 120, false);
    assert!(checkpoint.lpips && checkpoint.lpips_for(9) && !checkpoint.lpips_for(19));
    assert_eq!(checkpoint.seam_band_m, 0.0, "seam split only from 214 received keyframes on");
    assert_eq!(checkpoint.save_dir, Some(dir.join("eval_renders")));
    assert_eq!(at(true, 214, false).seam_band_m, 0.5);
    assert_eq!(at(true, 50, true).seam_band_m, 0.5, "finish always splits");
    let periodic = at(false, 300, false);
    assert!(!periodic.lpips && periodic.seam_band_m == 0.0 && periodic.save_dir.is_none());
    let all = EvalExtras::for_eval(&GuideConfig { eval_lpips: true, ..GuideConfig::default() }, true, 1, false, dir, 0);
    assert!(all.lpips_for(19), "no list: every view");
}

#[test]
fn zero_holds_nothing_out() {
    let mut h = Holdout::new(0, 30.0);
    for _ in 0..20 {
        assert!(!h.next_is_held());
        h.note_trained();
    }
}

#[test]
fn one_is_treated_as_off() {
    let h = Holdout::new(1, 30.0);
    assert!(!h.next_is_held());
}

#[test]
fn corrected_poses_move_held_out_views() {
    let dir = std::env::temp_dir().join(format!("holdout-poses-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let camera = Camera::new(
        glam::Vec3::ZERO,
        glam::Quat::IDENTITY,
        1.0,
        1.0,
        glam::vec2(0.5, 0.5),
        brush_render::kernels::camera_model::CameraModel::Pinhole,
    );
    let mut h = Holdout::new(2, 30.0);
    h.add_held(4, camera, view(&dir, 4), &dir);
    let p = crate::live::ViewPose {
        id: 4,
        position: glam::Vec3::new(0.3, 0.0, 0.0),
        rotation: glam::Quat::IDENTITY,
    };
    assert_eq!(
        h.update_poses(&[p, crate::live::ViewPose { id: 5, ..p }]),
        1
    );
    assert_eq!(h.views[0].1.camera.position, glam::Vec3::new(0.3, 0.0, 0.0));
    assert_eq!(h.update_poses(&[p]), 0);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn eval_scores_views_that_load_and_skips_the_missing_one() {
    let dir = std::env::temp_dir().join(format!("holdout-eval-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let camera = Camera::new(
        glam::Vec3::ZERO,
        glam::Quat::IDENTITY,
        1.0,
        1.0,
        glam::vec2(0.5, 0.5),
        brush_render::kernels::camera_model::CameraModel::Pinhole,
    );
    let mut h = Holdout::new(2, 30.0);
    h.add_held(1, camera.clone(), view(&dir, 1), &dir);
    h.add_held(2, camera, view(&dir, 2), &dir);
    std::fs::remove_file(dir.join("2.jpg")).unwrap();

    let device = brush_cube::test_helpers::test_device().await.into();
    let n = 20;
    let splats = Splats::from_raw(
        (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, 2.0]).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![-3.0; n * 3],
        vec![0.5; n * 3],
        vec![inverse_sigmoid(0.5); n],
        SplatRenderMode::Default,
        &device,
    );
    let result = h.eval(&splats, 0.0, &EvalExtras::none(), None).await.expect("one view scores");
    assert_eq!(result.views, 1);
    assert_eq!((result.lpips_ms, result.seam_ms), (0.0, 0.0));
    assert!(result.psnr.is_finite(), "psnr {}", result.psnr);
    assert_eq!(result.per_view.len(), 1);
    assert_eq!(result.per_view[0].id, 1);
    assert!((result.per_view[0].psnr - result.psnr).abs() < 1e-6);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn extras_add_lpips_and_save_renders() {
    let dir = std::env::temp_dir().join(format!("holdout-extras-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rel = PathBuf::from("5.jpg");
    image::RgbImage::from_pixel(64, 64, image::Rgb([90, 120, 30])).save(dir.join(&rel)).unwrap();
    let vfs = Arc::new(BrushVfs::from_directory_files(&dir, vec![rel.clone()]));
    let view = SceneView { image: LoadImage::new(vfs, rel, None, 64, None, false), camera: Camera::default() };
    let camera = Camera::new(glam::Vec3::ZERO, glam::Quat::IDENTITY, 1.0, 1.0, glam::vec2(0.5, 0.5),
        brush_render::kernels::camera_model::CameraModel::Pinhole);
    let mut h = Holdout::new(2, 30.0);
    h.add_held(5, camera, view, &dir);
    let device = brush_cube::test_helpers::test_device().await.into();
    // An opaque splat 1 m wide, 2 m ahead on the tile border x = 0: its surface spans the 0.5 m band and the
    // interior beyond it.
    let splats = Splats::from_raw(vec![0.0, 0.0, 2.0], vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 3], vec![0.5; 3],
        vec![inverse_sigmoid(0.99)], SplatRenderMode::Default, &device);
    let extras = EvalExtras { lpips: true, lpips_ids: Vec::new(), seam_band_m: 0.5, tile_size_m: 10.0,
        save_dir: Some(dir.join("eval_renders")), save_ids: vec![5], iter: 7 };
    let r = h.eval(&splats, 0.0, &extras, None).await.expect("scores");
    assert!(r.lpips_ms > 0.0 && r.seam_ms > 0.0, "{} {}", r.lpips_ms, r.seam_ms);
    assert!(r.per_view[0].lpips.is_some_and(f32::is_finite));
    assert!(r.per_view[0].psnr_band.is_some_and(f32::is_finite), "{:?}", r.per_view[0]);
    assert!(r.per_view[0].psnr_interior.is_some_and(f32::is_finite), "{:?}", r.per_view[0]);
    assert!(dir.join("eval_renders/5_7.png").exists());
    assert!(dir.join("eval_renders/gt_5.png").exists());
    std::fs::remove_dir_all(dir).ok();
}

const GT_RGB: [u8; 3] = [90, 120, 30];

/// A holdout of one 64x64 view of `GT_RGB` (id 5) in `dir`, seen by a pinhole camera at the origin looking down +z.
fn solid_holdout(dir: &Path) -> Holdout {
    std::fs::create_dir_all(dir).unwrap();
    let rel = PathBuf::from("5.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb(GT_RGB)).save(dir.join(&rel)).unwrap();
    let vfs = Arc::new(BrushVfs::from_directory_files(dir, vec![rel.clone()]));
    let view = SceneView { image: LoadImage::new(vfs, rel, None, 64, None, false), camera: Camera::default() };
    let camera = Camera::new(glam::Vec3::ZERO, glam::Quat::IDENTITY, 1.0, 1.0, glam::vec2(0.5, 0.5),
        brush_render::kernels::camera_model::CameraModel::Pinhole);
    let mut h = Holdout::new(2, 30.0);
    h.add_held(5, camera, view, dir);
    h
}

/// A globe of `GT_RGB` everywhere.
fn gt_globe(device: &burn::tensor::Device) -> ShBackground {
    ShBackground::constant(&device.clone().autodiff(), glam::Vec3::from_array(GT_RGB.map(|v| f32::from(v) / 255.0)))
}

/// One opaque splat 1 m wide, 2 m ahead on the tile border x = 0, or (`behind`) out of view behind the camera.
fn one_splat(device: &burn::tensor::Device, behind: bool) -> Splats {
    let z = if behind { -2.0 } else { 2.0 };
    Splats::from_raw(vec![0.0, 0.0, z], vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 3], vec![0.5; 3],
        vec![inverse_sigmoid(0.99)], SplatRenderMode::Default, device)
}

#[tokio::test]
async fn saved_renders_show_the_sh_background() {
    let dir = std::env::temp_dir().join(format!("holdout-globe-save-{}", std::process::id()));
    let mut h = solid_holdout(&dir);
    let device = brush_cube::test_helpers::test_device().await.into();
    let splats = one_splat(&device, true);
    let globe = gt_globe(&device);
    let save = |iter| EvalExtras { save_dir: Some(dir.join("eval_renders")), save_ids: vec![5], iter,
        ..EvalExtras::none() };
    let black = h.eval(&splats, 0.0, &save(1), None).await.expect("scores");
    let over = h.eval(&splats, 0.0, &save(2), Some(&globe)).await.expect("scores");
    assert!(black.psnr < 15.0 && over.psnr > 40.0, "black {} dB, over the globe {} dB", black.psnr, over.psnr);
    let pixel = |iter: u32| image::open(dir.join(format!("eval_renders/5_{iter}.png"))).unwrap().to_rgb8()[(32, 32)].0;
    assert_eq!(pixel(1), [0, 0, 0]);
    let saved = pixel(2);
    assert!(saved.iter().zip(GT_RGB).all(|(a, b)| a.abs_diff(b) <= 1), "saved {saved:?} vs {GT_RGB:?}");
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn seam_split_keeps_its_pixels_over_the_sh_background() {
    let dir = std::env::temp_dir().join(format!("holdout-globe-seam-{}", std::process::id()));
    let mut h = solid_holdout(&dir);
    let device = brush_cube::test_helpers::test_device().await.into();
    // A splat of the ground truth's colour 2 m ahead whose surface spans the 0.5 m band at x = 0 and the interior
    // beyond, leaving the left of the view open.
    let dc: Vec<f32> = GT_RGB.iter().map(|v| (f32::from(*v) / 255.0 - 0.5) / 0.282_094_8).collect();
    let splats = Splats::from_raw(vec![0.4, 0.0, 2.0], vec![1.0, 0.0, 0.0, 0.0], vec![-1.2; 3], dc,
        vec![inverse_sigmoid(0.99)], SplatRenderMode::Default, &device);
    let globe = gt_globe(&device);
    let seam = EvalExtras { seam_band_m: 0.5, ..EvalExtras::none() };
    let black = h.eval(&splats, 0.0, &seam, None).await.expect("scores").per_view.remove(0);
    let over = h.eval(&splats, 0.0, &seam, Some(&globe)).await.expect("scores").per_view.remove(0);
    let (Some(band_black), Some(interior_black), Some(band), Some(interior)) =
        (black.psnr_band, black.psnr_interior, over.psnr_band, over.psnr_interior)
    else {
        panic!("both groups labelled either way: {black:?} {over:?}");
    };
    assert!(black.psnr < 20.0 && over.psnr > 40.0, "open view: {} -> {} dB", black.psnr, over.psnr);
    // The same pixels, now with the globe filling what the splat's alpha leaves open: exact.
    assert!(band_black < 40.0 && band > 40.0, "band {band_black} -> {band}");
    assert!(interior_black < 40.0 && interior > 40.0, "interior {interior_black} -> {interior}");
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn warm_up_composites_over_a_globe_at_the_keyframe_size() {
    let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
    let config = GuideConfig { holdout_every: 2, sh_background: true, eval_lpips: false, eval_seam_band_m: 0.0,
        ..GuideConfig::default() };
    let camera = Camera::new(glam::Vec3::ZERO, glam::Quat::IDENTITY, 1.0, 0.8, glam::vec2(0.5, 0.5),
        brush_render::kernels::camera_model::CameraModel::Pinhole);
    warm_up_eval(&config, &one_splat(&device, false), &camera, glam::uvec2(500, 375)).await;
}
