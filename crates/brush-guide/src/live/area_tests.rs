use super::*;
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::ViewPose;
use crate::protocol::KeyframeHeader;
use glam::{Mat4, Vec3};

pub(in crate::live) fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("live-area-{name}-{}", std::process::id()))
}

/// A 64×48 keyframe at `pos` looking at `target` (y up, f = 50 px) with a 5×5 grid of feature points 0.25 m apart
/// around `target`, so its seeds land at `target`.
pub(in crate::live) fn keyframe_toward(id: u64, pos: Vec3, target: Vec3) -> (KeyframeHeader, Vec<u8>) {
    let (w, h) = (64u32, 48u32);
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 4) as u8, (y * 5) as u8, ((x ^ y) * 8) as u8]));
    let mut jpeg = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
        .unwrap();
    let pose = Mat4::look_at_rh(pos, target, Vec3::Y).inverse();
    let (right, up) = (pose.x_axis.truncate(), pose.y_axis.truncate());
    let points: Vec<Vec3> = (-2..=2)
        .flat_map(|i| (-2..=2).map(move |j| (i, j)))
        .map(|(i, j)| target + right * (i as f32 * 0.25) + up * (j as f32 * 0.25))
        .collect();
    let header = KeyframeHeader {
        id,
        timestamp: id as f64,
        pose: pose.to_cols_array(),
        fx: 50.0,
        fy: 50.0,
        cx: 32.0,
        cy: 24.0,
        width: w,
        height: h,
        jpeg_len: jpeg.len() as u32,
        depth_size: None,
        depth_confidence: false,
        num_points: points.len() as u32,
        mono_depth_size: None,
    };
    let mut payload = jpeg;
    for p in &points {
        for v in p.to_array() {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    (header, payload)
}

/// Small tiles and radius, no hysteresis or windows, a refine every 5 steps, and the all-views loader rebuilt on
/// every added area view (the default waits for 10). Left tiles are held and the views within the radius train
/// (`area_freeze` left); [`active_config`] trains the active tiles instead. Area training is engaged
/// from the first keyframe (`area_engage_on_leave` false).
pub(in crate::live) fn area_config(radius: f32) -> GuideConfig {
    GuideConfig {
        area_freeze: crate::config::AreaFreeze::Left,
        refine_every: 5,
        all_loader_rebuild_every: 1,
        max_splats: 20_000,
        area_radius_m: radius,
        area_tile_size_m: 1.0,
        area_hysteresis_m: 0.0,
        area_seen_window_steps: 0,
        area_min_resident_keyframes: 0,
        area_junction_steps: 10,
        area_engage_on_leave: false,
        ..GuideConfig::default()
    }
}

/// [`area_config`] training only the active tiles: the tiles within 2.5 m and up to 2 tiles in view within 8 m, any
/// with rows.
pub(in crate::live) fn active_config(radius: f32) -> GuideConfig {
    GuideConfig {
        area_freeze: crate::config::AreaFreeze::Inactive,
        area_view_m: 8.0,
        area_active_tiles: 2,
        area_active_min_rows: 1,
        ..area_config(radius)
    }
}

pub(in crate::live) async fn add(live: &mut LiveModel, dir: &std::path::Path, id: u64, pos: Vec3, target: Vec3) {
    let (h, p) = keyframe_toward(id, pos, target);
    assert!(live.add_keyframe(decode_keyframe(&h, &p, dir).await.unwrap()).await);
}

pub(in crate::live) async fn device() -> burn::tensor::Device {
    burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

/// Adds keyframes at `(x, 0, 2)` looking at `(x, 0, 0)`, ids from `first_id`.
pub(in crate::live) async fn add_at(live: &mut LiveModel, dir: &std::path::Path, first_id: u64, xs: &[f32]) {
    for (i, x) in xs.iter().enumerate() {
        add(live, dir, first_id + i as u64, Vec3::new(*x, 0.0, 2.0), Vec3::new(*x, 0.0, 0.0)).await;
    }
}

/// The x of the camera of the next batch from both loaders.
async fn served_x(live: &mut LiveModel) -> [f32; 2] {
    let a = live.recent.as_mut().unwrap().next_batch().await.camera.position.x;
    let b = live.all.as_mut().unwrap().next_batch().await.camera.position.x;
    [a, b]
}

/// Both loaders hand out only views within the radius of the newest keyframe.
#[tokio::test]
async fn loaders_serve_only_area_views() {
    let dir = tmp("views");
    let mut live = LiveModel::new(area_config(1.0), device().await);
    live.enable_area(dir.join("tiles"));
    for (i, x) in [0.0f32, 0.5, 5.0].iter().enumerate() {
        add(&mut live, &dir, i as u64, Vec3::new(*x, 0.0, 2.0), Vec3::new(*x, 0.0, 0.0)).await;
    }
    for _ in 0..24 {
        for loader in [live.recent.as_mut().unwrap(), live.all.as_mut().unwrap()] {
            let p = loader.next_batch().await.camera.position;
            assert!((p.x - 5.0).abs() < 1e-3, "a view outside the area was served: {p}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Without `enable_area` (or with radius 0) every view trains, as before.
#[tokio::test]
async fn area_off_trains_every_view() {
    let dir = tmp("views-off");
    let mut live = LiveModel::new(area_config(0.0), device().await);
    live.enable_area(dir.join("tiles"));
    for (i, x) in [0.0f32, 5.0].iter().enumerate() {
        add(&mut live, &dir, i as u64, Vec3::new(*x, 0.0, 2.0), Vec3::new(*x, 0.0, 0.0)).await;
    }
    assert!(live.area.is_none());
    assert_eq!(live.training_views(), vec![0, 1]);
    std::fs::remove_dir_all(dir).ok();
}

/// With radius 0 `enable_area` and `reset` leave the disk alone: no tiles directory is created, an existing one keeps
/// its files.
#[tokio::test]
async fn area_off_leaves_the_tiles_dir_untouched() {
    let dir = tmp("off-disk");
    std::fs::remove_dir_all(&dir).ok();
    let mut live = LiveModel::new(area_config(0.0), device().await);
    live.enable_area(dir.join("tiles"));
    live.reset();
    assert!(!dir.join("tiles").exists(), "area off created a tiles directory");
    let kept = dir.join("kept");
    std::fs::create_dir_all(&kept).unwrap();
    std::fs::write(kept.join("0_0.bin"), b"other").unwrap();
    live.enable_area(kept.clone());
    live.reset();
    assert!(live.area.is_none());
    assert_eq!(std::fs::read_dir(&kept).unwrap().count(), 1, "area off emptied a directory");
    std::fs::remove_dir_all(dir).ok();
}

/// A model without `enable_area` keeps every view even with a radius in its config, and reports no area count.
#[tokio::test]
async fn area_needs_enable_area() {
    let dir = tmp("not-enabled");
    let mut live = LiveModel::new(area_config(1.0), device().await);
    add_at(&mut live, &dir, 0, &[0.0, 5.0]).await;
    assert!(live.area.is_none());
    assert_eq!(live.training_views(), vec![0, 1]);
    assert_eq!(live.area_view_count(), None);
    std::fs::remove_dir_all(dir).ok();
}

/// `enable_area` empties a reused tiles directory, and `reset` keeps area training on with the same directory.
#[tokio::test]
async fn enable_area_wipes_the_dir_and_reset_keeps_it() {
    let dir = tmp("wipe");
    let tiles = dir.join("tiles");
    std::fs::create_dir_all(&tiles).unwrap();
    std::fs::write(tiles.join("0_0.tile"), b"stale").unwrap();
    let mut live = LiveModel::new(area_config(1.0), device().await);
    live.enable_area(tiles.clone());
    assert!(tiles.is_dir());
    assert_eq!(std::fs::read_dir(&tiles).unwrap().count(), 0, "stale tile file kept");
    add_at(&mut live, &dir, 0, &[0.0, 5.0]).await;
    std::fs::write(tiles.join("0_0.tile"), b"stale").unwrap();
    live.reset();
    assert!(live.area.is_some());
    assert_eq!(live.tiles_dir.as_deref(), Some(tiles.as_path()));
    assert_eq!(std::fs::read_dir(&tiles).unwrap().count(), 0, "reset kept a tile file");
    add_at(&mut live, &dir, 10, &[0.0]).await;
    assert_eq!(live.training_views(), vec![0]);
    std::fs::remove_dir_all(dir).ok();
}

/// Membership follows the views' current cameras and ignores height: a pose update moves a view into or out of the
/// area and both loaders follow.
#[tokio::test]
async fn pose_updates_move_views_into_and_out_of_the_area() {
    let dir = tmp("poses");
    let mut live = LiveModel::new(area_config(0.8), device().await);
    live.enable_area(dir.join("tiles"));
    add_at(&mut live, &dir, 0, &[0.0, 0.5, 1.0]).await;
    assert_eq!(live.training_views(), vec![1, 2]);
    assert_eq!(live.area_view_count(), Some(2));
    let pose = |x: f32, y: f32| Mat4::look_at_rh(Vec3::new(x, y, 2.0), Vec3::new(x, y, 0.0), Vec3::Y).inverse();
    // 30 m higher but 0.4 m away horizontally: inside.
    live.update_poses(&[ViewPose::from_arkit(0, &pose(0.6, 30.0).to_cols_array()).unwrap()]);
    assert_eq!(live.training_views(), vec![0, 1, 2]);
    live.update_poses(&[ViewPose::from_arkit(1, &pose(5.0, 0.0).to_cols_array()).unwrap()]);
    assert_eq!(live.training_views(), vec![0, 2]);
    for _ in 0..24 {
        for x in served_x(&mut live).await {
            assert!((x - 5.0).abs() > 1e-3, "view 1 served after it left the area");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Views leaving the area rebuild both loaders, though too few views were added for the usual rebuild: the recent
/// loader when any of its views left, the all-views loader when `all_loader_rebuild_every` of its views left.
#[tokio::test]
async fn views_leaving_the_area_rebuild_the_loaders() {
    let dir = tmp("leaving");
    // The all-views loader is built at the first and the third view; the fourth adds one view and removes three.
    let config = GuideConfig { all_loader_rebuild_every: 2, ..area_config(1.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add_at(&mut live, &dir, 0, &[0.0, 0.2, 0.4]).await;
    add_at(&mut live, &dir, 3, &[10.0]).await;
    assert_eq!(live.training_views(), vec![3]);
    for _ in 0..24 {
        for x in served_x(&mut live).await {
            assert!((x - 10.0).abs() < 1e-3, "a view the walker left was served: x = {x}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Views added since the recent loader was built train directly only while they are in the area.
#[tokio::test]
async fn pending_views_are_area_views() {
    let dir = tmp("pending");
    let mut live = LiveModel::new(area_config(1.0), device().await);
    live.enable_area(dir.join("tiles"));
    // Views 0–2 stay in the area of views 3 and 4, which lie 1.6 m apart: view 4 leaves view 3 behind while no
    // recent-loader view left the area.
    add_at(&mut live, &dir, 0, &[10.0, 10.0, 10.0, 10.8]).await;
    assert_eq!(live.pending_views(), vec![3]);
    add_at(&mut live, &dir, 4, &[9.2]).await;
    assert_eq!(live.training_views(), vec![0, 1, 2, 4]);
    assert_eq!(live.pending_views(), vec![4]);
    std::fs::remove_dir_all(dir).ok();
}

/// After the walker left, the all-views loader caches only the views of the new area.
#[tokio::test]
async fn the_loader_cache_holds_only_area_views() {
    let dir = tmp("cache");
    let view_bytes = 64 * 48 * 4;
    let mut live = LiveModel::new(area_config(1.0), device().await);
    live.enable_area(dir.join("tiles"));
    add_at(&mut live, &dir, 0, &[0.0, 0.2, 0.4]).await;
    for _ in 0..30 {
        served_x(&mut live).await;
    }
    assert_eq!(live.all.as_ref().unwrap().cached_bytes(), 3 * view_bytes);
    add_at(&mut live, &dir, 3, &[10.0, 10.2]).await;
    for _ in 0..30 {
        served_x(&mut live).await;
    }
    assert_eq!(live.all.as_ref().unwrap().cached_bytes(), 2 * view_bytes);
    std::fs::remove_dir_all(dir).ok();
}

/// Training steps run with the area on, and an eval reports the area's view count.
#[tokio::test]
async fn training_steps_run_in_the_area() {
    let dir = tmp("steps");
    let mut live = LiveModel::new(area_config(1.0), device().await);
    live.enable_area(dir.join("tiles"));
    add_at(&mut live, &dir, 0, &[0.0, 0.3, 5.0, 5.3]).await;
    for _ in 0..12 {
        live.train_step().await;
    }
    assert_eq!(live.iter(), 12);
    assert_eq!(live.area_view_count(), Some(2));
    std::fs::remove_dir_all(dir).ok();
}

/// Views leaving the all-views loader's area below `all_loader_rebuild_every` do not rebuild it: it keeps its
/// members until enough left (or a park, see `area_park`).
#[tokio::test]
async fn departures_below_the_threshold_do_not_rebuild() {
    let dir = tmp("below-threshold");
    let config = GuideConfig { all_loader_rebuild_every: 3, ..area_config(1.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    // Built at view 0 and again at view 3, when three views have joined.
    add_at(&mut live, &dir, 0, &[0.0, 0.1, 0.2, 0.3]).await;
    assert_eq!(live.all_members, vec![0, 1, 2, 3]);
    // Views 0 and 1 leave and view 4 joins: two departures, one addition.
    add_at(&mut live, &dir, 4, &[1.15]).await;
    assert_eq!(live.training_views(), vec![2, 3, 4]);
    assert_eq!(live.all_members, vec![0, 1, 2, 3], "the all-views loader was rebuilt");
    std::fs::remove_dir_all(dir).ok();
}

/// A newest view whose camera position is not finite still trains: the area never comes out empty, so the loaders
/// always get a view.
#[tokio::test]
async fn the_area_always_holds_the_newest_view() {
    let dir = tmp("never-empty");
    let mut live = LiveModel::new(area_config(1.0), device().await);
    live.enable_area(dir.join("tiles"));
    add_at(&mut live, &dir, 0, &[0.0, 0.2]).await;
    live.view_cams[1].0 = Vec3::NAN;
    live.refresh_area_views();
    assert_eq!(live.training_views(), vec![1]);
    std::fs::remove_dir_all(dir).ok();
}

/// A tile file left from an earlier capture in a reused session directory is wiped, and so is
/// everything on reset.
#[tokio::test]
async fn enable_area_wipes_stale_tiles_and_reset_wipes_again() {
    let dir = tmp("wipe-stale");
    let tiles = dir.join("tiles");
    std::fs::create_dir_all(&tiles).unwrap();
    std::fs::write(tiles.join("0_0.bin"), b"stale").unwrap();
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(tiles.clone());
    assert_eq!(std::fs::read_dir(&tiles).unwrap().count(), 0);
    std::fs::write(tiles.join("1_1.bin"), b"from this session").unwrap();
    live.reset();
    assert!(live.area.is_some(), "reset keeps area training on");
    assert_eq!(std::fs::read_dir(&tiles).unwrap().count(), 0);
    std::fs::remove_dir_all(dir).ok();
}

/// A junction draws on every other call until its steps run out, cycling through its views, and then forgets its
/// boxes and views.
#[test]
fn a_junction_draws_every_other_step_and_clears_when_done() {
    let mut j = Junction { boxes: vec![(Vec3::ZERO, Vec3::ONE)], views: vec![3, 7], steps_left: 4, flip: false, cache: Vec::new() };
    let picks: Vec<Option<usize>> = (0..6).map(|_| j.pick()).collect();
    let mut drawn: Vec<usize> = picks.iter().flatten().copied().collect();
    drawn.sort_unstable();
    assert_eq!(drawn, vec![3, 7], "{picks:?}");
    assert!(picks[0].is_some() && picks[1].is_none() && picks[4..].iter().all(Option::is_none), "{picks:?}");
    assert_eq!(j.steps_left, 0);
    assert!(j.boxes.is_empty() && j.views.is_empty());
}
