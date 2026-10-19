//! Task 20c: area training engages once the walker has left a tile (or a tile parked); until then the engine trains as
//! with area training off. The scenes face the wall of [`crate::live::area_tests::keyframe_toward`] from 5 m.

use super::*;
use crate::config::{AreaFreeze, GuideConfig};
use crate::live::area::tests::{active_config, add, device, tmp};
use crate::live::area_restore::tests::train;
use glam::Vec3;

/// inactive_at_cap with the defaults' engagement, 1 m tiles, a per-tile cap of 5 % of `max_splats` and `hysteresis`.
fn engage_config(radius: f32, hysteresis: f32) -> GuideConfig {
    GuideConfig {
        area_freeze: AreaFreeze::InactiveAtCap,
        area_engage_on_leave: true,
        area_tile_max_fraction: 0.05,
        area_hysteresis_m: hysteresis,
        ..active_config(radius)
    }
}

/// Keyframe looking straight at the wall from 5 m at x.
async fn facing_wall(live: &mut LiveModel, dir: &std::path::Path, id: u64, x: f32) {
    add(live, dir, id, Vec3::new(x, 0.0, 5.0), Vec3::new(x, 0.0, 0.0)).await;
}

/// Sets `max_splats` so the resident rows stand at 0.9 of it: at the cap.
fn at_the_cap(live: &mut LiveModel) {
    let n = live.splats().unwrap().num_splats() as f32;
    live.config.max_splats = (n / 0.9).floor() as u32;
}

/// Whether the last refine got a row budget (its tiles are recorded; the refine consumes the budget).
fn budgeted(live: &LiveModel) -> bool {
    live.area.as_ref().is_some_and(|a| !a.budget_keys.is_empty())
}

/// Whether seeds past the per-tile cap into the origin's wall tile are turned away.
fn caps_tiles(live: &mut LiveModel) -> bool {
    let seeds = [0.2, 0.0, 0.0].repeat(live.config.area_tile_cap() as usize + 1);
    live.tile_cap_keep(&seeds).is_some()
}

fn engaged_at(live: &LiveModel) -> u32 {
    live.area_report().unwrap().counters.engaged_at_keyframe
}

/// What the engine does that area training would change: splat counts after each keyframe and its steps (seeds,
/// growth, eviction), the training views and both loaders' members and builds, held rows, the row budget and the
/// floor's and score round's cameras.
#[derive(Debug, PartialEq)]
struct Trace {
    splats: Vec<u32>,
    views: Vec<usize>,
    members: (Vec<usize>, Vec<usize>, Vec<usize>, u32),
    held: bool,
    budget: bool,
    sight: usize,
}

/// Four keyframes within 1 m of the first at the cap, with steps and refines between: no visited tile lies beyond
/// radius 3 + hysteresis 2. Area training traces as area training off does (twin with radius 0), seeds nothing past
/// the per-tile cap, parks nothing, and still ranks the active set.
#[tokio::test]
async fn until_a_tile_is_left_the_engine_trains_as_with_area_off() {
    let mut traces = Vec::new();
    for radius in [3.0, 0.0] {
        let dir = tmp(&format!("engage-twin-{radius}"));
        let mut live = LiveModel::new(engage_config(radius, 2.0), device().await);
        live.enable_area(dir.join("tiles"));
        let mut splats = Vec::new();
        for (id, x) in [0.0, 0.5, -0.5, 1.0].into_iter().enumerate() {
            facing_wall(&mut live, &dir, id as u64, x).await;
            if id == 0 {
                at_the_cap(&mut live);
            }
            splats.push(live.splats().unwrap().num_splats());
            train(&mut live, 6).await;
            splats.push(live.splats().unwrap().num_splats());
        }
        let trainer = live.trainer.as_ref().unwrap();
        traces.push(Trace {
            splats,
            views: live.training_views(),
            members: (live.all_members.clone(), live.recent_members.clone(), live.pending_views(), live.all_loader_builds),
            held: trainer.frozen_rows().is_some(),
            budget: budgeted(&live),
            sight: live.sight_cameras().len(),
        });
        if radius > 0.0 {
            let r = live.area_report().unwrap();
            let c = r.counters;
            assert_eq!((c.engaged_at_keyframe, c.parks, c.seeds_over_tile_cap, c.parks_for_room), (0, 0, 0, 0), "{c:?}");
            assert!(!r.at_cap && r.frozen_rows == 0 && !r.active_tiles.is_empty(), "{r:?}");
            assert!(live.area.as_ref().unwrap().registry.left_keys(Vec3::new(1.0, 0.0, 5.0)).is_empty());
        }
        std::fs::remove_dir_all(dir).ok();
    }
    assert_eq!(traces[0], traces[1], "area training before any tile is left differs from area off");
    assert_eq!(traces[0].views, (0..4).collect::<Vec<_>>());
}

/// At the cap, keyframe 1 (5.5 m on) leaves the origin's tiles: area training engages within the keyframe (the
/// loaders are built once, with its view in), the rows outside the active tiles are held, the active views train,
/// the next refine gets a row budget and seeds meet the per-tile cap. Keyframe 3 comes back
/// (2.5 m: nothing left within radius 2 + hysteresis 3), yet area training stays on. A reset turns it off again.
#[tokio::test]
async fn a_keyframe_that_leaves_a_tile_engages_area_training_for_good() {
    let dir = tmp("engage-leave");
    let mut live = LiveModel::new(engage_config(2.0, 3.0), device().await);
    live.enable_area(dir.join("tiles"));
    facing_wall(&mut live, &dir, 0, 0.0).await;
    train(&mut live, 6).await;
    at_the_cap(&mut live);
    train(&mut live, 1).await;
    assert_eq!(engaged_at(&live), 0);
    assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none() && !live.area_report().unwrap().at_cap);
    assert!(!budgeted(&live) && !caps_tiles(&mut live));

    let builds = live.all_loader_builds;
    facing_wall(&mut live, &dir, 1, 5.5).await;
    assert_eq!(engaged_at(&live), 2);
    assert!(!live.area.as_ref().unwrap().registry.left_keys(Vec3::new(5.5, 0.0, 5.0)).is_empty());
    assert!(live.area_report().unwrap().at_cap);
    assert!(live.trainer.as_ref().unwrap().frozen_rows().is_some(), "nothing held at the cap once engaged");
    let views = live.training_views();
    assert_eq!(views, vec![1], "keyframe 0 does not see keyframe 1's wall");
    assert_eq!(live.all_members, views);
    assert!(live.recent_members.iter().chain(&live.pending_views()).all(|v| views.contains(v)));
    assert_eq!(live.all_loader_builds - builds, 1);
    train(&mut live, 6).await;
    assert!(budgeted(&live), "no row budget once engaged");
    assert!(caps_tiles(&mut live), "no per-tile cap once engaged");
    facing_wall(&mut live, &dir, 2, 5.6).await;

    facing_wall(&mut live, &dir, 3, 2.5).await;
    assert!(live.area.as_ref().unwrap().registry.left_keys(Vec3::new(2.5, 0.0, 5.0)).is_empty());
    train(&mut live, 6).await;
    assert_eq!(engaged_at(&live), 2, "area training switched back");
    assert!(budgeted(&live) && caps_tiles(&mut live));

    live.reset();
    assert_eq!(engaged_at(&live), 0);
    facing_wall(&mut live, &dir, 10, 0.0).await;
    train(&mut live, 6).await;
    assert_eq!(engaged_at(&live), 0);
    assert!(!budgeted(&live));
    std::fs::remove_dir_all(dir).ok();
}

/// A pose update that moves the only keyframe 6 m away leaves the tiles it visited: area training engages and the
/// views are recomputed.
#[tokio::test]
async fn a_pose_update_that_leaves_a_tile_engages_area_training() {
    let dir = tmp("engage-poses");
    let config = GuideConfig { move_anchored_splats: false, ..engage_config(2.0, 0.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    facing_wall(&mut live, &dir, 0, 0.0).await;
    assert_eq!(engaged_at(&live), 0);
    let (h, _) = crate::live::area::tests::keyframe_toward(0, Vec3::new(6.0, 0.0, 5.0), Vec3::new(6.0, 0.0, 0.0));
    live.update_poses(&[crate::live::ViewPose::from_arkit(0, &h.pose).unwrap()]);
    assert_eq!(engaged_at(&live), 1);
    train(&mut live, 6).await;
    assert!(budgeted(&live));
    std::fs::remove_dir_all(dir).ok();
}
