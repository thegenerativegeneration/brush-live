//! One `restore_tiles` call that brings back many tiles appends them one after another
//! onto the same splats; the model must hold each tile's rows exactly as its file does.

use super::*;
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::{drift_rows, train};
use brush_train::rows::SplatRows;

/// Tiles restored in one call.
const TILES: i32 = 48;

/// Per row of `rows` (sorted): its mean and raw opacity as bits.
fn means_and_opacities(rows: &SplatRows) -> Vec<[u32; 4]> {
    let mut v: Vec<[u32; 4]> = (0..rows.len())
        .map(|i| {
            let m = rows.mean(i);
            [m.x.to_bits(), m.y.to_bits(), m.z.to_bits(), rows.raw_opacities[i].to_bits()]
        })
        .collect();
    v.sort_unstable();
    v
}

/// 48 tiles of three rows each, 3 m and more from the walker, park in one event and come back in one restore call:
/// every tile's rows in the model match what its file held (as a restore moves them), and the model trains on.
#[tokio::test]
async fn restoring_48_tiles_in_one_call_keeps_every_row() {
    let dir = tmp("restore-bulk");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    let keys: Vec<TileKey> = (0..TILES).map(|i| TileKey { x: 3 + i % 8, z: 3 + i / 8 }).collect();
    let means: Vec<f32> = keys
        .iter()
        .flat_map(|k| (0..3).flat_map(move |j| [k.x as f32 + 0.2 + 0.3 * j as f32, 0.1 * j as f32, k.z as f32 + 0.5]))
        .collect();
    drift_rows(&mut live, means).await;
    let all: Vec<f32> = live.splats().unwrap().means().into_data_async().await.unwrap().try_to_vec().unwrap();
    let iter = live.iter;
    live.area.as_mut().unwrap().registry.note_resident(&all, &[], iter);
    let splats = live.splats.take().unwrap();
    let splats = live.park_tiles(splats, &keys).await;
    live.splats = Some(splats);
    assert_eq!(live.area_report().unwrap().counters.parks, TILES as u32);
    let want: Vec<(TileKey, SplatRows)> = keys.iter().map(|&k| (k, live.load_parked(k).unwrap())).collect();

    let splats = live.splats.take().unwrap();
    let splats = live.restore_tiles(splats, &keys, RestoreReason::Distance, Vec3::new(0.0, 0.0, 2.0)).await;
    live.splats = Some(splats);
    assert_eq!(live.area_report().unwrap().counters.restores, TILES as u32);

    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    for (key, file) in &want {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| TileKey::of(rows.mean(i), 1.0) == *key).collect();
        assert_eq!(means_and_opacities(&rows.select(&idx)), means_and_opacities(file), "tile {key:?}");
    }
    train(&mut live, 5).await;
    let means: Vec<f32> = live.splats().unwrap().means().into_data_async().await.unwrap().try_to_vec().unwrap();
    assert!(means.iter().all(|v| v.is_finite()), "the model broke after the restore");
    std::fs::remove_dir_all(dir).ok();
}
