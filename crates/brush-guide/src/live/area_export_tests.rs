use super::*;
use crate::area::{PlyStreamWriter, TileKey, read_tile, tile_path, write_tile};
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::{correction, parked_origin, train};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_train::rows::SplatRows;
use glam::Vec3;

/// The rows of a PLY as `splat_to_ply` lays them out: one byte chunk per vertex, sorted (order-free comparison).
fn sorted_vertices(ply: &[u8]) -> Vec<Vec<u8>> {
    let end = b"end_header\n";
    let at = ply.windows(end.len()).position(|w| w == end).unwrap() + end.len();
    let header = std::str::from_utf8(&ply[..at]).unwrap();
    let stride = header.matches("property float").count() * 4;
    let mut v: Vec<Vec<u8>> = ply[at..].chunks(stride).map(<[u8]>::to_vec).collect();
    assert!(v.last().is_none_or(|c| c.len() == stride), "a partial vertex");
    v.sort();
    v
}

/// What the export must hold: the resident rows and every parked tile but `skip`, as [`LiveModel::load_parked`]
/// moves them, floors baked, in [`sorted_vertices`] form.
async fn expected_vertices(live: &LiveModel, skip: &[TileKey]) -> Vec<Vec<u8>> {
    let resident = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let mut parts = vec![resident];
    for key in live.area.as_ref().unwrap().registry.parked_keys() {
        if !skip.contains(&key) {
            parts.push(live.load_parked(key).unwrap());
        }
    }
    let total: usize = parts.iter().map(SplatRows::len).sum();
    let degree = brush_render::sh::sh_degree_from_coeffs(parts[0].sh_coeffs_per_channel);
    let mut w = PlyStreamWriter::new(Vec::new(), total as u64, degree, SplatRenderMode::Default).unwrap();
    for p in &parts {
        w.write_rows(&p.baked()).unwrap();
    }
    sorted_vertices(&w.finish().unwrap())
}

fn ply_path(dir: &std::path::Path) -> std::path::PathBuf {
    let ply = dir.join("out/splat.ply");
    std::fs::create_dir_all(ply.parent().unwrap()).unwrap();
    ply
}

/// Stop writes the whole walk once: resident and parked splats (pose-corrected, floors baked) in one PLY whose
/// header counts them.
#[tokio::test]
async fn final_export_holds_resident_and_parked_tiles() {
    let (live, dir) = parked_origin("final").await;
    let r = live.area_report().unwrap();
    let resident = live.splats().unwrap().num_splats();
    let total = u64::from(resident + r.summary.parked_splats);
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    assert!(!rows.min_scale.is_empty(), "the scenario's rows carry a floor, so the bake is exercised");
    let ply = ply_path(&dir);
    let e = live.write_final(&ply).await.unwrap();
    assert_eq!((e.splats, e.skipped_tiles), (total, 0));
    let bytes = std::fs::read(&ply).unwrap();
    assert_eq!(e.ply_bytes, bytes.len() as u64);
    let msg = brush_serde::load_splat_from_ply(std::io::Cursor::new(bytes.clone()), None).await.unwrap();
    assert_eq!(msg.data.num_splats() as u64, total);
    assert_eq!(sorted_vertices(&bytes), expected_vertices(&live, &[]).await);
    let keys: std::collections::BTreeSet<TileKey> = crate::area::group_rows(&rows.means, 1.0)
        .into_keys()
        .chain(live.area.as_ref().unwrap().registry.parked_keys())
        .collect();
    assert_eq!(e.tiles as usize, keys.len());
    assert!(!dir.join("out/splat.ply.tmp").exists());
    std::fs::remove_dir_all(dir).ok();
}

/// A pose correction made after a tile parked moves its rows in the export, as a restore would.
#[tokio::test]
async fn final_export_moves_parked_rows_by_later_pose_corrections() {
    let (mut live, dir) = parked_origin("final-moved").await;
    let key = live.area.as_ref().unwrap().registry.parked_keys()[0];
    let stored = read_tile(&dir.join("tiles"), key).unwrap().rows;
    live.update_poses(&[correction(0, Vec3::new(0.5, 0.0, 2.0), Vec3::new(0.5, 0.0, 0.0))]);
    let moved = live.load_parked(key).unwrap();
    assert!((moved.means[0] - stored.means[0] - 0.5).abs() < 1e-3, "keyframe 0 seeded the tile");
    let ply = ply_path(&dir);
    live.write_final(&ply).await.unwrap();
    assert_eq!(sorted_vertices(&std::fs::read(&ply).unwrap()), expected_vertices(&live, &[]).await);
    std::fs::remove_dir_all(dir).ok();
}

/// One unreadable tile file and one of another SH degree must not break Stop: both are left out and the totals stay
/// consistent.
#[tokio::test]
async fn final_export_skips_unreadable_and_other_degree_tiles() {
    let (live, dir) = parked_origin("final-skip").await;
    let keys = live.area.as_ref().unwrap().registry.parked_keys();
    assert!(keys.len() >= 2, "the scenario parks several origin tiles");
    let tiles = dir.join("tiles");
    std::fs::write(tile_path(&tiles, keys[0]), b"broken").unwrap();
    let mut other = read_tile(&tiles, keys[1]).unwrap();
    let n = other.rows.len();
    let coeffs = other.rows.sh_coeffs_per_channel as usize;
    let dc = |i: usize| other.rows.sh_coeffs[i * coeffs * 3..i * coeffs * 3 + 3].to_vec();
    other.rows.sh_coeffs = (0..n).flat_map(dc).collect();
    other.rows.sh_coeffs_per_channel = 1;
    other.rows.moments = None;
    assert_ne!(coeffs, 1, "the model trains higher SH degrees");
    write_tile(&tiles, &other).unwrap();
    let left_out: u32 = keys[..2].iter().map(|&k| live.area.as_ref().unwrap().registry.get(k).unwrap().splats).sum();
    let ply = ply_path(&dir);
    let e = live.write_final(&ply).await.unwrap();
    assert_eq!(e.skipped_tiles, 2);
    let r = live.area_report().unwrap();
    assert_eq!(e.splats, u64::from(live.splats().unwrap().num_splats() + r.summary.parked_splats - left_out));
    let bytes = std::fs::read(&ply).unwrap();
    let msg = brush_serde::load_splat_from_ply(std::io::Cursor::new(bytes.clone()), None).await.unwrap();
    assert_eq!(msg.data.num_splats() as u64, e.splats);
    assert_eq!(sorted_vertices(&bytes), expected_vertices(&live, &keys[..2]).await);
    std::fs::remove_dir_all(dir).ok();
}

/// Area training off, or on with nothing parked: the file is `splat_to_ply` of the trainer's splats, byte for byte.
#[tokio::test]
async fn without_parked_tiles_the_export_is_todays() {
    for radius in [0.0, 2.0] {
        let dir = tmp(&format!("final-plain-{radius}"));
        let mut live = LiveModel::new(area_config(radius), device().await);
        live.enable_area(dir.join("tiles"));
        add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
        train(&mut live, 5).await;
        assert_eq!(live.area.is_some(), radius > 0.0);
        let ply = ply_path(&dir);
        let e = live.write_final(&ply).await.unwrap();
        let today = brush_serde::splat_to_ply(live.splats().unwrap().clone(), None).await.unwrap();
        assert_eq!(std::fs::read(&ply).unwrap(), today, "radius {radius}");
        let splats = u64::from(live.splats().unwrap().num_splats());
        assert_eq!((e.ply_bytes, e.splats, e.skipped_tiles), (today.len() as u64, splats, 0));
        std::fs::remove_dir_all(dir).ok();
    }
}

/// The export command's bytes are the file Stop writes.
#[tokio::test]
async fn final_ply_bytes_match_the_written_file() {
    let (live, dir) = parked_origin("final-bytes").await;
    let ply = ply_path(&dir);
    let e = live.write_final(&ply).await.unwrap();
    let (bytes, e2) = live.final_ply().await.unwrap();
    assert_eq!(bytes, std::fs::read(&ply).unwrap());
    assert_eq!((e2.splats, e2.tiles, e2.ply_bytes), (e.splats, e.tiles, e.ply_bytes));
    std::fs::remove_dir_all(dir).ok();
}

/// Without splats there is nothing to write, and no file is left behind.
#[tokio::test]
async fn no_splats_no_file() {
    let dir = tmp("final-empty");
    let live = LiveModel::new(area_config(2.0), device().await);
    let ply = ply_path(&dir);
    assert!(live.write_final(&ply).await.is_err());
    assert!(std::fs::read_dir(ply.parent().unwrap()).unwrap().next().is_none());
    std::fs::remove_dir_all(dir).ok();
}

/// A failed export leaves an earlier `splat.ply` as it was, removes its tmp file, and names the path: the tmp path
/// taken by a directory (create fails), then the target a directory (rename fails after the file was written).
#[tokio::test]
async fn a_failed_export_keeps_the_earlier_file_and_removes_the_tmp_file() {
    let (live, dir) = parked_origin("final-fails").await;
    let ply = ply_path(&dir);
    let tmp_path = dir.join("out/splat.ply.tmp");
    std::fs::write(&ply, b"earlier export").unwrap();
    std::fs::create_dir(&tmp_path).unwrap();
    let e = live.write_final(&ply).await.unwrap_err();
    assert!(e.to_string().starts_with(&format!("writing splat {}", tmp_path.display())), "{e}");
    assert_eq!(std::fs::read(&ply).unwrap(), b"earlier export");
    std::fs::remove_dir(&tmp_path).unwrap();
    let target = dir.join("out/taken.ply");
    std::fs::create_dir_all(target.join("inside")).unwrap();
    let e = live.write_final(&target).await.unwrap_err();
    assert!(e.to_string().starts_with(&format!("writing splat {}", target.display())), "{e}");
    assert!(!dir.join("out/taken.ply.tmp").exists(), "the tmp file was left behind");
    assert_eq!(std::fs::read(&ply).unwrap(), b"earlier export");
    std::fs::remove_dir_all(dir).ok();
}
