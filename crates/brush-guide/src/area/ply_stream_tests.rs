use super::*;
use brush_train::rows::SplatRows;

/// `n` rows with `coeffs` SH coefficients per channel, unnormalised rotations and no floor.
pub(crate) fn rows(n: usize, coeffs: u32) -> SplatRows {
    SplatRows {
        means: (0..n * 3).map(|i| i as f32 * 0.37 - 5.0).collect(),
        rotations: (0..n).flat_map(|i| [1.0, 0.1 * i as f32, -0.2, 0.3]).collect(),
        log_scales: (0..n * 3).map(|i| -3.0 + 0.01 * i as f32).collect(),
        sh_coeffs: (0..n * coeffs as usize * 3).map(|i| (i as f32 * 0.13).sin()).collect(),
        raw_opacities: (0..n).map(|i| i as f32 * 0.2 - 1.0).collect(),
        min_scale: Vec::new(),
        origin: vec![None; n],
        age: vec![0.0; n],
        importance: vec![f32::INFINITY; n],
        sh_coeffs_per_channel: coeffs,
        ..SplatRows::default()
    }
}

async fn device() -> burn::tensor::Device {
    brush_cube::test_helpers::test_device().await.into()
}

/// The f32 values after `end_header`.
fn body(ply: &[u8]) -> Vec<f32> {
    let end = b"end_header\n";
    let at = ply.windows(end.len()).position(|w| w == end).unwrap() + end.len();
    ply[at..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// The streamed file is byte-identical to `splat_to_ply` on the same splats, written in one or two batches.
#[tokio::test]
async fn streamed_ply_matches_splat_to_ply_byte_for_byte() {
    let device = device().await;
    for coeffs in [1u32, 4, 9, 16] {
        let r = rows(7, coeffs);
        let reference = brush_serde::splat_to_ply(r.to_splats_raw(&device), None).await.unwrap();
        let degree = brush_render::sh::sh_degree_from_coeffs(coeffs);
        let mut w = PlyStreamWriter::new(Vec::new(), 7, degree, SplatRenderMode::Default).unwrap();
        w.write_rows(&r.select(&[0, 1, 2])).unwrap();
        w.write_rows(&r.select(&[3, 4, 5, 6])).unwrap();
        assert_eq!(w.bytes(), reference.len() as u64);
        assert_eq!(w.finish().unwrap(), reference, "sh coeffs {coeffs}");
        let mut one = PlyStreamWriter::new(Vec::new(), 7, degree, SplatRenderMode::Default).unwrap();
        one.write_rows(&r).unwrap();
        assert_eq!(one.finish().unwrap(), reference, "one batch, sh coeffs {coeffs}");
    }
}

/// The header names the render mode as `splat_to_ply` does.
#[tokio::test]
async fn the_header_names_the_render_mode() {
    let device = device().await;
    let r = rows(3, 1);
    let mut splats = r.to_splats_raw(&device);
    splats.render_mip = true;
    let reference = brush_serde::splat_to_ply(splats, None).await.unwrap();
    let mut w = PlyStreamWriter::new(Vec::new(), 3, 0, SplatRenderMode::Mip).unwrap();
    w.write_rows(&r).unwrap();
    assert_eq!(w.finish().unwrap(), reference);
}

/// Rows with a Mip floor are written with it folded in, as `splat_to_ply` bakes it on the GPU (to rounding).
#[tokio::test]
async fn a_floor_is_baked_in() {
    let device = device().await;
    let mut r = rows(5, 4);
    r.min_scale = (0..5).map(|i| 0.02 + 0.01 * i as f32).collect();
    let reference = brush_serde::splat_to_ply(r.to_splats(&device), None).await.unwrap();
    let mut w = PlyStreamWriter::new(Vec::new(), 5, 1, SplatRenderMode::Default).unwrap();
    w.write_rows(&r).unwrap();
    let ours = w.finish().unwrap();
    assert_eq!(ours.len(), reference.len());
    let (a, b) = (body(&ours), body(&reference));
    assert_ne!(a, body(&{
        let mut raw = PlyStreamWriter::new(Vec::new(), 5, 1, SplatRenderMode::Default).unwrap();
        raw.write_rows(&SplatRows { min_scale: Vec::new(), ..r.clone() }).unwrap();
        raw.finish().unwrap()
    }), "the floor changed nothing");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 1e-4 * y.abs().max(1.0), "value {i}: {x} vs {y}");
    }
}

#[test]
fn a_short_file_is_an_error() {
    let mut w = PlyStreamWriter::new(Vec::new(), 3, 0, SplatRenderMode::Default).unwrap();
    w.write_rows(&rows(2, 1)).unwrap();
    assert!(w.finish().is_err());
}

#[test]
fn rows_beyond_the_header_count_are_an_error() {
    let mut w = PlyStreamWriter::new(Vec::new(), 3, 0, SplatRenderMode::Default).unwrap();
    w.write_rows(&rows(2, 1)).unwrap();
    assert!(w.write_rows(&rows(2, 1)).is_err());
}

#[test]
fn rows_of_another_sh_degree_or_out_of_shape_are_an_error() {
    let mut w = PlyStreamWriter::new(Vec::new(), 4, 1, SplatRenderMode::Default).unwrap();
    assert!(w.write_rows(&rows(2, 1)).is_err(), "degree 0 rows into a degree 1 file");
    let mut bad = rows(2, 4);
    bad.means.pop();
    assert!(w.write_rows(&bad).is_err(), "a short column");
    w.write_rows(&rows(2, 4)).unwrap();
    assert_eq!(w.rows(), 2, "rejected rows are not counted");
}
