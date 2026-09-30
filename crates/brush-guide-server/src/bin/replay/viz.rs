//! Rerun logging of what the server sends: score cells gated into colours,
//! status scalars and mesh bricks.

use brush_guide::geometry::mesh::BrickMesh;
use brush_guide::protocol::{Cell, MeshBrick};
use clap::ValueEnum;

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum Mode {
    /// Red/yellow by coverage.
    Coverage,
    /// Red/yellow by Fisher uncertainty.
    Uncertainty,
    /// Red/yellow if either metric flags the cell.
    Both,
    /// Weak cells only, coloured by which metric flags them.
    Agreement,
}

const GREY: [u8; 3] = [150, 150, 150];
const RED: [u8; 3] = [230, 40, 40];
const YELLOW: [u8; 3] = [240, 200, 40];
const BLUE: [u8; 3] = [40, 110, 240];
const PURPLE: [u8; 3] = [180, 60, 220];

pub(crate) const AGREEMENT_LEGEND: &str = "Agreement mode, weak cells only:\n\n\
    * red: weak by coverage and by uncertainty\n\
    * blue: weak by coverage only\n\
    * purple: weak by uncertainty only\n\
    * grey: first seen less than 3 s ago\n\n\
    Uncertainty is ranked within each round, so its share of weak cells stays roughly \
    constant; compare where the colours sit, not how many there are.";

fn weak_coverage(c: &Cell) -> bool {
    c.coverage < 80
}

fn weak_uncertainty(c: &Cell) -> bool {
    c.uncertainty > 200
}

fn cell_color(c: &Cell, mode: Mode) -> Option<[u8; 3]> {
    if c.age < 3 {
        return Some(GREY);
    }
    let (wc, wu) = (weak_coverage(c), weak_uncertainty(c));
    if matches!(mode, Mode::Agreement) {
        return match (wc, wu) {
            (true, true) => Some(RED),
            (true, false) => Some(BLUE),
            (false, true) => Some(PURPLE),
            (false, false) => None,
        };
    }
    let border_cov = c.coverage < 160;
    let border_unc = c.uncertainty > 140;
    let (weak, border) = match mode {
        Mode::Coverage => (wc, border_cov),
        Mode::Uncertainty => (wu, border_unc),
        Mode::Both | Mode::Agreement => (wc || wu, border_cov || border_unc),
    };
    if weak {
        Some(RED)
    } else if border {
        Some(YELLOW)
    } else {
        None
    }
}

/// Per-round counts of settled cells (not pending) flagged weak by each metric.
fn log_agreement_counts(rec: &rerun::RecordingStream, cells: &[Cell]) {
    let settled = cells.iter().filter(|c| c.age >= 3);
    let (mut both, mut cov_only, mut unc_only) = (0u32, 0u32, 0u32);
    for c in settled {
        match (weak_coverage(c), weak_uncertainty(c)) {
            (true, true) => both += 1,
            (true, false) => cov_only += 1,
            (false, true) => unc_only += 1,
            (false, false) => {}
        }
    }
    let _ = rec.log("agreement/both", &rerun::Scalars::new(vec![both as f64]));
    let _ = rec.log(
        "agreement/coverage_only",
        &rerun::Scalars::new(vec![cov_only as f64]),
    );
    let _ = rec.log(
        "agreement/uncertainty_only",
        &rerun::Scalars::new(vec![unc_only as f64]),
    );
}

/// Logs a score set's cells coloured by `mode`, the agreement counts and
/// the cell normals, and prints how many cells have a normal.
pub(crate) fn log_score_set(
    rec: &rerun::RecordingStream,
    version: u64,
    voxel_size: f32,
    cells: &[Cell],
    mode: Mode,
) {
    let (pos, col): (Vec<[f32; 3]>, Vec<[u8; 3]>) = cells
        .iter()
        .filter_map(|c| cell_color(c, mode).map(|col| (c.center, col)))
        .unzip();
    log_agreement_counts(rec, cells);
    let _ = rec.log(
        "scores",
        &rerun::Points3D::new(pos)
            .with_colors(col)
            .with_radii([voxel_size * 0.4]),
    );
    let (origins, vectors): (Vec<[f32; 3]>, Vec<[f32; 3]>) = cells
        .iter()
        .filter_map(|c| {
            c.normal
                .map(|n| (c.center, [n[0] * 0.08, n[1] * 0.08, n[2] * 0.08]))
        })
        .unzip();
    let dense: Vec<&Cell> = cells.iter().filter(|c| c.density >= 16).collect();
    eprintln!(
        "score set v{version}: {} of {} cells have a normal; {} of {} with density >= 16",
        origins.len(),
        cells.len(),
        dense.iter().filter(|c| c.normal.is_some()).count(),
        dense.len()
    );
    let _ = rec.log(
        "cells/normals",
        &rerun::Arrows3D::from_vectors(vectors).with_origins(origins),
    );
}

pub(crate) fn log_status(
    rec: &rerun::RecordingStream,
    num_splats: u32,
    train_iters_per_s: f32,
    last_score_ms: u32,
) {
    let _ = rec.log(
        "status/splats",
        &rerun::Scalars::new(vec![num_splats as f64]),
    );
    let _ = rec.log(
        "status/iters_per_s",
        &rerun::Scalars::new(vec![train_iters_per_s as f64]),
    );
    let _ = rec.log(
        "status/score_ms",
        &rerun::Scalars::new(vec![last_score_ms as f64]),
    );
}

pub(crate) fn brick_name(b: &MeshBrick) -> String {
    let [x, y, z] = b.key().0.to_array();
    format!("brick_{x}_{y}_{z}")
}

/// A brick's mesh for rerun, with its vertex colours if it has them.
pub(crate) fn mesh_3d(m: &BrickMesh) -> rerun::Mesh3D {
    let mesh = rerun::Mesh3D::new(m.positions.iter().copied())
        .with_vertex_normals(m.normals.iter().copied())
        .with_triangle_indices(m.indices.as_chunks::<3>().0.iter().copied());
    if m.colours.is_empty() {
        return mesh;
    }
    mesh.with_vertex_colors(
        m.colours
            .iter()
            .map(|&[r, g, b]| rerun::Color::from_rgb(r, g, b)),
    )
}

/// Logs a round of mesh bricks under `mesh/brick_<x>_<y>_<z>` and its
/// statistics under `mesh/stats`.
pub(crate) fn log_mesh_bricks(
    rec: &rerun::RecordingStream,
    mesh_ms: u32,
    frame_bytes: usize,
    bricks: &[MeshBrick],
) {
    for b in bricks {
        let path = format!("mesh/{}", brick_name(b));
        let _ = match b {
            MeshBrick::Mesh(m) => rec.log(path, &mesh_3d(m)),
            MeshBrick::Removed(_) => rec.log(path, &rerun::Clear::flat()),
        };
    }
    let _ = rec.log(
        "mesh/stats/bricks",
        &rerun::Scalars::new(vec![bricks.len() as f64]),
    );
    let _ = rec.log(
        "mesh/stats/bytes",
        &rerun::Scalars::new(vec![frame_bytes as f64]),
    );
    let _ = rec.log(
        "mesh/stats/mesh_ms",
        &rerun::Scalars::new(vec![mesh_ms as f64]),
    );
}
