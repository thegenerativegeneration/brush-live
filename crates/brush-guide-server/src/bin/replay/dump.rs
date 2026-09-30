//! Files written for offline analysis: score sets as JSON lines, mesh
//! bricks as ASCII PLY with one JSON line per round.

use brush_guide::geometry::mesh::BrickMesh;
use brush_guide::protocol::{Cell, MeshBrick};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use super::viz::brick_name;

/// Appends one JSON line per score set to a file.
pub(crate) struct ScoreDump {
    file: std::fs::File,
    start: Instant,
}

impl ScoreDump {
    pub(crate) fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file,
            start: Instant::now(),
        })
    }

    /// Each cell as `[x, y, z, coverage, uncertainty, age, nx, ny, nz,
    /// density]`, normal 0 when the cell has none.
    pub(crate) fn write(&mut self, version: u64, voxel_size: f32, cells: &[Cell]) {
        let cells_json: Vec<[f64; 10]> = cells
            .iter()
            .map(|c| {
                let [nx, ny, nz] = c.normal.unwrap_or([0.0, 0.0, 0.0]);
                [
                    c.center[0] as f64,
                    c.center[1] as f64,
                    c.center[2] as f64,
                    c.coverage as f64,
                    c.uncertainty as f64,
                    c.age as f64,
                    nx as f64,
                    ny as f64,
                    nz as f64,
                    c.density as f64,
                ]
            })
            .collect();
        let line = serde_json::json!({
            "version": version,
            "received_s": self.start.elapsed().as_secs_f64(),
            "voxel_size": voxel_size,
            "cells": cells_json,
        });
        let _ = writeln!(self.file, "{line}");
    }
}

fn write_ply(path: &Path, mesh: &BrickMesh) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        out,
        "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face {}\nproperty list uchar uint vertex_indices\nend_header",
        mesh.positions.len(),
        mesh.indices.len() / 3
    )?;
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        writeln!(out, "{} {} {} {} {} {}", p[0], p[1], p[2], n[0], n[1], n[2])?;
    }
    for t in mesh.indices.as_chunks::<3>().0 {
        writeln!(out, "3 {} {} {}", t[0], t[1], t[2])?;
    }
    out.flush()
}

/// Writes each brick's mesh as `v<version>_brick_<x>_<y>_<z>.ply` in `dir`
/// and appends the round to `rounds.jsonl`.
pub(crate) fn dump_mesh_round(
    dir: &Path,
    version: u64,
    mesh_ms: u32,
    frame_bytes: usize,
    bricks: &[MeshBrick],
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut entries = Vec::new();
    for b in bricks {
        let [x, y, z] = b.key().0.to_array();
        match b {
            MeshBrick::Mesh(m) => {
                let file = format!("v{version:05}_{}.ply", brick_name(b));
                write_ply(&dir.join(&file), m)?;
                entries.push(serde_json::json!({"key": [x, y, z], "file": file, "vertices": m.positions.len()}));
            }
            MeshBrick::Removed(_) => {
                entries.push(serde_json::json!({"key": [x, y, z], "removed": true}));
            }
        }
    }
    let line = serde_json::json!({
        "version": version,
        "bytes": frame_bytes,
        "mesh_ms": mesh_ms,
        "bricks": entries,
    });
    let mut rounds = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("rounds.jsonl"))?;
    writeln!(rounds, "{line}")
}
