//! Score sets written for offline analysis, as JSON lines.

use brush_guide::protocol::Cell;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Appends one JSON line per score set to a file.
pub(crate) struct ScoreDump {
    file: std::fs::File,
    start: Instant,
}

impl ScoreDump {
    /// `started` is the replay's connect clock, shared with [`super::receive::Receiver`]
    /// so dumped and logged timestamps read from the same origin.
    pub(crate) fn open(path: &Path, started: Instant) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file,
            start: started,
        })
    }

    /// Each cell as `[x, y, z, coverage, uncertainty, age, nx, ny, nz,
    /// density, uninformed]`, normal 0 when the cell has none, uninformed 0
    /// or 1.
    pub(crate) fn write(&mut self, version: u64, voxel_size: f32, cells: &[Cell]) {
        let cells_json: Vec<[f64; 11]> = cells
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
                    f64::from(u8::from(c.uninformed)),
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
