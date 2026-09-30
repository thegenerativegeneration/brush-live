//! Handling of server frames: visualised, optionally dumped, summarised
//! on stderr.

use brush_guide::protocol::{
    CELL_BYTES, MeshBrick, ServerHeader, decode_cells, decode_frame, decode_mesh_bricks,
};
use futures_util::{Stream, StreamExt};
use std::path::{Path, PathBuf};
use tokio_tungstenite::tungstenite::Message;

use super::dump::{ScoreDump, dump_mesh_round};
use super::viz::{Mode, log_mesh_bricks, log_score_set, log_status};

pub(crate) struct Receiver {
    pub(crate) rec: rerun::RecordingStream,
    pub(crate) mode: Mode,
    pub(crate) score_dump: Option<ScoreDump>,
    pub(crate) mesh_dump: Option<PathBuf>,
}

impl Receiver {
    /// Handles binary frames until the connection ends.
    pub(crate) async fn run<E>(
        mut self,
        mut source: impl Stream<Item = Result<Message, E>> + Unpin,
    ) {
        while let Some(Ok(msg)) = source.next().await {
            if let Message::Binary(bytes) = msg {
                self.handle(&bytes);
            }
        }
    }

    fn handle(&mut self, bytes: &[u8]) {
        let Ok((header, payload)) = decode_frame::<ServerHeader>(bytes) else {
            return;
        };
        match header {
            ServerHeader::ScoreSet {
                version,
                voxel_size,
                cell_bytes,
                ..
            } => self.score_set(version, voxel_size, cell_bytes, payload),
            ServerHeader::Status {
                num_splats,
                train_iters_per_s,
                last_score_ms,
                ..
            } => log_status(&self.rec, num_splats, train_iters_per_s, last_score_ms),
            ServerHeader::MeshBricks {
                version,
                num_bricks,
                mesh_ms,
            } => match decode_mesh_bricks(payload, num_bricks) {
                Ok(bricks) => {
                    let dump = self.mesh_dump.as_deref();
                    match handle_mesh_bricks(
                        &self.rec,
                        dump,
                        version,
                        mesh_ms,
                        bytes.len(),
                        &bricks,
                    ) {
                        Ok(line) => eprintln!("{line}"),
                        Err(e) => eprintln!("mesh dump failed: {e}"),
                    }
                }
                Err(e) => eprintln!("bad mesh_bricks v{version}: {e}"),
            },
            ServerHeader::Error { message } => eprintln!("server error: {message}"),
            _ => {}
        }
    }

    fn score_set(&mut self, version: u64, voxel_size: f32, cell_bytes: u32, payload: &[u8]) {
        if cell_bytes != CELL_BYTES as u32 {
            eprintln!(
                "server and replay cell formats differ: server cell_bytes={cell_bytes}, replay expects {CELL_BYTES}"
            );
            return;
        }
        let cells = decode_cells(payload).unwrap_or_default();
        if let Some(dump) = self.score_dump.as_mut() {
            dump.write(version, voxel_size, &cells);
        }
        log_score_set(&self.rec, version, voxel_size, &cells, self.mode);
    }
}

/// Logs a round of mesh bricks to rerun and optionally dumps it; returns
/// the per-round summary line.
pub(crate) fn handle_mesh_bricks(
    rec: &rerun::RecordingStream,
    dump: Option<&Path>,
    version: u64,
    mesh_ms: u32,
    frame_bytes: usize,
    bricks: &[MeshBrick],
) -> std::io::Result<String> {
    let removed = bricks
        .iter()
        .filter(|b| matches!(b, MeshBrick::Removed(_)))
        .count();
    let triangles: usize = bricks
        .iter()
        .map(|b| match b {
            MeshBrick::Mesh(m) => m.indices.len() / 3,
            MeshBrick::Removed(_) => 0,
        })
        .sum();
    log_mesh_bricks(rec, mesh_ms, frame_bytes, bricks);
    if let Some(dir) = dump {
        dump_mesh_round(dir, version, mesh_ms, frame_bytes, bricks)?;
    }
    Ok(format!(
        "mesh bricks v{version}: {} bricks ({removed} removed, {triangles} triangles), {frame_bytes} bytes, TSDF+mesh {mesh_ms} ms",
        bricks.len()
    ))
}
