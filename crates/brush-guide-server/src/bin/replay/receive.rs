//! Handling of server frames: visualised, optionally dumped, summarised
//! on stderr.

use brush_guide::protocol::{CELL_BYTES, Cell, ServerHeader, decode_cells, decode_frame};
use futures_util::{Stream, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use super::dump::ScoreDump;
use super::viz::{Mode, log_score_set, log_status};

/// How long the replay keeps reading frames after `splat`.
const LINGER: std::time::Duration = std::time::Duration::from_secs(3);

pub(crate) struct Receiver {
    pub(crate) rec: rerun::RecordingStream,
    pub(crate) mode: Mode,
    pub(crate) score_dump: Option<ScoreDump>,
    /// Return `LINGER` after the server answers `finish` with `splat`.
    pub(crate) stop_on_splat: bool,
    /// Dump a score set only this many seconds after the last dumped one.
    pub(crate) dump_every_s: f32,
    /// When the replay connected; received frames are stamped relative to it.
    pub(crate) started: std::time::Instant,
    pub(crate) last_dump: Option<std::time::Instant>,
    /// The newest score set `dump_every_s` skipped, written at `splat` so the
    /// dump ends with the set the server published before it.
    pub(crate) skipped: Option<(u64, f32, Vec<Cell>)>,
}

impl Receiver {
    /// Handles binary frames until the connection ends, or until `LINGER`
    /// after `splat` with `stop_on_splat`, dumping every score set then.
    pub(crate) async fn run<E>(
        mut self,
        mut source: impl Stream<Item = Result<Message, E>> + Unpin,
    ) {
        while let Some(Ok(msg)) = source.next().await {
            if let Message::Binary(bytes) = msg
                && self.handle(&bytes)
                && self.stop_on_splat
            {
                // The server publishes the final score set before it answers
                // `finish`, but the socket may deliver `splat` first.
                if let (Some(dump), Some((version, voxel_size, cells))) =
                    (self.score_dump.as_mut(), self.skipped.take())
                {
                    dump.write(version, voxel_size, &cells);
                }
                self.dump_every_s = 0.0;
                let deadline = tokio::time::Instant::now() + LINGER;
                while let Ok(Some(Ok(msg))) = tokio::time::timeout_at(deadline, source.next()).await
                {
                    if let Message::Binary(bytes) = msg {
                        self.handle(&bytes);
                    }
                }
                return;
            }
        }
    }

    /// Returns true for `splat`.
    fn handle(&mut self, bytes: &[u8]) -> bool {
        let Ok((header, payload)) = decode_frame::<ServerHeader>(bytes) else {
            return false;
        };
        if matches!(
            header,
            ServerHeader::ScoreSet { .. } | ServerHeader::Status { .. }
        ) {
            eprint!("[{:8.2} s] ", self.started.elapsed().as_secs_f64());
        }
        match header {
            ServerHeader::ScoreSet {
                version,
                voxel_size,
                cell_bytes,
                ..
            } => self.score_set(version, voxel_size, cell_bytes, payload),
            ServerHeader::Status {
                num_keyframes,
                num_splats,
                train_iters_per_s,
                last_score_ms,
                ..
            } => {
                eprintln!(
                    "status: {num_keyframes} keyframes, {num_splats} splats, {train_iters_per_s:.1} it/s, score {last_score_ms} ms"
                );
                log_status(&self.rec, num_splats, train_iters_per_s, last_score_ms);
            }
            ServerHeader::Error { message } => eprintln!("server error: {message}"),
            ServerHeader::Splat { ply_len } => {
                eprintln!("splat written on the server ({ply_len} bytes)");
                return true;
            }
            _ => {}
        }
        false
    }

    fn score_set(&mut self, version: u64, voxel_size: f32, cell_bytes: u32, payload: &[u8]) {
        if cell_bytes != CELL_BYTES as u32 {
            eprintln!(
                "server and replay cell formats differ: server cell_bytes={cell_bytes}, replay expects {CELL_BYTES}"
            );
            return;
        }
        let cells = decode_cells(payload).unwrap_or_default();
        eprintln!("score_set v{version}, {} cells", cells.len());
        let due = self
            .last_dump
            .is_none_or(|t| t.elapsed().as_secs_f32() >= self.dump_every_s);
        log_score_set(&self.rec, version, voxel_size, &cells, self.mode);
        if let Some(dump) = self.score_dump.as_mut() {
            if due {
                dump.write(version, voxel_size, &cells);
                self.last_dump = Some(std::time::Instant::now());
                self.skipped = None;
            } else if self.stop_on_splat {
                self.skipped = Some((version, voxel_size, cells));
            }
        }
    }
}
