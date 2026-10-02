mod geometry;
mod splat_read;
#[cfg(test)]
mod tests;
mod worker;
mod frames;

pub use frames::forward_frames;

use crate::config::GuideConfig;
use crate::protocol::{CELL_BYTES, Cell, KeyframeHeader, ServerHeader, encode_cells, encode_frame};
use brush_async::Actor;
use burn::tensor::Device;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};
use worker::{Channels, worker};

pub use geometry::{MeshBricksMsg, MeshLog};

pub struct ScoreSetMsg {
    pub version: u64,
    pub based_on_keyframe_id: u64,
    pub voxel_size: f32,
    pub cells: Vec<Cell>,
}

impl ScoreSetMsg {
    pub fn to_frame(&self) -> Vec<u8> {
        let header = ServerHeader::ScoreSet {
            version: self.version,
            based_on_keyframe_id: self.based_on_keyframe_id,
            voxel_size: self.voxel_size,
            num_cells: self.cells.len() as u32,
            cell_bytes: CELL_BYTES as u32,
        };
        encode_frame(&header, &encode_cells(&self.cells))
    }
}

#[derive(Clone, Debug, Default)]
pub struct StatusMsg {
    pub num_keyframes: u32,
    pub num_splats: u32,
    pub train_iters_per_s: f32,
    pub last_score_ms: u32,
    /// Training iterations since the session started; not sent on the wire.
    pub train_iters: u64,
}

impl StatusMsg {
    pub fn to_frame(&self) -> Vec<u8> {
        encode_frame(
            &ServerHeader::Status {
                num_keyframes: self.num_keyframes,
                num_splats: self.num_splats,
                train_iters_per_s: self.train_iters_per_s,
                last_score_ms: self.last_score_ms,
            },
            &[],
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The worker is gone (panicked or shut down); the session is unusable.
    #[error("session stopped")]
    Stopped,
    #[error("{0}")]
    Rejected(String),
    #[error("writing splat: {0}")]
    Io(#[from] std::io::Error),
}

enum Command {
    Keyframe(KeyframeHeader, Vec<u8>, oneshot::Sender<Result<(), String>>),
    /// Export the splats; `true` also pauses training until the next new keyframe.
    Export(bool, oneshot::Sender<Result<Vec<u8>, String>>),
    Reset(oneshot::Sender<()>),
    /// `true` stops training, rounds and keyframe decoding until `false`.
    Pause(bool, oneshot::Sender<()>),
}

/// Handle to one capture session: a worker that trains on incoming keyframes
/// and periodically publishes voxel scores. Clones share the same worker.
#[derive(Clone)]
pub struct GuideSession {
    tx: mpsc::Sender<Command>,
    scores: watch::Receiver<Option<Arc<ScoreSetMsg>>>,
    meshes: watch::Receiver<MeshLog>,
    status: watch::Receiver<StatusMsg>,
    /// The worker runs pinned to this actor's thread: GPU streams are keyed on
    /// the OS thread, so the worker must not migrate between tokio workers.
    _actor: Actor,
}

impl GuideSession {
    /// `device` must be the autodiff device.
    pub fn start(config: GuideConfig, device: Device, session_dir: PathBuf) -> Self {
        Self::start_when(config, device, session_dir, None)
    }

    /// Like [`Self::start`], but the worker takes no commands until `ready`
    /// turns true (e.g. the server's GPU warm-up); commands queue meanwhile
    /// and status messages already flow.
    pub fn start_when(
        config: GuideConfig,
        device: Device,
        session_dir: PathBuf,
        ready: Option<watch::Receiver<bool>>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let (scores_tx, scores) = watch::channel(None);
        let (meshes_tx, meshes) = watch::channel(MeshLog::default());
        let (status_tx, status) = watch::channel(StatusMsg::default());
        let actor = Actor::new("brush-guide-session");
        let channels = Channels {
            scores: scores_tx,
            meshes: meshes_tx,
            status: status_tx,
        };
        actor
            .run(move || worker(config, device, session_dir, rx, channels, ready))
            .detach();
        Self {
            tx,
            scores,
            meshes,
            status,
            _actor: actor,
        }
    }

    /// Resolves once the keyframe is decoded and added, or recognised as a duplicate.
    pub async fn push_keyframe(
        &self,
        header: KeyframeHeader,
        payload: Vec<u8>,
    ) -> Result<(), SessionError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Keyframe(header, payload, tx))
            .await
            .map_err(|_closed| SessionError::Stopped)?;
        rx.await
            .map_err(|_closed| SessionError::Stopped)?
            .map_err(SessionError::Rejected)
    }

    /// True while the worker's command channel is open. A panicked worker
    /// closes its receiver, so this goes false without waiting for a reply.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }

    pub fn scores(&self) -> watch::Receiver<Option<Arc<ScoreSetMsg>>> {
        self.scores.clone()
    }

    /// Brick meshes. A round's bricks are recorded before its score set is
    /// published, so after a score set `since` yields bricks of at least
    /// that version; later rounds may already be included.
    pub fn meshes(&self) -> watch::Receiver<MeshLog> {
        self.meshes.clone()
    }

    pub fn status(&self) -> watch::Receiver<StatusMsg> {
        self.status.clone()
    }

    /// The current splats as PLY bytes.
    pub async fn export_splat(&self) -> Result<Vec<u8>, SessionError> {
        self.export(false).await
    }

    /// Writes the current splats as PLY to `path` and returns its size. The
    /// worker stops training until a new keyframe arrives or the session resets.
    pub async fn finish(&self, path: &Path) -> Result<u64, SessionError> {
        let ply = self.export(true).await?;
        if let Some(dir) = path.parent() {
            tokio::fs::create_dir_all(dir).await?;
        }
        tokio::fs::write(path, &ply).await?;
        Ok(ply.len() as u64)
    }

    async fn export(&self, finish: bool) -> Result<Vec<u8>, SessionError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Export(finish, tx))
            .await
            .map_err(|_closed| SessionError::Stopped)?;
        rx.await
            .map_err(|_closed| SessionError::Stopped)?
            .map_err(SessionError::Rejected)
    }

    pub async fn reset(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Command::Reset(tx)).await.is_ok() {
            let _ = rx.await;
        }
    }

    /// Pauses or resumes the worker; returns once it has applied the change,
    /// i.e. after the training step or round in progress. While paused no GPU
    /// work is submitted and `push_keyframe` waits for the resume.
    pub async fn set_paused(&self, paused: bool) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Command::Pause(paused, tx)).await.is_ok() {
            let _ = rx.await;
        }
    }
}
