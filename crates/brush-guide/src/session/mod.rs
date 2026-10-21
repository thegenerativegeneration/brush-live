mod frames;
mod preview;
pub(crate) mod splat_read;
#[cfg(test)]
mod tests;
mod worker;
pub(crate) use worker::warm_up_eval;

pub use frames::forward_frames;

use crate::config::GuideConfig;
use crate::live::{PoseUpdateStats, StepTotals, ViewPose};
use crate::protocol::{
    CELL_BYTES, Cell, ClientHeader, KeyframeHeader, PoseUpdate, ServerHeader, decode_frame,
    encode_cells, encode_frame,
};
use brush_async::Actor;
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};
use worker::{Channels, worker};

pub use preview::{PREVIEW_FLOATS, PreviewSnapshot};

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

/// Cumulative wall time of the session worker per activity since the session
/// started, in ms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerTimes {
    pub train_ms: u64,
    pub ingest_ms: u64,
    pub preview_ms: u64,
    pub voxel_ms: u64,
    pub fisher_ms: u64,
    /// Worker uptime over the same span: the denominator for the shares.
    pub uptime_ms: u64,
    /// Time the iteration cap held the next training step back.
    pub throttle_ms: u64,
    /// The training step's phases, from the live model.
    pub step: StepTotals,
}

#[derive(Clone, Debug, Default)]
pub struct StatusMsg {
    pub num_keyframes: u32,
    pub num_splats: u32,
    pub train_iters_per_s: f32,
    pub last_score_ms: u32,
    /// Training iterations since the session started.
    pub train_iters: u64,
    pub times: WorkerTimes,
    /// Optimizer elements whose step was clamped since the session started, counted up to the last refine.
    pub adam_clamped: u64,
    /// Splats refine pruned for a diverged colour since the session started.
    pub sh_pruned: u64,
}

impl StatusMsg {
    pub fn to_frame(&self) -> Vec<u8> {
        encode_frame(
            &ServerHeader::Status {
                num_keyframes: self.num_keyframes,
                num_splats: self.num_splats,
                train_iters_per_s: self.train_iters_per_s,
                last_score_ms: self.last_score_ms,
                train_iters: self.train_iters,
                train_ms: self.times.train_ms,
                ingest_ms: self.times.ingest_ms,
                preview_ms: self.times.preview_ms,
                voxel_ms: self.times.voxel_ms,
                fisher_ms: self.times.fisher_ms,
                uptime_ms: self.times.uptime_ms,
                throttle_ms: self.times.throttle_ms,
                adam_clamped: self.adam_clamped,
                sh_pruned: self.sh_pruned,
                batch_ms: self.times.step.batch_ms,
                forward_ms: self.times.step.forward_ms,
                loss_ms: self.times.step.loss_ms,
                backward_ms: self.times.step.backward_ms,
                optimizer_ms: self.times.step.optimizer_ms,
                stats_ms: self.times.step.stats_ms,
                refine_ms: self.times.step.refine_ms,
                area_ms: self.times.step.area_ms,
                hold_ms: self.times.step.hold_ms,
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
}

/// A client frame [`GuideSession::push_frame`] accepted.
#[derive(Debug, PartialEq)]
pub enum Pushed {
    /// The keyframe was added, or was a resend: the host acknowledges it.
    Keyframe(u64),
    /// The poses were applied: nothing is sent back.
    Poses(PoseUpdateStats),
}

enum Command {
    Keyframe(KeyframeHeader, Vec<u8>, oneshot::Sender<Result<(), String>>),
    /// Corrected camera poses, applied after the keyframes queued before them; held like keyframes while paused.
    Poses(
        Vec<ViewPose>,
        oneshot::Sender<Result<PoseUpdateStats, String>>,
    ),
    /// The whole model as PLY bytes (resident and parked splats, [`crate::live::LiveModel::final_ply`]); training
    /// goes on.
    Export(oneshot::Sender<Result<Vec<u8>, String>>),
    /// Write the whole model to the path as PLY ([`crate::live::LiveModel::write_final`]) and reply with its size;
    /// training pauses until the next new keyframe.
    Finish(PathBuf, oneshot::Sender<Result<u64, String>>),
    /// Clear the session; later preview snapshots carry the generation.
    Reset(u64, oneshot::Sender<()>),
    /// `true` stops training, rounds and keyframe decoding until `false`.
    Pause(bool, oneshot::Sender<()>),
    /// Snapshot interval for the preview; `None` turns it off. No reply.
    SetPreview(Option<std::time::Duration>),
}

/// Handle to one capture session: a worker that trains on incoming keyframes
/// and periodically publishes voxel scores. Clones share the same worker.
#[derive(Clone)]
pub struct GuideSession {
    tx: mpsc::Sender<Command>,
    scores: watch::Receiver<Option<Arc<ScoreSetMsg>>>,
    status: watch::Receiver<StatusMsg>,
    preview: watch::Receiver<Option<Arc<PreviewSnapshot>>>,
    /// [`GuideConfig::status_every_step`]: [`forward_frames`] sends status on every change, not once a second.
    status_every_step: bool,
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
        let status_every_step = config.status_every_step;
        let (scores_tx, scores) = watch::channel(None);
        let (status_tx, status) = watch::channel(StatusMsg::default());
        let (preview_tx, preview) = watch::channel(None);
        let actor = Actor::new("brush-guide-session");
        let channels = Channels {
            scores: scores_tx,
            status: status_tx,
            preview: preview_tx,
        };
        actor
            .run(move || worker(config, device, session_dir, rx, channels, ready))
            .detach();
        Self {
            tx,
            scores,
            status,
            preview,
            status_every_step,
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

    /// Moves added views to corrected poses (ARKit camera-to-world per keyframe id, as in the keyframe header) after
    /// every keyframe pushed before; waits while paused, like `push_keyframe`. An invalid pose rejects the whole
    /// update before it reaches the worker.
    pub async fn update_poses(
        &self,
        poses: Vec<PoseUpdate>,
    ) -> Result<PoseUpdateStats, SessionError> {
        let poses = poses
            .iter()
            .map(|p| {
                ViewPose::from_arkit(p.id, &p.pose)
                    .map_err(|e| SessionError::Rejected(format!("keyframe {} pose: {e}", p.id)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Poses(poses, tx))
            .await
            .map_err(|_closed| SessionError::Stopped)?;
        rx.await
            .map_err(|_closed| SessionError::Stopped)?
            .map_err(SessionError::Rejected)
    }

    /// Adds one `keyframe` frame or applies one `poses` frame, in the order pushed. The error is the text of the
    /// `error` frame to send: `keyframe <id>: …`, `poses: …`, or the decode error.
    pub async fn push_frame(&self, frame: &[u8]) -> Result<Pushed, String> {
        match decode_frame::<ClientHeader>(frame) {
            Ok((ClientHeader::Keyframe(h), payload)) => {
                let id = h.id;
                self.push_keyframe(h, payload.to_vec())
                    .await
                    .map(|()| Pushed::Keyframe(id))
                    .map_err(|e| format!("keyframe {id}: {e}"))
            }
            Ok((ClientHeader::Poses { poses }, _)) => self
                .update_poses(poses)
                .await
                .map(Pushed::Poses)
                .map_err(|e| format!("poses: {e}")),
            Ok(_) => Err("expected a keyframe or poses frame".to_owned()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// True while the worker's command channel is open. A panicked worker
    /// closes its receiver, so this goes false without waiting for a reply.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }

    pub fn scores(&self) -> watch::Receiver<Option<Arc<ScoreSetMsg>>> {
        self.scores.clone()
    }

    pub fn status(&self) -> watch::Receiver<StatusMsg> {
        self.status.clone()
    }

    /// Whether status goes out after every training step ([`GuideConfig::status_every_step`]).
    pub fn status_every_step(&self) -> bool {
        self.status_every_step
    }

    /// Newest preview snapshot; `None` until the first one.
    pub fn preview(&self) -> watch::Receiver<Option<Arc<PreviewSnapshot>>> {
        self.preview.clone()
    }

    /// Turns preview snapshots on (`Some(interval)`, zero = every training
    /// step) or off. Returns once queued, without waiting for the worker.
    pub async fn set_preview(&self, interval: Option<std::time::Duration>) {
        let _ = self.tx.send(Command::SetPreview(interval)).await;
    }

    /// The whole model as PLY bytes: the resident splats and every parked tile, as [`Self::finish`] writes them.
    pub async fn export_splat(&self) -> Result<Vec<u8>, SessionError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Export(tx))
            .await
            .map_err(|_closed| SessionError::Stopped)?;
        rx.await
            .map_err(|_closed| SessionError::Stopped)?
            .map_err(SessionError::Rejected)
    }

    /// Writes the whole model (the resident splats and every parked tile, streamed tile by tile) as PLY to `path`,
    /// creating its directory, and returns its size. The worker stops training until a new keyframe arrives or the
    /// session resets.
    pub async fn finish(&self, path: &Path) -> Result<u64, SessionError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Finish(path.to_path_buf(), tx))
            .await
            .map_err(|_closed| SessionError::Stopped)?;
        rx.await
            .map_err(|_closed| SessionError::Stopped)?
            .map_err(SessionError::Rejected)
    }

    /// Clears the session. Preview snapshots published after the reset carry
    /// `generation`, so the host can tell them from snapshots of the old one.
    pub async fn reset(&self, generation: u64) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Command::Reset(generation, tx)).await.is_ok() {
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
