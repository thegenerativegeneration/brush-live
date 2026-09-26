use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::LiveModel;
use crate::protocol::{Cell, KeyframeHeader, ServerHeader, encode_cells, encode_frame};
use crate::schedule::ScoreScheduler;
use crate::scores::metrics::{gaussian_metrics, uncertainty_cap};
use crate::scores::pass::{PassView, score_pass};
use crate::scores::voxel::{GaussianScore, VoxelAggregator};
use brush_async::Actor;
use burn::tensor::{Device, Tensor};
use glam::{UVec2, Vec3};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};
use web_time::Instant;

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

enum Command {
    Keyframe(KeyframeHeader, Vec<u8>, oneshot::Sender<Result<(), String>>),
    Export(oneshot::Sender<Result<Vec<u8>, String>>),
    Reset(oneshot::Sender<()>),
}

/// Handle to one capture session: a worker that trains on incoming keyframes
/// and periodically publishes voxel scores. Clones share the same worker.
#[derive(Clone)]
pub struct GuideSession {
    tx: mpsc::Sender<Command>,
    scores: watch::Receiver<Option<Arc<ScoreSetMsg>>>,
    status: watch::Receiver<StatusMsg>,
    /// The worker runs pinned to this actor's thread: GPU streams are keyed on
    /// the OS thread, so the worker must not migrate between tokio workers.
    _actor: Actor,
}

impl GuideSession {
    /// `device` must be the autodiff device.
    pub fn start(config: GuideConfig, device: Device, session_dir: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let (scores_tx, scores) = watch::channel(None);
        let (status_tx, status) = watch::channel(StatusMsg::default());
        let actor = Actor::new("brush-guide-session");
        actor
            .run(move || worker(config, device, session_dir, rx, scores_tx, status_tx))
            .detach();
        Self {
            tx,
            scores,
            status,
            _actor: actor,
        }
    }

    /// Resolves once the keyframe is decoded and added, or recognised as a duplicate.
    pub async fn push_keyframe(
        &self,
        header: KeyframeHeader,
        payload: Vec<u8>,
    ) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Keyframe(header, payload, tx))
            .await
            .map_err(stopped)?;
        rx.await.map_err(stopped)?
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

    /// The current splats as PLY bytes.
    pub async fn export_splat(&self) -> Result<Vec<u8>, String> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Command::Export(tx)).await.map_err(stopped)?;
        rx.await.map_err(stopped)?
    }

    pub async fn reset(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Command::Reset(tx)).await.is_ok() {
            let _ = rx.await;
        }
    }
}

fn stopped(e: impl std::fmt::Display) -> String {
    format!("session stopped: {e}")
}

async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("splat readback")
        .try_to_vec::<f32>()
        .expect("f32 splat data")
}

fn publish_counts(status_tx: &watch::Sender<StatusMsg>, live: &LiveModel) {
    status_tx.send_modify(|s| {
        s.num_keyframes = live.views().len() as u32;
        s.num_splats = live.splats().map_or(0, |s| s.num_splats());
    });
}

async fn worker(
    config: GuideConfig,
    device: Device,
    session_dir: PathBuf,
    mut rx: mpsc::Receiver<Command>,
    scores_tx: watch::Sender<Option<Arc<ScoreSetMsg>>>,
    status_tx: watch::Sender<StatusMsg>,
) {
    let clock = Instant::now();
    let mut live = LiveModel::new(config.clone(), device.clone());
    let mut voxels = VoxelAggregator::new(
        config.voxel_size,
        config.min_cell_opacity,
        uncertainty_cap(config.fisher_ridge()),
    );
    let mut scheduler = ScoreScheduler::new(config.score_budget, config.min_score_interval_s);
    let mut version = 0u64;
    let mut last_score_ms = 0u32;
    let mut rate_window = (clock.elapsed().as_secs_f64(), live.iter());
    // Sent JPEG size per view, in the same order as `live.views()`.
    let mut sizes: Vec<UVec2> = Vec::new();

    loop {
        // With nothing to train, block for the next command; otherwise just drain.
        let cmd = if live.views().is_empty() {
            match rx.recv().await {
                Some(c) => Some(c),
                None => return,
            }
        } else {
            match rx.try_recv() {
                Ok(c) => Some(c),
                Err(mpsc::error::TryRecvError::Empty) => None,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        };

        match cmd {
            Some(Command::Keyframe(h, payload, reply)) => {
                let size = UVec2::new(h.width, h.height);
                let result = match decode_keyframe(&h, &payload, &session_dir).await {
                    Ok(kf) => {
                        if live.add_keyframe(kf).await {
                            sizes.push(size);
                        }
                        Ok(())
                    }
                    Err(e) => Err(e.to_string()),
                };
                // Counts are visible to the caller as soon as the push resolves.
                publish_counts(&status_tx, &live);
                let _ = reply.send(result);
            }
            Some(Command::Export(reply)) => {
                let result = match live.splats() {
                    Some(s) => brush_serde::splat_to_ply(s.clone(), None)
                        .await
                        .map_err(|e| e.to_string()),
                    None => Err("no splats yet".to_owned()),
                };
                let _ = reply.send(result);
            }
            Some(Command::Reset(reply)) => {
                live = LiveModel::new(config.clone(), device.clone());
                voxels.reset();
                scheduler = ScoreScheduler::new(config.score_budget, config.min_score_interval_s);
                sizes.clear();
                last_score_ms = 0;
                // `live.iter()` restarts at 0; an old window would underflow.
                rate_window = (clock.elapsed().as_secs_f64(), live.iter());
                scores_tx.send_replace(None);
                status_tx.send_replace(StatusMsg::default());
                let _ = reply.send(());
            }
            None => {}
        }

        if live.views().is_empty() {
            continue;
        }
        live.train_step().await;

        let now = clock.elapsed().as_secs_f64();
        if scheduler.due(now) {
            let splats = live.splats().expect("views imply splats").clone();
            let views: Vec<PassView> = live
                .views()
                .iter()
                .zip(&sizes)
                .map(|(v, s)| PassView {
                    camera: v.camera,
                    img_size: *s,
                })
                .collect();
            let out = score_pass(&splats, &views, &config.pass).await;
            let (coverage, uncertainty) =
                gaussian_metrics(&out, &config.coverage, config.fisher_ridge());
            let means = read_f32(splats.means()).await;
            let opac = read_f32(splats.opacities()).await;
            let gaussians: Vec<GaussianScore> = (0..opac.len())
                .map(|i| GaussianScore {
                    pos: Vec3::new(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]),
                    opacity: opac[i],
                    coverage: coverage[i],
                    uncertainty: uncertainty[i],
                })
                .collect();
            let end = clock.elapsed().as_secs_f64();
            let cells = voxels.aggregate(&gaussians, end);
            scheduler.record(end, end - now);
            last_score_ms = ((end - now) * 1000.0) as u32;
            version += 1;
            scores_tx.send_replace(Some(Arc::new(ScoreSetMsg {
                version,
                based_on_keyframe_id: live.last_keyframe_id().unwrap_or(0),
                voxel_size: config.voxel_size,
                cells,
            })));
        }

        let now = clock.elapsed().as_secs_f64();
        if now - rate_window.0 >= 1.0 {
            let rate = (live.iter() - rate_window.1) as f64 / (now - rate_window.0);
            rate_window = (now, live.iter());
            status_tx.send_replace(StatusMsg {
                num_keyframes: live.views().len() as u32,
                num_splats: live.splats().map_or(0, |s| s.num_splats()),
                train_iters_per_s: rate as f32,
                last_score_ms,
            });
        } else {
            publish_counts(&status_tx, &live);
        }
        brush_async::yield_now().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the channel-closed check directly, without spinning up the
    /// GPU-backed worker: a real "panicked worker" is covered by the
    /// server's Hello-reuse behaviour instead.
    #[tokio::test]
    async fn is_alive_reflects_worker_channel() {
        let (tx, rx) = mpsc::channel(1);
        let (_scores_tx, scores) = watch::channel(None);
        let (_status_tx, status) = watch::channel(StatusMsg::default());
        let session = GuideSession {
            tx,
            scores,
            status,
            _actor: Actor::new("test"),
        };
        assert!(session.is_alive());
        drop(rx);
        assert!(!session.is_alive());
    }
}
