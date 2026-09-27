use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::LiveModel;
use crate::protocol::{CELL_BYTES, Cell, KeyframeHeader, ServerHeader, encode_cells, encode_frame};
use crate::schedule::{ScoreScheduler, select_score_views};
use crate::scores::metrics::{gaussian_metrics, uncertainty_cap};
use crate::scores::pass::{PassView, score_pass};
use crate::scores::voxel::{GaussianScore, ViewCone, VoxelAggregator};
use brush_async::Actor;
use burn::tensor::{Device, Tensor};
use glam::{Quat, UVec2, Vec3};
use std::path::{Path, PathBuf};
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
}

async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("splat readback")
        .try_to_vec::<f32>()
        .expect("f32 splat data")
}

/// World direction of the Gaussian's shortest scale axis; zero for a
/// degenerate rotation.
fn shortest_axis(r: &[f32], s: &[f32]) -> Vec3 {
    // Brush stores rotations as [w, x, y, z].
    let q = Quat::from_xyzw(r[1], r[2], r[3], r[0]);
    if !q.is_finite() || q.length_squared() == 0.0 {
        return Vec3::ZERO;
    }
    let k = if s[0] <= s[1] && s[0] <= s[2] {
        0
    } else if s[1] <= s[2] {
        1
    } else {
        2
    };
    q.normalize() * Vec3::AXES[k]
}

/// `1 − s_min / s_mid` of the Gaussian's scales: 0 when round (or
/// degenerate), towards 1 for a flat disc.
fn flatness(s: &[f32]) -> f32 {
    let mut v = [s[0], s[1], s[2]];
    v.sort_by(f32::total_cmp);
    if !(v[0].is_finite() && v[1].is_finite()) || v[1] <= 0.0 {
        return 0.0;
    }
    (1.0 - v[0] / v[1]).clamp(0.0, 1.0)
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
    // Set by Finish: no training or scoring until a new keyframe or Reset.
    let mut finished = false;

    loop {
        // With nothing to train, block for the next command; otherwise just drain.
        let cmd = if live.views().is_empty() || finished {
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
                // A resend must not overwrite the stored image of the first send.
                let result = if live.contains(h.id) {
                    Ok(())
                } else {
                    match decode_keyframe(&h, &payload, &session_dir).await {
                        Ok(kf) => {
                            if live.add_keyframe(kf).await {
                                sizes.push(size);
                                if finished {
                                    finished = false;
                                    rate_window = (clock.elapsed().as_secs_f64(), live.iter());
                                }
                            }
                            Ok(())
                        }
                        Err(e) => Err(e.to_string()),
                    }
                };
                // Counts are visible to the caller as soon as the push resolves.
                publish_counts(&status_tx, &live);
                let _ = reply.send(result);
            }
            Some(Command::Export(finish, reply)) => {
                let result = match live.splats() {
                    Some(s) => brush_serde::splat_to_ply(s.clone(), None)
                        .await
                        .map_err(|e| e.to_string()),
                    None => Err("no splats yet".to_owned()),
                };
                if finish && result.is_ok() {
                    finished = true;
                    status_tx.send_modify(|s| s.train_iters_per_s = 0.0);
                }
                let _ = reply.send(result);
            }
            Some(Command::Reset(reply)) => {
                live = LiveModel::new(config.clone(), device.clone());
                voxels.reset();
                scheduler = ScoreScheduler::new(config.score_budget, config.min_score_interval_s);
                sizes.clear();
                finished = false;
                last_score_ms = 0;
                // `live.iter()` restarts at 0; an old window would underflow.
                rate_window = (clock.elapsed().as_secs_f64(), live.iter());
                scores_tx.send_replace(None);
                status_tx.send_replace(StatusMsg::default());
                let _ = reply.send(());
            }
            None => {}
        }

        if live.views().is_empty() || finished {
            continue;
        }
        live.train_step().await;

        let now = clock.elapsed().as_secs_f64();
        if scheduler.due(now) {
            let splats = live.splats().expect("views imply splats").clone();
            let num_views = live.views().len();
            let views: Vec<PassView> = select_score_views(
                num_views,
                config.max_score_views,
                config.seed.wrapping_add(version),
            )
            .into_iter()
            .map(|i| PassView {
                camera: live.views()[i].camera,
                img_size: sizes[i],
            })
            .collect();
            let mut out = score_pass(&splats, &views, &config.pass).await;
            // Observation counts are over the sampled views only; rescale so
            // `CoverageParams::n_target` keeps meaning views of the whole capture.
            out.scale_observations(num_views as f32 / views.len() as f32);
            let (coverage, uncertainty) =
                gaussian_metrics(&out, &config.coverage, config.fisher_ridge());
            let means = read_f32(splats.means()).await;
            let opac = read_f32(splats.opacities()).await;
            let rots = read_f32(splats.rotations()).await;
            let scales = read_f32(splats.scales()).await;
            let gaussians: Vec<GaussianScore> = (0..opac.len())
                .map(|i| GaussianScore {
                    pos: Vec3::new(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]),
                    opacity: opac[i],
                    coverage: coverage[i],
                    uncertainty: uncertainty[i],
                    axis: shortest_axis(&rots[i * 4..i * 4 + 4], &scales[i * 3..i * 3 + 3]),
                    flatness: flatness(&scales[i * 3..i * 3 + 3]),
                })
                .collect();
            let cones: Vec<ViewCone> = live
                .views()
                .iter()
                .map(|v| {
                    let c = &v.camera;
                    ViewCone {
                        position: c.position,
                        forward: c.rotation * Vec3::Z,
                        cos_half_fov: (0.5 * c.fov_x.max(c.fov_y) as f32).cos(),
                    }
                })
                .collect();
            let end = clock.elapsed().as_secs_f64();
            let cells = voxels.aggregate(&gaussians, &cones, end);
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

    #[test]
    fn shortest_axis_follows_rotation() {
        let h = std::f32::consts::FRAC_1_SQRT_2;
        // 90° about x in [w, x, y, z]; the flat local z axis maps to ±y.
        let a = shortest_axis(&[h, h, 0.0, 0.0], &[1.0, 1.0, 0.01]);
        assert!(a.dot(Vec3::Y).abs() > 0.999, "{a}");
    }

    #[test]
    fn flatness_compares_smallest_to_middle_scale() {
        assert!((flatness(&[1.0, 0.25, 0.5]) - 0.5).abs() < 1e-6);
        assert_eq!(flatness(&[2.0, 2.0, 2.0]), 0.0);
        assert!(flatness(&[1.0, 1.0, 0.001]) > 0.99);
        assert_eq!(flatness(&[0.0, 0.0, 1.0]), 0.0);
    }
}
