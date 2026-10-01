//! The session worker: trains on incoming keyframes and, when the
//! scheduler says so, runs a round of scoring, TSDF fusion and meshing.

use super::geometry::{Geometry, MeshLog};
use super::{Command, ScoreSetMsg, StatusMsg};
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::LiveModel;
use crate::protocol::{Cell, KeyframeHeader, MeshBrick};
use crate::schedule::{ScoreScheduler, score_view_weight, select_score_views};
use crate::scores::importance::importances;
use crate::scores::metrics::gaussian_metrics;
use crate::scores::pass::{PassView, score_pass};
use crate::scores::voxel::{GaussianScore, RawVoxel, ViewCone, VoxelAggregator};
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::Splats;
use burn::module::Module;
use burn::tensor::{Device, Tensor};
use glam::{Quat, UVec2, Vec3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use web_time::Instant;

pub(super) struct Channels {
    pub(super) scores: watch::Sender<Option<Arc<ScoreSetMsg>>>,
    pub(super) meshes: watch::Sender<MeshLog>,
    pub(super) status: watch::Sender<StatusMsg>,
}

pub(super) async fn worker(
    config: GuideConfig,
    device: Device,
    session_dir: PathBuf,
    mut rx: mpsc::Receiver<Command>,
    channels: Channels,
) {
    let mut w = Worker::new(config, device, session_dir, channels);
    loop {
        // With nothing to train, block for the next command; otherwise just drain.
        let cmd = if w.idle() {
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
        if let Some(cmd) = cmd {
            w.handle(cmd).await;
        }

        if w.idle() {
            continue;
        }
        w.live.train_step().await;
        let now = w.clock.elapsed().as_secs_f64();
        if w.scheduler.due(now) {
            w.score_round(now).await;
        }
        w.publish_status();
        brush_async::yield_now().await;
    }
}

struct Worker {
    config: GuideConfig,
    device: Device,
    session_dir: PathBuf,
    channels: Channels,
    clock: Instant,
    live: LiveModel,
    voxels: VoxelAggregator,
    scheduler: ScoreScheduler,
    geometry: Geometry,
    /// Version of the last score set.
    version: u64,
    last_score_ms: u32,
    /// Start time and training iteration of the current rate window.
    rate_window: (f64, u32),
    /// Sent JPEG size per view, in the same order as `live.views()`.
    sizes: Vec<UVec2>,
    /// Set by Finish: no training or scoring until a new keyframe or Reset.
    finished: bool,
    /// `live.num_evicted()` at the end of the previous round.
    evicted_at_round: u64,
}

impl Worker {
    fn new(config: GuideConfig, device: Device, session_dir: PathBuf, channels: Channels) -> Self {
        let clock = Instant::now();
        let live = LiveModel::new(config.clone(), device.clone());
        let mut voxels = VoxelAggregator::new(
            config.voxel_size,
            config.min_cell_opacity,
            config.uncertainty_scale(),
        );
        voxels.record_raw(config.dump_raw_uncertainty);
        let scheduler = ScoreScheduler::new(config.score_budget, config.min_score_interval_s);
        let rate_window = (clock.elapsed().as_secs_f64(), live.iter());
        Self {
            config,
            device,
            session_dir,
            channels,
            clock,
            live,
            voxels,
            scheduler,
            geometry: Geometry::new(),
            version: 0,
            last_score_ms: 0,
            rate_window,
            sizes: Vec::new(),
            finished: false,
            evicted_at_round: 0,
        }
    }

    /// Nothing to train: no views yet, or paused by Finish.
    fn idle(&self) -> bool {
        self.live.views().is_empty() || self.finished
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Keyframe(h, payload, reply) => {
                let result = self.add_keyframe(&h, &payload).await;
                // Counts are visible to the caller as soon as the push resolves.
                publish_counts(&self.channels.status, &self.live);
                let _ = reply.send(result);
            }
            Command::Export(finish, reply) => {
                let result = self.export(finish).await;
                let _ = reply.send(result);
            }
            Command::Reset(reply) => {
                self.reset();
                let _ = reply.send(());
            }
        }
    }

    async fn add_keyframe(&mut self, h: &KeyframeHeader, payload: &[u8]) -> Result<(), String> {
        // A resend must not overwrite the stored image of the first send.
        if self.live.contains(h.id) {
            return Ok(());
        }
        let kf = decode_keyframe(h, payload, &self.session_dir)
            .await
            .map_err(|e| e.to_string())?;
        if self.live.add_keyframe(kf).await {
            self.sizes.push(UVec2::new(h.width, h.height));
            if self.finished {
                self.finished = false;
                self.rate_window = (self.clock.elapsed().as_secs_f64(), self.live.iter());
            }
        }
        Ok(())
    }

    /// The splats as PLY; `finish` also pauses training.
    async fn export(&mut self, finish: bool) -> Result<Vec<u8>, String> {
        let result = match self.live.splats() {
            Some(s) => brush_serde::splat_to_ply(s.clone(), None)
                .await
                .map_err(|e| e.to_string()),
            None => Err("no splats yet".to_owned()),
        };
        if finish && result.is_ok() {
            self.finished = true;
            self.channels
                .status
                .send_modify(|s| s.train_iters_per_s = 0.0);
        }
        result
    }

    /// Clears everything but the score set version, which keeps counting.
    fn reset(&mut self) {
        let config = &self.config;
        self.live = LiveModel::new(config.clone(), self.device.clone());
        self.voxels.reset();
        self.geometry = Geometry::new();
        self.scheduler = ScoreScheduler::new(config.score_budget, config.min_score_interval_s);
        self.sizes.clear();
        self.finished = false;
        self.evicted_at_round = 0;
        self.last_score_ms = 0;
        // `live.iter()` restarts at 0; an old window would underflow.
        self.rate_window = (self.clock.elapsed().as_secs_f64(), self.live.iter());
        self.channels.scores.send_replace(None);
        self.channels.meshes.send_replace(MeshLog::default());
        self.channels.status.send_replace(StatusMsg::default());
    }

    /// Scores the splats, fuses and meshes the TSDF, and publishes the
    /// round's bricks, then its score set. `now` is when the round started.
    async fn score_round(&mut self, now: f64) {
        let splats = self.live.splats().expect("views imply splats").clone();
        let cells = self.score_cells(&splats).await;
        let scored = self.clock.elapsed().as_secs_f64();
        self.last_score_ms = ((scored - now) * 1000.0) as u32;
        self.version += 1;
        let version = self.version;
        if self.config.dump_raw_uncertainty {
            let path = self.session_dir.join("raw_uncertainty.jsonl");
            if let Err(e) = append_raw_round(&path, version, self.voxels.raw_round()) {
                log::warn!("raw uncertainty dump to {}: {e}", path.display());
            }
        }

        let (bricks, pending, num_fused) = self.mesh_round(&splats).await;
        let end = self.clock.elapsed().as_secs_f64();
        let mesh_ms = ((end - scored) * 1000.0) as u32;
        let evicted = self.live.num_evicted() - self.evicted_at_round;
        self.evicted_at_round = self.live.num_evicted();
        log::info!(
            "round {version}: {} bricks sent of {pending} stale, {num_fused} views fused, {mesh_ms} ms; \
             {} splats, {evicted} evicted since last round",
            bricks.len(),
            splats.num_splats()
        );
        self.scheduler.record(end, end - now);
        self.channels
            .meshes
            .send_modify(|log| log.record(version, mesh_ms, bricks));
        self.channels
            .scores
            .send_replace(Some(Arc::new(ScoreSetMsg {
                version,
                based_on_keyframe_id: self.live.last_keyframe_id().unwrap_or(0),
                voxel_size: self.config.voxel_size,
                cells,
            })));
    }

    /// Voxel cells of this round's score pass over a sample of the views.
    async fn score_cells(&mut self, splats: &Splats) -> Vec<Cell> {
        let config = &self.config;
        let num_views = self.live.views().len();
        let views: Vec<PassView> = select_score_views(
            num_views,
            config.max_score_views,
            config.seed.wrapping_add(self.version),
        )
        .into_iter()
        .map(|i| PassView {
            camera: self.live.views()[i].camera,
            img_size: self.sizes[i],
            // Sums over the sample estimate sums over every view, so
            // `CoverageParams::n_target` and σ refer to the whole capture.
            weight: score_view_weight(i, num_views, config.max_score_views),
        })
        .collect();
        let out = score_pass(splats, &views, &config.pass).await;
        let (coverage, fisher_pos) = gaussian_metrics(&out, &config.coverage);
        let read = SplatRead::new(splats).await;
        if config.evict {
            let importance = importances(&out, &read.rots, &read.scales);
            self.live.set_importance(&importance);
        }
        let gaussians = gaussian_scores(&read, &coverage, &fisher_pos);
        let cones = view_cones(self.live.views());
        self.voxels
            .aggregate(&gaussians, &cones, self.clock.elapsed().as_secs_f64())
    }

    /// Fuses and meshes the TSDF: the bricks to publish, how many were
    /// stale, and how many views were fused.
    async fn mesh_round(&mut self, splats: &Splats) -> (Vec<MeshBrick>, usize, usize) {
        if splats.num_splats() == 0 {
            return (Vec::new(), 0, 0);
        }
        let views = self.live.views();
        let num_fused = self
            .geometry
            .fuse(&splats.valid(), views, &self.sizes)
            .await;
        let eye = views.last().expect("views exist").camera.position;
        let (bricks, pending) = self.geometry.mesh_changed(eye);
        (bricks, pending, num_fused)
    }

    /// Publishes the training rate once a second, the counts otherwise.
    fn publish_status(&mut self) {
        let now = self.clock.elapsed().as_secs_f64();
        if now - self.rate_window.0 >= 1.0 {
            let rate = (self.live.iter() - self.rate_window.1) as f64 / (now - self.rate_window.0);
            self.rate_window = (now, self.live.iter());
            self.channels.status.send_replace(StatusMsg {
                num_keyframes: self.live.views().len() as u32,
                num_splats: self.live.splats().map_or(0, |s| s.num_splats()),
                train_iters_per_s: rate as f32,
                last_score_ms: self.last_score_ms,
            });
        } else {
            publish_counts(&self.channels.status, &self.live);
        }
    }
}

fn publish_counts(status_tx: &watch::Sender<StatusMsg>, live: &LiveModel) {
    status_tx.send_modify(|s| {
        s.num_keyframes = live.views().len() as u32;
        s.num_splats = live.splats().map_or(0, |s| s.num_splats());
    });
}

/// One JSON line: the round's version and each voxel as
/// `[kx, ky, kz, coverage, sigma]` (voxel index, mean coverage, positional
/// σ in metres, `null` when infinite).
fn append_raw_round(path: &Path, version: u64, raw: &[RawVoxel]) -> std::io::Result<()> {
    use std::io::Write;
    let voxels: Vec<(i32, i32, i32, f32, f32)> = raw
        .iter()
        .map(|r| (r.key.x, r.key.y, r.key.z, r.coverage, r.sigma))
        .collect();
    let line = serde_json::json!({ "version": version, "voxels": voxels });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{line}")
}

async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("splat readback")
        .try_to_vec::<f32>()
        .expect("f32 splat data")
}

/// Splat parameters read back once per round, flat per splat.
struct SplatRead {
    means: Vec<f32>,
    opac: Vec<f32>,
    rots: Vec<f32>,
    scales: Vec<f32>,
}

impl SplatRead {
    async fn new(splats: &Splats) -> Self {
        Self {
            means: read_f32(splats.means()).await,
            opac: read_f32(splats.opacities()).await,
            rots: read_f32(splats.rotations()).await,
            scales: read_f32(splats.scales()).await,
        }
    }
}

/// Per-Gaussian inputs of the voxel aggregation.
fn gaussian_scores(
    read: &SplatRead,
    coverage: &[f32],
    fisher_pos: &[[f32; 9]],
) -> Vec<GaussianScore> {
    let SplatRead {
        means,
        opac,
        rots,
        scales,
    } = read;
    (0..opac.len())
        .map(|i| GaussianScore {
            pos: Vec3::new(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]),
            opacity: opac[i],
            coverage: coverage[i],
            fisher_pos: fisher_pos[i],
            axis: shortest_axis(&rots[i * 4..i * 4 + 4], &scales[i * 3..i * 3 + 3]),
            flatness: flatness(&scales[i * 3..i * 3 + 3]),
        })
        .collect()
}

fn view_cones(views: &[SceneView]) -> Vec<ViewCone> {
    views
        .iter()
        .map(|v| {
            let c = &v.camera;
            ViewCone {
                position: c.position,
                forward: c.rotation * Vec3::Z,
                cos_half_fov: (0.5 * c.fov_x.max(c.fov_y) as f32).cos(),
            }
        })
        .collect()
}

/// World direction of the Gaussian's shortest scale axis; zero for a
/// degenerate rotation.
pub(super) fn shortest_axis(r: &[f32], s: &[f32]) -> Vec3 {
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
pub(super) fn flatness(s: &[f32]) -> f32 {
    let mut v = [s[0], s[1], s[2]];
    v.sort_by(f32::total_cmp);
    if !(v[0].is_finite() && v[1].is_finite()) || v[1] <= 0.0 {
        return 0.0;
    }
    (1.0 - v[0] / v[1]).clamp(0.0, 1.0)
}
