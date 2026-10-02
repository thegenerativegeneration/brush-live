//! The session worker: trains on incoming keyframes and, when the
//! scheduler says so, runs a voxel round (score set from the splat
//! parameters, TSDF fusion, meshing) or a Fisher pass (coverage and
//! uncertainty per voxel, eviction importance).

mod fisher;

use super::geometry::{Geometry, MeshLog};
use super::splat_read::{SplatRead, view_cones};
use super::{Command, ScoreSetMsg, StatusMsg};
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::LiveModel;
use crate::protocol::{Cell, KeyframeHeader, MeshBrick};
use crate::schedule::{Cadence, FisherCost, IterThrottle, Round, RoundScheduler};
use crate::scores::voxel::{RawVoxel, VoxelAggregator};
use brush_render::gaussian_splats::Splats;
use burn::module::Module;
use burn::tensor::Device;
use glam::UVec2;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use web_time::Instant;

pub(super) struct Channels {
    pub(super) scores: watch::Sender<Option<Arc<ScoreSetMsg>>>,
    pub(super) meshes: watch::Sender<MeshLog>,
    pub(super) status: watch::Sender<StatusMsg>,
}

/// Runs the session. With `ready`, waits for it to turn true (the server's
/// GPU warm-up) before taking commands; they queue meanwhile.
pub(super) async fn worker(
    config: GuideConfig,
    device: Device,
    session_dir: PathBuf,
    mut rx: mpsc::Receiver<Command>,
    channels: Channels,
    ready: Option<watch::Receiver<bool>>,
) {
    if let Some(mut ready) = ready {
        let t = Instant::now();
        let _ = ready.wait_for(|r| *r).await;
        let waited = t.elapsed().as_secs_f64();
        if waited > 0.05 {
            log::info!("session waited {waited:.1} s for the GPU warm-up");
        }
    }
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
            let t = Instant::now();
            let is_kf = matches!(cmd, Command::Keyframe(..));
            w.handle(cmd).await;
            if is_kf {
                if let Some(s) = w.live.splats() {
                    crate::timing::sync_splats(s).await;
                }
                w.acc.kf_s += t.elapsed().as_secs_f64();
                w.acc.kfs += 1;
            }
        }

        if w.idle() {
            continue;
        }
        // Over the iteration cap: sleep until the next step is allowed, waking early for a command.
        if let Some(wait) = w.throttle.wait_s(w.clock.elapsed().as_secs_f64()) {
            match tokio::time::timeout(Duration::from_secs_f64(wait), rx.recv()).await {
                Ok(Some(cmd)) => {
                    w.handle(cmd).await;
                    continue;
                }
                Ok(None) => return,
                Err(_elapsed) => {}
            }
        }
        w.throttle.record_step(w.clock.elapsed().as_secs_f64());
        let t = Instant::now();
        w.live.train_step().await;
        w.acc.train_s += t.elapsed().as_secs_f64();
        w.acc.steps += 1;
        let now = w.clock.elapsed().as_secs_f64();
        if let Some(round) = w.scheduler.next(now, w.fisher_cost()) {
            let t = Instant::now();
            if let Some(s) = w.live.splats() {
                crate::timing::sync_splats(s).await;
            }
            w.acc.train_s += t.elapsed().as_secs_f64();
            let now = w.clock.elapsed().as_secs_f64();
            match round {
                Round::Voxel => w.voxel_round(now).await,
                Round::Fisher(max_views) => w.fisher_pass(now, max_views).await,
            }
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
    scheduler: RoundScheduler,
    throttle: IterThrottle,
    /// Fisher passes so far; rotates their view sample.
    fisher_passes: u64,
    /// Cost of the last Fisher pass and the splat count it ran at.
    fisher_cost: Option<(FisherCost, u32)>,
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
    /// Time spent between rounds, for the debug timing log.
    acc: Between,
}

#[derive(Default)]
struct Between {
    train_s: f64,
    steps: u32,
    kf_s: f64,
    kfs: u32,
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
        let scheduler = scheduler(&config);
        let throttle = IterThrottle::new(config.max_iters_per_s);
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
            throttle,
            fisher_passes: 0,
            fisher_cost: None,
            geometry: Geometry::new(),
            version: 0,
            last_score_ms: 0,
            rate_window,
            sizes: Vec::new(),
            finished: false,
            evicted_at_round: 0,
            acc: Between::default(),
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
        self.scheduler = scheduler(config);
        self.throttle = IterThrottle::new(config.max_iters_per_s);
        self.fisher_passes = 0;
        self.fisher_cost = None;
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

    /// Builds the score set from the splat parameters, fuses and meshes the
    /// TSDF, and publishes the round's bricks, then its score set. `start`
    /// is when the round started.
    async fn voxel_round(&mut self, start: f64) {
        let acc = std::mem::take(&mut self.acc);
        let refine = self.live.take_refine_stats();
        log::debug!(
            target: crate::timing::TARGET,
            "between rounds: {} train steps in {:.0} ms ({:.1} ms/step), of which {} refines {:.0} ms; \
             {} keyframes in {:.0} ms; {} views, {} splats",
            acc.steps,
            acc.train_s * 1e3,
            acc.train_s * 1e3 / f64::from(acc.steps.max(1)),
            refine.0,
            refine.1 * 1e3,
            acc.kfs,
            acc.kf_s * 1e3,
            self.live.views().len(),
            self.live.splats().map_or(0, |s| s.num_splats())
        );
        let splats = self.live.splats().expect("views imply splats").clone();
        let cells = self.cells(&splats).await;
        let scored = self.clock.elapsed().as_secs_f64();
        self.last_score_ms = ((scored - start) * 1000.0) as u32;
        self.version += 1;
        let version = self.version;

        let (bricks, pending, num_fused) = self.mesh_round(&splats).await;
        let end = self.clock.elapsed().as_secs_f64();
        let mesh_ms = ((end - scored) * 1000.0) as u32;
        let evicted = self.live.num_evicted() - self.evicted_at_round;
        self.evicted_at_round = self.live.num_evicted();
        log::info!(
            "round {version} at {start:.2} s: {} cells in {} ms; {} bricks sent of {pending} stale, \
             {num_fused} views fused, {mesh_ms} ms; {} splats, {evicted} evicted since last round",
            cells.len(),
            self.last_score_ms,
            bricks.len(),
            splats.num_splats()
        );
        self.scheduler.voxel.record(start, end - start);
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

    /// The score set's cells from the splat parameters, with each voxel's
    /// coverage and uncertainty from the latest Fisher pass that scored it.
    async fn cells(&mut self, splats: &Splats) -> Vec<Cell> {
        let t = Instant::now();
        let read = SplatRead::new(splats).await;
        let t_read = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let geoms = read.geoms();
        let cones = view_cones(self.live.views());
        let t_prep = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let cells = self
            .voxels
            .cells(&geoms, &cones, self.clock.elapsed().as_secs_f64());
        log::debug!(
            target: crate::timing::TARGET,
            "cells: splat readback {:.0} ms, gaussian prep {:.0} ms ({} cones), voxel aggregate {:.0} ms, {} cells",
            t_read * 1e3,
            t_prep * 1e3,
            cones.len(),
            t.elapsed().as_secs_f64() * 1e3,
            cells.len()
        );
        cells
    }

    /// The last Fisher pass's cost, scaled to the current splat count.
    fn fisher_cost(&self) -> Option<FisherCost> {
        let (c, n) = self.fisher_cost?;
        let now = self.live.splats().map_or(n, Splats::num_splats);
        let k = f64::from(now.max(1)) / f64::from(n.max(1));
        Some(FisherCost {
            per_view_s: c.per_view_s * k,
            fixed_s: c.fixed_s * k,
        })
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
                train_iters: u64::from(self.live.iter()),
            });
        } else {
            publish_counts(&self.channels.status, &self.live);
        }
    }
}

fn scheduler(config: &GuideConfig) -> RoundScheduler {
    RoundScheduler::new(
        Cadence::new(config.score_budget, config.min_score_interval_s),
        Cadence::new(config.fisher_budget, config.min_fisher_interval_s),
        config.max_fisher_views,
        config.min_fisher_views,
    )
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
