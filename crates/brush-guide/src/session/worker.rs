//! The session worker: trains on incoming keyframes and, when the
//! scheduler says so, runs a voxel round (score set from the splat
//! parameters) or a Fisher pass (coverage and uncertainty per voxel,
//! eviction importance).

mod drain;
mod fisher;
mod holdout;

use super::preview::{PREVIEW_FLOATS, PreviewClock, PreviewSnapshot};
use super::splat_read::{SplatRead, view_cones};
use super::{Command, ScoreSetMsg, StatusMsg, WorkerTimes};
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::LiveModel;
use crate::protocol::{Cell, KeyframeHeader};
use crate::schedule::{Cadence, FisherCost, IterThrottle, Round, RoundScheduler};
use crate::scores::voxel::{RawVoxel, VoxelAggregator};
use brush_render::gaussian_splats::Splats;
use burn::tensor::Device;
use drain::{Drain, try_take};
use glam::UVec2;
use holdout::Holdout;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
use web_time::Instant;

pub(super) struct Channels {
    pub(super) scores: watch::Sender<Option<Arc<ScoreSetMsg>>>,
    pub(super) status: watch::Sender<StatusMsg>,
    pub(super) preview: watch::Sender<Option<Arc<PreviewSnapshot>>>,
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
    let mut last_step = Duration::ZERO;
    loop {
        // With nothing to train, block for the next command; otherwise just drain.
        let mut next = if w.idle() {
            match rx.recv().await {
                Some(c) => Some(c),
                None => return,
            }
        } else {
            let Ok(c) = try_take(&mut rx) else { return };
            c
        };
        // Take further queued commands before the next step, within a
        // budget, so ingest keeps up when training steps get slow.
        let mut drain = Drain::new(Instant::now(), last_step);
        while let Some(cmd) = next {
            w.handle(cmd).await;
            drain.handled();
            next = None;
            if drain.more(Instant::now()) {
                let Ok(c) = try_take(&mut rx) else { return };
                next = c;
            }
        }

        if w.idle() {
            continue;
        }
        // Over the iteration cap: sleep until the next step is allowed, waking early for a command.
        if let Some(wait) = w.throttle.wait_s(w.clock.elapsed().as_secs_f64()) {
            let t = Instant::now();
            let woken = tokio::time::timeout(Duration::from_secs_f64(wait), rx.recv()).await;
            w.totals_s.throttle += t.elapsed().as_secs_f64();
            match woken {
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
        last_step = t.elapsed();
        let dt = last_step.as_secs_f64();
        w.acc.train_s += dt;
        w.totals_s.train += dt;
        w.acc.steps += 1;
        if w.step_cap_reached() {
            log::info!(
                "max_train_steps reached: training stopped at step {}",
                w.live.iter()
            );
        }
        if w.preview.due(Instant::now()) {
            w.wait_for_training();
            let t = Instant::now();
            w.publish_preview().await;
            w.totals_s.preview += t.elapsed().as_secs_f64();
        }
        let now = w.clock.elapsed().as_secs_f64();
        if let Some(round) = w.scheduler.next(now, w.fisher_cost()) {
            w.wait_for_training();
            let now = w.clock.elapsed().as_secs_f64();
            let t = Instant::now();
            match round {
                Round::Voxel => {
                    w.voxel_round(now).await;
                    w.totals_s.voxel += t.elapsed().as_secs_f64();
                }
                Round::Fisher(max_views) => {
                    w.fisher_pass(now, max_views).await;
                    w.totals_s.fisher += t.elapsed().as_secs_f64();
                }
            }
        }
        if w.holdout.due(w.clock.elapsed().as_secs_f64()) {
            w.eval_holdout().await;
        }
        w.publish_status();
        brush_async::yield_now().await;
    }
}

/// Seconds spent per activity; see [`WorkerTimes`].
struct ActivityTotals {
    /// When the totals started; `uptime_ms` counts from here.
    start: Instant,
    train: f64,
    ingest: f64,
    preview: f64,
    voxel: f64,
    fisher: f64,
    throttle: f64,
}

impl ActivityTotals {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            train: 0.0,
            ingest: 0.0,
            preview: 0.0,
            voxel: 0.0,
            fisher: 0.0,
            throttle: 0.0,
        }
    }

    fn to_times(&self) -> WorkerTimes {
        let ms = |s: f64| (s * 1e3) as u64;
        WorkerTimes {
            train_ms: ms(self.train),
            ingest_ms: ms(self.ingest),
            preview_ms: ms(self.preview),
            voxel_ms: ms(self.voxel),
            fisher_ms: ms(self.fisher),
            uptime_ms: ms(self.start.elapsed().as_secs_f64()),
            throttle_ms: ms(self.throttle),
        }
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
    /// Wall time per activity since the session started; unlike `acc` it
    /// survives voxel rounds.
    totals_s: ActivityTotals,
    /// Cost of the last Fisher pass and the splat count it ran at.
    fisher_cost: Option<(FisherCost, u32)>,
    /// Version of the last score set.
    version: u64,
    last_score_ms: u32,
    /// Start time and training iteration of the current rate window.
    rate_window: (f64, u32),
    /// Sent JPEG size per view, in the same order as `live.views()`.
    sizes: Vec<UVec2>,
    /// Set by Finish: no training or scoring until a new keyframe or Reset.
    finished: bool,
    /// Set by `Command::Pause(true)`: no GPU work until resumed.
    paused: bool,
    /// Keyframes received while paused, added in order on resume.
    held: Vec<(KeyframeHeader, Vec<u8>, oneshot::Sender<Result<(), String>>)>,
    /// `live.num_evicted()` at the end of the previous round.
    evicted_at_round: u64,
    /// Time spent between rounds, for the debug timing log.
    acc: Between,
    preview: PreviewClock,
    preview_version: u64,
    /// From the last `Command::Reset`; stamped on every preview snapshot.
    generation: u64,
    holdout: Holdout,
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
        let holdout = Holdout::new(config.holdout_every, config.eval_interval_s);
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
            totals_s: ActivityTotals::new(),
            fisher_cost: None,
            version: 0,
            last_score_ms: 0,
            rate_window,
            sizes: Vec::new(),
            finished: false,
            paused: false,
            held: Vec::new(),
            evicted_at_round: 0,
            acc: Between::default(),
            preview: PreviewClock::default(),
            preview_version: 0,
            generation: 0,
            holdout,
        }
    }

    /// Nothing to train: no views yet, paused by Finish, or paused by the app.
    fn idle(&self) -> bool {
        self.live.views().is_empty() || self.finished || self.paused || self.step_cap_reached()
    }

    fn step_cap_reached(&self) -> bool {
        self.config
            .max_train_steps
            .is_some_and(|n| self.live.iter() >= n)
    }

    /// Waits for all queued GPU work, which is training's unless an
    /// activity left some behind, and counts the wait as training. Without
    /// it the next activity's timer would include the wait.
    fn wait_for_training(&mut self) {
        let t = Instant::now();
        if let Err(e) = self.device.sync() {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                log::warn!("worker: device sync failed ({e:?}); activity times are unreliable");
            });
        }
        let dt = t.elapsed().as_secs_f64();
        self.acc.train_s += dt;
        self.totals_s.train += dt;
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Keyframe(h, payload, reply) if self.paused => {
                self.held.push((h, payload, reply));
            }
            Command::Pause(paused, reply) => {
                self.paused = paused;
                if !paused {
                    for (h, payload, held_reply) in std::mem::take(&mut self.held) {
                        let result = self.add_keyframe(&h, &payload).await;
                        publish_counts(&self.channels.status, &self.live);
                        let _ = held_reply.send(result);
                    }
                }
                let _ = reply.send(());
            }
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
            Command::SetPreview(interval) => self.preview.set(interval),
            Command::Reset(generation, reply) => {
                self.generation = generation;
                self.reset();
                let _ = reply.send(());
            }
        }
    }

    /// [`Self::ingest`], counting the keyframe and its time as ingest.
    async fn add_keyframe(&mut self, h: &KeyframeHeader, payload: &[u8]) -> Result<(), String> {
        self.wait_for_training();
        let t = Instant::now();
        let result = self.ingest(h, payload).await;
        if let Some(s) = self.live.splats() {
            crate::timing::sync_splats(s).await;
        }
        let dt = t.elapsed().as_secs_f64();
        self.acc.kf_s += dt;
        self.totals_s.ingest += dt;
        self.acc.kfs += 1;
        result
    }

    async fn ingest(&mut self, h: &KeyframeHeader, payload: &[u8]) -> Result<(), String> {
        // A resend must not overwrite the stored image of the first send.
        if self.live.contains(h.id) || self.holdout.contains(h.id) {
            return Ok(());
        }
        let kf = decode_keyframe(h, payload, &self.session_dir)
            .await
            .map_err(|e| e.to_string())?;
        if self.holdout.next_is_held() {
            // Fail before recording, so a resend is tried again.
            kf.view.image.load().await.map_err(|e| e.to_string())?;
            self.holdout
                .add_held(h.id, kf.camera, kf.view, &self.session_dir);
            return Ok(());
        }
        if self.live.add_keyframe(kf).await {
            self.holdout.note_trained();
            self.sizes.push(UVec2::new(h.width, h.height));
            if self.finished {
                self.finished = false;
                self.rate_window = (self.clock.elapsed().as_secs_f64(), self.live.iter());
            }
        }
        Ok(())
    }

    /// Scores the splats on the held-out views and logs the result.
    async fn eval_holdout(&mut self) {
        let now = self.clock.elapsed().as_secs_f64();
        let Some(splats) = self.live.splats() else {
            return;
        };
        if let Some(r) = self.holdout.eval(splats, now).await {
            // Eval time does not count against the training step rate.
            self.rate_window.0 += r.ms / 1e3;
            log::info!(
                "eval at {now:.2} s: {} held-out views, psnr {:.2} dB, ssim {:.4}; iter {}, {} splats ({:.0} ms, black background{})",
                r.views,
                r.psnr,
                r.ssim,
                self.live.iter(),
                splats.num_splats(),
                r.ms,
                if self.config.sh_background {
                    ", SH background not composited"
                } else {
                    ""
                }
            );
        }
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
            self.eval_holdout().await;
            if self.config.finish_fisher {
                let t = Instant::now();
                self.finish_pass(self.clock.elapsed().as_secs_f64()).await;
                self.voxel_round(self.clock.elapsed().as_secs_f64()).await;
                self.totals_s.fisher += t.elapsed().as_secs_f64();
            }
            self.finished = true;
            self.channels
                .status
                .send_modify(|s| s.train_iters_per_s = 0.0);
            // Dump the SH background's coefficients alongside the export,
            // when the flag is on: the exported `splat.ply` does not bake
            // the globe in, so it keeps holes where sky was, and this JSON
            // is the only record of what the globe learned for this
            // session.
            if let Some(json) = self.live.sh_background_json().await {
                let path = self.session_dir.join("sh_background.json");
                let _ = tokio::fs::write(&path, json).await;
            }
        }
        result
    }

    /// Clears everything but the score set version, which keeps counting.
    fn reset(&mut self) {
        for (_, _, held_reply) in self.held.drain(..) {
            let _ = held_reply.send(Err("session reset".to_owned()));
        }
        let config = &self.config;
        self.live = LiveModel::new(config.clone(), self.device.clone());
        self.holdout = Holdout::new(config.holdout_every, config.eval_interval_s);
        self.voxels.reset();
        self.scheduler = scheduler(config);
        self.throttle = IterThrottle::new(config.max_iters_per_s);
        self.fisher_passes = 0;
        self.totals_s = ActivityTotals::new();
        self.fisher_cost = None;
        self.sizes.clear();
        self.finished = false;
        self.evicted_at_round = 0;
        self.last_score_ms = 0;
        // `live.iter()` restarts at 0; an old window would underflow.
        self.rate_window = (self.clock.elapsed().as_secs_f64(), self.live.iter());
        self.channels.scores.send_replace(None);
        self.channels.status.send_replace(StatusMsg::default());
        if self.channels.preview.borrow().is_some() {
            self.preview_version += 1;
            self.channels
                .preview
                .send_replace(Some(Arc::new(PreviewSnapshot {
                    version: self.preview_version,
                    generation: self.generation,
                    ..PreviewSnapshot::default()
                })));
        }
    }

    /// Reads and publishes a preview snapshot of the current splats.
    async fn publish_preview(&mut self) {
        let Some(splats) = self.live.splats() else {
            return;
        };
        let t = Instant::now();
        let data = super::preview::read(splats).await;
        self.preview_version += 1;
        let count = (data.len() / PREVIEW_FLOATS) as u32;
        self.channels
            .preview
            .send_replace(Some(Arc::new(PreviewSnapshot {
                version: self.preview_version,
                generation: self.generation,
                count,
                readback_ms: t.elapsed().as_secs_f32() * 1000.0,
                data,
            })));
        self.preview.taken(Instant::now());
    }

    /// Builds the score set from the splat parameters and publishes it.
    /// `start` is when the round started.
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

        let end = self.clock.elapsed().as_secs_f64();
        let evicted = self.live.num_evicted() - self.evicted_at_round;
        self.evicted_at_round = self.live.num_evicted();
        log::info!(
            "round {version} at {start:.2} s: {} cells in {} ms; {} splats, {evicted} evicted since last round",
            cells.len(),
            self.last_score_ms,
            splats.num_splats()
        );
        self.scheduler.voxel.record(start, end - start);
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
            "cells: splat readback {:.0} ms, gaussian prep {:.0} ms ({} cones), voxel aggregate {:.0} ms, {} cells, {} voxels tracked",
            t_read * 1e3,
            t_prep * 1e3,
            cones.len(),
            t.elapsed().as_secs_f64() * 1e3,
            cells.len(),
            self.voxels.tracked_voxels()
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
                times: self.totals_s.to_times(),
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

/// One JSON line: the round's version and each voxel as
/// `[kx, ky, kz, n, spread, px_per_m, coverage]`.
fn append_ingredient_round(
    path: &Path,
    version: u64,
    rows: &[(glam::IVec3, f32, f32, f32, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let voxels: Vec<(i32, i32, i32, f32, f32, f32, f32)> = rows
        .iter()
        .map(|(k, n, s, p, c)| (k.x, k.y, k.z, *n, *s, *p, *c))
        .collect();
    let line = serde_json::json!({ "version": version, "voxels": voxels });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{line}")
}
