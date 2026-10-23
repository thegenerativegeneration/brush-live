//! The session worker: trains on incoming keyframes and, when its cadence
//! says so, runs a voxel round (score set from the splat parameters). With
//! `finish_fisher` a final Fisher pass adds coverage and uncertainty.

mod drain;
mod eval_log;
mod eval_schedule;
mod fisher;
mod holdout;
mod holdout_seam;
mod jsonl;
mod poses;

use super::preview::{PREVIEW_FLOATS, PreviewClock, PreviewSnapshot};
use super::splat_read::{SplatRead, view_cones};
use super::{Command, ScoreSetMsg, StatusMsg, WorkerTimes};
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::{LiveModel, StepTotals};
use crate::protocol::{Cell, KeyframeHeader};
use crate::schedule::{Cadence, IterThrottle};
use crate::scores::voxel::VoxelAggregator;
use brush_render::gaussian_splats::Splats;
use burn::tensor::Device;
use drain::{Drain, try_take};
use glam::UVec2;
use eval_schedule::EvalSchedule;
use holdout::{EvalExtras, Holdout};
pub(crate) use holdout::warm_up_eval;
use poses::Held;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
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
        if w.voxel_cadence.due(now) {
            w.wait_for_training();
            let now = w.clock.elapsed().as_secs_f64();
            let t = Instant::now();
            w.voxel_round(now).await;
            w.totals_s.voxel += t.elapsed().as_secs_f64();
        }
        let (received, now) = (w.holdout.received(), w.clock.elapsed().as_secs_f64());
        if w.holdout.has_views()
            && let Some(extras) = w.eval_schedule.due(received, now)
        {
            let scored = w.eval_holdout(extras, false).await;
            w.eval_schedule.record(received, now, extras, scored);
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

    fn to_times(&self, step: StepTotals) -> WorkerTimes {
        let ms = |s: f64| (s * 1e3) as u64;
        WorkerTimes {
            train_ms: ms(self.train),
            ingest_ms: ms(self.ingest),
            preview_ms: ms(self.preview),
            voxel_ms: ms(self.voxel),
            fisher_ms: ms(self.fisher),
            uptime_ms: ms(self.start.elapsed().as_secs_f64()),
            throttle_ms: ms(self.throttle),
            step,
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
    voxel_cadence: Cadence,
    throttle: IterThrottle,
    /// Wall time per activity since the session started; unlike `acc` it
    /// survives voxel rounds.
    totals_s: ActivityTotals,
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
    /// Commands received while paused, run in order on resume.
    held: Vec<Held>,
    /// `live.num_evicted()` at the end of the previous round.
    evicted_at_round: u64,
    /// Time spent between rounds, for the debug timing log.
    acc: Between,
    preview: PreviewClock,
    preview_version: u64,
    /// From the last `Command::Reset`; stamped on every preview snapshot.
    generation: u64,
    holdout: Holdout,
    eval_schedule: EvalSchedule,
}

#[derive(Default)]
struct Between {
    train_s: f64,
    steps: u32,
    kf_s: f64,
    kfs: u32,
}

fn eval_schedule_of(config: &GuideConfig) -> EvalSchedule {
    EvalSchedule::new(
        config.eval_every_keyframes,
        config.eval_interval_s,
        config.eval_extras_at.clone(),
    )
}

impl Worker {
    fn new(config: GuideConfig, device: Device, session_dir: PathBuf, channels: Channels) -> Self {
        let clock = Instant::now();
        let mut live = LiveModel::new(config.clone(), device.clone());
        live.enable_area(session_dir.join("tiles"));
        let holdout = Holdout::from_config(&config);
        let eval_schedule = eval_schedule_of(&config);
        let mut voxels = VoxelAggregator::new(
            config.voxel_size,
            config.min_cell_opacity,
            config.uncertainty_scale(),
        );
        voxels.record_raw(config.dump_raw_uncertainty);
        let voxel_cadence = voxel_cadence(&config);
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
            voxel_cadence,
            throttle,
            totals_s: ActivityTotals::new(),
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
            eval_schedule,
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
                self.held.push(Held::Keyframe(h, payload, reply));
            }
            Command::Poses(poses, reply) if self.paused => {
                self.held.push(Held::Poses(poses, reply));
            }
            Command::Pause(paused, reply) => {
                self.paused = paused;
                if !paused {
                    self.run_held().await;
                }
                let _ = reply.send(());
            }
            Command::Poses(poses, reply) => {
                let _ = reply.send(Ok(self.update_poses(&poses).await));
            }
            Command::Keyframe(h, payload, reply) => {
                let result = self.add_keyframe(&h, &payload).await;
                // Counts are visible to the caller as soon as the push resolves.
                publish_counts(&self.channels.status, &self.live);
                let _ = reply.send(result);
            }
            Command::Export(reply) => {
                let result = self.live.final_ply().await.map(|(bytes, _)| bytes).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            Command::Finish(path, reply) => {
                let result = self.finish(&path).await;
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

    /// Scores the splats on the held-out views and logs the result; `extras` adds LPIPS, the seam split and
    /// saved renders as configured ([`EvalExtras::for_eval`]), `finish` marks the finish eval. Whether any view
    /// was scored (not without splats or renderable views).
    async fn eval_holdout(&mut self, extras: bool, finish: bool) -> bool {
        // Nothing to score: skip the full model, which loads every parked tile.
        if !self.holdout.has_views() {
            return false;
        }
        let now = self.clock.elapsed().as_secs_f64();
        let eval_extras = EvalExtras::for_eval(
            &self.config,
            extras,
            self.holdout.received(),
            finish,
            &self.session_dir,
            self.live.iter(),
        );
        let t_load = Instant::now();
        let Some(splats) = self.live.eval_splats().await else {
            return false;
        };
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
        // Loading the parked tiles does not count against the training step rate.
        self.rate_window.0 += load_ms / 1e3;
        let background = self.live.sh_background();
        let composited = if background.is_some() { eval_log::EvalBackground::Sh } else { eval_log::EvalBackground::Black };
        let Some(r) = self.holdout.eval(&splats, now, &eval_extras, background).await else {
            return false;
        };
        // Nor does the eval itself.
        self.rate_window.0 += r.ms / 1e3;
        log::info!(
            "eval at {now:.2} s: {} held-out views, psnr {:.2} dB, ssim {:.4}; iter {}, {} splats ({:.0} ms, {} background)",
            r.views,
            r.psnr,
            r.ssim,
            self.live.iter(),
            splats.num_splats(),
            r.ms,
            if composited == eval_log::EvalBackground::Sh { "SH" } else { "black" }
        );
        let record = eval_log::EvalRecord {
            unix_s: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64()),
            t_s: now,
            iter: self.live.iter(),
            keyframes: self.live.views().len() as u32,
            received_keyframes: self.holdout.received(),
            last_keyframe_id: self.live.last_keyframe_id(),
            splats: splats.num_splats(),
            resident_splats: self.live.splats().map_or(0, |s| s.num_splats()),
            loader_cache_bytes: self.live.loader_cache_bytes(),
            train_ms: (self.totals_s.train * 1e3) as u64,
            refine_ms: self.live.refine_total_ms(),
            eval_ms: r.ms,
            lpips_ms: r.lpips_ms,
            seam_ms: r.seam_ms,
            bounds_median_size: self.live.bounds_median_size(),
            lr_mean: self.live.last_step_lr().lr_mean,
            position_scale_m: self.live.last_step_lr().position_scale_m,
            adam_clamped: self.live.adam_clamped(),
            masked_by_radius: self.live.masked_by_radius().await,
            area_views: self.live.area_view_count(),
            sight_views: self.live.sight_view_count(),
            sight_voters: self.live.sight_voter_count(),
            parked_load_ms: self.live.area_view_count().map(|_| load_ms),
            area: self.live.area_report(),
            extras,
            background: composited,
            views: r.per_view.clone(),
        };
        if let Err(e) = jsonl::append_jsonl(&self.session_dir.join("eval.jsonl"), &record) {
            log::warn!("could not append eval.jsonl: {e}");
        }
        true
    }

    /// Writes the whole model to `path` ([`LiveModel::write_final`]), then runs the finish eval (and with
    /// `finish_fisher` the Fisher pass and a last round) and pauses training; returns the file size.
    async fn finish(&mut self, path: &Path) -> Result<u64, String> {
        if self.live.splats().is_none() {
            return Err("no splats yet".to_owned());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let result = self.live.write_final(path).await.map(|e| e.ply_bytes).map_err(|e| e.to_string());
        if result.is_ok() {
            self.eval_holdout(true, true).await;
            if self.config.finish_fisher {
                let t = Instant::now();
                self.finish_pass(self.clock.elapsed().as_secs_f64()).await;
                self.totals_s.fisher += t.elapsed().as_secs_f64();
                let t = Instant::now();
                self.voxel_round(self.clock.elapsed().as_secs_f64()).await;
                self.totals_s.voxel += t.elapsed().as_secs_f64();
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

    /// Clears everything but the score set version, which keeps counting,
    /// and the mono scale history (`LiveModel::reset`).
    fn reset(&mut self) {
        for held in self.held.drain(..) {
            held.reset();
        }
        let config = &self.config;
        self.live.reset();
        self.holdout = Holdout::from_config(config);
        self.eval_schedule = eval_schedule_of(&config);
        self.voxels.reset();
        self.voxel_cadence = voxel_cadence(config);
        self.throttle = IterThrottle::new(config.max_iters_per_s);
        self.totals_s = ActivityTotals::new();
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
        self.voxel_cadence.record(start, end - start);
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
        let cones = view_cones(&self.live.sight_cameras());
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
                times: self.totals_s.to_times(self.live.step_totals()),
                adam_clamped: self.live.adam_clamped(),
                sh_pruned: self.live.sh_pruned(),
            });
        } else {
            publish_counts(&self.channels.status, &self.live);
        }
    }
}

fn voxel_cadence(config: &GuideConfig) -> Cadence {
    Cadence::new(config.score_budget, config.min_score_interval_s)
}

/// Keyframe, splat and step counts into the status; the rate and times stay at their last once-a-second value.
fn publish_counts(status_tx: &watch::Sender<StatusMsg>, live: &LiveModel) {
    status_tx.send_modify(|s| {
        s.num_keyframes = live.views().len() as u32;
        s.num_splats = live.splats().map_or(0, |s| s.num_splats());
        s.train_iters = u64::from(live.iter());
    });
}

#[cfg(test)]
#[path = "worker/eval_tests.rs"]
mod eval_tests;
