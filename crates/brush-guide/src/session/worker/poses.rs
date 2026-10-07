//! Commands held while paused, and pose updates.

use super::Worker;
use super::publish_counts;
use crate::live::{PoseUpdateStats, ViewPose};
use crate::protocol::KeyframeHeader;
use tokio::sync::oneshot;
use web_time::Instant;

/// A command received while paused; run in arrival order on resume.
pub(super) enum Held {
    Keyframe(KeyframeHeader, Vec<u8>, oneshot::Sender<Result<(), String>>),
    Poses(
        Vec<ViewPose>,
        oneshot::Sender<Result<PoseUpdateStats, String>>,
    ),
}

impl Held {
    /// Answers the waiting caller: the session was reset before the command ran.
    pub(super) fn reset(self) {
        match self {
            Held::Keyframe(_, _, reply) => {
                let _ = reply.send(Err("session reset".to_owned()));
            }
            Held::Poses(_, reply) => {
                let _ = reply.send(Err("session reset".to_owned()));
            }
        }
    }
}

impl Worker {
    /// Runs the commands held while paused, in order.
    pub(super) async fn run_held(&mut self) {
        for held in std::mem::take(&mut self.held) {
            match held {
                Held::Keyframe(h, payload, reply) => {
                    let result = self.add_keyframe(&h, &payload).await;
                    publish_counts(&self.channels.status, &self.live);
                    let _ = reply.send(result);
                }
                Held::Poses(poses, reply) => {
                    let _ = reply.send(Ok(self.update_poses(&poses).await));
                }
            }
        }
    }

    /// [`crate::live::LiveModel::update_poses`] plus the held-out views; its time counts as ingest.
    pub(super) async fn update_poses(&mut self, poses: &[ViewPose]) -> PoseUpdateStats {
        self.wait_for_training();
        let t = Instant::now();
        let stats = self.live.update_poses(poses);
        let held = self.holdout.update_poses(poses);
        if let Some(s) = self.live.splats() {
            crate::timing::sync_splats(s).await;
        }
        let dt = t.elapsed().as_secs_f64();
        self.totals_s.ingest += dt;
        log::info!(
            "poses: {} views moved, {} unchanged, {} unknown, {held} held-out views moved in {:.0} ms",
            stats.moved,
            stats.unchanged,
            stats.unknown,
            dt * 1e3
        );
        stats
    }
}
