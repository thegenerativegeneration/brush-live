//! The session's surface reconstruction: which keyframe views are fused
//! into the TSDF each round, which stale bricks are meshed, and the log of
//! published brick meshes that connections catch up from.

#[cfg(test)]
mod tests;

use crate::geometry::depth::{render_colour, render_expected_depth};
use crate::geometry::mesh::mesh_brick;
use crate::geometry::tsdf::{BRICK, BrickKey, Tsdf, VOXEL};
use crate::protocol::{MeshBrick, ServerHeader, encode_frame, encode_mesh_bricks};
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::Splats;
use glam::{UVec2, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Bricks whose mesh changed since the reader's last message, in their
/// latest state: cumulative over every round recorded since then.
pub struct MeshBricksMsg {
    /// Version of the last round included. A reader may take several rounds
    /// in one message, so this can be newer than the score set it follows.
    pub version: u64,
    pub mesh_ms: u32,
    pub bricks: Vec<Arc<MeshBrick>>,
}

impl MeshBricksMsg {
    pub fn to_frame(&self) -> Vec<u8> {
        let header = ServerHeader::MeshBricks {
            version: self.version,
            num_bricks: self.bricks.len() as u32,
            mesh_ms: self.mesh_ms,
        };
        encode_frame(
            &header,
            &encode_mesh_bricks(self.bricks.iter().map(|b| &**b)),
        )
    }
}

/// Latest published state of every brick that has had a mesh, so a reader
/// that missed rounds, or joins late, catches up with one message.
#[derive(Default)]
pub struct MeshLog {
    version: u64,
    mesh_ms: u32,
    next_seq: u64,
    bricks: HashMap<BrickKey, LoggedBrick>,
}

struct LoggedBrick {
    version: u64,
    /// Order in which bricks were published, across rounds.
    seq: u64,
    brick: Arc<MeshBrick>,
}

impl MeshLog {
    /// Version of the last recorded round; 0 before the first.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Records round `version`'s bricks in the order given.
    pub fn record(&mut self, version: u64, mesh_ms: u32, bricks: Vec<MeshBrick>) {
        self.version = version;
        self.mesh_ms = mesh_ms;
        for brick in bricks {
            self.bricks.insert(
                brick.key(),
                LoggedBrick {
                    version,
                    seq: self.next_seq,
                    brick: Arc::new(brick),
                },
            );
            self.next_seq += 1;
        }
    }

    /// Every brick published in a round after `since`, in its latest state,
    /// in publication order; `None` if no round was recorded after `since`.
    pub fn since(&self, since: u64) -> Option<MeshBricksMsg> {
        if self.version <= since {
            return None;
        }
        let mut changed: Vec<&LoggedBrick> =
            self.bricks.values().filter(|b| b.version > since).collect();
        changed.sort_unstable_by_key(|b| b.seq);
        Some(MeshBricksMsg {
            version: self.version,
            mesh_ms: self.mesh_ms,
            bricks: changed.into_iter().map(|b| b.brick.clone()).collect(),
        })
    }
}

/// Bricks meshed per round at most; further changed bricks wait.
const MAX_BRICKS_PER_ROUND: usize = 24;
/// Older keyframes re-integrated per round besides the new ones.
const REFRESH_VIEWS_PER_ROUND: usize = 4;
/// New keyframes integrated per round at most; further ones wait.
const MAX_NEW_VIEWS_PER_ROUND: usize = 8;
/// Expected depth and colour are rendered at the sent image size divided
/// by this.
const DEPTH_DOWNSCALE: u32 = 4;
/// Factor on all TSDF weights per round, so the surface follows the splat.
const WEIGHT_DECAY: f32 = 0.95;

/// TSDF fused from the splat's expected depth and colour at keyframe
/// poses, meshed per brick.
pub(super) struct Geometry {
    tsdf: Tsdf,
    /// Views `0..integrated` were integrated at least once.
    integrated: usize,
    /// Next older view to re-integrate.
    cursor: usize,
    /// Bricks whose last published state is a mesh.
    meshed: HashSet<BrickKey>,
    /// Bricks meshed at least once, with or without a result.
    visited: HashSet<BrickKey>,
    /// Round in which each stale brick was first seen stale.
    stale_since: HashMap<BrickKey, u64>,
    /// Meshing rounds so far.
    round: u64,
}

impl Geometry {
    pub(super) fn new() -> Self {
        Self {
            tsdf: Tsdf::new(),
            integrated: 0,
            cursor: 0,
            meshed: HashSet::new(),
            visited: HashSet::new(),
            stale_since: HashMap::new(),
            round: 0,
        }
    }

    /// Views to integrate this round: up to `REFRESH_VIEWS_PER_ROUND` older
    /// views, rotating, then up to `MAX_NEW_VIEWS_PER_ROUND` views not
    /// integrated yet, oldest first; the rest wait for later rounds.
    fn views_for_round(&mut self, num_views: usize) -> Vec<usize> {
        let older = self.integrated.min(num_views);
        let mut views = Vec::new();
        for _ in 0..REFRESH_VIEWS_PER_ROUND.min(older) {
            self.cursor %= older;
            views.push(self.cursor);
            self.cursor += 1;
        }
        let new_end = num_views.min(older + MAX_NEW_VIEWS_PER_ROUND);
        views.extend(older..new_end);
        self.integrated = new_end;
        views
    }

    /// Decays the TSDF and integrates this round's views, depth and colour;
    /// `splats` must have Gaussians. Returns the number of views integrated.
    pub(super) async fn fuse(
        &mut self,
        splats: &Splats,
        views: &[SceneView],
        sizes: &[UVec2],
    ) -> usize {
        self.tsdf.decay(WEIGHT_DECAY);
        let round = self.views_for_round(views.len());
        for &i in &round {
            let size = (sizes[i] / DEPTH_DOWNSCALE).max(UVec2::ONE);
            let camera = &views[i].camera;
            let depth = render_expected_depth(splats, camera, size).await;
            let colour = render_colour(splats, camera, size).await;
            self.tsdf.integrate(&depth, Some(&colour), camera);
        }
        round.len()
    }

    /// Meshes up to `MAX_BRICKS_PER_ROUND` stale bricks and marks those
    /// meshed; the rest stay stale for later rounds. A brick that had a
    /// mesh and has none now is returned as removed. Also returns how many
    /// bricks were stale.
    pub(super) fn mesh_changed(&mut self, eye: Vec3) -> (Vec<MeshBrick>, usize) {
        self.round += 1;
        let stale = self.tsdf.changed();
        let pending = stale.len();
        let chosen = self.select(stale, eye);
        let bricks = chosen.iter().filter_map(|&key| self.mesh(key)).collect();
        self.tsdf.mark_meshed(&chosen);
        (bricks, pending)
    }

    /// The first `MAX_BRICKS_PER_ROUND` of `stale`: bricks never meshed
    /// come first, then those stale for the most rounds, then those
    /// nearest to `eye`, so near bricks that keep changing cannot starve
    /// the others.
    fn select(&mut self, mut stale: Vec<BrickKey>, eye: Vec3) -> Vec<BrickKey> {
        let round = self.round;
        let current: HashSet<BrickKey> = stale.iter().copied().collect();
        self.stale_since.retain(|k, _| current.contains(k));
        for &k in &stale {
            self.stale_since.entry(k).or_insert(round);
        }
        let centre = |k: &BrickKey| ((k.0 * BRICK).as_vec3() + 0.5 * BRICK as f32) * VOXEL;
        stale.sort_by(|a, b| {
            let rank = |k: &BrickKey| (self.visited.contains(k), self.stale_since[k]);
            rank(a).cmp(&rank(b)).then_with(|| {
                centre(a)
                    .distance_squared(eye)
                    .total_cmp(&centre(b).distance_squared(eye))
            })
        });
        stale.truncate(MAX_BRICKS_PER_ROUND);
        for k in &stale {
            self.stale_since.remove(k);
            self.visited.insert(*k);
        }
        stale
    }

    /// Meshes brick `key`: its mesh, its removal if it had a mesh before,
    /// or nothing.
    fn mesh(&mut self, key: BrickKey) -> Option<MeshBrick> {
        match mesh_brick(&self.tsdf, key) {
            Some(mesh) => {
                self.meshed.insert(key);
                Some(MeshBrick::Mesh(mesh))
            }
            None => self.meshed.remove(&key).then_some(MeshBrick::Removed(key)),
        }
    }
}
