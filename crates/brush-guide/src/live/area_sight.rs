//! The views that see the resident model with area training on: the views that see a resident tile's box up to
//! [`SIGHT_FAR_M`] deep ([`crate::area::resident_boxes`], left tiles included), the area's training views and the
//! newest view. Only these views' cameras floor the splats (Mip-Splatting 3D filter), so a camera that sees none of
//! them never sets a resident splat's floor. Parked rows keep the floor they were parked with, and held rows keep
//! theirs through a refine.
//!
//! Views are bucketed by pose ([`crate::area::buckets`]): the frustum tests run on one representative per bucket and
//! every view of the bucket takes its result, and only the representatives (with the newest view) orient the score
//! round's normals, so that vote stops growing where the walk passes the same place again.

use super::LiveModel;
use crate::area::buckets::representatives;
use crate::area::{resident_boxes, views_seeing, widen_all};
use brush_render::camera::Camera;
use glam::Vec3;
use std::collections::BTreeSet;

/// How deep a view sees a resident tile for the floor and the score round, metres.
pub const SIGHT_FAR_M: f32 = 100.0;

/// The views that see the resident model, and the voters among them.
#[derive(Debug, Default)]
pub(super) struct Sight {
    /// The views that see the resident model, ascending.
    pub(super) views: Vec<usize>,
    /// The views among `views` that vote in the score round: one per pose bucket, and the newest view.
    pub(super) reps: Vec<usize>,
}

impl LiveModel {
    /// Recomputes the views that see the resident model from the current poses: the views of every pose bucket whose
    /// representative sees a resident tile's box (height bands of the last readback widened by the rows seeded since)
    /// up to [`SIGHT_FAR_M`], the area's views and the newest view; and the voters among them, each such bucket's
    /// representative, the representatives of the area's views and the newest view. Before area training engages
    /// ([`super::area_engage`]) both are every view, as with area training off. No-op with area training off.
    pub(super) fn refresh_sight(&mut self) {
        let n = self.views.len();
        let Some(area) = self.area.as_mut() else {
            return;
        };
        if !area.engaged {
            area.sight.views = (0..n).collect();
            area.sight.reps = (0..n).collect();
            return;
        }
        let mut heights = area.heights.last.clone();
        widen_all(&mut heights, &area.heights.fresh);
        let boxes = resident_boxes(&area.registry, &heights);
        let cameras: Vec<Camera> = self.views.iter().map(|v| v.camera).collect();
        let focals: Vec<f32> = self.view_cams.iter().take(cameras.len()).map(|(_, f)| *f).collect();
        let rep_of = representatives(&cameras, &focals);
        let reps: Vec<usize> = rep_of.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
        let rep_cameras: Vec<Camera> = reps.iter().map(|&i| cameras[i]).collect();
        let seeing: BTreeSet<usize> =
            views_seeing(&rep_cameras, &boxes, SIGHT_FAR_M).into_iter().map(|k| reps[k]).collect();
        let newest = cameras.len().checked_sub(1);
        let area_views = area.views.iter().copied().filter(|&i| i < cameras.len());
        let mut sight: BTreeSet<usize> = (0..cameras.len()).filter(|&i| seeing.contains(&rep_of[i])).collect();
        sight.extend(area_views.clone().chain(newest));
        let mut voters = seeing;
        voters.extend(area_views.map(|i| rep_of[i]).chain(newest));
        area.sight.views = sight.into_iter().collect();
        area.sight.reps = voters.into_iter().collect();
    }

    /// The floor's cameras: every view's with area training off; otherwise those of the views that see the resident
    /// model and of a keyframe being added (in `view_cams`, not yet in `views`).
    fn floor_cams(&self) -> Vec<(Vec3, f32)> {
        let cams = match &self.area {
            None => self.view_cams.clone(),
            Some(a) => {
                let adding = self.views.len().min(self.view_cams.len())..self.view_cams.len();
                a.sight.views.iter().copied().chain(adding).filter_map(|i| self.view_cams.get(i).copied()).collect()
            }
        };
        debug_assert!(self.view_cams.is_empty() || !cams.is_empty(), "views exist but no camera floors the splats");
        cams
    }

    /// Hands the trainer the floor's cameras ([`Self::floor_cams`]).
    pub(super) fn push_floor_cams(&mut self) {
        let cams = self.floor_cams();
        if let Some(t) = self.trainer.as_mut() {
            t.set_view_cams(cams);
        }
    }

    /// [`Self::refresh_sight`], then [`Self::push_floor_cams`].
    pub(super) fn refresh_floor_cams(&mut self) {
        self.refresh_sight();
        self.push_floor_cams();
    }

    /// The cameras the score round orients normals by: every view's with area training off, otherwise those of the
    /// voters among the views that see the resident model (one per pose bucket, and the newest view).
    pub fn sight_cameras(&self) -> Vec<Camera> {
        match &self.area {
            None => self.views.iter().map(|v| v.camera).collect(),
            Some(a) => a.sight.reps.iter().filter_map(|&i| self.views.get(i)).map(|v| v.camera).collect(),
        }
    }

    /// Views that see the resident model; `None` with area training off.
    pub fn sight_view_count(&self) -> Option<u32> {
        self.area.as_ref().map(|a| a.sight.views.len() as u32)
    }

    /// Voters of the score round among them ([`Self::sight_cameras`]); `None` with area training off.
    pub fn sight_voter_count(&self) -> Option<u32> {
        self.area.as_ref().map(|a| a.sight.reps.len() as u32)
    }
}

#[cfg(test)]
#[path = "area_sight_tests.rs"]
mod tests;
