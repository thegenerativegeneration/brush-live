//! The views that see the resident model with area training on: the views that see a resident tile's box up to
//! [`SIGHT_FAR_M`] deep ([`crate::area::resident_boxes`], left tiles included), and the area's training views. Only
//! their cameras floor the splats (Mip-Splatting 3D filter) and orient the score round's normals, so neither grows
//! with the length of the walk. Parked rows keep the floor they were parked with, and held rows keep theirs through a
//! refine, so the views that drop out never move a floor they no longer see.

use super::LiveModel;
use crate::area::{resident_boxes, views_seeing, widen_all};
use brush_render::camera::Camera;
use glam::Vec3;

/// How deep a view sees a resident tile for the floor and the score round, metres.
pub const SIGHT_FAR_M: f32 = 100.0;

impl LiveModel {
    /// Recomputes the views that see the resident model: the views that see a resident tile's box (height bands of
    /// the last readback widened by the rows seeded since) up to [`SIGHT_FAR_M`], and the area's views. No-op with
    /// area training off.
    pub(super) fn refresh_sight(&mut self) {
        let Some(area) = self.area.as_mut() else {
            return;
        };
        let mut heights = area.heights.clone();
        widen_all(&mut heights, &area.fresh);
        let boxes = resident_boxes(&area.registry, &heights);
        let cameras: Vec<Camera> = self.views.iter().map(|v| v.camera).collect();
        let mut sight = views_seeing(&cameras, &boxes, SIGHT_FAR_M);
        sight.extend(area.views.iter().copied().filter(|&i| i < cameras.len()));
        sight.sort_unstable();
        sight.dedup();
        area.sight = sight;
    }

    /// The floor's cameras: every view's with area training off; otherwise those of the views that see the resident
    /// model and of a keyframe being added (in `view_cams`, not yet in `views`).
    fn floor_cams(&self) -> Vec<(Vec3, f32)> {
        match &self.area {
            None => self.view_cams.clone(),
            Some(a) => {
                let adding = self.views.len().min(self.view_cams.len())..self.view_cams.len();
                a.sight.iter().copied().chain(adding).filter_map(|i| self.view_cams.get(i).copied()).collect()
            }
        }
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
    /// views that see the resident model.
    pub fn sight_cameras(&self) -> Vec<Camera> {
        match &self.area {
            None => self.views.iter().map(|v| v.camera).collect(),
            Some(a) => a.sight.iter().filter_map(|&i| self.views.get(i)).map(|v| v.camera).collect(),
        }
    }

    /// Views that see the resident model; `None` with area training off.
    pub fn sight_view_count(&self) -> Option<u32> {
        self.area.as_ref().map(|a| a.sight.len() as u32)
    }
}

#[cfg(test)]
#[path = "area_sight_tests.rs"]
mod tests;
