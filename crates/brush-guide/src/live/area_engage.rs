//! When area training engages (`area_engage_on_leave`): once the walker has left a tile (a visited tile lies beyond
//! radius + hysteresis of the newest camera, [`crate::area::TileRegistry::left_keys`]) or a tile parked, and from then
//! on until the session resets. Until then the engine trains as with area training off: every view trains and floors
//! the splats, every view's camera votes in the score round, nothing is held or parked, and seeds, growth and
//! eviction see no per-tile cap, row budget or quota. The active set is still ranked and reported.

use super::LiveModel;

impl LiveModel {
    /// Whether area training is on and engaged.
    pub(super) fn area_engaged(&self) -> bool {
        self.area.as_ref().is_some_and(|a| a.engaged)
    }

    /// Engages area training once the walker at the newest camera has left a tile or a tile is parked; called after
    /// the walker is noted at a keyframe and after a pose update. The switch marks the mask stale, recomputes the
    /// training views and the floor's cameras and rebuilds both loaders, or during a keyframe (its view not yet in)
    /// leaves that to the keyframe's own rebuild. Logged and recorded with the keyframes received. No-op once engaged
    /// or without area training. Returns whether it recomputed the views and rebuilt the loaders.
    pub(super) fn note_engaged(&mut self) -> bool {
        let Some(&(centre, _)) = self.view_cams.last() else {
            return false;
        };
        let received = self.view_cams.len() as u32;
        let Some(area) = self.area.as_mut().filter(|a| !a.engaged) else {
            return false;
        };
        let left = area.registry.left_keys(centre).len();
        let parked = area.registry.parked_keys().len();
        if left == 0 && parked == 0 {
            return false;
        }
        area.engaged = true;
        area.counters.engaged_at_keyframe = received;
        area.mask_stale = true;
        log::info!("area: training engages at keyframe {received} ({left} tiles left, {parked} parked)");
        if self.views.len() != self.view_cams.len() {
            area.loaders_stale = true;
            return false;
        }
        self.refresh_area_views();
        if !self.views.is_empty() {
            self.refresh_floor_cams();
            self.rebuild_all_loaders();
        }
        true
    }
}

#[cfg(test)]
#[path = "area_engage_tests.rs"]
mod tests;
