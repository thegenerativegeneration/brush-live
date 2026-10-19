//! The training loaders over the training views (see [`LiveModel::training_views`]): a recent-window loader over the
//! newest ones and an all-views loader over all of them, and when each is rebuilt.

use super::LiveModel;
use crate::area::membership_additions;
use brush_dataset::scene::Scene;
use brush_dataset::scene_loader::SceneLoader;

/// Recent-window loader is rebuilt every this many new keyframes (and for each
/// of the first ones): every rebuild spawns a fresh set of loader threads.
const RECENT_LOADER_REBUILD_EVERY: usize = 3;

impl LiveModel {
    /// Bytes both loaders' caches hold.
    pub fn loader_cache_bytes(&self) -> u64 {
        [&self.recent, &self.all]
            .iter()
            .filter_map(|l| l.as_ref())
            .map(SceneLoader::cached_bytes)
            .sum()
    }

    /// Training views added since the recent loader was built, ascending: a recent-window draw trains on them
    /// directly until the loader is rebuilt.
    pub(super) fn pending_views(&self) -> Vec<usize> {
        self.training_views_from(self.views_in_recent)
    }

    /// Builds both loaders from the current training views now: their prefetched batches and cached images, which
    /// carry the camera they were built with, are dropped. The seeds take the step count in, so the new loaders do
    /// not replay the shuffle order the old ones started with.
    pub(super) fn rebuild_all_loaders(&mut self) {
        let salt = u64::from(self.iter) << 32;
        self.build_recent_loader(salt);
        self.build_all_loader(salt);
    }

    /// After a new view: rebuilds the recent loader every `RECENT_LOADER_REBUILD_EVERY` views (each of the first
    /// ones) or when one of its views left the area, and the all-views loader when `all_loader_rebuild_every` views
    /// joined or left its training views. With area training off no view leaves, so these are the count rules alone.
    pub(super) fn rebuild_loaders(&mut self) {
        let n = self.views.len();
        let training = self.training_views();
        if n <= RECENT_LOADER_REBUILD_EVERY
            || n >= self.views_in_recent + RECENT_LOADER_REBUILD_EVERY
            || membership_additions(&training, &self.recent_members) > 0
        {
            self.build_recent_loader(0);
        }
        let every = self.config.all_loader_rebuild_every;
        if self.all.is_none()
            || membership_additions(&self.all_members, &training) >= every
            || membership_additions(&training, &self.all_members) >= every
        {
            self.build_all_loader(0);
        }
    }

    /// Rebuilds the all-views loader when it holds views that left the area, which the count rule in
    /// [`Self::rebuild_loaders`] tolerates; after a park, so no such view keeps training toward the tiles it saw.
    pub(super) fn drop_departed_views(&mut self) {
        if membership_additions(&self.training_views(), &self.all_members) > 0 {
            self.build_all_loader(u64::from(self.iter) << 32);
        }
    }

    /// The recent-window loader over the newest `recent_window` training views; `salt` 0 gives the usual seed.
    fn build_recent_loader(&mut self, salt: u64) {
        let n = self.views.len();
        let mut members = self.training_views();
        members.drain(..members.len().saturating_sub(self.config.recent_window));
        let recent = Scene::new(members.iter().map(|&i| self.views[i].clone()).collect());
        self.recent = Some(SceneLoader::new(
            &recent,
            (self.config.seed + n as u64) ^ salt,
            &self.load_config,
        ));
        self.recent_members = members;
        self.views_in_recent = n;
    }

    /// The loader over all training views; `salt` 0 gives the usual seed.
    fn build_all_loader(&mut self, salt: u64) {
        let members = self.training_views();
        let all = Scene::new(members.iter().map(|&i| self.views[i].clone()).collect());
        self.all = Some(SceneLoader::new(
            &all,
            self.config.seed ^ salt,
            &self.load_config,
        ));
        self.all_members = members;
        self.all_loader_builds += 1;
    }
}
