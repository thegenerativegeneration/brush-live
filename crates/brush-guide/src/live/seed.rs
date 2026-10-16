//! Seeding: splats for a new keyframe where the current model is transparent, from its depth, mono depth and
//! feature points, within the splat budget. With area training, seeds that land in parked tiles bring nearby tiles
//! back first or are dropped.

use super::LiveModel;
use super::area_restore::RestoreReason;
use crate::area::{SeedSort, TileKey};
use crate::keyframe::DecodedKeyframe;
use crate::mono::{MonoSeed, ScaleFit, mono_seed_for};
use crate::seed::{SeedInput, Seeds, seed_points, stride_for};
use brush_render::gaussian_splats::{SplatRenderMode, Splats, TextureMode, render_splats};
use brush_render::sh::rgb_to_sh;
use brush_serde::SplatData;
use brush_train::{RandomSplatsConfig, create_random_splats, to_init_splats};
use burn::tensor::s;
use glam::{UVec2, Vec3};
use std::collections::BTreeSet;

impl LiveModel {
    /// Keeps a seeded random subset of `seeds` that fits under `max_splats`
    /// next to `current` existing splats.
    /// Also marks the cells `seeds` fall in as newly observed for eviction.
    fn cap_seeds(&mut self, seeds: Seeds, current: u32) -> Seeds {
        if let Some(t) = self.trainer.as_mut() {
            t.note_keyframe_seeds(&seeds.means);
        }
        let room = self.config.max_splats.saturating_sub(current) as usize;
        if seeds.colors.len() <= room {
            return seeds;
        }
        if let Some(t) = self.trainer.as_mut() {
            t.note_seed_shortfall((seeds.colors.len() - room) as u32);
        }
        let mut keep = rand::seq::index::sample(&mut self.rng, seeds.colors.len(), room).into_vec();
        keep.sort_unstable();
        Seeds {
            means: keep
                .iter()
                .flat_map(|&i| seeds.means[i * 3..i * 3 + 3].iter().copied())
                .collect(),
            colors: keep.iter().map(|&i| seeds.colors[i]).collect(),
            mono: seeds.mono,
        }
    }

    /// The keyframe's mono seeding input; records its fit in the scale
    /// history, so call it exactly once per keyframe.
    pub(super) fn mono_input<'a>(
        &mut self,
        kf: &'a DecodedKeyframe,
        size: UVec2,
    ) -> (Option<MonoSeed<'a>>, ScaleFit) {
        let (seed, fit) = mono_seed_for(
            kf.mono.as_ref(),
            kf.depth.as_ref(),
            &kf.camera,
            size,
            &kf.points,
            &self.config,
            self.scale_history.median(),
        );
        self.scale_history.note(&fit);
        (seed, fit)
    }

    /// Seed grid spacing for a keyframe decoded at `size`.
    fn seed_stride(&self, size: UVec2) -> u32 {
        stride_for(
            size.x,
            size.y,
            self.config.max_splats,
            self.config.seed_view_fraction,
            self.config.seed_stride_px,
        )
    }

    pub(super) fn initial_splats(&mut self, kf: &DecodedKeyframe, size: UVec2) -> Splats {
        let (mono, fit) = self.mono_input(kf, size);
        let seeds = seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &vec![0.0; (size.x * size.y) as usize],
            alpha_size: size,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: self.seed_stride(size),
            alpha_threshold: self.config.seed_alpha_threshold,
            mono,
        });
        log::debug!(
            target: crate::timing::TARGET,
            "keyframe {}: {} initial seeds (depth {}, mono {fit}, {} mono seeds)",
            kf.id,
            seeds.colors.len(),
            kf.depth.is_some(),
            seeds.mono
        );
        self.refresh_active(&kf.camera, &seeds.means);
        let seeds = self.cap_seeds(seeds, 0);
        self.note_seeded(&seeds.means);
        let splats = if seeds.colors.len() >= 3 {
            self.seeds_to_splats(seeds.means, &seeds.colors)
        } else {
            let count = self
                .config
                .init_random_count
                .min(self.config.max_splats as usize);
            let cfg = RandomSplatsConfig::new().with_init_count(count);
            create_random_splats(
                &cfg,
                &[kf.camera],
                None,
                &mut self.rng,
                SplatRenderMode::Default,
                &self.device.clone().inner(),
            )
        };
        splats.with_sh_degree(self.config.sh_degree)
    }

    /// Seeds the keyframe where `splats` render transparent. With area training, parked tiles its seeds hit come
    /// back first ([`Self::triage_parked`]), so the returned splats may hold restored rows; the new splats come
    /// second, `None` with fewer than 3 seeds.
    pub(super) async fn seed_from_mask(
        &mut self,
        splats: Splats,
        kf: &DecodedKeyframe,
        size: UVec2,
    ) -> (Splats, Option<Splats>) {
        let (mono, fit) = self.mono_input(kf, size);
        let seeds = self.seeds_over(&splats, kf, size, mono).await;
        log::debug!(
            target: crate::timing::TARGET,
            "keyframe {}: {} seeds (depth {}, mono {fit}, {} mono seeds) onto {} splats",
            kf.id,
            seeds.colors.len(),
            kf.depth.is_some(),
            seeds.mono,
            splats.num_splats()
        );
        self.refresh_active(&kf.camera, &seeds.means);
        let (splats, seeds) = self.triage_parked(splats, seeds, kf, size, mono).await;
        let seeds = self.cap_seeds(seeds, splats.num_splats());
        self.note_seeded(&seeds.means);
        let new = (seeds.colors.len() >= 3).then(|| {
            self.seeds_to_splats(seeds.means, &seeds.colors)
                .with_sh_degree(self.config.sh_degree)
        });
        (splats, new)
    }

    /// The keyframe's seeds where `splats` render transparent at a quarter of its size.
    async fn seeds_over(
        &self,
        splats: &Splats,
        kf: &DecodedKeyframe,
        size: UVec2,
        mono: Option<MonoSeed<'_>>,
    ) -> Seeds {
        let small = (size / 4).max(UVec2::ONE);
        let (img, _) = render_splats(
            splats.clone(),
            &kf.camera,
            small,
            Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;
        let alpha = img
            .slice(s![.., .., 3..4])
            .into_data_async()
            .await
            .expect("alpha readback")
            .try_to_vec::<f32>()
            .expect("f32 alpha");
        seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &alpha,
            alpha_size: small,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: (self.seed_stride(size) / 4).max(1),
            alpha_threshold: self.config.seed_alpha_threshold,
            mono,
        })
    }

    /// With area training, sorts the keyframe's seeds against the parked tiles: tiles beyond the radius (the tiles
    /// within it were already tried by distance) that at least `area_restore_min_seeds` seeds hit within the radius
    /// plus `area_far_seed_m` come back ([`RestoreReason::Seeds`]), and when one did the keyframe seeds again over
    /// them. Seeds that land in a parked tile, as the registry stands after that, are dropped, and with `area_freeze`
    /// inactive so are seeds in tiles with rows outside the active set ([`Self::keep_active`]). Without area training
    /// the seeds pass unchanged.
    async fn triage_parked(
        &mut self,
        splats: Splats,
        seeds: Seeds,
        kf: &DecodedKeyframe,
        size: UVec2,
        mono: Option<MonoSeed<'_>>,
    ) -> (Splats, Seeds) {
        let Some(area) = self.area.as_ref() else {
            return (splats, seeds);
        };
        let pos = kf.camera.position;
        let min_seeds = self.config.area_restore_min_seeds as usize;
        let mut triage = area.registry.triage_seeds(&seeds.means, pos, min_seeds);
        let tried: BTreeSet<TileKey> = area.registry.parked_within_radius(pos).into_iter().collect();
        triage.restore.retain(|k| !tried.contains(k));
        if !triage.restore.is_empty() {
            let restores = area.counters.restores;
            let splats = self.restore_tiles(splats, &triage.restore, RestoreReason::Seeds, pos).await;
            let restored = self.area.as_ref().is_some_and(|a| a.counters.restores > restores);
            let seeds = if restored {
                let second = self.seeds_over(&splats, kf, size, mono).await;
                let counts = (seeds.colors.len() as u32, second.colors.len() as u32);
                log::info!("area: reseed {} -> {} seeds", counts.0, counts.1);
                self.area.as_mut().expect("area training is on").last_reseed = Some(counts);
                // The restored tiles may be in view now: rank them.
                self.refresh_active(&kf.camera, &second.means);
                second
            } else {
                seeds
            };
            // The restore may have parked tiles for room or dropped tiles: sort the seeds against the registry as
            // it is now (a dropped tile's area takes seeds again).
            let registry = &self.area.as_ref().expect("area training is on").registry;
            let mut keep = registry.triage_seeds(&seeds.means, pos, usize::MAX).keep;
            let sort = self.keep_active(&seeds.means, &mut keep);
            return (splats, kept_logged(seeds, &keep, sort));
        }
        let mut keep = triage.keep;
        let sort = self.keep_active(&seeds.means, &mut keep);
        (splats, kept_logged(seeds, &keep, sort))
    }

    fn seeds_to_splats(&self, means: Vec<f32>, colors: &[Vec3]) -> Splats {
        let sh: Vec<f32> = colors
            .iter()
            .flat_map(|c| rgb_to_sh(*c).to_array())
            .collect();
        let data = SplatData {
            means,
            rotations: None,
            log_scales: None,
            sh_coeffs: Some(sh),
            raw_opacities: None,
        };
        to_init_splats(data, SplatRenderMode::Default, &self.device.clone().inner())
    }
}

/// The seeds `keep` flags (see [`keep_seeds`]); logs how many were dropped, `sort.dropped` of them for landing in
/// tiles outside the active set, the rest in parked or left tiles.
fn kept_logged(seeds: Seeds, keep: &[bool], sort: SeedSort) -> Seeds {
    let kept = keep_seeds(seeds, keep);
    let dropped = keep.len() - kept.colors.len();
    if dropped > 0 {
        log::info!("area: dropped {dropped} seeds ({} outside the active tiles)", sort.dropped);
    }
    kept
}

/// The seeds `keep` flags; the mono count is clamped to what is left, since which seeds came from mono depth is not
/// recorded.
fn keep_seeds(seeds: Seeds, keep: &[bool]) -> Seeds {
    let means: Vec<f32> = seeds
        .means
        .chunks_exact(3)
        .zip(keep)
        .filter(|(_, k)| **k)
        .flat_map(|(m, _)| m.iter().copied())
        .collect();
    let colors: Vec<Vec3> = seeds.colors.iter().zip(keep).filter(|(_, k)| **k).map(|(c, _)| *c).collect();
    Seeds { mono: seeds.mono.min(colors.len()), means, colors }
}
