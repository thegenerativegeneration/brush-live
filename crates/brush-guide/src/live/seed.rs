//! Seeding: splats for a new keyframe where the current model is transparent, from its depth, mono depth and
//! feature points, within the splat budget.

use super::LiveModel;
use crate::keyframe::DecodedKeyframe;
use crate::mono::{MonoSeed, ScaleFit, mono_seed_for};
use crate::seed::{SeedInput, Seeds, seed_points, stride_for};
use brush_render::gaussian_splats::{SplatRenderMode, Splats, TextureMode, render_splats};
use brush_render::sh::rgb_to_sh;
use brush_serde::SplatData;
use brush_train::{RandomSplatsConfig, create_random_splats, to_init_splats};
use burn::tensor::s;
use glam::{UVec2, Vec3};

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
        let seeds = self.cap_seeds(seeds, 0);
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

    pub(super) async fn seed_from_mask(
        &mut self,
        splats: Splats,
        kf: &DecodedKeyframe,
        size: UVec2,
    ) -> Option<Splats> {
        let current = splats.num_splats();
        let small = (size / 4).max(UVec2::ONE);
        let (img, _) = render_splats(
            splats,
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
        let (mono, fit) = self.mono_input(kf, size);
        let seeds = seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &alpha,
            alpha_size: small,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: (self.seed_stride(size) / 4).max(1),
            alpha_threshold: self.config.seed_alpha_threshold,
            mono,
        });
        log::debug!(
            target: crate::timing::TARGET,
            "keyframe {}: {} seeds (depth {}, mono {fit}, {} mono seeds) onto {current} splats",
            kf.id,
            seeds.colors.len(),
            kf.depth.is_some(),
            seeds.mono
        );
        let seeds = self.cap_seeds(seeds, current);
        (seeds.colors.len() >= 3).then(|| {
            self.seeds_to_splats(seeds.means, &seeds.colors)
                .with_sh_degree(self.config.sh_degree)
        })
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
