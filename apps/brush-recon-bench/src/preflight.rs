use crate::cli::PreflightArgs;
use crate::io;
use brush_recon::{
    ReconError, RenderRequest,
    capture::{
        AssetReference, CaptureManifest, CaptureSelection, CaptureSplit, FrozenAssets,
        load_capture, mask::rasterize_roi,
    },
};
use brush_triangle_render::{
    TriangleSeed,
    reference::{ReferenceConfig, render_reference},
};
use brush_triangle_train::{SeedConfig, TrainConfig, seed_from_training};
use glam::{UVec2, Vec3};
use image::{ImageEncoder, RgbImage};
use serde::Serialize;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Serialize)]
pub struct PreflightSummary {
    pub manifest: PathBuf,
    pub train_views: usize,
    pub holdout_views: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub training_support: Vec<ViewSupport>,
    pub nonanchor_median_relative_depth_error: f32,
}
#[derive(Debug, Serialize)]
pub struct ViewSupport {
    pub frame_id: u64,
    pub roi_pixels: usize,
    pub total_pixels: usize,
    pub native_depth_pairs: usize,
    pub median_relative_depth_error: Option<f32>,
}
pub struct FrozenInput {
    pub split: CaptureSplit,
    pub seed: TriangleSeed,
    pub config: TrainConfig,
}

fn value<T: serde::Serialize>(v: &T) -> Result<Value, ReconError> {
    serde_json::to_value(v)
        .map_err(|e| ReconError::input(format!("configuration serialization: {e}")))
}
fn median(v: &mut [f32]) -> Option<f32> {
    if v.is_empty() {
        None
    } else {
        v.sort_by(f32::total_cmp);
        let i = v.len() / 2;
        Some(if v.len() % 2 == 0 {
            (v[i - 1] + v[i]) * 0.5
        } else {
            v[i]
        })
    }
}

pub fn save_rgb(path: &Path, image: &RgbImage) -> Result<(), ReconError> {
    let mut bytes = vec![];
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| ReconError::input(format!("{}: {e}", path.display())))?;
    io::write_new(path, &bytes)
}

pub fn freeze_roi(split: &mut CaptureSplit, seed: &TriangleSeed) -> Result<(), ReconError> {
    for frame in split.train.iter_mut().chain(split.holdout.iter_mut()) {
        let f = Arc::make_mut(frame);
        f.roi = Arc::new(rasterize_roi(
            &seed.positions,
            &seed.faces,
            &f.camera,
            UVec2::new(f.rgb.width(), f.rgb.height()),
            0.05,
        )?);
        f.validate()?;
    }
    Ok(())
}

pub fn preflight(args: &PreflightArgs, environment: Value) -> Result<PreflightSummary, ReconError> {
    let output = io::validate_output(&args.segment, &args.output)?;
    if !(4..=4096).contains(&args.width) {
        return Err(ReconError::input("preflight width must be 4..4096"));
    }
    let selection = CaptureSelection {
        first_id: args.first_id,
        last_id: args.last_id,
        holdout_every: args.holdout_every,
        holdout_offset: args.holdout_offset,
        size: UVec2::new(args.width, args.width * 3 / 4),
    };
    let mut split = load_capture(&args.segment, &selection)?;
    let anchors = [0_u64, 10, 20, 30]
        .into_iter()
        .map(|n| {
            args.first_id
                .checked_add(n)
                .ok_or_else(|| ReconError::input("anchor ID overflow"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let seed_config = SeedConfig {
        anchor_ids: anchors,
        ..Default::default()
    };
    let seed = seed_from_training(&split.train, &seed_config)?;
    freeze_roi(&mut split, &seed)?;
    let logits: Vec<_> = seed
        .colors
        .iter()
        .map(|c| {
            c.map(|v| {
                let c = v.clamp(1e-4, 1. - 1e-4);
                (c / (1. - c)).ln()
            })
        })
        .collect();
    let weights = vec![(0.1_f32 / 0.9).ln(); seed.positions.len()];
    let mut support = vec![];
    let mut nonanchor = vec![];
    let mut depth_previews = vec![];
    for frame in &split.train {
        let roi_pixels = frame.roi.iter().filter(|v| **v == 255).count();
        if roi_pixels == 0 {
            return Err(ReconError::input(format!(
                "training frame {} has no projected seed support; do not silently select a new window",
                frame.id
            )));
        }
        let depth = frame
            .lidar
            .as_ref()
            .ok_or_else(|| ReconError::input(format!("training frame {} lacks LiDAR", frame.id)))?;
        let request = RenderRequest {
            camera: frame.camera,
            size: depth.size,
            background: Vec3::ZERO,
            depth: true,
            normals: false,
        };
        let rendered = render_reference(
            &seed,
            &logits,
            &weights,
            &request,
            &ReferenceConfig::default(),
        )?;
        let mut relative = vec![];
        for i in 0..depth.values_m.len() {
            let z = depth.values_m[i];
            let confidence = depth.confidence.as_ref().map_or(0, |c| c[i]);
            if confidence == 2
                && (0.1..=8.).contains(&z)
                && rendered.alpha[i] > 1e-6
                && rendered.depth_m[i] > 0.
            {
                relative.push((rendered.depth_m[i] - z).abs() / z);
            }
        }
        let pairs = relative.len();
        let med = median(&mut relative);
        if !seed_config.anchor_ids.contains(&frame.id) {
            if let Some(m) = med {
                nonanchor.push(m);
            }
        }
        let bytes = rendered
            .depth_m
            .iter()
            .zip(&rendered.alpha)
            .flat_map(|(&z, &a)| {
                if a <= 1e-6 {
                    [0, 0, 0]
                } else {
                    let t = ((z - 0.1) / 7.9).clamp(0., 1.);
                    [(t * 255.) as u8, ((1. - t) * 255.) as u8, 80]
                }
            })
            .collect();
        depth_previews.push((
            frame.id,
            RgbImage::from_raw(depth.size.x, depth.size.y, bytes)
                .ok_or_else(|| ReconError::input("depth preview shape"))?,
        ));
        support.push(ViewSupport {
            frame_id: frame.id,
            roi_pixels,
            total_pixels: frame.roi.len(),
            native_depth_pairs: pairs,
            median_relative_depth_error: med,
        });
    }
    let agreement = median(&mut nonanchor).ok_or_else(|| {
        ReconError::input("no non-anchor training LiDAR overlaps initial geometry")
    })?;
    if agreement > 0.15 {
        return Err(ReconError::input(format!(
            "training reprojection median relative depth error {agreement:.3} exceeds 0.15; inspect pose/depth alignment before proceeding"
        )));
    }
    let config = TrainConfig::default();
    let (revision, dirty) = io::provenance()?;
    io::create_output(&output)?;
    let seed_bytes =
        serde_json::to_vec(&seed).map_err(|e| ReconError::input(format!("seed: {e}")))?;
    io::write_new(&output.join("seed.json"), &seed_bytes)?;
    let mut roi_assets = std::collections::BTreeMap::new();
    for frame in split.train.iter().chain(&split.holdout) {
        let name = PathBuf::from(format!("roi-{}.bin", frame.id));
        io::write_new(&output.join(&name), &frame.roi)?;
        roi_assets.insert(
            frame.id,
            AssetReference {
                file: name,
                sha256: io::digest(&frame.roi),
            },
        );
    }
    for frame in &split.train {
        let mut overlay = (*frame.rgb).clone();
        for (i, p) in overlay.pixels_mut().enumerate() {
            if frame.roi[i] == 0 {
                p.0 = p.0.map(|c| c / 4);
            } else {
                p.0[1] = (p.0[1] as f32 * 0.7 + 255. * 0.3) as u8;
            }
        }
        save_rgb(
            &output.join(format!("training-{}-roi.png", frame.id)),
            &overlay,
        )?;
    }
    for (id, image) in depth_previews {
        save_rgb(
            &output.join(format!("training-{id}-initial-depth.png")),
            &image,
        )?;
    }
    split.manifest.frozen = Some(FrozenAssets {
        seed: AssetReference {
            file: "seed.json".into(),
            sha256: io::digest(&seed_bytes),
        },
        roi: roi_assets,
        seed_config: value(&seed_config)?,
        loss_config: value(&config.loss)?,
        schedule_config: value(&config.schedule)?,
        render_config: value(&config.render)?,
        triangle_train_config: value(&config)?,
        gaussian_train_config: value(&crate::gaussian::controlled_config())?,
        rng_seed: args.seed,
        git_revision: revision,
        dirty_patch_sha256: dirty,
        environment,
    });
    let summary = PreflightSummary {
        manifest: output.join("manifest.json"),
        train_views: split.train.len(),
        holdout_views: split.holdout.len(),
        vertices: seed.positions.len(),
        triangles: seed.faces.len(),
        training_support: support,
        nonanchor_median_relative_depth_error: agreement,
    };
    io::json(&output.join("preflight.json"), &summary)?;
    verify_sources(&split.manifest)?;
    io::json(&summary.manifest, &split.manifest)?; // Completion marker is written last.
    Ok(summary)
}

pub fn verify_sources(manifest: &CaptureManifest) -> Result<(), ReconError> {
    for (name, expected) in &manifest.source_sha256 {
        let path = Path::new(name);
        if path.is_absolute()
            || path
                .components()
                .any(|p| !matches!(p, std::path::Component::Normal(_)))
        {
            return Err(ReconError::input("unsafe source path in manifest"));
        }
        let actual = manifest.source.join(path);
        let canonical = actual
            .canonicalize()
            .map_err(|e| ReconError::io(&actual, e))?;
        if !canonical.starts_with(&manifest.source) || io::digest(&io::read(&actual)?) != *expected
        {
            return Err(ReconError::input(format!(
                "frozen source changed: {}",
                actual.display()
            )));
        }
    }
    Ok(())
}

pub fn load_frozen(path: &Path) -> Result<FrozenInput, ReconError> {
    let manifest: CaptureManifest = io::read_json(path)?;
    if manifest.format_version != 1 || manifest.source_kind != "arkit_raw" {
        return Err(ReconError::input(
            "unsupported frozen capture version/source",
        ));
    }
    let assets = manifest
        .frozen
        .as_ref()
        .ok_or_else(|| ReconError::input("manifest is not frozen; run preflight first"))?;
    verify_sources(&manifest)?;
    let root = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let seed: TriangleSeed = serde_json::from_slice(&io::asset(root, &assets.seed)?)
        .map_err(|e| ReconError::input(format!("frozen seed: {e}")))?;
    seed.validate()?;
    let mut split = load_capture(&manifest.source, &manifest.selection)?;
    if split.manifest.source_sha256 != manifest.source_sha256
        || split.manifest.train_ids != manifest.train_ids
        || split.manifest.holdout_ids != manifest.holdout_ids
    {
        return Err(ReconError::input("frozen split/source identity changed"));
    }
    let selection_count = split.train.len() + split.holdout.len();
    if assets.roi.len() != selection_count {
        return Err(ReconError::input(
            "frozen ROI manifest has missing/extra frame IDs",
        ));
    }
    for frame in split.train.iter_mut().chain(split.holdout.iter_mut()) {
        let roi = assets
            .roi
            .get(&frame.id)
            .ok_or_else(|| ReconError::input("frozen ROI missing frame"))?;
        let mask = io::asset(root, roi)?;
        let expected = rasterize_roi(
            &seed.positions,
            &seed.faces,
            &frame.camera,
            manifest.selection.size,
            0.05,
        )?;
        if mask != expected {
            return Err(ReconError::input(format!(
                "frame {}: frozen ROI differs from initial geometry",
                frame.id
            )));
        }
        Arc::make_mut(frame).roi = Arc::new(mask);
        frame.validate()?;
    }
    let seed_config: SeedConfig = serde_json::from_value(assets.seed_config.clone())
        .map_err(|e| ReconError::input(format!("seed config: {e}")))?;
    if seed_from_training(&split.train, &seed_config)?.fingerprint()? != seed.fingerprint()? {
        return Err(ReconError::input(
            "frozen seed is not the training-only initialization",
        ));
    }
    let config: TrainConfig = serde_json::from_value(assets.triangle_train_config.clone())
        .map_err(|e| ReconError::input(format!("triangle config: {e}")))?;
    if value(&config.loss)? != assets.loss_config
        || value(&config.schedule)? != assets.schedule_config
        || value(&config.render)? != assets.render_config
        || value(&crate::gaussian::controlled_config())? != assets.gaussian_train_config
    {
        return Err(ReconError::input(
            "frozen backend configurations differ from recorded component/control configurations",
        ));
    }
    split.manifest = manifest;
    Ok(FrozenInput {
        split,
        seed,
        config,
    })
}
