//! Inspection artifacts. Image encoding clips for display only; metrics never clip predictions.
use crate::{
    io,
    metrics::{Evaluation, research_gate},
    preflight::save_rgb,
    run::RunRecord,
};
use brush_recon::{BackendKind, ReconError, capture::CaptureManifest};
use brush_triangle_train::artifact::{decode_state, mesh_ply};
use image::RgbImage;
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct ProcessFootprint {
    pub current_bytes: Option<u64>,
    pub lifetime_peak_bytes: Option<u64>,
    pub note: String,
}
pub fn process_footprint() -> ProcessFootprint {
    #[cfg(target_os = "macos")]
    {
        // Same read-only OS accounting used by brush-guide-bench; no renderer-only claim.
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let code = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                (&raw mut info).cast::<libc::rusage_info_t>(),
            )
        };
        if code == 0 && info.ri_phys_footprint > 0 {
            return ProcessFootprint {current_bytes:Some(info.ri_phys_footprint),lifetime_peak_bytes:(info.ri_lifetime_max_phys_footprint>0).then_some(info.ri_lifetime_max_phys_footprint),note:"Measured process footprint; lifetime peak includes warmup, evaluation, artifacts and runtime, not just renderer/GPU buffers. Unavailable values are null.".into()};
        }
    }
    ProcessFootprint {
        current_bytes: None,
        lifetime_peak_bytes: None,
        note: "Process physical footprint unavailable on this platform or accounting call failed"
            .into(),
    }
}

fn rgb_image(size: glam::UVec2, rgb: &[f32]) -> Result<RgbImage, ReconError> {
    if rgb.iter().any(|v| !v.is_finite()) {
        return Err(ReconError::ModelState(
            "cannot encode nonfinite RGB artifact".into(),
        ));
    }
    let bytes = rgb
        .iter()
        .map(|v| (v.clamp(0., 1.) * 255.).round() as u8)
        .collect();
    RgbImage::from_raw(size.x, size.y, bytes)
        .ok_or_else(|| ReconError::input("RGB artifact dimensions do not match"))
}
fn depth_image(
    size: glam::UVec2,
    depth: &[f32],
    valid: impl Fn(usize) -> bool,
) -> Result<RgbImage, ReconError> {
    let bytes = depth
        .iter()
        .enumerate()
        .flat_map(|(i, &z)| {
            if !valid(i) || !z.is_finite() || z <= 0. {
                [0, 0, 0]
            } else {
                let t = ((z - 0.1) / 7.9).clamp(0., 1.);
                [
                    (t * 255.).round() as u8,
                    ((1. - t) * 255.).round() as u8,
                    80,
                ]
            }
        })
        .collect();
    RgbImage::from_raw(size.x, size.y, bytes)
        .ok_or_else(|| ReconError::input("depth artifact dimensions do not match"))
}
fn save_images(
    evaluation: &Evaluation,
    stage: &str,
    output: &Path,
    targets: bool,
) -> Result<(), ReconError> {
    for view in &evaluation.images {
        let id = view.frame_id;
        save_rgb(
            &output.join(format!("{stage}-{id}.png")),
            &rgb_image(view.size, &view.rgb)?,
        )?;
        if targets {
            save_rgb(&output.join(format!("target-{id}.png")), &view.target)?;
            let roi = RgbImage::from_raw(
                view.size.x,
                view.size.y,
                view.roi.iter().flat_map(|v| [*v; 3]).collect(),
            )
            .ok_or_else(|| ReconError::input("ROI artifact dimensions"))?;
            save_rgb(&output.join(format!("roi-{id}.png")), &roi)?;
        }
        if let Some(depth) = &view.depth {
            save_rgb(
                &output.join(format!("{stage}-{id}-depth.png")),
                &depth_image(depth.size, &depth.prediction, |i| depth.alpha[i] > 1e-6)?,
            )?;
            io::write_new(
                &output.join(format!("{stage}-{id}-depth.f32")),
                &depth
                    .prediction
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            if targets {
                save_rgb(
                    &output.join(format!("target-{id}-depth.png")),
                    &depth_image(depth.size, &depth.target.values_m, |i| {
                        depth.target.confidence.as_ref().map_or(0, |c| c[i]) == 2
                            && (0.1..=8.).contains(&depth.target.values_m[i])
                    })?,
                )?;
                let roi = RgbImage::from_raw(
                    depth.size.x,
                    depth.size.y,
                    depth.roi.iter().flat_map(|v| [*v; 3]).collect(),
                )
                .ok_or_else(|| ReconError::input("native ROI artifact dimensions"))?;
                save_rgb(&output.join(format!("roi-{id}-depth.png")), &roi)?;
            }
        }
    }
    Ok(())
}

pub fn save_artifacts(run: &RunRecord, output: &Path) -> Result<(), ReconError> {
    if run.initial_state.kind != run.backend
        || run.final_state.kind != run.backend
        || run.initial_state.format_version != 1
        || run.final_state.format_version != 1
    {
        return Err(ReconError::input("backend snapshot kind/version mismatch"));
    }
    if run.initial_evaluation.images.len() != run.final_evaluation.images.len()
        || run
            .initial_evaluation
            .images
            .iter()
            .zip(&run.final_evaluation.images)
            .any(|(a, b)| {
                a.frame_id != b.frame_id
                    || a.size != b.size
                    || a.roi != b.roi
                    || a.target.as_raw() != b.target.as_raw()
            })
    {
        return Err(ReconError::input(
            "evaluation targets or immutable masks changed during run",
        ));
    }
    io::json(&output.join("curve.json"), &run.curve)?;
    let mut last_renderer = None;
    match run.backend {
        BackendKind::Triangle => {
            for (stage, state) in [("initial", &run.initial_state), ("final", &run.final_state)] {
                let decoded = decode_state(&state.payload)?;
                if stage == "final" {
                    last_renderer = decoded.renderer_last_render;
                }
                io::write_new(&output.join(format!("{stage}.state.json")), &state.payload)?;
                io::write_new(&output.join(format!("{stage}.ply")), &mesh_ply(&decoded)?)?;
            }
        }
        BackendKind::Gaussian => {
            io::write_new(&output.join("initial.ply"), &run.initial_state.payload)?;
            io::write_new(&output.join("final.ply"), &run.final_state.payload)?;
        }
    }
    let images = output.join("images");
    std::fs::create_dir(&images).map_err(|e| ReconError::io(&images, e))?;
    save_images(&run.initial_evaluation, "initial", &images, true)?;
    save_images(&run.final_evaluation, "final", &images, false)?;
    let layouts:Vec<_>=run.initial_evaluation.images.iter().filter_map(|v|v.depth.as_ref().map(|d|serde_json::json!({"frame_id":v.frame_id,"width":d.size.x,"height":d.size.y,"raw_depth":"little-endian f32 camera-z metres, row-major, 0=unavailable; inspect alpha/metrics for availability"}))).collect();
    io::json(&images.join("native-layout.json"), &layouts)?;
    let minimum = run
        .curve
        .iter()
        .min_by(|a, b| a.stats.total_loss.total_cmp(&b.stats.total_loss));
    io::json(
        &output.join("metrics.json"),
        &serde_json::json!({
            "initial":run.initial_evaluation,"final":run.final_evaluation,"research_gate":research_gate(run.backend,&run.initial_evaluation.aggregate,&run.final_evaluation.aggregate),
            "process_footprint":process_footprint(),"renderer_buffer_reservation_bytes":if run.backend==BackendKind::Triangle {Some(run.stats.renderer_peak_bytes)}else{None},
            "triangle_last_render_diagnostics":last_renderer,
            "sampled_training_loss_minimum":minimum,"checkpoint_note":"Initial and final parameters are saved. The sampled training-loss minimum is diagnostic, not a saved/held-out-selected checkpoint.",
            "comparison_note":"Both use frozen RGB ROI and degree-0/per-vertex appearance; triangle additionally uses metric depth and seed prior. Gaussian depth is unsupported, not zero error. This is not paper parity or full-scene quality.",
            "memory_note":"Triangle renderer bytes are a conservative buffer reservation upper bound; Gaussian buffer reservation is uninstrumented/null. Process footprint is separately measured."
        }),
    )?;
    Ok(())
}

pub fn copy_frozen_assets(
    manifest_path: &Path,
    manifest: &CaptureManifest,
    output: &Path,
) -> Result<(), ReconError> {
    let root = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let frozen = manifest
        .frozen
        .as_ref()
        .ok_or_else(|| ReconError::input("capture manifest is not frozen"))?;
    for asset in std::iter::once(&frozen.seed).chain(frozen.roi.values()) {
        let bytes = io::asset(root, asset)?;
        let path = output.join(&asset.file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ReconError::io(parent, e))?;
        }
        io::write_new(&path, &bytes)?;
    }
    io::write_new(
        &output.join("capture-manifest.json"),
        &io::read(manifest_path)?,
    )
}
