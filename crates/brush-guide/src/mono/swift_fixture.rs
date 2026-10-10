//! Golden cases for the phone's Swift port of the mono scale fit.

use super::tests::{cam, size};
use super::*;
use crate::config::GuideConfig;
use brush_render::kernels::camera_model::CameraModel;

fn cam_at(position: Vec3, rotation: glam::Quat) -> Camera {
    let fov = 90f64.to_radians();
    Camera::new(position, rotation, fov, fov, glam::vec2(0.5, 0.5), CameraModel::Pinhole)
}

/// LiDAR map at 40×30 whose pixel values are the mono value at the same image fraction times `factor`.
fn lidar_from(mono: &DepthMap, factor: f32, confidence: Option<u8>) -> DepthMap {
    let values = (0..1200)
        .map(|i| {
            let (x, y) = (i % 40, i / 40);
            mono.sample_uv((x as f32 + 0.5) / 40.0, (y as f32 + 0.5) / 30.0).unwrap_or(1.0) * factor
        })
        .collect();
    DepthMap { width: 40, height: 30, values, confidence: confidence.map(|c| vec![c; 1200]) }
}

fn constant_mono(value: f32) -> DepthMap {
    DepthMap { width: 8, height: 6, values: vec![value; 48], confidence: None }
}

/// Mono map whose columns carry `columns`, repeated down 6 rows.
fn column_mono(columns: [f32; 8]) -> DepthMap {
    DepthMap { width: 8, height: 6, values: (0..48).map(|i| columns[i % 8]).collect(), confidence: None }
}

/// Image-fraction samples at the centre of each of the eight mono columns, on the middle row.
fn column_uvs() -> Vec<[f32; 2]> {
    (0..8).map(|c| [(c as f32 + 0.5) / 8.0, 0.5]).collect()
}

/// Feature points straight ahead of an identity camera whose depth over the (constant) mono value is `ratio`.
fn feature_points(mono_value: f32, ratio: f32) -> Vec<Vec3> {
    (0..24).map(|i| Vec3::new((i % 5) as f32 * 0.01, (i / 5) as f32 * 0.01, mono_value * ratio)).collect()
}

struct Golden<'a> {
    mono: &'a DepthMap,
    lidar: Option<&'a DepthMap>,
    camera: &'a Camera,
    points: &'a [Vec3],
    uvs: &'a [[f32; 2]],
    recent: Option<f32>,
}

fn golden_case(g: &Golden, config: &GuideConfig) -> serde_json::Value {
    let (seed, fit) = mono_seed_for(Some(g.mono), g.lidar, g.camera, size(), g.points, config, g.recent);
    let depth_at: Vec<Option<f32>> =
        g.uvs.iter().map(|uv| seed.as_ref().and_then(|s| s.depth_at(Vec2::new(uv[0], uv[1])))).collect();
    let (kind, source, scale, samples) = match fit {
        ScaleFit::Fitted { source, scale, samples } => ("fitted", source, scale, samples),
        ScaleFit::Rejected { source, scale, samples } => ("rejected", source, scale, samples),
        ScaleFit::Absent => unreachable!("mono is present and seeding is on"),
    };
    let source = match source {
        ScaleSource::Lidar => "lidar",
        ScaleSource::Features => "features",
        ScaleSource::Metric => "metric",
        ScaleSource::Recent => "recent",
    };
    // The phone's camera looks down -z with y up: its camera-to-world has the engine's y and z axes negated.
    let rotation = glam::Mat3::from_quat(g.camera.rotation);
    let columns = [rotation.x_axis, -rotation.y_axis, -rotation.z_axis, g.camera.position];
    let camera_to_world: Vec<f32> =
        columns.iter().flat_map(|c| [c.x, c.y, c.z, if *c == g.camera.position { 1.0 } else { 0.0 }]).collect();
    let mut case = serde_json::json!({
        "mono": { "width": g.mono.width, "height": g.mono.height, "values": g.mono.values },
        "lidar": g.lidar.map(|l| serde_json::json!({
            "width": l.width, "height": l.height, "values": l.values, "confidence": l.confidence
        })),
        "camera_to_world_phone_column_major": camera_to_world,
        "points_world": g.points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
        "fit": { "kind": kind, "source": source, "scale": scale, "samples": samples },
        "min_depth_m": seed.as_ref().map(|s| s.min_depth_m),
        "uvs": g.uvs,
        "depth_at": depth_at,
    });
    // Only the recent-scale cases carry the key, so the older cases keep their bytes.
    if let Some(recent) = g.recent {
        case["recent"] = serde_json::json!(recent);
    }
    case
}

/// Deterministic cases at the fit's and the seeding gate's boundaries.
fn edge_cases(config: &GuideConfig) -> Vec<serde_json::Value> {
    let camera = cam();
    let min_depth = config.mono_min_depth_with_lidar_m;
    let (low, high) = config.mono_scale_range;
    let mut cases = Vec::new();
    let mut push = |mono: &DepthMap, lidar: Option<&DepthMap>, points: &[Vec3], uvs: &[[f32; 2]]| {
        cases.push(golden_case(&Golden { mono, lidar, camera: &camera, points, uvs, recent: None }, config));
    };
    // Feature fits just outside the range, and a LiDAR fit exactly on each bound (inclusive).
    let flat = constant_mono(2.0);
    push(&flat, None, &feature_points(2.0, low * 0.9997), &column_uvs());
    push(&flat, None, &feature_points(2.0, high * 1.0001), &column_uvs());
    push(&flat, None, &feature_points(2.0, low * 1.0003), &column_uvs());
    let one = constant_mono(1.0);
    push(&one, Some(&lidar_from(&one, low, Some(2))), &[], &column_uvs());
    push(&one, Some(&lidar_from(&one, high, Some(2))), &[], &column_uvs());
    // Scale 2 over large values: 60 and 52 pass the 100 m cap on the wrong side, 50 sits on it.
    let large = column_mono([60.0, 52.0, 50.0, 49.9, 40.0, 2.0, 1.0, 0.1]);
    push(&large, Some(&lidar_from(&large, 2.0, Some(2))), &[], &column_uvs());
    // Scale 1 with depths around the LiDAR keyframe's minimum depth.
    let near = column_mono([
        min_depth * 0.9,
        min_depth * 0.999,
        min_depth,
        min_depth * 1.001,
        min_depth * 1.1,
        0.0,
        f32::MIN_POSITIVE,
        min_depth * 2.0,
    ]);
    push(&near, Some(&lidar_from(&near, 1.0, Some(2))), &[], &column_uvs());
    // The same depths without a LiDAR map have no minimum depth: the features fit scale 1.
    push(&near, None, &feature_points(min_depth * 2.0, 1.0), &column_uvs());
    // A LiDAR map without confidence counts every pixel.
    push(&near, Some(&lidar_from(&near, 1.5, None)), &[], &column_uvs());
    cases
}

/// Cases where `recent` is set: too few samples use it, enough samples ignore it.
fn recent_cases(config: &GuideConfig) -> Vec<serde_json::Value> {
    let camera = cam();
    let flat = constant_mono(2.0);
    let one = constant_mono(1.0);
    let few_lidar = {
        let mut lidar = lidar_from(&flat, 1.5, Some(2));
        lidar.values.iter_mut().skip(100).for_each(|v| *v = 0.0);
        lidar
    };
    let uvs = column_uvs();
    let mut cases = Vec::new();
    let mut push = |mono: &DepthMap, lidar: Option<&DepthMap>, points: &[Vec3], recent: f32| {
        let g = Golden { mono, lidar, camera: &camera, points, uvs: &uvs, recent: Some(recent) };
        cases.push(golden_case(&g, config));
    };
    // Too few samples: none at all, 19 feature points, 100 LiDAR pixels.
    push(&flat, None, &[], 0.7);
    push(&flat, None, &feature_points(2.0, 1.5)[..19], 0.7);
    push(&flat, Some(&few_lidar), &[], 0.7);
    // A recent scale outside the range is used as given.
    push(&flat, None, &[], 5.0);
    // Enough features or LiDAR pixels: the fit ignores it.
    push(&flat, None, &feature_points(2.0, 1.5), 0.7);
    push(&one, Some(&lidar_from(&one, 1.5, Some(2))), &[], 0.7);
    // A feature fit outside the range is rejected, not replaced by the recent scale.
    push(&flat, None, &feature_points(2.0, config.mono_scale_range.1 * 1.0001), 0.7);
    // Recent scale with large mono values: the seeding cap applies to the scaled depth.
    let large = column_mono([60.0, 52.0, 50.0, 49.9, 40.0, 2.0, 1.0, 0.1]);
    push(&large, None, &[], 2.0);
    cases
}

/// A fit as the history sees it: kind, source, scale.
type HistoryFit = (&'static str, &'static str, f32);

/// Sequences of fits fed to the scale history, with its median after each.
fn history_cases() -> Vec<serde_json::Value> {
    let fitted = |source, scale| ("fitted", source, scale);
    let ramp: Vec<HistoryFit> = (1..=25).map(|i| fitted("features", i as f32 * 0.1)).collect();
    let sequences: Vec<Vec<HistoryFit>> = vec![
        vec![
            fitted("metric", 1.0),
            fitted("recent", 0.5),
            ("rejected", "lidar", 9.0),
            ("absent", "none", 0.0),
            fitted("lidar", 0.8),
            fitted("recent", 0.1),
            fitted("metric", 1.0),
            fitted("features", 1.2),
            ("rejected", "features", 0.01),
            fitted("lidar", 0.4),
        ],
        ramp,
        vec![fitted("lidar", 3.0), fitted("features", 1.0), fitted("lidar", 2.0), fitted("features", 2.0)],
    ];
    sequences
        .into_iter()
        .map(|seq| {
            let mut history = ScaleHistory::default();
            let mut fits = Vec::new();
            let mut medians = Vec::new();
            for &(kind, source, scale) in &seq {
                let source_enum = match source {
                    "lidar" => ScaleSource::Lidar,
                    "features" => ScaleSource::Features,
                    "recent" => ScaleSource::Recent,
                    "metric" | "none" => ScaleSource::Metric,
                    other => unreachable!("unknown source {other:?} in the history sequences"),
                };
                let fit = match kind {
                    "fitted" => ScaleFit::Fitted { source: source_enum, scale, samples: 1 },
                    "rejected" => ScaleFit::Rejected { source: source_enum, scale, samples: 1 },
                    "absent" => ScaleFit::Absent,
                    other => unreachable!("unknown kind {other:?} in the history sequences"),
                };
                history.note(&fit);
                fits.push(serde_json::json!({ "kind": kind, "source": source, "scale": scale }));
                medians.push(history.median());
            }
            serde_json::json!({ "fits": fits, "medians": medians })
        })
        .collect()
}

/// Writes the golden cases of the phone's Swift port (`MonoScaleFit`) to the path in `SWIFT_FIXTURE_OUT`
/// (`tools/fixtures/mono_scale_fit.json` of the app repo). Run with
/// `SWIFT_FIXTURE_OUT=… tools/cargo-host.sh test --crate brush-guide -- write_swift_fixture --ignored`.
#[test]
#[ignore]
fn write_swift_fixture() {
    let mut state = 0x2545_f491u32;
    let mut rnd = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / 16_777_216.0
    };
    let config = GuideConfig::default();
    let focal = cam().focal(size());
    let center = cam().center(size());
    let mut cases = Vec::new();
    for case in 0..36usize {
        let (mw, mh) = [(16u32, 12u32), (10, 5), (32, 24)][case % 3];
        let mono_values: Vec<f32> = (0..mw * mh)
            .map(|_| if rnd() < 0.1 { 0.0 } else { 0.5 + 3.0 * rnd() })
            .collect();
        let mono = DepthMap { width: mw, height: mh, values: mono_values, confidence: None };
        let ratio = 0.2 + 3.2 * rnd();
        let lidar_share = [1.0f32, 0.5, 0.1, 0.0][case % 4];
        // 40×30 LiDAR (1200 pixels): with `lidar_share` 1 and confidence 2 on 80 % of them, well over the 200
        // samples a LiDAR fit needs; the smaller shares fall back to features or the metric scale.
        let lidar = (case % 6 != 5).then(|| {
            let values: Vec<f32> = (0..1200)
                .map(|i| {
                    let (x, y) = (i % 40, i / 40);
                    let m = mono.sample_uv((x as f32 + 0.5) / 40.0, (y as f32 + 0.5) / 30.0).unwrap_or(1.0);
                    if rnd() < lidar_share { m * ratio * (0.9 + 0.2 * rnd()) } else { 0.0 }
                })
                .collect();
            let confidence: Vec<u8> =
                (0..1200).map(|_| if rnd() < 0.8 { 2 } else { (rnd() * 2.0) as u8 }).collect();
            DepthMap { width: 40, height: 30, values, confidence: Some(confidence) }
        });
        // Every third camera is the identity; the others are rotated and translated.
        let camera = if case % 3 == 0 {
            cam()
        } else {
            let axis = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
            let rotation = glam::Quat::from_axis_angle(axis, 6.0 * rnd() - 3.0);
            cam_at(Vec3::new(10.0 * rnd() - 5.0, 10.0 * rnd() - 5.0, 10.0 * rnd() - 5.0), rotation)
        };
        // Points in front of the camera, a fifth of them behind it, placed in world space.
        let n_points = [0usize, 10, 40, 60][(case / 4) % 4];
        let points: Vec<Vec3> = (0..n_points)
            .map(|_| {
                let z = if rnd() < 0.2 { -1.0 - 3.0 * rnd() } else { 1.0 + 7.0 * rnd() };
                let local = Vec3::new((rnd() - 0.5) * 1.6 * z, (rnd() - 0.5) * 1.6 * z, z);
                camera.local_to_world().transform_point3(local)
            })
            .collect();
        let uvs: Vec<[f32; 2]> = (0..12).map(|_| [rnd(), rnd()]).collect();
        let g = Golden { mono: &mono, lidar: lidar.as_ref(), camera: &camera, points: &points, uvs: &uvs, recent: None };
        cases.push(golden_case(&g, &config));
    }
    cases.extend(edge_cases(&config));
    let fixture = serde_json::json!({
        "image": { "width": size().x, "height": size().y, "fx": focal.x, "fy": focal.y, "cx": center.x, "cy": center.y },
        "min_confidence": config.min_depth_confidence,
        "range": [config.mono_scale_range.0, config.mono_scale_range.1],
        "min_depth_with_lidar_m": config.mono_min_depth_with_lidar_m,
        "case_count": cases.len(),
        "cases": cases,
        "recent_cases": recent_cases(&config),
        "history_len": ScaleHistory::LEN,
        "history_cases": history_cases(),
    });
    let path = std::env::var("SWIFT_FIXTURE_OUT").expect("set SWIFT_FIXTURE_OUT to the fixture's path");
    std::fs::write(path, serde_json::to_string(&fixture).unwrap()).unwrap();
}
