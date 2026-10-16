use brush_recon::{MetricDepth, ReconFrame};
use brush_render::{
    camera::{Camera, focal_to_fov},
    kernels::camera_model::CameraModel,
};
use brush_triangle_train::{SeedConfig, seed_from_training};
use glam::{Quat, UVec2, Vec2, Vec3};
use image::{Rgb, RgbImage};
use std::sync::Arc;

fn frame(id: u64, x: f32) -> Arc<ReconFrame> {
    Arc::new(ReconFrame {
        id,
        camera: Camera::new(
            Vec3::new(x, 0., 0.),
            Quat::IDENTITY,
            focal_to_fov(4., 4, &CameraModel::Pinhole),
            focal_to_fov(4., 4, &CameraModel::Pinhole),
            Vec2::splat(0.5),
            CameraModel::Pinhole,
        ),
        rgb: Arc::new(RgbImage::from_pixel(4, 4, Rgb([255, 0, 128]))),
        lidar: Some(Arc::new(MetricDepth {
            size: UVec2::new(4, 4),
            values_m: vec![2.; 16],
            confidence: Some(vec![2; 16]),
        })),
        roi: Arc::new(vec![255; 16]),
    })
}

fn cfg() -> SeedConfig {
    SeedConfig {
        anchor_ids: vec![174],
        stride: 1,
        ..Default::default()
    }
}

#[test]
fn plane_backprojects_and_shares_grid_vertices() {
    let f = frame(174, 1.);
    let s = seed_from_training(&[f.clone()], &cfg()).unwrap();
    assert_eq!(s.faces.len(), 18);
    assert_eq!(s.positions.len(), 16);
    assert!((Vec3::from(s.positions[0]) - Vec3::new(0.25, -0.75, 2.)).length() < 1e-5);
    for p in &s.positions {
        let local = f.camera.world_to_local().transform_point3(Vec3::from(*p));
        let pixel = local.truncate() / local.z * f.camera.focal(UVec2::splat(4))
            + f.camera.center(UVec2::splat(4));
        assert!((pixel.x.fract() - 0.5).abs() < 1e-5 && (pixel.y.fract() - 0.5).abs() < 1e-5);
    }
    assert_eq!(s.colors[0], [1., 0., 128. / 255.]);
    assert_eq!(
        s.fingerprint().unwrap(),
        seed_from_training(&[f], &cfg())
            .unwrap()
            .fingerprint()
            .unwrap()
    );
}

#[test]
fn holes_and_depth_discontinuities_do_not_create_bridges() {
    let mut f = (*frame(174, 0.)).clone();
    let mut d = (**f.lidar.as_ref().unwrap()).clone();
    d.confidence.as_mut().unwrap()[0] = 0;
    f.lidar = Some(Arc::new(d));
    let s = seed_from_training(&[Arc::new(f)], &cfg()).unwrap();
    assert_eq!(s.faces.len(), 17);
    assert_eq!(s.positions.len(), 15);
    let mut f = (*frame(174, 0.)).clone();
    let mut d = (**f.lidar.as_ref().unwrap()).clone();
    for y in 0..4 {
        for x in 2..4 {
            d.values_m[y * 4 + x] = 3.;
        }
    }
    f.lidar = Some(Arc::new(d));
    let s = seed_from_training(&[Arc::new(f)], &cfg()).unwrap();
    assert_eq!(s.faces.len(), 12);
    for face in s.faces {
        let z = face.map(|i| s.positions[i as usize][2]);
        assert!(z.iter().all(|v| *v == z[0]));
    }
}

#[test]
fn budget_is_round_robin_and_anchors_are_training_only() {
    let c = SeedConfig {
        anchor_ids: vec![174, 184],
        stride: 1,
        max_faces: 2,
        ..Default::default()
    };
    let a = frame(174, 0.);
    let b = frame(184, 10.);
    assert!(seed_from_training(&[a.clone()], &c).is_err());
    let s = seed_from_training(&[a, b], &c).unwrap();
    assert_eq!(s.faces.len(), 2);
    assert!(s.faces[0].iter().all(|i| s.positions[*i as usize][0] < 2.));
    assert!(s.faces[1].iter().all(|i| s.positions[*i as usize][0] > 8.));
    assert_eq!(s.positions.len(), 6, "unused vertices must be compacted");
}

#[test]
fn empty_and_degenerate_initialization_fail_explicitly() {
    let f = frame(174, 0.);
    let mut c = cfg();
    c.min_area_m2 = 10.;
    assert!(seed_from_training(&[f.clone()], &c).is_err());
    c = cfg();
    c.stride = 0;
    assert!(seed_from_training(&[f], &c).is_err());
    assert!(seed_from_training(&[], &cfg()).is_err());
}
