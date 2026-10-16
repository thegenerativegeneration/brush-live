use brush_recon::RenderRequest;
use brush_render::{
    camera::{Camera, focal_to_fov},
    kernels::camera_model::CameraModel,
};
use brush_triangle_render::{
    TriangleSeed,
    reference::{ReferenceConfig, render_reference},
};
use glam::{Quat, UVec2, Vec2, Vec3};

fn request(size: u32) -> RenderRequest {
    let fov = focal_to_fov(1., size, &CameraModel::Pinhole);
    RenderRequest {
        camera: Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            fov,
            fov,
            Vec2::ZERO,
            CameraModel::Pinhole,
        ),
        size: UVec2::splat(size),
        background: Vec3::new(0.2, 0.3, 0.4),
        depth: true,
        normals: true,
    }
}
fn seed() -> TriangleSeed {
    TriangleSeed {
        positions: vec![[0., 0., 2.], [8., 0., 2.], [0., 8., 2.]],
        faces: vec![[0, 1, 2]],
        colors: vec![[0.8, 0.1, 0.1]; 3],
    }
}
fn logits(colors: &[[f32; 3]]) -> Vec<[f32; 3]> {
    colors
        .iter()
        .map(|c| c.map(|x| (x / (1. - x)).ln()))
        .collect()
}
fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 2e-5, "{a} != {b}");
}

#[test]
fn inradius_coverage_weight_and_alpha_normalized_outputs_are_analytical() {
    let s = seed();
    let r = request(4);
    let out = render_reference(
        &s,
        &logits(&s.colors),
        &[0., 0., 0.],
        &r,
        &ReferenceConfig::default(),
    )
    .unwrap();
    let radius = 4. - 2. * 2_f32.sqrt();
    let alpha = 0.5 * (0.5 / radius);
    close(out.alpha[0], alpha);
    for ch in 0..3 {
        close(
            out.rgb[0][ch],
            s.colors[0][ch] * alpha + r.background[ch] * (1. - alpha),
        );
    }
    close(out.depth_m[0], 2.);
    for (a, b) in out.normals[0].into_iter().zip([0., 0., -1.]) {
        close(a, b);
    }
    assert_eq!(out.rgb[15], r.background.to_array());
    assert_eq!(out.depth_m[15], 0.);
    assert_eq!(out.visible_faces, 1);
}

#[test]
fn tilted_plane_interpolation_is_perspective_correct_for_color_and_camera_z() {
    let mut s = seed();
    s.positions = vec![[0., 0., 1.], [8., 0., 2.], [0., 16., 4.]];
    s.colors = vec![[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];
    let mut r = request(4);
    r.background = Vec3::ZERO;
    let out = render_reference(
        &s,
        &logits(&s.colors),
        &[0.; 3],
        &r,
        &ReferenceConfig::default(),
    )
    .unwrap();
    // At (.5,.5), screen barycentrics (.75,.125,.125); divide by depths (1,2,4).
    close(out.depth_m[0], 32. / 27.);
    for (ch, w) in [24. / 27., 2. / 27., 1. / 27.].into_iter().enumerate() {
        close(out.rgb[0][ch] / out.alpha[0], 0.1 + 0.7 * w);
    }
    let n = -Vec3::new(-0.125, -0.1875, 1.).normalize();
    assert!((Vec3::from(out.normals[0]) - n).length() < 2e-5);
}

#[test]
fn weight_floor_scales_sigmoid_and_minimum_vertex_controls_alpha() {
    let s = seed();
    let r = request(4);
    let c = ReferenceConfig {
        weight_floor: 0.4,
        ..Default::default()
    };
    let weights = [(0.2_f32 / 0.8).ln(), 0., (0.8_f32 / 0.2).ln()];
    let out = render_reference(&s, &logits(&s.colors), &weights, &r, &c).unwrap();
    close(
        out.alpha[0],
        (0.4 + 0.6 * 0.2) * 0.5 / (4. - 2. * 2_f32.sqrt()),
    );
}

#[test]
fn alpha_cap_is_shared_and_not_opaque_one() {
    let mut s = seed();
    let a = 1.5 + 2_f32.sqrt();
    s.positions = vec![[-1., -1., 2.], [2. * a, -1., 2.], [-1., 2. * a, 2.]];
    let c = ReferenceConfig {
        weight_floor: 1.,
        ..Default::default()
    };
    let out = render_reference(&s, &logits(&s.colors), &[0.; 3], &request(1), &c).unwrap();
    close(out.alpha[0], 0.99);
}

#[test]
fn empty_and_all_culled_return_background_without_fake_depth() {
    let r = request(4);
    for mode in 0..5 {
        let mut s = seed();
        match mode {
            0 => s.faces.clear(),
            1 => s.positions[0][2] = 0.05,
            2 => s.faces[0] = [0, 0, 2],
            3 => s.positions[2] = s.positions[1],
            _ => s.positions.iter_mut().for_each(|p| p[0] += 100.),
        }
        let out = render_reference(
            &s,
            &logits(&s.colors),
            &[0.; 3],
            &r,
            &ReferenceConfig::default(),
        )
        .unwrap();
        assert!(out.alpha.iter().all(|v| *v == 0.));
        assert!(out.depth_m.iter().all(|v| *v == 0.));
        assert!(out.rgb.iter().all(|v| *v == r.background.to_array()));
    }
}

#[test]
fn reversed_winding_produces_identical_channels() {
    let s = seed();
    let r = request(4);
    let c = ReferenceConfig::default();
    let a = render_reference(&s, &logits(&s.colors), &[0.; 3], &r, &c).unwrap();
    let mut b = s.clone();
    b.faces[0] = [0, 2, 1];
    let b = render_reference(&b, &logits(&b.colors), &[0.; 3], &r, &c).unwrap();
    for (a, b) in a.rgb.iter().flatten().zip(b.rgb.iter().flatten()) {
        close(*a, *b);
    }
    for (a, b) in a.normals.iter().flatten().zip(b.normals.iter().flatten()) {
        close(*a, *b);
    }
}

#[test]
fn depth_order_is_front_to_back_and_equal_depth_ties_use_face_id() {
    let mut s = seed();
    s.positions
        .extend([[0., 0., 4.], [16., 0., 4.], [0., 16., 4.]]);
    s.colors.extend([[0.1, 0.1, 0.8]; 3]);
    s.faces = vec![[3, 4, 5], [0, 1, 2]];
    let mut r = request(4);
    r.background = Vec3::ZERO;
    let out = render_reference(
        &s,
        &logits(&s.colors),
        &[0.; 6],
        &r,
        &ReferenceConfig::default(),
    )
    .unwrap();
    let a = 0.25 / (4. - 2. * 2_f32.sqrt());
    close(out.rgb[0][0], a * 0.8 + (1. - a) * a * 0.1);
    // Coincident faces, blue face ID 0 precedes red face ID 1.
    for v in &mut s.positions[3..] {
        *v = v.map(|x| x * 0.5);
    }
    let out = render_reference(
        &s,
        &logits(&s.colors),
        &[0.; 6],
        &r,
        &ReferenceConfig::default(),
    )
    .unwrap();
    close(out.rgb[0][0], a * 0.1 + (1. - a) * a * 0.8);
}

#[test]
fn invalid_indices_and_nonfinite_configuration_fail() {
    let mut s = seed();
    s.faces[0][0] = 9;
    assert!(
        render_reference(
            &s,
            &logits(&s.colors),
            &[0.; 3],
            &request(4),
            &ReferenceConfig::default()
        )
        .is_err()
    );
    let s = seed();
    for c in [
        ReferenceConfig {
            sigma: 0.,
            ..Default::default()
        },
        ReferenceConfig {
            sigma: f32::NAN,
            ..Default::default()
        },
        ReferenceConfig {
            near_m: -1.,
            ..Default::default()
        },
    ] {
        assert!(render_reference(&s, &logits(&s.colors), &[0.; 3], &request(4), &c).is_err());
    }
}

#[test]
fn exact_edge_is_zero_even_with_fractional_sigma() {
    let mut s = seed();
    s.positions = vec![[1., 0., 2.], [8., 0., 2.], [1., 8., 2.]];
    let out = render_reference(
        &s,
        &logits(&s.colors),
        &[0.; 3],
        &request(4),
        &ReferenceConfig {
            sigma: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(out.alpha[0], 0.);
    assert!(out.rgb.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn crossing_triangles_follow_documented_centroid_order_not_pixel_depth() {
    let mut s = seed();
    s.positions = vec![
        [0., 0., 1.],
        [16., 0., 4.],
        [0., 16., 4.],
        [0., 0., 2.],
        [8., 0., 2.],
        [0., 8., 2.],
    ];
    s.faces = vec![[0, 1, 2], [3, 4, 5]];
    s.colors.extend([[0.1, 0.1, 0.8]; 3]);
    let mut r = request(4);
    r.background = Vec3::ZERO;
    let out = render_reference(
        &s,
        &logits(&s.colors),
        &[0.; 6],
        &r,
        &ReferenceConfig::default(),
    )
    .unwrap();
    let a = 0.25 / (4. - 2. * 2_f32.sqrt());
    // Blue centroid z=2 precedes red centroid z=3, even where red is locally nearer.
    close(out.rgb[0][0], a * 0.1 + (1. - a) * a * 0.8);
}
