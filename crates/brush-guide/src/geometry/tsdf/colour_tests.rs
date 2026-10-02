use glam::{IVec3, Vec3, vec3};

use super::fixtures::{colour_image, depth_image, look_at, plane_z};
use super::*;
use crate::geometry::colour::srgb_to_linear;

const A: [f32; 3] = [0.9, 0.2, 0.05];
const B: [f32; 3] = [0.1, 0.6, 0.8];

fn assert_rgb(got: Option<[f32; 3]>, want: [f32; 3], what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: no colour"));
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 2e-3, "{what}: {got:?}, want {want:?}");
    }
}

/// Fuses plane `z` seen from the origin, painted `rgb`.
fn fuse_plane(tsdf: &mut Tsdf, z: f32, rgb: [f32; 3]) {
    let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 2.5));
    let colour = colour_image(&cam, plane_z(z), |_| rgb);
    tsdf.integrate(&depth_image(&cam, plane_z(z)), Some(&colour), &cam);
}

#[test]
fn integration_without_colour_leaves_none() {
    let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 2.5));
    let mut tsdf = Tsdf::new();
    tsdf.integrate(&depth_image(&cam, plane_z(2.5)), None, &cam);
    let p = vec3(0.013, -0.2, 2.5);
    assert!(tsdf.sdf(p).is_some());
    assert!(tsdf.rgb(p).is_none());
}

/// Voxel centre (0.025, 0.025, 2.625) lies 12.5 cm behind plane 2.5
/// (drop-off weight 0.25) and 2.5 cm behind plane 2.6 (weight 1): fused
/// with A behind the first and B behind the second, it holds
/// (0.25·A + B) / 1.25, as its distance averages with those weights.
#[test]
fn colour_averages_with_the_distance_weights() {
    let mut tsdf = Tsdf::new();
    fuse_plane(&mut tsdf, 2.5, A);
    fuse_plane(&mut tsdf, 2.6, B);
    let g = IVec3::new(0, 0, 52);
    assert!((voxel_centre(g) - vec3(0.025, 0.025, 2.625)).length() < 1e-6);
    let (t, w) = tsdf.voxel(g).unwrap();
    assert!((w - 1.25).abs() < 1e-5, "weight {w}");
    let want_t = (0.25 * (-0.125 / TRUNC) + 1.0 * (-0.025 / TRUNC)) / 1.25;
    assert!((t - want_t).abs() < 1e-4, "distance {t} vs {want_t}");
    let want = std::array::from_fn(|c| (0.25 * A[c] + B[c]) / 1.25);
    assert_rgb(tsdf.voxel_colour(g), want, "voxel behind both planes");
}

/// Weights are capped and decay; the colour keeps following new
/// observations with the capped weight.
#[test]
fn colour_follows_a_change_at_the_weight_cap() {
    let mut tsdf = Tsdf::new();
    for _ in 0..40 {
        fuse_plane(&mut tsdf, 2.5, A);
    }
    fuse_plane(&mut tsdf, 2.5, B);
    let g = voxel_of(vec3(0.01, 0.01, 2.49));
    let want = std::array::from_fn(|c| (MAX_WEIGHT * A[c] + B[c]) / (MAX_WEIGHT + 1.0));
    assert_rgb(tsdf.voxel_colour(g), want, "after the change");
}

/// Fuses plane 2.5 painted `rgb` `n` times.
fn repaint(tsdf: &mut Tsdf, rgb: [f32; 3], n: usize) {
    for _ in 0..n {
        fuse_plane(tsdf, 2.5, rgb);
    }
}

/// A converged grey plane drifting by 5/255 in sRGB is not colour-stale;
/// by 12/255 it is.
#[test]
fn colour_change_below_the_threshold_is_ignored() {
    let grey = |v: f32| [srgb_to_linear(v / 255.0); 3];
    let brick = BrickKey(IVec3::new(0, 0, 2));
    let mut tsdf = Tsdf::new();
    repaint(&mut tsdf, grey(128.0), 30);
    assert!(tsdf.take_changed().contains(&brick));
    repaint(&mut tsdf, grey(133.0), 150);
    assert!(tsdf.changed().is_empty(), "geometry unchanged");
    assert!(tsdf.colour_changed().is_empty(), "drift of 5");
    repaint(&mut tsdf, grey(140.0), 150);
    assert!(tsdf.colour_changed().contains(&brick), "change of 12");
    tsdf.mark_meshed(&[brick]);
    assert!(!tsdf.colour_changed().contains(&brick), "snapshot taken");
}
