use brush_recon::capture::mask::rasterize_roi;
use brush_render::{
    camera::{Camera, focal_to_fov},
    kernels::camera_model::CameraModel,
};
use glam::{Quat, UVec2, Vec2, Vec3};

fn camera() -> Camera {
    let f = focal_to_fov(1., 4, &CameraModel::Pinhole);
    Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        f,
        f,
        Vec2::ZERO,
        CameraModel::Pinhole,
    )
}

#[test]
fn hard_roi_includes_exact_edges_and_unions_faces_without_soft_coverage() {
    let p = [[1., 0., 2.], [8., 0., 2.], [1., 8., 2.]];
    let a = rasterize_roi(&p, &[[0, 1, 2]], &camera(), UVec2::splat(4), 0.05).unwrap();
    assert_eq!(
        a[0], 255,
        "pixel center on x=.5 edge is included even though soft coverage is zero"
    );
    assert_eq!(a[15], 0);
    let b = rasterize_roi(
        &p,
        &[[0, 2, 1], [0, 1, 2]],
        &camera(),
        UVec2::splat(4),
        0.05,
    )
    .unwrap();
    assert_eq!(a, b, "mask is a binary union, not alpha accumulation");
}

#[test]
fn hard_roi_matches_culling_and_rejects_invalid_indices() {
    let p = [[1., 0., 0.05], [8., 0., 2.], [1., 8., 2.]];
    assert!(
        rasterize_roi(&p, &[[0, 1, 2]], &camera(), UVec2::splat(4), 0.05)
            .unwrap()
            .iter()
            .all(|v| *v == 0)
    );
    assert!(rasterize_roi(&p, &[[0, 1, 3]], &camera(), UVec2::splat(4), 0.05).is_err());
}
