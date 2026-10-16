//! Corrected camera poses for views already added: a map correction on the phone moved their keyframes.

use super::{LiveModel, view_cone};
use crate::keyframe::{KeyframeError, arkit_pose};
use crate::pose_eq::same_pose;
use brush_render::camera::Camera;
use glam::{Affine3A, Quat, Vec3};

/// One keyframe's corrected pose in Brush's camera convention (see [`arkit_pose`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewPose {
    pub id: u64,
    pub position: Vec3,
    pub rotation: Quat,
}

impl ViewPose {
    /// From an ARKit camera-to-world pose as the wire carries it; an error for a non-rigid or non-finite pose.
    pub fn from_arkit(id: u64, pose: &[f32; 16]) -> Result<Self, KeyframeError> {
        let (position, rotation) = arkit_pose(pose)?;
        Ok(Self {
            id,
            position,
            rotation,
        })
    }

    /// The same camera at this pose: field of view, centre and model kept.
    pub fn applied_to(&self, camera: &Camera) -> Camera {
        Camera {
            position: self.position,
            rotation: self.rotation,
            ..*camera
        }
    }

    /// Whether `camera` already sits at this pose.
    pub fn matches(&self, camera: &Camera) -> bool {
        same_pose(
            (camera.position, camera.rotation),
            (self.position, self.rotation),
        )
    }
}

/// What one [`LiveModel::update_poses`] call did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoseUpdateStats {
    /// Views whose camera changed.
    pub moved: u32,
    /// Views already at the given pose.
    pub unchanged: u32,
    /// Ids of no trained view: never added, rejected, held out, or of another session.
    pub unknown: u32,
}

impl LiveModel {
    /// Per splat, the view (index into `views`) whose seeds created it, `None` for none; `None` when origins are
    /// not tracked (`move_anchored_splats` off) or before the first keyframe.
    pub async fn splat_origins(&self) -> Option<Vec<Option<u32>>> {
        self.trainer.as_ref()?.origin_slots().await
    }

    /// Moves views to corrected poses: the view's camera, its Mip-filter camera and, for the newest view, the
    /// protected cone, and moves the splats each moved view seeded by that view's correction (when
    /// `move_anchored_splats` is on); then ranks the active tiles again, recomputes the area's views and rebuilds both
    /// loaders so no batch with an old camera, or of a view that left the area, is drawn again. The score set needs
    /// nothing: each round reads the splats and the current views.
    pub fn update_poses(&mut self, poses: &[ViewPose]) -> PoseUpdateStats {
        let mut stats = PoseUpdateStats::default();
        // Per moved view: its slot and the world transform that takes the old camera onto the new one.
        let mut moves: Vec<(u32, Affine3A)> = Vec::new();
        for p in poses {
            let Some(&i) = self.index.get(&p.id) else {
                stats.unknown += 1;
                continue;
            };
            let old = self.views[i].camera;
            if p.matches(&old) {
                stats.unchanged += 1;
                continue;
            }
            let new = p.applied_to(&old);
            moves.push((i as u32, new.local_to_world() * old.world_to_local()));
            self.views[i].camera = new;
            self.view_cams[i].0 = new.position;
            stats.moved += 1;
        }
        if moves.is_empty() {
            return stats;
        }
        let newest = self.views.len() - 1;
        if let Some(t) = self.trainer.as_mut() {
            t.set_view_cams(self.view_cams.clone());
            if moves.iter().any(|(i, _)| *i as usize == newest) {
                t.set_protect_cone(Some(view_cone(&self.views[newest].camera)));
            }
            if let Some(splats) = self.splats.take() {
                self.splats = Some(t.move_by_origin(splats, &moves));
            }
        }
        let camera = self.views[newest].camera;
        self.refresh_active(&camera, &[]);
        if let Some(area) = self.area.as_mut() {
            // Rows moved with their views and may lie in other tiles now: the next step holds them afresh.
            area.mask_stale = true;
        }
        self.refresh_area_views();
        self.rebuild_all_loaders();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GuideConfig;
    use crate::keyframe::decode_keyframe;
    use crate::protocol::KeyframeHeader;
    use glam::Mat4;

    /// `tests/integration/test_scene::arkit_pose_at`: integration-test helpers are not visible to unit tests, and
    /// these need `LiveModel`'s private loaders.
    fn pose_at(pos: Vec3) -> [f32; 16] {
        Mat4::look_at_rh(pos, Vec3::ZERO, Vec3::Y)
            .inverse()
            .to_cols_array()
    }

    fn keyframe(id: u64, pos: Vec3) -> (KeyframeHeader, Vec<u8>) {
        let (w, h) = (16u32, 12u32);
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x * 16) as u8, (y * 20) as u8, 90])
        });
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        let header = KeyframeHeader {
            id,
            timestamp: 0.0,
            pose: pose_at(pos),
            fx: 14.0,
            fy: 14.0,
            cx: 8.0,
            cy: 6.0,
            width: w,
            height: h,
            jpeg_len: jpeg.len() as u32,
            depth_size: None,
            depth_confidence: false,
            num_points: 0,
            mono_depth_size: None,
        };
        (header, jpeg)
    }

    /// After an update neither loader hands out a batch with the old camera: prefetched and cached batches are gone,
    /// and the Mip-filter camera follows the view.
    #[tokio::test]
    async fn loaders_serve_only_corrected_cameras() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let dir = std::env::temp_dir().join(format!("live-poses-loaders-{}", std::process::id()));
        let mut live = LiveModel::new(GuideConfig::default(), device);
        for (i, x) in [0.0f32, 0.5, 1.0].iter().enumerate() {
            let (h, p) = keyframe(i as u64, Vec3::new(*x, 0.0, 2.0));
            live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
                .await;
        }
        // Fill the prefetch channels and the caches with batches of the old cameras.
        for _ in 0..12 {
            let _ = live.recent.as_mut().unwrap().next_batch().await;
            let _ = live.all.as_mut().unwrap().next_batch().await;
        }
        let old = Vec3::new(0.5, 0.0, 2.0);
        let stats =
            live.update_poses(&[
                ViewPose::from_arkit(1, &pose_at(Vec3::new(0.5, 0.3, 2.2))).unwrap()
            ]);
        assert_eq!(stats.moved, 1);
        for _ in 0..24 {
            let a = live
                .recent
                .as_mut()
                .unwrap()
                .next_batch()
                .await
                .camera
                .position;
            let b = live
                .all
                .as_mut()
                .unwrap()
                .next_batch()
                .await
                .camera
                .position;
            for p in [a, b] {
                assert!(
                    (p - old).length() > 0.1,
                    "a batch still carries view 1's old camera: {p}"
                );
            }
        }
        assert_eq!(live.view_cams[1].0, live.views[1].camera.position);
        std::fs::remove_dir_all(dir).ok();
    }
}
