//! Packed splat snapshots for the phone's preview overlay.

use super::splat_read::read_f32;
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Tensor, s};
use std::time::{Duration, Instant};

/// Floats per splat: position 3, rotation `[w, x, y, z]` 4, linear scale 3,
/// opacity 1, SH0 3.
pub const PREVIEW_FLOATS: usize = 14;

/// The splats at one training step, `count * PREVIEW_FLOATS` floats.
/// `count == 0` after a reset.
#[derive(Debug, Default)]
pub struct PreviewSnapshot {
    pub version: u64,
    /// The `generation` of the last reset before this snapshot (0 before any).
    pub generation: u64,
    pub count: u32,
    /// Time the worker spent reading the snapshot back.
    pub readback_ms: f32,
    pub data: Vec<f32>,
}

/// When the next snapshot is due; `None` interval is off.
#[derive(Default)]
pub(super) struct PreviewClock {
    interval: Option<Duration>,
    last: Option<Instant>,
}

impl PreviewClock {
    pub(super) fn set(&mut self, interval: Option<Duration>) {
        self.interval = interval;
        self.last = None;
    }

    pub(super) fn due(&self, now: Instant) -> bool {
        match (self.interval, self.last) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(i), Some(last)) => now.duration_since(last) >= i,
        }
    }

    pub(super) fn taken(&mut self, now: Instant) {
        self.last = Some(now);
    }
}

/// Interleaves per-splat parameters into the snapshot layout. A zero or
/// non-finite rotation becomes the identity.
pub(super) fn pack(
    means: &[f32],
    rots: &[f32],
    scales: &[f32],
    opac: &[f32],
    sh0: &[f32],
) -> Vec<f32> {
    let n = opac.len();
    let mut out = Vec::with_capacity(n * PREVIEW_FLOATS);
    for i in 0..n {
        out.extend_from_slice(&means[i * 3..i * 3 + 3]);
        let r = &rots[i * 4..i * 4 + 4];
        let len = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        if len > 0.0 && len.is_finite() {
            out.extend(r.iter().map(|v| v / len));
        } else {
            out.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }
        out.extend_from_slice(&scales[i * 3..i * 3 + 3]);
        out.push(opac[i]);
        out.extend_from_slice(&sh0[i * 3..i * 3 + 3]);
    }
    out
}

/// Reads the splats back from the GPU in the snapshot layout.
pub(super) async fn read(splats: &Splats) -> Vec<f32> {
    let n = splats.num_splats() as usize;
    let sh0: Tensor<2> = splats
        .sh_coeffs
        .val()
        .slice(s![.., 0..1, ..])
        .reshape([n, 3]);
    pack(
        &read_f32(splats.means()).await,
        &read_f32(splats.rotations()).await,
        &read_f32(splats.scales()).await,
        &read_f32(splats.opacities()).await,
        &read_f32(sh0).await,
    )
}
