//! Debug-level stage timings (`RUST_LOG=brush_guide::timing=debug`). With
//! the target enabled, stage boundaries wait for the GPU so each stage's
//! time includes the GPU work it queued.

use brush_render::gaussian_splats::Splats;
use burn::tensor::{Tensor, s};

pub const TARGET: &str = "brush_guide::timing";

pub fn enabled() -> bool {
    log::log_enabled!(target: TARGET, log::Level::Debug)
}

/// Waits for queued GPU work that `t` depends on, when timing is enabled.
pub async fn sync<const D: usize>(t: &Tensor<D>) {
    if enabled() {
        let n = t.dims()[0].min(1);
        let _ = t.clone().narrow(0, 0, n).into_data_async().await;
    }
}

pub async fn sync_splats(splats: &Splats) {
    if enabled() && splats.num_splats() > 0 {
        let _ = splats.means().slice(s![0..1, ..]).into_data_async().await;
    }
}
