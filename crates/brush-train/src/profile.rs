//! Opt-in per-phase timing of training steps. Each phase boundary waits for
//! the GPU, so a phase includes the work it queued; the syncs also split
//! fused work at the boundaries, so the phases add up to more than an
//! unprofiled step.

use burn::tensor::Device;
use web_time::Instant;

/// Seconds spent per phase over `steps` training steps.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepProfile {
    pub steps: u32,
    /// Ground-truth upload and the forward render.
    pub forward_s: f64,
    /// Image loss (L1 + SSIM), SH background and LPIPS if on.
    pub loss_s: f64,
    /// Backward pass, as far as the refine gradient needs it.
    pub backward_s: f64,
    /// The rest of the backward and the Adam steps.
    pub optimizer_s: f64,
    /// Refine statistics and mean noise.
    pub noise_s: f64,
}

impl StepProfile {
    pub(crate) fn add(&mut self, phases: [f64; 5]) {
        self.steps += 1;
        self.forward_s += phases[0];
        self.loss_s += phases[1];
        self.backward_s += phases[2];
        self.optimizer_s += phases[3];
        self.noise_s += phases[4];
    }
}

/// Flushes all lazily queued (fused) work on `device` and waits for the GPU.
/// Syncing on one tensor would only run that tensor's dependencies, leaving
/// e.g. optimizer moments to be charged to a later phase.
pub(crate) fn device_sync(device: &Device) {
    let _ = device.sync();
}

/// Seconds since `clock`, which then restarts.
pub(crate) fn lap(clock: &mut Instant) -> f64 {
    let now = Instant::now();
    let dt = now.duration_since(*clock).as_secs_f64();
    *clock = now;
    dt
}
