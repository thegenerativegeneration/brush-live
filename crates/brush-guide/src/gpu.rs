//! The process-wide wgpu setup shared by the FFI, the bench and the server.

pub use burn_wgpu::graphics::{AutoGraphicsApi, Metal};
use burn_wgpu::graphics::GraphicsApi;
use burn_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice};

/// Sets up the process-wide wgpu device with graphics API `G` and `tasks_max`
/// kernels per command buffer, and returns the autodiff device. Panics when
/// there is no adapter. Callers guard against running it twice.
pub async fn init_device<G: GraphicsApi>(tasks_max: usize) -> burn::tensor::Device {
    burn_wgpu::init_setup_async::<G>(
        &WgpuDevice::default(),
        RuntimeOptions { tasks_max, memory_config: MemoryConfiguration::ExclusivePages },
    )
    .await;
    burn::tensor::Device::from(WgpuDevice::default()).autodiff()
}
