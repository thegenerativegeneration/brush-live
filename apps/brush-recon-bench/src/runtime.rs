use brush_recon::ReconError;
use burn::tensor::Device;
use burn_wgpu::{RuntimeOptions, WgpuDevice, graphics::Metal};
use serde_json::{Value, json};

pub async fn initialize() -> Result<(Device, Value), ReconError> {
    let setup =
        burn_wgpu::init_setup_async::<Metal>(&WgpuDevice::default(), RuntimeOptions::default())
            .await;
    let info = setup.adapter.get_info();
    let compiler = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(&compiler)
        .arg("--version")
        .output()
        .map_err(|e| ReconError::io(&compiler, e))?;
    if !version.status.success() {
        return Err(ReconError::input("cannot identify Rust compiler"));
    }
    let environment = json!({"adapter":info.name,"backend":format!("{:?}",info.backend),"driver":info.driver_info,
        "os":std::env::consts::OS,"arch":std::env::consts::ARCH,"rustc":String::from_utf8_lossy(&version.stdout).trim(),"shader_stack":"workspace-pinned Burn/CubeCL/wgpu; native Metal feature"});
    Ok((Device::from(WgpuDevice::default()).autodiff(), environment))
}
