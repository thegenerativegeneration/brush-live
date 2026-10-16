//! Small, deliberately dense differentiable triangle reference on Metal.
//! No research-repository code, custom kernels, topology changes, or live-engine changes.
//! Run: cargo run --release -p brush-render --features brush-cube/metal --example triangle_probe
//! Optional: --output <directory> (PPM images), --steps 200.
//! This checks correctness, NOT scalable renderer performance or paper parity.

use burn::tensor::{Device, Tensor, TensorData};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

const SIZE: usize = 48;
const PIXELS: usize = SIZE * SIZE;
const SOFTNESS: f32 = 0.035; // Camera-plane distance, fixed during optimization.
const TARGET_VERTICES: [f32; 18] = [
    -0.65, -0.55, 2.2, 0.65, -0.5, 2.2, 0.0, 0.65, 2.2, -0.95, -0.7, 3.3, 1.0, -0.65, 3.3, 0.1,
    1.0, 3.3,
];
const TARGET_COLORS: [f32; 18] = [
    0.9, 0.15, 0.1, 0.85, 0.3, 0.1, 0.95, 0.15, 0.3, 0.1, 0.3, 0.85, 0.15, 0.55, 0.9, 0.3, 0.25,
    0.8,
];
const CAMERAS: [[f32; 2]; 3] = [[0.0, 0.0], [0.45, 0.1], [-0.35, -0.2]];
const HELD_OUT: [f32; 2] = [0.2, -0.25];

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 200)]
    steps: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn tensor(values: &[f32], device: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), device)
}

fn scalar(t: &Tensor<1>, i: usize) -> Tensor<1> {
    t.clone().slice([i..i + 1])
}

fn component(t: &Tensor<1>, i: usize) -> Tensor<1> {
    scalar(t, i).expand([PIXELS])
}

struct Grid {
    x: Tensor<1>,
    y: Tensor<1>,
}

impl Grid {
    fn new(device: &Device) -> Self {
        let xy: Vec<_> = (0..PIXELS).map(pixel_center).collect();
        Self {
            x: tensor(&xy.iter().map(|p| p[0]).collect::<Vec<_>>(), device),
            y: tensor(&xy.iter().map(|p| p[1]).collect::<Vec<_>>(), device),
        }
    }
}

fn pixel_center(i: usize) -> [f32; 2] {
    [
        ((i % SIZE) as f32 + 0.5) / SIZE as f32 * 1.2 - 0.6,
        0.6 - ((i / SIZE) as f32 + 0.5) / SIZE as f32 * 1.2,
    ]
}

// Positive inside for counterclockwise triangles. Fixed topology, winding, positive
// depth and front-to-back ordering are intentional constraints of this probe.
fn edge(a: &[Tensor<1>; 2], b: &[Tensor<1>; 2], x: Tensor<1>, y: Tensor<1>) -> Tensor<1> {
    (b[0].clone() - a[0].clone()) * (y - a[1].clone())
        - (b[1].clone() - a[1].clone()) * (x - a[0].clone())
}

fn render(vertices: &Tensor<1>, colors: &Tensor<1>, camera: [f32; 2], grid: &Grid) -> Tensor<3> {
    let device = vertices.device();
    let mut transmittance = Tensor::<1>::ones([PIXELS], &device);
    let mut rgb: [Tensor<1>; 3] = std::array::from_fn(|_| Tensor::zeros([PIXELS], &device));
    for triangle in 0..2 {
        let base = triangle * 9;
        let z: [Tensor<1>; 3] = std::array::from_fn(|v| component(vertices, base + v * 3 + 2));
        let p: [[Tensor<1>; 2]; 3] = std::array::from_fn(|v| {
            std::array::from_fn(|axis| {
                (component(vertices, base + v * 3 + axis) - camera[axis]) / z[v].clone()
            })
        });
        let e: [Tensor<1>; 3] =
            std::array::from_fn(|v| edge(&p[v], &p[(v + 1) % 3], grid.x.clone(), grid.y.clone()));
        let mut alpha = Tensor::<1>::ones([PIXELS], &device);
        for v in 0..3 {
            let next = (v + 1) % 3;
            let len = ((p[next][0].clone() - p[v][0].clone()).powi_scalar(2)
                + (p[next][1].clone() - p[v][1].clone()).powi_scalar(2)
                + 1e-12)
                .sqrt();
            let distance = e[v].clone() / len / SOFTNESS;
            let coverage = (distance.neg().exp() + 1.0).recip();
            alpha = alpha * coverage;
        }
        // Barycentric numerator / depth; common signed-area divisor cancels.
        // Clamp only outside the triangle, where soft coverage extends beyond its edges.
        let weights: [Tensor<1>; 3] =
            std::array::from_fn(|v| e[(v + 1) % 3].clone().clamp_min(0.0) / z[v].clone());
        let denom = (weights[0].clone() + weights[1].clone() + weights[2].clone()).clamp_min(1e-12);
        let blend = transmittance.clone() * alpha.clone();
        for (channel, out) in rgb.iter_mut().enumerate() {
            let color = (0..3)
                .map(|v| weights[v].clone() * component(colors, base + v * 3 + channel))
                .reduce(|a, b| a + b)
                .unwrap()
                / denom.clone();
            *out = out.clone() + blend.clone() * color;
        }
        transmittance = transmittance * (alpha.neg() + 1.0);
    }
    Tensor::stack::<2>(rgb.to_vec(), 1).reshape([SIZE, SIZE, 3])
}

async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("GPU readback")
        .try_to_vec()
        .expect("f32 tensor")
}

async fn value(t: Tensor<1>) -> f32 {
    read(t).await[0]
}

fn cpu_edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn cpu_pixel(vertices: &[f32], colors: &[f32], camera: [f32; 2], pixel: [f32; 2]) -> [f32; 3] {
    let mut out = [0.0; 3];
    let mut transmission = 1.0;
    for triangle in 0..2 {
        let base = triangle * 9;
        let p: [[f32; 2]; 3] = std::array::from_fn(|v| {
            std::array::from_fn(|axis| {
                (vertices[base + v * 3 + axis] - camera[axis]) / vertices[base + v * 3 + 2]
            })
        });
        let e: [f32; 3] = std::array::from_fn(|v| cpu_edge(p[v], p[(v + 1) % 3], pixel));
        let mut alpha = 1.0;
        for v in 0..3 {
            let next = (v + 1) % 3;
            let len =
                ((p[next][0] - p[v][0]).powi(2) + (p[next][1] - p[v][1]).powi(2) + 1e-12).sqrt();
            alpha *= 1.0 / (1.0 + (-e[v] / len / SOFTNESS).exp());
        }
        let w: [f32; 3] =
            std::array::from_fn(|v| e[(v + 1) % 3].max(0.0) / vertices[base + v * 3 + 2]);
        let sum = w.iter().sum::<f32>().max(1e-12);
        for ch in 0..3 {
            out[ch] += transmission
                * alpha
                * (0..3)
                    .map(|v| w[v] * colors[base + v * 3 + ch])
                    .sum::<f32>()
                / sum;
        }
        transmission *= 1.0 - alpha;
    }
    out
}

fn valid_geometry(vertices: &[f32]) -> bool {
    if !vertices.iter().all(|v| v.is_finite()) {
        return false;
    }
    let front_max = (0..3)
        .map(|v| vertices[v * 3 + 2])
        .fold(f32::NEG_INFINITY, f32::max);
    let back_min = (0..3)
        .map(|v| vertices[9 + v * 3 + 2])
        .fold(f32::INFINITY, f32::min);
    if front_max >= back_min || vertices.chunks_exact(3).any(|v| v[2] <= 0.1) {
        return false;
    }
    CAMERAS.into_iter().chain([HELD_OUT]).all(|camera| {
        (0..2).all(|t| {
            let p: [[f32; 2]; 3] = std::array::from_fn(|v| {
                let i = t * 9 + v * 3;
                [
                    (vertices[i] - camera[0]) / vertices[i + 2],
                    (vertices[i + 1] - camera[1]) / vertices[i + 2],
                ]
            });
            cpu_edge(p[0], p[1], p[2]) > 1e-4
        })
    })
}

fn initial() -> (Vec<f32>, Vec<f32>) {
    let vertices = TARGET_VERTICES
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            v + if i % 3 == 2 {
                0.12
            } else if i % 2 == 0 {
                0.07
            } else {
                -0.06
            }
        })
        .collect();
    let colors = TARGET_COLORS.iter().map(|&v| v * 0.75 + 0.08).collect();
    (vertices, colors)
}

fn mse(a: Tensor<3>, b: Tensor<3>) -> Tensor<1> {
    (a - b).powi_scalar(2).mean()
}

// Host Adam keeps this correctness experiment independent of Brush's live optimizer.
// All rendering and differentiation still execute on the GPU.
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
}
impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
    fn step(&mut self, values: &mut [f32], grads: &[f32], step: usize) {
        for i in 0..values.len() {
            assert!(grads[i].is_finite(), "non-finite gradient {i}");
            self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i];
            self.v[i] = 0.999 * self.v[i] + 0.001 * grads[i].powi(2);
            let m = self.m[i] / (1.0 - 0.9_f32.powi(step as i32));
            let v = self.v[i] / (1.0 - 0.999_f32.powi(step as i32));
            values[i] -= 0.01 * m / (v.sqrt() + 1e-8);
        }
    }
}

async fn check_forward(device: &Device, grid: &Grid) {
    let mut max_error = 0.0_f32;
    for camera in CAMERAS {
        let gpu = read(render(
            &tensor(&TARGET_VERTICES, device),
            &tensor(&TARGET_COLORS, device),
            camera,
            grid,
        ))
        .await;
        for i in 0..PIXELS {
            let reference = cpu_pixel(&TARGET_VERTICES, &TARGET_COLORS, camera, pixel_center(i));
            for ch in 0..3 {
                assert!(gpu[i * 3 + ch].is_finite());
                max_error = max_error.max((gpu[i * 3 + ch] - reference[ch]).abs());
            }
        }
    }
    assert!(max_error < 2e-4, "CPU/GPU mismatch: {max_error}");
    // Check observable occlusion, not only two implementations of the same formula.
    let center = cpu_pixel(&TARGET_VERTICES, &TARGET_COLORS, CAMERAS[0], [0.0, 0.0]);
    assert!(
        center[0] > center[2] * 2.0,
        "front red triangle must occlude blue triangle"
    );
    let outside = cpu_pixel(&TARGET_VERTICES, &TARGET_COLORS, CAMERAS[0], [0.59, 0.59]);
    assert!(
        outside.iter().all(|&v| v < 0.01),
        "outside should approach black"
    );
    println!(
        "PASS forward: 3 views, CPU/GPU max absolute error {max_error:.2e}; overlap and background"
    );
}

async fn check_gradients(device: &Device, grid: &Grid) {
    let (vertices, colors) = initial();
    let v = tensor(&vertices, device).require_grad();
    let c = tensor(&colors, device).require_grad();
    let target = render(
        &tensor(&TARGET_VERTICES, device),
        &tensor(&TARGET_COLORS, device),
        CAMERAS[1],
        grid,
    )
    .detach();
    let grads = mse(render(&v, &c, CAMERAS[1], grid), target.clone()).backward();
    let analytic = [
        read(v.grad(&grads).expect("vertex gradients")).await,
        read(c.grad(&grads).expect("color gradients")).await,
    ];
    let mut worst = 0.0_f32;
    let mut nonzero = 0;
    for group in 0..2 {
        for index in 0..18 {
            let mut losses = [0.0; 2];
            for (side, delta) in [-0.001, 0.001].into_iter().enumerate() {
                let mut params = [vertices.clone(), colors.clone()];
                params[group][index] += delta;
                losses[side] = value(mse(
                    render(
                        &tensor(&params[0], device),
                        &tensor(&params[1], device),
                        CAMERAS[1],
                        grid,
                    ),
                    target.clone(),
                ))
                .await;
            }
            let numeric = (losses[1] - losses[0]) / 0.002;
            let exact = analytic[group][index];
            assert!(exact.is_finite() && numeric.is_finite());
            let error = (numeric - exact).abs() / (numeric.abs() + exact.abs()).max(1e-3);
            assert!(
                error < 0.03,
                "gradient group={group} index={index}: autodiff={exact}, numeric={numeric}, error={error}"
            );
            worst = worst.max(error);
            if exact.abs() > 1e-6 {
                nonzero += 1;
            }
        }
    }
    assert!(
        gradients_exercised(nonzero),
        "gradient check must exercise geometry and colors, got {nonzero}/36"
    );
    println!(
        "PASS gradients: all 36 position/color components, {nonzero} nonzero; worst normalized error {worst:.2e}"
    );
}

fn gradients_exercised(nonzero: usize) -> bool {
    nonzero == 36
}

fn write_ppm(path: &std::path::Path, rgb: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    write!(file, "P6\n{SIZE} {SIZE}\n255\n")?;
    file.write_all(
        &rgb.iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect::<Vec<_>>(),
    )
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.steps < 100 {
        return Err("use at least 100 steps for the recovery gate".into());
    }
    let setup = burn_wgpu::init_setup_async::<burn_wgpu::graphics::Metal>(
        &burn_wgpu::WgpuDevice::default(),
        burn_wgpu::RuntimeOptions::default(),
    )
    .await;
    let info = setup.adapter.get_info();
    println!(
        "Triangle correctness probe: {} ({:?}), {SIZE}x{SIZE}, 2 triangles",
        info.name, info.backend
    );
    let device = Device::from(burn_wgpu::WgpuDevice::default()).autodiff();
    let grid = Grid::new(&device);
    assert!(valid_geometry(&TARGET_VERTICES));
    check_forward(&device, &grid).await;
    check_gradients(&device, &grid).await;

    let targets: Vec<_> = CAMERAS
        .into_iter()
        .map(|camera| {
            render(
                &tensor(&TARGET_VERTICES, &device),
                &tensor(&TARGET_COLORS, &device),
                camera,
                &grid,
            )
            .detach()
        })
        .collect();
    let held_target = render(
        &tensor(&TARGET_VERTICES, &device),
        &tensor(&TARGET_COLORS, &device),
        HELD_OUT,
        &grid,
    )
    .detach();
    let (mut vertices, mut colors) = initial();
    let before = render(
        &tensor(&vertices, &device),
        &tensor(&colors, &device),
        HELD_OUT,
        &grid,
    );
    let initial_error = value(mse(before.clone(), held_target.clone())).await;
    let mut vertex_adam = Adam::new(18);
    let mut color_adam = Adam::new(18);
    let start = Instant::now();
    for step in 1..=args.steps {
        let v = tensor(&vertices, &device).require_grad();
        let c = tensor(&colors, &device).require_grad();
        let loss = CAMERAS
            .iter()
            .zip(&targets)
            .map(|(&camera, target)| mse(render(&v, &c, camera, &grid), target.clone()))
            .reduce(|a, b| a + b)
            .unwrap()
            / 3.0;
        let grads = loss.backward();
        vertex_adam.step(
            &mut vertices,
            &read(v.grad(&grads).expect("vertex gradients")).await,
            step,
        );
        color_adam.step(
            &mut colors,
            &read(c.grad(&grads).expect("color gradients")).await,
            step,
        );
        assert!(
            valid_geometry(&vertices),
            "step {step}: violated positive depth, winding or fixed depth-order assumptions"
        );
    }
    let elapsed = start.elapsed();
    let after = render(
        &tensor(&vertices, &device),
        &tensor(&colors, &device),
        HELD_OUT,
        &grid,
    );
    let final_error = value(mse(after.clone(), held_target.clone())).await;
    assert!(
        final_error.is_finite() && final_error < initial_error * 0.1,
        "recovery failed: held-out MSE {initial_error:.6e} -> {final_error:.6e}"
    );
    println!(
        "PASS recovery: held-out MSE {initial_error:.6e} -> {final_error:.6e} ({:.1}% lower)",
        (1.0 - final_error / initial_error) * 100.0
    );
    println!(
        "{} steps in {:.2}s ({:.2} ms/step, dense reference including host optimizer/readback; not a throughput benchmark)",
        args.steps,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / args.steps as f64
    );
    if let Some(dir) = args.output {
        std::fs::create_dir_all(&dir)?;
        for (name, img) in [
            ("target", held_target),
            ("before", before),
            ("after", after),
        ] {
            write_ppm(&dir.join(format!("{name}.ppm")), &read(img).await)?;
        }
        println!("Held-out images: {}", dir.display());
    }
    println!(
        "PASS all gates. No claim of exact geometry recovery, live suitability, or Triangle Splatting+ parity."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_geometry_and_color_gradients_are_exercised() {
        assert!(gradients_exercised(36));
        // A disconnected depth component used to pass the >=30 gate.
        assert!(!gradients_exercised(35));
        assert!(!gradients_exercised(30));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tilted_triangle_detects_missing_perspective_weights() {
        let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let grid = Grid::new(&device);
        let mut vertices = TARGET_VERTICES;
        vertices[..9].copy_from_slice(&[-0.375, -0.375, 1.5, 0.5, -0.5, 2.0, 0.0, 0.625, 2.5]);
        let colors = [
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        let i = SIZE / 2 * SIZE + SIZE / 2;
        let [x, y] = pixel_center(i);
        // Projected vertices (-.25,-.25), (.25,-.25), (0,.25).
        let b2 = (y + 0.25) / 0.5;
        let b1 = (x + 0.25 - 0.25 * b2) / 0.5;
        let b0 = 1.0 - b1 - b2;
        let w = [b0 / 1.5, b1 / 2.0, b2 / 2.5];
        let expected = w.map(|v| v / w.iter().sum::<f32>());
        assert!(
            (expected[0] - b0).abs() > 0.05,
            "fixture must distinguish affine interpolation"
        );
        let gpu = read(render(
            &tensor(&vertices, &device),
            &tensor(&colors, &device),
            CAMERAS[0],
            &grid,
        ))
        .await;
        let cpu = cpu_pixel(&vertices, &colors, CAMERAS[0], [x, y]);
        for actual in [&gpu[i * 3..i * 3 + 3], &cpu[..]] {
            let alpha: f32 = actual.iter().sum();
            for ch in 0..3 {
                assert!((actual[ch] / alpha - expected[ch]).abs() < 2e-4);
            }
        }
    }

    #[test]
    fn geometry_guard_rejects_invalid_probe_scenes() {
        assert!(valid_geometry(&TARGET_VERTICES));
        assert!(valid_geometry(&initial().0));
        let mut v = TARGET_VERTICES;
        v[2] = -1.0;
        assert!(!valid_geometry(&v));
        v = TARGET_VERTICES;
        v.swap(0, 3);
        v.swap(1, 4);
        v.swap(2, 5);
        assert!(!valid_geometry(&v));
        v = TARGET_VERTICES;
        v[2] = 4.0;
        assert!(!valid_geometry(&v));
        v = TARGET_VERTICES;
        v[0] = f32::NAN;
        assert!(!valid_geometry(&v));
    }

    #[test]
    fn scalar_reference_has_occlusion_and_background() {
        let center = cpu_pixel(&TARGET_VERTICES, &TARGET_COLORS, CAMERAS[0], [0.0, 0.0]);
        assert!(center[0] > 0.7 && center[2] < 0.4);
        let outside = cpu_pixel(&TARGET_VERTICES, &TARGET_COLORS, CAMERAS[0], [0.59, 0.59]);
        assert!(outside.iter().all(|&v| v < 0.01));
    }
}
