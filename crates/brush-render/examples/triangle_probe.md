# Native-Mac triangle correctness probe

From the Brush workspace root, on a Mac with a Metal GPU:

```sh
cargo test --release -p brush-render --features brush-cube/metal --example triangle_probe
cargo run --release -p brush-render --features brush-cube/metal --example triangle_probe -- --output /path/to/probe-images
```

The executable exits unsuccessfully if any correctness gate fails:

1. GPU rendering agrees with an independent scalar CPU implementation in three views
   (maximum absolute channel error below `2e-4`). Also check foreground occlusion
   and near-black background pixels.
2. Autodiff agrees with central finite differences for all 36 position/color
   components (`epsilon = 1e-3`, normalized error below `0.03`, with a `1e-3`
   denominator floor). All 36 components, including every depth component,
   must have nontrivial gradients. A tilted unequal-depth fixture independently
   checks perspective-correct color weights on CPU and GPU.
3. Jointly optimize geometry and colors against three known views for 200 Adam
   steps. A fourth, held-out view must improve its MSE by more than 90%.

Optional `--output` writes `target.ppm`, `before.ppm`, and `after.ppm` for the
held-out camera. Without it, no images are written. `--steps` changes the recovery
run length (minimum 100); the same acceptance criterion still applies.

## What this implements

Two overlapping triangles at 48 × 48 pixels, pinhole projection, per-vertex RGB,
perspective-correct color weights inside each triangle, and black background.
Coverage is the product of three sigmoid signed edge distances; distances are
normalized by projected edge lengths. Coverage supplies differentiable soft edges.
Outside the triangle, negative barycentric numerators are clamped to avoid
extrapolated colors. Front-to-back alpha compositing exercises occlusion.

All image and gradient operations use Burn on Metal. A tiny host-side Adam
optimizer reads back the 36 gradients each iteration. GPU readbacks synchronize
the timing. First-use compilation can be included in the reported time.

## Deliberate limitations

- This is an independently written soft-triangle reference, **not** a reproduction
  of Triangle Splatting+ or MeshSplatting, and includes no code from those projects.
- Depth order is fixed. Every front-triangle vertex must remain in front of every
  back-triangle vertex. Positive depth, winding, non-degeneracy and ordering are
  checked for all optimization steps and all cameras used by the experiment.
- There is no clipping, learned opacity, topology optimization, depth loss, SH,
  tile binning, sorting, custom kernel, live ingestion, or Gaussian baseline.
- Dense tensor operations are intentionally unoptimized. Timings are diagnostic,
  not evidence of scalable renderer speed or on-phone viability.
- Recovery means lower held-out image error; it does not establish exact vertex
  recovery, surface quality on real data, or suitability for guidance scoring.

Only this optional example and its manifest registration are added. Existing
renderer APIs and the live engine are unchanged.
