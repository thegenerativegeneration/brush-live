# Offline Gaussian / vertex reconstruction experiment

Milestones 1–3 only: a common offline backend contract, independently implemented differentiable triangle rendering on Metal, and a frozen fixed-capture comparison. Existing live Gaussian sessions, guidance, iOS and production preview/checkpoint formats are unchanged.

**Measured outcome:** the native pipeline works, but this prototype is not a quality/performance replacement for Brush. On the Lenzburg subset, the 300-second triangle run reached 21.78 dB ROI PSNR versus 24.63 dB for the constrained Gaussian control. Coverage improved, while depth accuracy inside the original ROI worsened. See [the experiment report](../../docs/experiments/2026-10-16-lenzburg-vertex-backend.md).

## Freeze inputs, then run

From the Brush workspace root:

```sh
cargo build --release -p brush-recon-bench --features brush-cube/metal
cargo run --release -p brush-recon-bench --features brush-cube/metal -- preflight \
  --segment '/Volumes/T7 Shield/Temp/data/castle_lenzburg/2026-10-15T14-25-34Z/segment-0' \
  --output target/vertex-runs/lenzburg-preflight
cargo run --release -p brush-recon-bench --features brush-cube/metal -- run \
  --manifest target/vertex-runs/lenzburg-preflight/manifest.json \
  --backend triangle --train-seconds 60 \
  --output target/vertex-runs/lenzburg-triangle-60
```

Use fresh output names if these directories already exist. On the development Mac, this linked worktree does not inherit the original parent's compiler configuration; the measured runs used matching overrides:

```sh
export RUSTC="$HOME/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc"
export RUSTDOC="$HOME/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustdoc"
```

Preflight defaults: IDs 174–213; every fifth item held out at offset 4 (32 train/8 held-out); RGB width 320; seed 0. It loads raw `transforms.json`, JPEG and float16 metric LiDAR/confidence—not an existing trained splat, mono depth, or separate COLMAP calibration. Training anchors are first ID + 0/10/20/30 and must be training views. Nonempty support and training-only depth reprojection must pass before a manifest is written.

The output directory must be new and outside the capture. Inputs remain read-only. The manifest hashes source files, seed, masks and configurations; `run` verifies them and rejects raw selection overrides. Changing a subset/configuration requires a new preflight, never a silent fallback. macOS may request removable-volume access; a blocked OS file open must be resolved before benchmarking.

Choose `gaussian` or `triangle` and exactly one of `--train-seconds` or `--steps` (at least 2). Default warmup is 2 steps in a discarded model. Every run starts with fresh parameters and optimizer state; a 300-second result does not resume a 60-second result.

Optimization time sums synchronized `train_step` wall intervals. Decode, initialization, snapshots, held-out evaluation and artifact writing are excluded and separately reported. The last completed step may overshoot the budget; actual time is recorded. Source hashes are checked again afterward.

A successful CLI exit means the computational/artifact pipeline completed, **not** that the research gate passed. Inspect `metrics.json`.

## Outputs

- `run.json`: completion marker, provenance/configuration, per-view/aggregate metrics, timings and curve.
- `metrics.json`, `curve.json`: quality/gate, sampled training-loss minimum, memory accounting and optimization history.
- `initial.ply`, `final.ply`: Gaussian PLY or indexed RGB triangle mesh, according to backend.
- Triangle `initial.state.json`, `final.state.json`: versioned experimental parameters/configuration; no optimizer-resume claim. Mesh PLY does not encode training coverage/weight logits and is for geometry inspection.
- `images/`: initial/final/target RGB, immutable masks, native-grid depth views and little-endian float32 depth arrays with layout metadata.
- `capture-manifest.json`, `seed.json`, `roi-*.bin`: copied frozen inputs that can be revalidated/replayed against the original capture.

Outputs are never overwritten. Backend failures write `failure.json` when possible and exit unsuccessfully; interrupted runs without `run.json` are incomplete.

## Interpretation and limitations

RGB MAE/PSNR are reported over both the full image and fixed initial-geometry ROI, micro-aggregated by channel count. Depth uses the native LiDAR grid, confidence 2 and 0.1–8 m. Every valid target stays in the denominator; an unavailable prediction costs 8 m. Coverage/availability are separate metrics. Gaussian depth is unsupported/null, never zero error.

The triangle feasibility gate requires at least 10% held-out ROI RGB MAE improvement and no more than 10% worsening of full-native-grid depth error, with documented near-zero tolerances. This gate can pass through increased coverage even when depth accuracy within the original ROI degrades; always inspect the secondary ROI depth metric and images.

Held-out data never seeds or optimizes the model. The Lenzburg window was nominated using depth-support statistics including eventual hold-outs, so it is not a selection-blind benchmark. The supported region is mainly nearby courtyard/path and low-wall surfaces, not the whole castle or sky.

Triangle training adds depth and a seed-position prior; the Gaussian control is degree-0 RGB-only L1 with the same masks/normalization and no refinement/noise. Initial opacity, parameterization and learning rates differ. Triangle sigma/opacity-floor schedules change appearance independently of geometry. These are single implementation-feasibility runs, not paper parity, isolated renderer throughput or an unbiased method ranking.

Renderer constraints: pinhole cameras, fixed indexed topology, per-vertex RGB/weights, centroid depth ordering, conservative near-plane culling, no densification/pruning/remeshing/SH. Intersecting triangles retain approximate splat ordering rather than exact per-pixel z-buffer visibility. No phone/live integration is included.

Triangle renderer memory is a conservative buffer-reservation bound, not a measured GPU peak. Gaussian reservation is uninstrumented/null. Process footprint is measured separately; its lifetime peak includes warmup/evaluation/artifacts/runtime.

## Verification

```sh
cargo test --release -p brush-recon-bench -p brush-triangle-render -p brush-triangle-train \
  -p brush-recon -p brush-render -p brush-train -p brush-dataset --features brush-cube/metal
cargo test --release -p brush-render --features brush-cube/metal --example triangle_probe
```

See the [design](../../docs/superpowers/specs/2026-10-16-vertex-backend-design.md) for formulas, pinned research references, gates and deliberate deviations.
