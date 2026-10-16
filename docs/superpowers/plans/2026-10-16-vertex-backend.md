# Vertex Backend Milestones 1–3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This document is a proposal awaiting user review; do not begin implementation merely because this file exists.

**Goal:** Run a native-Metal fixed-topology triangle reconstruction and Gaussian baseline on a frozen Lenzburg capture subset through one offline backend contract.

**Architecture:** Add a narrow representation-neutral input/training/render contract, separate triangle renderer/trainer crates, and an offline benchmark app containing both adapters. Preserve the live Gaussian engine. Establish a mathematical CPU reference before custom GPU forward/backward and real-data training.

**Tech Stack:** Rust 2024, existing pinned Burn/CubeCL/wgpu, native Metal, existing image/serde/glam/half dependencies; no CUDA/PyTorch runtime.

**Spec:** `docs/superpowers/specs/2026-10-16-vertex-backend-design.md` (read in full; it owns formulas, units, masks, thresholds, defaults and stop conditions).

## Global Constraints

- Work only in the `triangle-metal-probe` worktree; preserve its existing synthetic-probe changes and unrelated original-checkout edits.
- Capture source is read-only. Output must be explicit, outside the source tree, and never overwrite source images/models or a previous run without explicit permission.
- Scope is milestones 1–3 only. No live/FFI, guidance, pose correction, adaptive topology, remeshing or iOS integration.
- Pinhole-only, metric camera-z depth, fixed topology, independent implementation of the documented reference subset; no paper-parity claim.
- Raw ARKit/LiDAR and COLMAP are separate calibration/world sources; never mix them.
- Training/initialization consumes training frames only. Immutable ROI masks cannot shrink with optimized coverage.
- No copying inherited research-only implementation code. Use the pinned links as research references, not vendored source.
- Empty input and excessive intersection counts return defined results/errors, not invalid GPU dispatches or silent truncation.
- Acceptance values and numeric defaults are copied from the spec through the tasks below; changes require a recorded rationale and a new manifest for data/config changes.
- Each task: write failing tests, verify intended failure, implement, run targeted and affected-package suites, then commit only this task's files. Never claim a full-workspace suite passed from a package-level run.

## Review Focus

1. Held-out leakage through seeding, masks, baseline initialization or checkpoint selection → tests in Tasks 3, 8 and 12.
2. Capture camera orientation, pixel centers and metric depth alignment → tests in Tasks 3 and 4.
3. Shared vertices and gradients of coverage/depth/normal normalization → tests in Tasks 5 and 7.
4. Tile-list overflow, empty GPU dispatches and conservative near culling → tests in Tasks 5–7.
5. Misleading quality/performance comparisons from changing support, objective differences or asynchronous timing → tests in Tasks 9–12.

## File ownership and dependency graph

- `crates/brush-recon/src/{lib,frame,backend,error}.rs`: neutral types and fixed-capture contract; depends on existing camera/image/tensor types, not either trainer.
- `crates/brush-recon/src/capture/{mod,manifest,raw,mask}.rs`: validated raw-capture loading, immutable split/masks and provenance.
- `crates/brush-triangle-render/src/{lib,model,reference,project,binning,forward,backward,burn_glue}.rs`, `src/kernels/`: triangle state, reference math, custom GPU path and autodiff glue.
- `crates/brush-triangle-train/src/{lib,seed,loss,optimizer,train,schedule,artifact}.rs`: fixed initialization, training, schedules and experimental state export.
- `apps/brush-recon-bench/src/{main,run,gaussian,triangle,metrics,report}.rs`: adapters, one generic runner, timing and inspection artifacts.
- Each new crate/app has `Cargo.toml`; root `Cargo.toml` gains workspace members. Tests live in sibling `_tests.rs` modules or `tests/` named below.
- Existing probe changes remain in `crates/brush-render/examples/triangle_probe.{rs,md}`; root/current rendering APIs are not renamed.
- Dependency direction: triangle renderer → neutral contract → existing `brush-render`; triangle trainer → renderer + contract; app → contract + both trainers. `brush-recon` must not depend on triangle renderer/trainers (avoids cycles).

## Shared interface ledger

All items below are defined in Task 2 unless a producing task is named. Use these names consistently; errors use `ReconError` with contextual input paths/IDs and typed unsupported-output/budget failures.

```rust
// brush_recon::{frame,backend}; fields are public experiment interfaces.
struct MetricDepth { size: UVec2, values_m: Vec<f32>, confidence: Option<Vec<u8>> }
struct ReconFrame {
    id: u64, camera: Camera, rgb: Arc<RgbImage>,
    lidar: Option<Arc<MetricDepth>>, roi: Arc<Vec<u8>>,
}
enum BackendKind { Gaussian, Triangle }
struct Capabilities { depth: bool, normals: bool }
struct RenderRequest { camera: Camera, size: UVec2, background: Vec3, depth: bool, normals: bool }
struct RenderResult {
    rgb: Tensor<3>, alpha: Tensor<2>,
    depth_m: Option<Tensor<2>>, normals: Option<Tensor<3>>,
}
struct StepStats { iteration: u64, rgb_l1: f32, depth_l1_m: Option<f32>, total_loss: f32 }
struct StepContext { iteration: u64, progress: f32 } // zero-based iteration, progress in [0,1]
struct ModelStats { vertices: u32, triangles: u32, gaussians: u32, renderer_peak_bytes: u64 }
struct ExperimentalState { kind: BackendKind, format_version: u32, payload: Vec<u8> }
trait ReconstructionBackend {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> Capabilities;
    async fn initialize(&mut self, train: &[Arc<ReconFrame>]) -> Result<(), ReconError>;
    async fn train_step(&mut self, frame: &ReconFrame, context: StepContext) -> Result<StepStats, ReconError>;
    async fn render(&self, request: &RenderRequest) -> Result<RenderResult, ReconError>;
    async fn synchronize(&self) -> Result<(), ReconError>;
    async fn export_state(&self) -> Result<ExperimentalState, ReconError>;
    fn stats(&self) -> ModelStats;
}
```

`rgb` and normals are `[H,W,3]`; alpha/depth `[H,W]`. Unsupported requested channels return `ReconError::UnsupportedOutput`; unrequested optional channels are `None`. `roi` has exactly `W*H` bytes with values 0/255. `StepStats.rgb_l1` is the training scalar normalized by `3*H*W`, not the evaluator's per-ROI MAE. Configs are constructor arguments on concrete adapters, not hidden globals. Generic async methods avoid dyn object-safety requirements. Task 9 adds `mask_scene_batch(frame: &ReconFrame) -> Result<SceneBatch, ReconError>`. `export_state` saves experimental final parameters (not a resumable optimizer); warmup reset reconstructs a fresh adapter from the frozen seed/config, never from an optimized state.

---

### Task 1: Strengthen the existing smoke test

**Files:** Modify `crates/brush-render/examples/triangle_probe.rs` and `.md` only.
**Consumes:** Existing smoke renderer. **Produces:** Stronger smoke gates; not the new reference renderer.

- [ ] Add `tilted_triangle_detects_missing_perspective_weights` with unequal vertex depths and an analytical pixel-color expectation that differs from screen-linear interpolation. Verify the expectation would reject removal of reciprocal-depth weights.
- [ ] Add `all_geometry_and_color_gradients_are_exercised`: require all 36 gradients above 1e-6, including six depth entries; show the gate fails with a disconnected depth component before restoring it.
- [ ] Run the explicit example test target, then the native probe; preserve 90% held-out recovery acceptance and update docs. Commit the smoke changes.

Verification: `cargo test --release -p brush-render --features brush-cube/metal --example triangle_probe` and `cargo run --release -p brush-render --features brush-cube/metal --example triangle_probe`. Expected: example tests plus forward, all-36 gradient and recovery gates pass.

### Task 2: Introduce the narrow offline backend contract

**Files:** Create `crates/brush-recon/Cargo.toml`, `src/{lib,frame,backend,error}.rs`, `tests/contract.rs`; modify root `Cargo.toml` workspace membership.
**Consumes:** Existing `Camera`, `SceneBatch`, Burn tensors, image/glam. **Produces:** Shared interface ledger above and validated constructors for frames/depth/results.

- [ ] Write `frame_rejects_mismatched_roi`, `depth_rejects_invalid_shape`, `requested_unsupported_channel_is_error`, and `result_channels_match_requested_size`; assert exact shapes and `None` semantics from the ledger.
- [ ] Run `cargo test --release -p brush-recon` and observe intended failures, then implement the types and validation.
- [ ] Add a small generic test runner using a deterministic test backend to prove async trait use without trait objects. No production Gaussian/session refactor in this task.
- [ ] Run the package suite and existing `brush-render` suite. Commit.

### Task 3: Load and freeze the real capture without side effects

**Files:** Create `brush-recon/src/capture/{mod,manifest,raw}.rs`, `brush-recon/tests/capture.rs`; expose module in `src/lib.rs` and add manifest dependencies as needed.
**Consumes:** Task 2 types. **Produces:**
`CaptureSelection { first_id:u64, last_id:u64, holdout_every:usize, holdout_offset:usize, size:UVec2 }`,
`CaptureSplit { train:Vec<Arc<ReconFrame>>, holdout:Vec<Arc<ReconFrame>>, manifest:CaptureManifest }`,
`load_capture(segment:&Path, selection:&CaptureSelection) -> Result<CaptureSplit,ReconError>`.

- [ ] Create tiny generated fixtures testing numeric frame order, per-frame intrinsics, ARKit/OpenGL → Brush orientation, pixel-center depth lookup, confidence masks, float16 endianness and metres.
- [ ] Add `split_174_213_is_32_train_8_holdout`, `heldout_payloads_are_not_loaded_for_training`, `malformed_depth_is_contextual_error`, `appledouble_entries_are_ignored`, and `loading_does_not_write_to_source`. No real capture in automated fixtures.
- [ ] Run the tests RED, then implement raw `transforms.json` + JPEG + `.f16/.conf` reading. Reuse documented camera conversion conventions rather than `decode_keyframe`, which writes JPEGs to a session directory. Normalize the homogeneous matrix scale where appropriate; validate finite rigid poses after conversion. Reject unsupported calibration.
- [ ] Freeze source hashes, source pose/calibration identity and exact splits. Do not auto-discover `splat.ply` as initialization (the existing generic dataset loader may otherwise choose it).
- [ ] Run package suite; commit. Integration preflight on the real capture waits for Task 10's CLI.

### Task 4: Define indexed triangle state and depth-grid initialization

**Files:** Create `brush-triangle-render/Cargo.toml`, `src/{lib,model}.rs`; create `brush-triangle-train/Cargo.toml`, `src/{lib,seed}.rs`, `tests/seed.rs`; update root members.
**Consumes:** Task 2 `ReconFrame`/`MetricDepth`. **Produces:**
`TriangleSeed { positions:Vec<[f32;3]>, faces:Vec<[u32;3]>, colors:Vec<[f32;3]> }` owned/exported by `brush_triangle_render::model`,
`seed_from_training(train:&[Arc<ReconFrame>], cfg:&SeedConfig) -> Result<TriangleSeed,ReconError>`,
`TriangleModel::from_seed(seed:&TriangleSeed, device:&Device) -> Result<TriangleModel,ReconError>` with the four tensor fields/shapes from the spec.

- [ ] Write plane backproject/reproject, shared grid vertex indexing, invalid-confidence holes, 5% depth-jump rejection, area rejection, round-robin face budget and train-only anchor tests. Assert anchors 174/184/194/204; missing anchors fail explicitly.
- [ ] Run RED, then implement seed generation with stride 4, confidence 2, 0.1–8 m, area 1e-8 m² and at most 20,000 faces. Do not fuse/weld separate anchors.
- [ ] Test index bounds, RGB/logit conversion using [1e-4,1-1e-4] clamping, initial weight 0.1 via `log(0.1/0.9)`, zero-face typed error for training initialization, and deterministic initialization hash.
- [ ] Run all three new package suites and commit.

### Task 5: Establish the mathematical CPU reference

**Files:** Create `brush-triangle-render/src/reference.rs`, `tests/reference.rs`; update `src/lib.rs`.
**Consumes:** Task 4 seed/model host representation and Task 2 requests. **Produces:**
`ReferenceConfig { sigma:f32, weight_floor:f32, near_m:f32 }`,
`render_reference(seed:&TriangleSeed, rgb_logits:&[[f32;3]], weight_logits:&[f32], request:&RenderRequest, cfg:&ReferenceConfig) -> Result<ReferenceImage,ReconError>`, where `ReferenceImage` holds host vectors for all four channels with ledger shapes.

- [ ] Write analytical tests for reference coverage/inradius, min vertex weight, shared 0.99 alpha cap, black/background compositing, tilted-plane perspective-correct color/depth, normalized geometric normals and alpha-normalized depth.
- [ ] Add zero/all-culled, invalid face indices, either winding, near-plane crossing conservative cull, degenerate and duplicate vertices, equal-depth ties and crossing-triangle approximation fixtures. Exact edge coverage/gradient is zero; min-distance ties choose lowest edge ID, min-weight ties lowest global vertex ID. Discrete policy is explicit; no claim of exact crossing visibility.
- [ ] Run RED, implement spec formulas in independent scalar Rust. Keep fixtures' expectations analytical rather than copies of the implementation.
- [ ] Run renderer suite and commit. This reference replaces neither the existing smoke formula nor Gaussian code.

### Task 6: Implement safe tiled GPU forward rendering

**Files:** Create `brush-triangle-render/src/{project,binning,forward,burn_glue}.rs`, focused `src/kernels/{project,intersections,ranges,rasterize}.rs`, `tests/forward.rs`; update crate dependencies/exports.
**Consumes:** Task 5 reference and Task 4 device model. **Produces:**
`RenderConfig { sigma:f32, weight_floor:f32, near_m:f32, buffer_budget_bytes:u64 }`,
`render_triangles(model:&TriangleModel, request:&RenderRequest, cfg:&RenderConfig) -> Result<RenderResult,ReconError>` plus private backward state.

- [ ] Write Metal CPU/GPU comparisons for all Task 5 fixtures: RGB/alpha/normals max absolute error <2e-4, depth <1e-3 m on valid pixels.
- [ ] Write `empty_model_does_not_dispatch`, `all_culled_returns_background`, `intersection_budget_fails_before_allocation`, `counts_use_checked_arithmetic`, `equal_depth_order_is_deterministic` and partial-edge-tile tests.
- [ ] Run RED, implement 16×16 tile projection/binning/rasterization. Reuse scan and 32-bit radix utilities. Preserve depth then tile ordering with stable passes; prove stability with tests, do not assume an available 64-bit key sort. Initialize all output/counter buffers.
- [ ] Enforce 512 MiB default renderer-buffer budget including sort scratch and saved backward state; validate GPU-read counts before sizing buffers. Return structured failure; no truncated lists.
- [ ] Run forward/reference suites, `brush-scan`/`brush-sort` suites if their APIs are changed (prefer no changes), and existing renderer suite. Commit.

### Task 7: Implement exact piecewise backward and Burn integration

**Files:** Create `brush-triangle-render/src/backward.rs`, `src/kernels/{rasterize_bwd,project_bwd,vertex_reduce}.rs`, `tests/gradients.rs`; extend `burn_glue.rs`.
**Consumes:** Task 6 saved forward state. **Produces:** Autodiff-capable `render_triangles` under the unchanged signature; gradients for positions/RGB logits/weight logits.

- [ ] Write finite-difference tests for all parameters on tilted multiview fixtures, with separate RGB, expected-depth and normal scalar objectives. Test min-weight winners without ties, clamped-alpha zero derivative, coverage/inradius derivatives, shared-vertex gradient sums and finite all-culled gradients.
- [ ] Use epsilon 1e-3; require `abs_error <= 2e-4 + 0.03*max(abs(analytic),abs(numeric))`. Assert chosen components are genuinely nonzero, not merely equal zeros. Test branch boundaries separately rather than applying finite differences across sort/cull/tie changes.
- [ ] Run RED, then implement backward from our mathematical forward, including normal/inradius/interpolation denominators. Accumulate shared vertex contributions safely using explicit reduction (per-face contributions then vertex adjacency reduction); no data races or assumed float-atomic portability.
- [ ] Register forward/backward/Fusion shapes using existing `brush-render` conventions, without modifying Gaussian parameter layouts. Run tests through the same dynamic Burn dispatch used by the CLI, not just direct kernels.
- [ ] Run all triangle-render suites and existing renderer suite; commit.

### Task 8: Freeze support masks and implement losses/schedules

**Files:** Create `brush-recon/src/capture/mask.rs`, `brush-triangle-train/src/{loss,schedule}.rs`, `tests/loss.rs`; extend manifest types.
**Consumes:** Tasks 3–7; initial seed and camera-only projection into all selected views. **Produces:**
`freeze_roi(split:&mut CaptureSplit, seed:&TriangleSeed) -> Result<(),ReconError>` belongs in the app orchestration, using `brush_recon::capture::mask::rasterize_roi(positions:&[[f32;3]], faces:&[[u32;3]], camera:&Camera, size:UVec2, near_m:f32) -> Result<Vec<u8>,ReconError>` (do not add a recon→triangle-render dependency);
`triangle_loss(render:&RenderResult, frame:&ReconFrame, seed_positions:&Tensor<2>, positions:&Tensor<2>, cfg:&LossConfig) -> Result<LossTerms,ReconError>`;
`schedule(progress:f32, cfg:&ScheduleConfig) -> Result<ScheduleState,ReconError>` with sigma/floor values.

- [ ] Write hard-interior mask pixel-center/boundary tests; fixed-mask independence from predicted alpha; held-out RGB/depth mutations cannot change seed/masks; all-zero ROI explicit rejection; RGB denominator `3*H*W` and fixed valid-depth denominator tests; no-valid-depth RGB-only behavior; missing predicted depth incurs 8 m error on the fixed valid-target denominator.
- [ ] Write endpoints/monotonicity/progress-clamping tests for linear sigma 1→0.1 and weight floor 0→0.99. Assert weight activation `floor+(1-floor)*sigmoid(logit)` and derivative scaling, not max-clamping. Test fixed-step first/last progress 0/1 (`N>=2`) and elapsed-budget progress at step start. Missing predicted depth carries finite explicit penalty; it must not introduce NaNs from dividing by zero alpha.
- [ ] Run RED, implement losses and immutable masks. Default objective RGB L1 +0.05×depth L1(m) +0.001×seed displacement MSE(m²). No real-data normal supervision.
- [ ] Run contract/render/loss suites; commit.

### Task 9: Add device-resident optimization and backend adapters

**Files:** Create `brush-triangle-train/src/{optimizer,train}.rs`, `tests/recovery.rs`; create `apps/brush-recon-bench/Cargo.toml`, `src/{gaussian,triangle}.rs`; root workspace member.
**Consumes:** Tasks 2–8 and existing `SplatTrainer`, `SceneBatch`, `SplatData`, `to_init_splats`. **Produces:**
`TriangleTrainer::new(model:TriangleModel, seed:TriangleSeed, config:TrainConfig) -> Result<Self,ReconError>`, `step(&mut self, frame:&ReconFrame, context:StepContext) -> Result<StepStats,ReconError>`;
`GaussianBackend` and `TriangleBackend` implementing Task 2 contract;
`mask_scene_batch(frame:&ReconFrame) -> Result<SceneBatch,ReconError>` packs ROI as RGBA with `AlphaMode::Masked`.

- [ ] Write device Adam versus scalar Adam tests, detach-between-steps and stable optimizer-state tests, nonfinite-state error, 0.01 m displacement cap, parameter-count invariance and absent per-step bulk host readback tests/instrumentation.
- [ ] Write RGB synthetic recovery (at least 90% held-out MSE improvement). For independent tilted-plane geometry recovery, use the spec's synthetic-only depth-MSE + normal cosine-error objective with a fixed interior mask; require each error to improve by 90% within 500 steps. Make learning rates/schedules explicit in fixtures; don't add this objective to production real-data training or change acceptance thresholds to fit failure.
- [ ] Run RED, implement device optimizer with defaults vertices 1e-3 m, RGB/weights 1e-2. Borrow existing optimizer patterns but do not modify the user's original-checkout optimizer work.
- [ ] Implement adapters and Gaussian capabilities (RGB/alpha only). Initialize Gaussian degree-0 colors/positions from the same training-seeded vertices, RGB L1 with same ROI and `3*H*W` normalization, refinement disabled. Set SSIM/LPIPS/background noise/globe/alpha-match/mean-noise/opacity-decay off and no mean-LR decay, using the spec's exact config fields. Log its absent depth/prior losses. New adapter tests verify unsupported channel requests, sparse-ROI loss equivalence to the triangle RGB term and that the existing trainer receives the mask. Preserve production Gaussian defaults/loss code.
- [ ] Run all new package tests plus existing `brush-render`, `brush-dataset`, `brush-train` suites; report pre-existing failures explicitly before proceeding. Commit.

### Task 10: Deliver a preflight and fixed-capture CLI

**Files:** Create `brush-recon-bench/src/{main,run}.rs`, `tests/cli.rs`; extend manifests/config serialization.
**Consumes:** Tasks 3, 4, 8, 9. **Produces:** CLI `brush-recon-bench preflight` and `brush-recon-bench run`; generic `run_backend<B:ReconstructionBackend>(backend:B, split:&CaptureSplit, config:&RunConfig) -> Result<RunRecord,ReconError>`.

- [ ] Write tests for quoted paths with spaces, source-output nesting/symlink alias rejection, existing output rejection, wrong calibration, ID-based subset selection, deterministic training-frame order, fixed-step versus budget exclusivity, and no held-out tensors passed to initialize/train_step.
- [ ] Run RED, implement flags: `--segment`, `--output`, `--first-id 174`, `--last-id 213`, `--holdout-every 5`, `--holdout-offset 4`, `--width 320`, `--seed 0`; run additionally takes `--backend gaussian|triangle` and exactly one of `--steps` / `--train-seconds`. Missing supported depth/overlap produces explicit preflight findings, not a fallback to COLMAP/mono/existing splat.
- [ ] Preflight writes inventory, frozen split/source hashes, seed hash, immutable hard-interior ROI masks and support/overlap visualizations below output. Include transform/wire source differences as metadata only; raw source remains authoritative. Training-view overlap alone controls go/no-go. Record that the nominated window was selected using depth-support statistics including eventual hold-outs; do not claim selection-blind evaluation. Stop if selected training views/anchors are unusable.
- [ ] Run the generated-fixture CLI suite; commit. Run real preflight read-only on the nominated source into a fresh `target/vertex-runs/lenzburg-preflight` directory; review geometric reprojection/support before Task 12.

### Task 11: Add honest metrics, timing and artifacts

**Files:** Create `brush-recon-bench/src/{metrics,report}.rs`, `brush-triangle-train/src/artifact.rs`, `tests/metrics.rs`, `tests/artifact.rs`; extend runner.
**Consumes:** Contract results, frozen masks/split, model state. **Produces:**
`evaluate(backend:&impl ReconstructionBackend, views:&[Arc<ReconFrame>]) -> Result<Evaluation,ReconError>`;
`save_artifacts(run:&RunRecord, output:&Path) -> Result<(),ReconError>`.

- [ ] Write analytical RGB MAE/PSNR tests including exact-match null PSNR plus flag; ROI/full-image separation; micro-aggregation by sample counts; depth fixed-valid-target denominator, missing prediction 8 m penalty and alpha>1e-6 availability; high alpha>=0.9; unsupported Gaussian depth represented explicitly; no held-out-score-driven checkpoint selection. Test zero-baseline gate tolerances (RGB 1e-8, depth 1e-6 m).
- [ ] Test native-grid depth evaluation: request 256×192 renders and consume each native valid LiDAR target once, not duplicated 320×240 samples. Generate native-resolution ROI from the same frozen seed for the additional ROI-depth metric; the primary depth gate still uses all valid native target pixels.
- [ ] Write timing tests with a controlled backend showing synchronization before stopping the optimization timer, no evaluation/decode charged to training, and warmup state discarded/reset before a fresh run. Preserve final completed-step timing even if it crosses the requested budget; report the actual time.
- [ ] Run RED, implement per-view/aggregate metrics, initialization/final images and geometry/cull counts; device timings exclude build/decode/init/eval and include optimization dispatch/readbacks. Report renderer buffer peak separately from measured process footprint; unavailable footprint is null with explanation, never zero.
- [ ] Write raw triangle-state roundtrip and PLY face/color tests. Save before/after/target, depth/ROI images, curves, manifest/config/hardware/revision/dirty hashes and raw state. Refuse overwrites.
- [ ] Run all new package suites; commit.

### Task 12: Run the frozen Lenzburg comparison and document the decision

**Files:** Create `docs/experiments/2026-10-16-lenzburg-vertex-backend.md` and `apps/brush-recon-bench/README.md`; add any missing synthetic regression tests to their owning packages. Large run artifacts remain ignored under `target/vertex-runs/`.
**Consumes:** All prior tasks and reviewed real preflight. **Produces:** Reproducible commands, artifacts, metrics and a go/no-go recommendation for later milestones, not an automatic live rollout.

- [ ] Verify the original capture is unchanged and manifest split remains 32/8, with no trained/existing splat initialization and no held-out leakage. Record source hashes before/after.
- [ ] Run fixed-step smoke and then separate fresh 60 s /300 s training-budget runs for Gaussian and triangle using the frozen manifest. Warm each separately and restore the same initialization; label objective/capability differences. Record failures without silently reducing resolution/faces or changing masks.
- [ ] Evaluate initialization and final models on the eight held-out views. Record both full-frame and immutable-ROI metrics, triangle depth error over all valid held-out pixels, depth availability, stability, memory and timing. Do not tune on these held-out results.
- [ ] Apply the research gate: ROI RGB MAE at least 10% better than initialization; full-native-grid fixed-denominator depth error no more than 10% worse; finite and artifact-complete. Apply spec absolute tolerances for a zero/near-zero initial error. If failed, record a negative result and hypotheses, not a weakened gate or implicit expansion into adaptive geometry. No requirement to beat Gaussian yet.
- [ ] Run `cargo test --release --workspace --features brush-cube/metal` before final handoff when dependencies/platform permit, plus the explicit smoke example target (excluded from default tests). Record every failure/ignored test and whether baseline or introduced; never erase unrelated failures. Run `cargo fmt --all -- --check` and `git diff --check`.
- [ ] Request one read-only whole-change review focused on the five Review Focus items; resolve blocking findings with regression tests. Commit report/docs only after recording actual outputs. Do not merge/push or start milestones 4–8.

## Execution command examples (after implementation)

```sh
cargo run --release -p brush-recon-bench --features brush-cube/metal -- preflight \
  --segment '/Volumes/T7 Shield/Temp/data/castle_lenzburg/2026-10-15T14-25-34Z/segment-0' \
  --output target/vertex-runs/lenzburg-preflight

cargo run --release -p brush-recon-bench --features brush-cube/metal -- run \
  --manifest target/vertex-runs/lenzburg-preflight/manifest.json \
  --backend triangle --train-seconds 60 \
  --output target/vertex-runs/lenzburg-triangle-60
```

Task 10 must make `--manifest` mutually exclusive with raw selection flags during `run`, verify source hashes/config against it, and load stored immutable masks/seed. It does not rerun subset selection or derive masks from the trained model. Repeat the run with `--backend gaussian` and fresh output; likewise each 300 s run starts from the frozen seed rather than continuing a 60 s result.

## Handoff

Review the spec and plan before implementation. Recommended execution: native/inline, one task at a time with one final independent review; most tasks depend on the renderer's exact forward/backward and input contracts, so parallel implementation is more likely to cause interface drift than save work. No implementation starts until user approval of these details.
