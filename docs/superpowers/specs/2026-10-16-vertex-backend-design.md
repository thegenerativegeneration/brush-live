# Vertex reconstruction backend: milestones 1–3

## Intent and scope

Build a native-Mac, fixed-capture vertex/triangle reconstruction experiment alongside Brush. The user approved planning milestones 1–3: a backend boundary, a native differentiable renderer, and real-capture reconstruction. This is not authorization to implement the later live pipeline.

Deliver a reproducible offline CLI that runs Gaussian Brush and a fixed-topology triangle backend on identical selected capture inputs, writes held-out renders and metrics, and makes differences in model capabilities explicit. Keep the existing Gaussian engine and iOS app unchanged.

Work in `/Users/philipp/.local/share/opencode/worktree/585856/triangle-metal-probe`, branch `triangle-metal-probe`. Existing uncommitted files there are the synthetic probe and its manifest entry. Do not touch the unrelated edits in the original checkout.

## Observed capture inventory

Read-only source: `/Volumes/T7 Shield/Temp/data/castle_lenzburg/2026-10-15T14-25-34Z`.

- `segment-0/transforms.json`: 757 frames, per-frame intrinsics and ARKit/OpenGL camera-to-world matrices, RGB dimensions 1920 × 1440.
- `segment-0/images/`: 757 JPEGs.
- `segment-0/depth/`: 757 little-endian float16 depth maps and 757 uint8 confidence maps, each 256 × 192. All pairs were checked for expected byte lengths during planning.
- `segment-0/wire/`: 757 keyframe `.bin` files; inspected headers contain 500 × 375 RGB, LiDAR/confidence, feature points, and mono depth. Additional `.stable`, `.occmono`, and `.farmono` sidecars exist. These are not the initial experiment's input.
- Separate `segment-0_colmap/sparse/0` reconstruction has OPENCV calibration with nonzero distortion. Do not mix this calibration/world frame with the raw ARKit/LiDAR source.
- A depth-support scan using confidence 2 and finite depth 0.1–8 m found a median valid fraction of 0.24945068359375. The highest-mean-support contiguous 40-frame window was IDs 174–213, mean fraction 0.4316762288411458. This nominates a window; it does not establish photometric quality, pose quality, or scene overlap.

## Input and evaluation decisions

These are proposed engineering defaults, not user-specified numeric requirements:

- Use the raw `segment-0` source first. Preserve source files; all caches/artifacts go below an explicit output directory outside the capture.
- Start with IDs 174–213 inclusive, numerically ordered. Hold out every fifth item starting at offset 4: IDs 178, 183, 188, 193, 198, 203, 208, 213. Remaining 32 are training views. Freeze the manifest before training.
- This is a depth-support-selected subset: nomination used depth statistics including the eventual held-out frames. The eight views are optimization-isolated, not selection-blind or independent scenes. No claim of an untouched benchmark test set.
- Preflight go/no-go checks use training-view reprojection/overlap only. Held-out support may be rendered for the frozen report but cannot choose a different window. If training overlap is unusable, stop and report the reason; changing the window creates a new manifest/run, not a silent fallback.
- Resize RGB to 320 × 240. Preserve field of view and normalized principal point via the existing `Camera` representation. Depth stays metric camera-z, not Euclidean ray distance, and is sampled from its native grid by normalized pixel-center coordinates.
- Use confidence 2 and finite depth within 0.1–8 m for seeds and depth supervision. Do not use held-out RGB, depth, features, or mono depth to initialize or optimize the scene. Poses/intrinsics are allowed for held-out rendering. No pose optimization in this scope.
- Initialize shared-vertex depth grids from training anchors 174, 184, 194, 204, stride 4 in the native depth grid. Reject a face if any vertex is invalid, a pair differs in depth by more than 5% of the nearer depth, or world area is below 1e-8 m². Keep at most 20,000 accepted faces, chosen round-robin across anchors. Connectivity is fixed after initialization; no welding across anchors yet.
- Build immutable per-view ROI masks as the union of hard projected triangle interiors at pixel centers, with boundaries included and the renderer's same near/degeneracy culling, from the initial training-seeded geometry. No soft-alpha threshold, erosion, dilation, or held-out image/depth input. Training loss uses these masks. Save masks and report their fractions. Report both ROI and full-image RGB metrics; this is a local-surface experiment, not reconstruction of every distant castle/sky pixel.
- Record input-file SHA-256s, explicit split, anchor IDs, masks, all configs, git revision plus dirty-patch hash, adapter/backend/compiler identity, initialization hash, RNG seed 0, and cache format version. Freeze everything before evaluation.

## Representation and renderer

Use Rust and the workspace's pinned Burn/CubeCL/wgpu dependencies. No CUDA/PyTorch runtime, downloads, or new heavy ML dependencies. Create focused crates `brush-recon` (neutral contracts/input), `brush-triangle-render` (reference and custom kernels), `brush-triangle-train` (seeding/losses/optimizer), and `apps/brush-recon-bench` (adapters/CLI/evaluation).

The initial renderer implements an explicitly documented subset inspired by Triangle Splatting+/MeshSplatting, not paper parity:

- World-space vertices `[V,3]`, indexed faces `[F,3]`, per-vertex RGB logits `[V,3]`, per-vertex weight logits `[V]`. RGB and weights use sigmoid. No SH yet.
- `TriangleSeed` is owned/exported by `brush-triangle-render::model`; the trainer produces that type, never the reverse dependency. Initial vertex weights are 0.1 (logits `log(0.1/0.9)`); initial RGB logits represent clamped seed RGB in [1e-4,1-1e-4].
- Pinhole cameras only, Brush convention +z forward/y down. Reject other camera models with an error rather than silently removing distortion.
- Triangle coverage is zero outside projected edges. Inside, let `d` be the minimum inward normalized edge distance divided by the triangle's projected inradius. Coverage is `max(d,0)^sigma`, multiplied by the minimum of the three activated vertex weights, with alpha capped at 0.99. This is not the synthetic probe's sigmoid-edge product.
- Fixed sigma 1.0 for first correctness/recovery tests. Offline training uses linear schedules `sigma=1-0.9*p`, `floor=0.99*p`, with `p` clamped to [0,1]. Weight activation is `floor+(1-floor)*sigmoid(logit)`, not `max(floor,sigmoid(logit))`; its logit derivative includes `(1-floor)`. Fixed-step runs use `p=i/(N-1)` for zero-based step i (require N>=2); time-budget runs use elapsed synchronized optimization time / budget at the start of each step. Pass this explicitly as step context and log the actual final progress reached. No claim that these schedules reproduce a paper's full optimization.
- Perspective-correct RGB and camera-z depth interpolation is a deliberate deviation from the pinned code's screen-linear interpolation. A tilted-plane analytical test defines the required behavior.
- Front-to-back alpha compositing, alpha-normalized expected depth and normals where alpha > 1e-6. Normal is the geometric face normal, oriented toward the camera; invalid pixels have zero depth/normals. Report alpha separately. Use the same definitions in CPU, GPU and backward.
- Tile size 16 × 16; sort by positive centroid camera-z, then deterministic face ID ties, grouped by tile. This retains the reference's approximate triangle depth ordering. Crossing/interpenetrating triangles are a documented limitation, not exact per-pixel z-buffer behavior.
- Near plane 0.05 m. Conservatively cull faces with any vertex at/below near, invalid coordinates, repeated indices, world area below 1e-8 m², or projected area below 1e-6 px². Two-sided rendering; no learned winding requirement. Geometric near-plane clipping is deferred and conservative culling must be visible in counters/tests.
- CPU/bounded dense reference and tiled GPU forward must agree. Exact differentiation within the selected branches, including edge normalization/inradius, projection, interpolation, geometric normals and shared vertices. Sorting, culling, branch/tie selection are discrete; gradient checks avoid those boundaries and separate tests assert boundary behavior.
- No alpha threshold or early transmittance termination in the initial reference-matching version. Add such approximations only in a separately tested/explicit future optimization.
- At an exact edge (`d<=0`) return zero coverage and zero coverage derivative, avoiding singular fractional-power derivatives. Min-distance/min-weight ties use the lowest edge/global vertex ID respectively; tie selection is a discrete branch, not a symmetric derivative claim.
- Handle no faces/all culled on the host without zero-sized GPU launches. Checked allocation arithmetic and a default 512 MiB renderer-buffer budget; return typed errors before allocation. No truncation of tile lists.

The existing tiny probe stays a smoke test, with tilted-depth coverage and all 36 nonzero gradients strengthened. Its previous success does not validate the new coverage model.

## Backend boundary

Expose only what this scope implements: initialize from training frames, one optimization step, render requested channels, capabilities and model statistics. Use async trait methods with a generic runner; do not require a dyn-compatible trait or refactor `LiveModel` now. No speculative park/restore, ingest, guidance, or phone interface methods.

Gaussian and triangle adapters both implement the contract inside the offline app. Gaussian rendering advertises RGB/alpha only unless additional channels really exist; unsupported depth/normals are `None`, not invented zeros. Reuse `SplatTrainer`, existing losses and camera types; do not replace Gaussian rendering/training internals. Triangle gradients have no Gaussian parameter packing.

## Optimization and comparison

- Implement device-side Adam with stable parameter/optimizer ownership, not per-step host parameter/gradient copies. Separate vertex, RGB and weight learning rates; validate finite state and bound displacements. Default learning rates: vertices 1e-3 m, RGB logits 1e-2, weight logits 1e-2; per-step vertex displacement cap 0.01 m. All values are recorded, not silently tuned.
- Triangle objective: masked RGB L1 + 0.05 × masked absolute camera-z depth error in metres + 0.001 × mean squared displacement from seed positions in m². Normalize masked RGB by `3*H*W`, matching the existing Gaussian loss (not by ROI population); depth uses its own fixed valid-target count. Missing predicted depth at a valid target pixel incurs the configured 8 m error; it cannot disappear from the loss denominator. Differentiate the depth term on valid predicted pixels. No normal prior in real-data training.
- Gaussian adapter uses the same RGB ROI mask, `3*H*W` normalization and RGB L1 objective with SH degree 0, same training-frame order and same training-seeded point positions/colors. Set `ssim_weight=0`, `ssim_every=0`, `lpips_loss_weight=0`, `background_noise_strength=0`, `sh_background=false`, `match_alpha_weight=0`, `mean_noise_weight=0`, `opac_decay=0`; use black background and no mean-LR decay. No depth loss or geometry prior is claimed for this baseline. Report the objective/capability difference. Refinement disabled for the controlled baseline; normal adaptive Brush is an optional labeled secondary run, not a prerequisite. No change to production Gaussian loss normalization/defaults is needed.
- Warm up separately and time device-synchronized optimization. Run equal training wall-clock budgets of 60 s and 300 s per backend in separate fresh runs. Exclude build, decode, initialization and held-out evaluation from the training budget; report them separately. Also provide fixed-step mode for repeatability. Report initialization, final and best recorded training-loss checkpoints, but do not choose checkpoints/configuration using held-out error.
- Persist training curves, per-view and aggregate held-out RGB MAE/PSNR (ROI and full image), triangle depth error over all valid held-out LiDAR pixels, predicted valid-depth fraction, high-alpha fraction, geometry stability/cull counters, model counts, renderer-buffer peak, process footprint, and synchronized throughput. Gaussian depth metrics are explicitly unsupported. RGB evaluation uses 320×240 and true per-ROI/full-image sample denominators (unlike the training scalar); aggregate sums errors/sample counts across views. Exact-match PSNR is represented as null plus `exact_match=true`, not nonfinite JSON.
- For held-out depth evaluation, render separately at the native 256×192 LiDAR grid. Targets are confidence-2 finite depths in 0.1–8 m; denominator is all such target pixels, independent of ROI/prediction. A prediction is available iff alpha>1e-6 and finite positive camera-z; unavailable predictions incur 8 m absolute error. Available ones use absolute metric error without clipping. Also report ROI-restricted depth with masks rasterized from the same initial geometry at this grid, its fixed target count, prediction availability, and high-alpha fraction (`alpha>=0.9`). Both initialization and final models use identical target validity/masks/aggregation.
- Save before/after/target RGB, depth visualizations, immutable ROI masks, raw final triangle state and an RGB triangle PLY for inspection. This is an experiment artifact, not a new production checkpoint/preview protocol.

## Acceptance gates and stop conditions

1. Both backends run through the offline contract; existing Gaussian tests remain green. Live/FFI behavior is unchanged.
2. CPU/GPU channel agreement max absolute RGB/alpha/normals error < 2e-4 and depth error < 1e-3 m on nondegenerate bounded fixtures. All selected nonzero vertex/RGB/weight gradients agree with central finite differences: `abs_error <= 2e-4 + 0.03 * max(abs(analytic),abs(numeric))`, epsilon 1e-3; normals/depth objectives checked independently.
3. Shared-vertex accumulation, zero/culled inputs, clipping policy, sort ties, memory-limit failure, masks, malformed capture data and held-out isolation have explicit automated tests.
4. Synthetic multiview recovery reduces held-out RGB MSE by at least 90% with finite model state. A separate tilted-plane fixture uses a synthetic-only objective `mean squared depth error(m²) + mean(1-dot(pred_normal,target_normal))` over a fixed interior mask; from a small geometry perturbation, require at least 90% depth-MSE reduction and at least 90% reduction in mean normal cosine error within 500 steps. Normal gradients are also finite-difference checked independently. This synthetic-only objective does not add a real-capture normal prior.
5. Lenzburg run is reproducible, finite and artifact-complete. Research outcome is positive if held-out ROI RGB MAE improves by at least 10% over initialization and full-native-grid depth error over the fixed evaluation denominator does not worsen by more than 10%. If initial RGB MAE<=1e-8, final must remain <=1e-8; if initial depth MAE<=1e-6 m, final must remain <=1e-6 m, instead of relative tests against zero. Report a negative outcome honestly; do not weaken masks/gates or tune on held-out frames. No requirement to beat Gaussian in this first experiment.

## Out of scope

Densification/pruning, learned connectivity, Delaunay/remeshing, incremental arrival, pose correction/optimization, guidance/Fisher changes, area parking, iOS/FFI/server integration, production preview/export formats, and claims of full-scene or paper parity.

## Research references and reuse policy

Pinned during planning:

- Triangle Splatting+: https://github.com/trianglesplatting2/triangle-splatting2/tree/fcde03dd0c3953ce1d47b210485a5e8a9e14b016
- Its rasterizer: https://github.com/trianglesplatting2/diff-triangle2-rasterization/tree/42e62aaf03f87835105ed84dd8ed513efd458716
- MeshSplatting: https://github.com/meshsplatting/mesh-splatting/tree/2a810a6c353215685307da3d4cc6ebd73b1c387b
- Its rasterizer: https://github.com/meshsplatting/diff-triangle-mesh-rasterization/tree/1886fbe3b6c4076b7792bee407ed374e823deeef

Implement the documented mathematics independently; do not copy inherited research-only code into Apache-licensed crates. Review licenses before any code reuse. In the pinned MeshSplatting rasterizer, forward caps alpha at 0.999 but backward uses 0.99: do not reproduce that inconsistency. Our 0.99 cap is one shared forward/backward constant. Screen-linear interpolation and discrete branch approximations in those references are not a substitute for our mathematical/finite-difference tests.
