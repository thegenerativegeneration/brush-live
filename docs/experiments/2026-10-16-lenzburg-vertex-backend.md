# Lenzburg native vertex-backend experiment

Executed 2026-10-17 from the plan approved 2026-10-16. **Measured implementation revision:** `0a5ebd415259b457d08208a9d060fb3d7e6125ba`; all four timed runs recorded a clean worktree SHA-256 (`e3b0c442…b855`). Documentation and final validation/review follow that implementation revision.

## Outcome

**Native-Mac feasibility demonstrated; not ready to replace `brush-live`.** The fixed-topology triangle pipeline completed on Metal, passed the predeclared numerical feasibility gate and produced inspectable geometry. The Gaussian control nevertheless achieved better RGB reconstruction and more optimization steps within the same budgets. Crucially, the triangle's depth accuracy inside the original supported region worsened, despite its improved whole-grid score from broader coverage. Rendered results are visibly faceted/fragmented and are not production-quality reconstructions.

This is an offline milestones-1–3 experiment. Live ingestion, guidance/Fisher scoring, pose updates, adaptive topology, connected remeshing, parking and iPhone integration remain excluded.

## Frozen input

- Read-only source: `/Volumes/T7 Shield/Temp/data/castle_lenzburg/2026-10-15T14-25-34Z/segment-0`.
- Raw ARKit/OpenGL poses/intrinsics, 1920×1440 RGB and 256×192 float16 camera-z LiDAR with uint8 confidence. Do not mix these with the separate OPENCV-distorted COLMAP reconstruction.
- IDs 174–213: 32 training frames; held out 178, 183, 188, 193, 198, 203, 208, 213.
- RGB evaluation/training 320×240; depth evaluation on native 256×192, confidence 2 and finite depths 0.1–8 m.
- Training anchors 174, 184, 194, 204; stride 4; depth-discontinuity/area rejection; fixed shared-vertex grids, no cross-anchor welding. **4,851 vertices / 8,941 faces.** The Gaussian control uses the same 4,851 seed positions/colors, not an existing trained PLY.
- Preflight non-anchor median relative depth reprojection error: **0.5266384%**. Training image ROI fractions range **18.6771–60.5247%**.
- 121 source files hashed, plus frozen seed/mask/configuration hashes. Each completed run reverified source hashes before/after; copied manifests agree. The initial removable-volume permission stall occurred before training and was resolved by the user granting access; it is not a renderer-performance result.

The nominated subset used depth-support statistics including eventual held-outs. It is optimization-isolated, **not selection-blind**. The fixed ROI mainly covers nearby courtyard/path and low-wall surfaces. Unmodelled distant scenery/sky is black, and full-image scores reflect that missing extent.

## Protocol and controls

Apple M1 Pro, macOS/aarch64, wgpu Metal, workspace-pinned Burn/CubeCL stack, Rust `1.98.1 (48a229cea 2026-09-01)`. Seed 0. Two discarded warmup steps; every timed run then initializes a new model/optimizer from the same frozen seed. Order: triangle smoke, Gaussian smoke, triangle 60 s, Gaussian 60 s, Gaussian 300 s, triangle 300 s. GPU runs were sequential; no GPU test suite overlapped them.

Optimization budgets count synchronized `train_step` wall intervals, including input preparation, gradients, optimizer operations and scalar readbacks. Decode, warmup, initialization, evaluation, snapshots and artifact writes are excluded and separately logged. Actual completed-step time is reported; the last step may overshoot slightly.

Both backends use the same immutable RGB ROI and `3*H*W` masked L1 normalization, degree-0/per-vertex RGB appearance, black background, fixed model counts and no refinement/noise. Triangle additionally uses metric depth supervision and a seed-position prior. Initial opacity, parameterization and learning rates differ. Triangle sigma/floor schedules are 1→0.1 and 0→0.99; gains cannot be attributed solely to geometry optimization. No hyperparameter sweep or held-out checkpoint selection was performed.

## Held-out RGB results

MAE is on [0,1] channel values; PSNR is dB. Aggregation sums errors/sample counts, not a mean of differently supported views.

| Backend | Budget | Actual optimization s | Steps | Steps/s | ROI MAE | ROI PSNR | Full-image MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Triangles | 60 s | 60.017 | 2,372 | 39.52 | 0.06158 | 21.65 | 0.36819 |
| Gaussian control | 60 s | 60.002 | 10,764 | 179.39 | 0.04526 | 24.26 | 0.34751 |
| Triangles | 300 s | 300.008 | 9,806 | 32.69 | 0.06084 | 21.78 | 0.34987 |
| Gaussian control | 300 s | 300.004 | 49,925 | 166.41 | 0.04368 | 24.63 | 0.32654 |

Initial ROI MAE/PSNR: triangles **0.39015 / 7.60 dB**; Gaussian control **0.15100 / 14.86 dB**. These different initial renderings matter when reading relative improvement. Triangle ROI MAE improved 84.41% at 300 s, but remained worse in absolute terms than the Gaussian control. The 2-step smoke already reduced triangle ROI MAE to 0.08866 while jumping the schedule from start to end; it is diagnostic, not convergence evidence.

These are single runs with uncontrolled thermal/background-system variation, not statistically established speed ratios. The two implementations also perform different objectives/work per step. No phone throughput is inferred.

## Triangle depth and coverage

Full-native-grid depth uses **166,086 fixed valid targets** across the eight held-outs. Unavailable predictions cost 8 m. The immutable-ROI secondary depth metric uses **123,315 fixed valid targets**. Gaussian depth is unsupported/null, not zero error.

| Triangle state | Full-native MAE m, including missing penalty | Available target fraction | ROI depth MAE cm | ROI availability |
|---|---:|---:|---:|---:|
| Initial | 2.08230 | 74.25% | 2.979 | 99.998% |
| 60 s | 1.11189 | 86.53% | 3.718 | 99.994% |
| 300 s | 0.63007 | 92.63% | 3.588 | 100.000% |

Both timed triangle runs pass the predeclared gate: ≥10% ROI RGB MAE improvement and ≤10% worsening of **full-native-grid** depth MAE, with finite state/artifacts. **This is not a local surface-accuracy win.** ROI depth MAE worsened **24.82% at 60 s / 20.45% at 300 s**. The whole-grid improvement is largely a coverage gain under the missing-depth penalty. The gate is deliberately reported as defined beforehand, not rewritten after observing this limitation.

At 300 s, high-alpha fraction is 90.28% of all valid native targets and 99.82% inside the ROI. These are rendering/coverage quantities, not calibrated reconstruction confidence or guidance scores.

## Memory and stability

| Backend / budget | Process lifetime peak MiB | Conservative renderer reservation MiB |
|---|---:|---:|
| Triangles / 60 s | 115.97 | 18.92 |
| Gaussian / 60 s | 108.22 | unsupported |
| Triangles / 300 s | 116.30 | 19.58 |
| Gaussian / 300 s | 99.59 | unsupported |

Process footprint is OS accounting and includes warmup, evaluation, artifacts and runtime. It is not renderer-only GPU memory. Triangle reservation is an upper-bound allocation guard, not a measured device peak; Gaussian reservation is uninstrumented/null. Model counts stayed fixed and state/losses were finite. Snapshot cull/intersection counters refer to the last render, not removed geometry or cumulative training.

Review caveat: `forward.rs::required_buffer_bytes` misses fixed minimum radix-sort scratch overhead on tiny scenes, so unusually small custom budgets are not strict allocation bounds. This is a nonblocking follow-up and does not affect the measured capture/default 512 MiB budget. The guard is not a general memory sandbox.

## Inspection artifacts and reproduction

The run artifacts were untracked local files under `target/vertex-runs/`:

- `lenzburg-preflight/manifest.json`, seed/masks and training ROI/depth previews.
- `lenzburg-triangle-smoke/`, `lenzburg-gaussian-smoke/`.
- `lenzburg-triangle-60/`, `lenzburg-gaussian-60/`, `lenzburg-gaussian-300/`, `lenzburg-triangle-300/`.
- Each completed run contains `run.json`, `metrics.json`, `curve.json`, initial/final PLY, frozen input assets, target/mask/prediction images and native depth arrays. Triangle raw states restore parameters, not optimizer history; mesh PLY does not encode the training coverage/weight model.

**Update 2026-10-17:** the worktree's entire `target/` directory, including all of the above and the
`lenzburg-comparison-300.png` contact sheet, has been deleted from disk (about 30 GB was freed) and is not
recoverable from the repository or the Trash. The numbers in this report, and the derived records
`results-summary.json` / `comparison-status.json` under `.superpowers/sdd/2026-10-16-vertex-backend/`, were
preserved before the loss and remain the evidence of record. The contact sheet showed target vs triangles vs
Gaussian control after 300 s on frames 178/193/213, rotated 90° clockwise for readability (metrics use
unchanged native orientation); frames were the first/middle/last listed hold-outs, not chosen by score. It
showed the black unsupported regions, faceting/fragmentation and unconstrained streak artifacts this report
describes. Re-running the commands below reproduces all artifacts.

Use the commands in [the CLI README](../../apps/brush-recon-bench/README.md). For the measured implementation, build revision `0a5ebd41`, reuse the frozen manifest, and run each backend separately at 60/300 seconds with fresh output directories. Never resume 60 s into 300 s, overwrite an old run, or change the masks/thresholds to improve held-out results.

## Validation and review

Before the real runs, all seven affected package suites passed: **240 tests passed, 0 failed, 1 existing Gaussian empty-dispatch ignore**. The original synthetic example is an additional explicit test target excluded from default suites. CPU/GPU forward agreement, all-parameter finite differences, shared-vertex reductions, RGB and depth/normal recovery, immutable masks, held-out isolation, native-grid target counts, timing, source safety and snapshot/PLY behavior are covered.

The explicit smoke-example target passed **4 tests**. Full `cargo test --release --workspace --features brush-cube/metal` is blocked at compile time by unchanged `crates/brush-bench-test/tests/fuzz.rs:177`: it supplies 13 arguments to the existing 14-argument `SplatOps::render`. Both the caller and renderer declaration are byte-identical to branch base `df859c1d`; this branch does not introduce that mismatch.

The workspace run excluding `brush-bench-test` reached **380 passes, 3 failures and 2 ignores** before Cargo stopped. These are a partial run, not additive to the affected-package total above. All three failures are unchanged `brush-guide-ffi/tests/ffi.rs` cases that cannot read the absent superproject fixture `datasets/segment-1/wire/0.bin` from this linked-worktree layout:

- `keyframes_in_acks_status_scores_pause_and_splat_out`
- `pause_after_warm_up_holds_keyframes_every_time`
- `pause_during_warm_up_returns_and_resume_lets_keyframes_through`

They fail at the recorded-frame read (`ffi.rs:21`), not in the new triangle backend. No fixture links were created outside the worktree, no unrelated tests were patched, and **no full-workspace green result is claimed**. Broader integration remains blocked until the legacy caller/fixture setup is repaired separately.

`cargo fmt --all -- --check` fails on unchanged baseline files only. Cross-checking `BASE..HEAD` shows no changed file among those format diffs; every changed Rust file passes an explicit rustfmt check. Unrelated files were not reformatted. The local rustup `rust-objcopy` emits a missing-`libLLVM.dylib` stripping warning; builds/test executables still ran.

Independent read-only whole-branch review: **PASS, ready to merge relative to the existing baseline; no Critical or Important findings**. The reviewer inspected implementation/tests/spec/ledger and local comparison JSON, but did not independently rerun tests, access/recalibrate T7 data, or certify legal provenance. Its one Minor finding is the tiny-scene radix-scratch reservation issue above, retained for follow-up. This review does not establish production readiness or change the experiment's negative replacement recommendation.

## Decision

Keep Gaussian `brush-live` as the working engine. This milestone establishes a tested native vertex backend and reproducible real-data experiment, not a drop-in quality replacement. Before live integration, investigate appearance/coverage artifacts, preservation of local depth accuracy, and measured bottlenecks; validate any changes on fresh development data rather than tuning against these held-outs. Adaptive topology, mesh connectivity refinement and live guidance remain separate future scope.

A follow-up [source audit](2026-10-16-lenzburg-vertex-backend-source-audit.md) compares the pinned Triangle Splatting+/MeshSplatting implementations line by line against this prototype. It finds no renderer-math defect relative to our specification, and attributes the quality gap to reference-method components that were out of scope (densification, pruning, global Delaunay init, SH, SSIM/normal losses, supersampling) — including the omission the paper itself calls essential once opacity is forced (pruning). It also explains the slower triangle steps as prototype orchestration overhead (per-render/per-step device readbacks, no early exit, quadratic backward reduction), not an inherent property of triangle splatting.

## Implementation decisions and remaining costs

These decisions were recorded during execution rather than silently changing the plan:

| Decision | Remaining cost/limitation |
|---|---|
| Exercise generic contracts with real adapters in Task 9 rather than a self-testing dummy backend in Task 2 | Interface defects could have surfaced later; real-adapter tests now cover the path. |
| Empty logical models use sentinel storage plus explicit logical counts | Consumers must never treat physical sentinel dimensions as scene counts. |
| Keep short host projection/binning orchestration in `forward.rs` | Split that file if orchestration grows. |
| Report estimated renderer reservations separately from measured process footprint | Reservations can reject feasible scenes; tiny custom budgets also have the known fixed-scratch accounting gap. |
| Register the concrete Cube/Wgpu custom backward directly at the dynamic Tensor boundary | Other backends need another bridge. |
| Use exact scalar tangent arithmetic for projection VJPs | Nine tangent evaluations can cost performance; this is analytical, not finite-difference training. |
| Put freeze/isolation orchestration with the app/manifest task | Caller-level isolation tests arrived later; they are now present. |
| Introduce experimental snapshots with the real adapters | The experimental schema may require future version changes; optimizer state is not saved. |
| Read one finite-state scalar per Adam group/update, never bulk parameter/gradient arrays | Synchronization can limit throughput. |
| Require a frozen manifest for `run` | Every new subset/configuration needs an explicit preflight. |
| Give the generic runner a backend factory | Callers supply a constructor closure so warmup can be discarded honestly. |
| Add a training-only reprojection gate at relative median error 0.15 | A noisy usable capture may be conservatively rejected; no silent window replacement. |
| Time synchronized optimization intervals rather than end-to-end elapsed time | Full runtime is longer and reported separately. |
| Pass frozen initial geometry explicitly to native-grid evaluation | One extra experiment-context argument avoids resampling/moving ROI support. |
| Save initial/final parameters; label sampled training-loss minimum diagnostic | Reproducing that intermediate model requires fixed-step replay. |
| Save last-render cull diagnostics, not cumulative claims | Those counters and process lifetime peaks are not renderer-only measured GPU peaks. |

The final reviewer did not independently replicate tests/GPU execution, survey physical calibration, certify legal provenance, or judge outside-scope paper/live/iOS/adaptive behavior. Executor logs/artifacts provide the reported evidence; depth remains sensor-relative, performance remains single-run/accounting-specific, and later distribution/reuse needs its own licensing review. Unchanged baseline formatting, the fuzz caller mismatch, missing FFI fixtures and the original checkout's eight user edits were deliberately left alone. The branch/worktree are retained without merge/push; diagnostic logs remain available locally under `.superpowers/sdd/2026-10-16-vertex-backend/` (the native run artifacts in `target/vertex-runs/` were later deleted with the worktree's `target/` directory; see the artifact-status update above).
