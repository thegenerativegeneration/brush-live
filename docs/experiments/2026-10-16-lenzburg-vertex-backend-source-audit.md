# Source audit: our triangle splatter vs. Triangle Splatting+ / MeshSplatting

Follow-up to the [Lenzburg experiment](2026-10-16-lenzburg-vertex-backend.md), performed 2026-10-17 at the
user's request: compare the pinned original sources and arXiv:2509.25122 against the written implementation,
explain the observed PSNR gap to the Gaussian control, and clarify timing claims. This is a read-only audit;
no renderer/trainer code was changed.

Pinned sources checked out and verified at the commits frozen in the
[design](../superpowers/specs/2026-10-16-vertex-backend-design.md):

| Reference | Commit |
|---|---|
| trianglesplatting2/triangle-splatting2 | `fcde03dd0c3953ce1d47b210485a5e8a9e14b016` |
| trianglesplatting2/diff-triangle2-rasterization | `42e62aaf03f87835105ed84dd8ed513efd458716` |
| meshsplatting/mesh-splatting | `2a810a6c353215685307da3d4cc6ebd73b1c387b` |
| meshsplatting/diff-triangle-mesh-rasterization | `1886fbe3b6c4076b7792bee407ed374e823deeef` |

## Verdict

**No mathematical defect was found in our renderer relative to our own specification.** The coverage window,
min-vertex opacity, shared 0.99 alpha cap, front-to-back centroid sorting, shared-vertex gradient reduction and
Adam updates match the pinned sources' math or the documented deliberate deviations, and are covered by the
finite-difference/CPU-GPU tests. The quality gap is explained by **components of the reference method that were
explicitly out of scope for milestones 1–3 and therefore never implemented** — chiefly densification, pruning,
global Delaunay initialization, view-dependent color and the paper's full loss stack — on top of a deliberately
constrained evaluation scene. The prototype's poor step throughput is explained by **orchestration and kernel
choices in our code**, not by an inherent cost of triangle rasterization, so it does not contradict the papers'
speed claims.

This means the measured result is a statement about *our subset*, not about triangle splatting as a method.

## Completeness matrix

Status meanings: **ours** = implemented in this branch; **omitted (scope)** = explicitly excluded by the
approved plan; **omitted (gap)** = present in the reference and relevant to quality but not part of the plan's
renderer-correctness scope.

| Reference component | Original source | Ours | Class | Expected quality/perf effect |
|---|---|---|---|---|
| 3D Delaunay tetrahedralization of the full SfM cloud → globally connected faces | `triangle_model.py:240-261` | Four per-anchor LiDAR depth grids, no cross-anchor welding (`seed.rs`) | omitted (scope) | Limited coverage (ROI 19–61% of image), seams between patches |
| Midpoint-subdivision densification every 500 its (outdoor 500–13k, indoor 1k–25k), split 100 largest, grow to ~2M vertices | `train.py:236-252`, `triangle_model.py:581-614` | None; 4,851 vertices / 8,941 faces fixed | omitted (scope) | Dominant: paper states the initial representation "does not contain enough triangles and vertices to achieve high visual quality" |
| Hard + blending-weight + oversize pruning every 500 its, threshold 0.235, final 0.5 pass | `train.py:191-227,274-285` | None | omitted (gap) | Paper calls this "essential" once opacity is forced; without it hidden/redundant triangles turn opaque and persist as artifacts — matches our visible streaks/floaters |
| SH degree 3 per vertex (48 color coefficients, degree ramp every 1k its) | `train.py:119-120`, `arguments/__init__.py:58` | Per-vertex RGB only (3 values) | omitted (scope) | View-dependent appearance; does not explain the gap vs our degree-0 Gaussian control, but does vs real 3DGS |
| Photometric loss L1 + 0.2·D-SSIM | `train.py:145-155` | ROI-masked L1 only (`loss.rs:119`) | omitted (gap) | SSIM term usually improves perceptual and moderately improves PSNR on held-out views |
| Opacity/weight-decay loss 1.9e-6 before iter 5k | `train.py:164-167` | None | omitted (gap) | Drives useless triangles low so pruning can remove them; without pruning it is moot |
| Supervised normal loss 0.05 after iter 5k (estimated normal maps) | `train.py:140-143,157-162` | No real-data normal prior (spec decision); depth + seed prior instead | omitted (scope) | Surface consistency; our substitute is metric depth supervision |
| Sigma schedule 1.0 → **0.0001** linearly over 30k | `train.py:78-82,109-116` | 1.0 → **0.1** over the budget (`schedule.rs:15-18`) | deliberate deviation | Mild: their Table 3 shows σ=0.1 ("soft") is ≈ +0.3 dB vs hard when opacity is opaque; full removal of decay costs −6.84 dB (Table 2) |
| Opacity floor 0.1→0.9999 from it 5k→24k, free sigmoid before | `train.py:83-85,164-167,255-263` | Floor 0→0.99 over the whole budget (`schedule.rs:17-18`) | similar in spirit | Comparable forcing; ours compresses the schedule to fit 60–300 s budgets |
| Supersampling 2× (it 20k) then 4× (it 25k), area downsample | `train.py:100-103`, `triangle_renderer/__init__.py:51-105` | 1× native resolution | omitted (gap) | Anti-aliasing; gains crisp texture edges in PSNR |
| 30k iterations, ~100–300 views, random-without-replacement view order | `train.py:97,128-133` | 2,372 / 9,806 steps, deterministic cyclic 32 views (`run.rs:126`) | deliberate protocol | ~3–15× fewer optimizer steps than the paper's runs |
| MeshSplatting extras: mono inverse-depth loss, vertex-depth consistency, restricted-Delaunay re-mesh after densification, SparseAdam | `mesh/train.py:120-128,215-245,350-364` | None (we use metric LiDAR depth instead of mono depth) | omitted (scope) | Mesh connectivity/surface quality |
| Screen-space barycentric color/depth interpolation | `forward.cu:623-663` | Perspective-correct interpolation (`kernels/mod.rs:128-143`) | deliberate deviation | Slightly different gradients/renders; ours is mathematically more faithful to the 3D triangle |
| Early exit: skip alpha < 1/255, stop at T < 1e-4; cull min-weight < 0.01 and 1–1600 px triangles | `forward.cu:266-268,395-399,595-599` | No early termination (spec: correctness-first), conservative near/area culls only | deliberate deviation | Large training/render speed cost for us; negligible quality effect |
| Alpha cap 0.99 (forward and backward consistent) | `forward.cu:594` | 0.99 in `kernels/mod.rs:133` and `rasterize_bwd.rs:127` | matches | MeshSplatting's 0.999/0.99 inconsistency deliberately not reproduced |
| Shared-vertex min opacity `o_T = min(o_i,o_j,o_k)` | paper §3.2, `forward.cu:170-185` | `kernels/mod.rs:133-136` + `face.weight` | matches | — |
| Coverage window `(d/d_incenter)^σ`, zero outside | paper Eq. 1, `forward.cu:328-354,590-592` | `kernels/mod.rs:119-135` | matches | — |
| Centroid-depth sort, front-to-back transmittance | `forward.cu:446,534-668` | `forward.rs:194` (ascending z) + `rasterize.rs:27-35` | matches | Crossing triangles remain approximate in both |

## How large is the gap "supposed" to be?

The paper's own Table 1 (Mip-NeRF360, mesh-based novel-view synthesis, radiance-rendering 3DGS shown for
reference): **3DGS 27.21 dB, Triangle Splatting+ 25.21 dB, opaque Triangle Splatting 21.05 dB**. So even the full
method sits ~2 dB under 3DGS rendering quality, and the prior triangle method sits ~6 dB under. Our measurement
(21.78 dB triangle vs 24.63 dB constrained Gaussian control, i.e. 2.85 dB gap, with a prototype missing
densification/pruning/SH/global mesh and covering only an ROI) is **not anomalously large relative to those
published numbers**. It cannot be read as "triangle splatting loses 3 dB to 3DGS": our Gaussian control is
itself far below real 3DGS (degree-0, no refinement, 4,851 fixed points).

The single most suspicious *implementation* alternative — a broken coverage or gradient path — is ruled out by
the finite-difference and recovery tests (240 package tests + 4 probe tests, CPU/GPU agreement, shared-vertex
accumulation). The audit found no divergence between our formulas and the pinned sources beyond the deviations
listed above.

## Timings: what we measured and what the papers claim

Measured on the M1 Pro (from the preserved run records; per-step times are derived from the recorded curve):

| Run | Optimization s | Steps | ms/step | Steps/s | Wall s (incl. decode/init/eval/artifacts) |
|---|---:|---:|---:|---:|---:|
| Triangles, 60 s budget | 60.017 | 2,372 | 25.3 | 39.5 | 65.5 |
| Gaussian control, 60 s | 60.002 | 10,764 | 5.6 | 179.4 | 65.9 |
| Triangles, 300 s budget | 300.008 | 9,806 | 30.6 | 32.7 | 305.7 |
| Gaussian control, 300 s | 300.004 | 49,925 | 6.0 | 166.4 | 305.8 |

So in this prototype a triangle training step is **~4–5× slower** than the Gaussian step. Reasons visible in
the code, without needing a profiler:

1. **Per-render GPU→CPU readback.** `forward.rs:197-214` executes a transaction readback of projection status
   and the tile-intersection count on *every* render, which stalls the Metal pipeline inside each training step.
   The original CUDA rasterizer allocates and launches without such a round trip.
2. **Per-step scalar readbacks.** `optimizer.rs:52-60` reads one finite-state scalar per Adam group per update
   (3 per step) and `train.rs:206-211` reads the loss metrics — at least 4 more device synchronizations per step.
3. **Backward face reduction.** `rasterize_bwd.rs:190-203` scans every tile-list entry in each face's bounding
   box with a linear `faces[i] == id` search — effectively quadratic in tile-list density — where the original
   scatters gradients with atomics or via prefix-sum addressing.
4. **No early exit.** We deliberately skip the original's `alpha < 1/255` skip and `T < 1e-4` termination
   (`forward.cu:595-599`), and its min-weight/screen-size culls (`forward.cu:266-268,395-399`), so every
   intersection is processed at full cost.
5. **Two radix sorts + prefix scan per render** (`forward.rs:194,245`) and large intermediates (`[F,32]`,
   `[N,32]`) versus the original's single tile-key sort.

None of these are inherent to triangle splatting. The papers' speed claims come from the optimized CUDA
rasterizer on an A100: Triangle Splatting+ trains Mip-NeRF360 in 39 min (vs MiLo's up to 2 h), and the earlier
Triangle Splatting reports ~3,000 FPS rendering — a claim about **inference of the final opaque mesh** (where
compositing collapses to one triangle per pixel and sorting can be skipped), not about differentiable training
steps. Our prototype does not implement that path and makes no such claim.

We did **not** run a render-only FPS benchmark: the timing artifacts of the experiment lived under
`target/vertex-runs/`, which has since been deleted from disk (see below), and re-deriving kernel-level numbers
would require rebuilding the toolchain artifacts and adding a benchmark target. That measurement remains a
follow-up; the step-throughput figures above are the only performance evidence this experiment produced.

## Smallest falsifiable follow-ups (priority order)

Ordered by the reference's own ablations and the observed artifacts; each should be run as a single changed
variable on fresh development data, not tuned against the frozen Lenzburg hold-outs.

1. **Pruning** (opacity < 0.235 then blending-weight based, plus a final 0.5 pass). Paper: −0.46/−0.51 dB
   without it; visually removes the streak/floater artifacts we observed. This is the one omission the paper
   itself calls essential under forced opacity.
2. **Densification** (midpoint 1→4 subdivision of selected/largest triangles). Paper: the route from 4.8k to
   ~2M vertices; expected dominant effect on fine detail and ROI PSNR.
3. **Remove per-step device readbacks** (compute intersection count via a two-pass fixed-capacity scheme or a
   budgeted upper bound; batch Adam finite checks). This targets step time directly; a render-only benchmark
   should accompany it to make any "faster than 3DGS" claim testable.
4. **Global Delaunay/welded initialization** instead of four independent depth grids, to remove inter-patch
   seams and extend coverage.
5. **SH degree ≥ 1 for vertices**, then SSIM term and supersampling — only after 1–4, and only on fresh data.

## Artifact status

The run artifacts (`run.json`, `metrics.json`, PLY/state files, images, `target/vertex-runs/…`) were untracked
local files. The worktree's entire `target/` directory is gone and roughly 30 GB of disk was freed; the files
are not recoverable from the repository or the Trash. All derived numbers used in this audit were preserved in
the experiment report, `results-summary.json` and `comparison-status.json` under
`.superpowers/sdd/2026-10-16-vertex-backend/` before the loss. Re-running the documented CLI commands
reproduces the artifacts; the source-capture hashes in the report remain the integrity record for the inputs.
