use crate::{TriangleModel, burn_glue, kernels, reference::ReferenceConfig};
use brush_cube::{CubeTensor, create_tensor, create_tensor_from_slice};
use brush_recon::{Capabilities, ReconError, RenderRequest, RenderResult};
use burn::backend::ops::{IntTensorOps, TransactionOps, TransactionPrimitive};
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, IntDType, Tensor, s};
use burn_cubecl::{CubeBackend, kernel::into_contiguous};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RenderConfig {
    pub sigma: f32,
    pub weight_floor: f32,
    pub near_m: f32,
    pub buffer_budget_bytes: u64,
}
impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            sigma: 1.,
            weight_floor: 0.,
            near_m: 0.05,
            buffer_budget_bytes: 512 * 1024 * 1024,
        }
    }
}

pub fn required_buffer_bytes(
    vertices: u64,
    faces: u64,
    pixels: u64,
    tiles: u64,
    intersections: u64,
) -> Result<u64, ReconError> {
    // Conservative reservation: current/projected + sort generations/scratch + output,
    // with headroom for backward state/reduction. Not process/GPU-resident footprint.
    [
        (vertices, 64),
        (faces, 512),
        (pixels, 128),
        (tiles, 32),
        (intersections, 256),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, (count, bytes)| {
        count.checked_mul(bytes).and_then(|n| sum.checked_add(n))
    })
    .ok_or_else(|| ReconError::input("renderer allocation arithmetic overflow"))
}

#[derive(Clone, Debug)]
pub(crate) struct ForwardBuffers {
    pub output: CubeTensor,
    pub projected: CubeTensor,
    pub sorted_faces: CubeTensor,
    pub tile_keys: CubeTensor,
    pub ranges: CubeTensor,
    pub params: CubeTensor,
}

fn check_budget(
    model: &TriangleModel,
    request: &RenderRequest,
    cfg: &RenderConfig,
    intersections: u64,
) -> Result<u64, ReconError> {
    let pixels = u64::from(request.size.x) * u64::from(request.size.y);
    let tiles = u64::from(request.size.x.div_ceil(16)) * u64::from(request.size.y.div_ceil(16));
    let required = required_buffer_bytes(
        model.num_vertices() as u64,
        model.num_faces() as u64,
        pixels,
        tiles,
        intersections,
    )?;
    if required > cfg.buffer_budget_bytes {
        return Err(ReconError::BufferBudget {
            requested: required,
            budget: cfg.buffer_budget_bytes,
        });
    }
    Ok(required)
}

pub(crate) fn uniforms(request: &RenderRequest, cfg: &RenderConfig) -> Vec<f32> {
    let view = glam::Mat4::from(request.camera.world_to_local()).to_cols_array_2d();
    let focal = request.camera.focal(request.size);
    let center = request.camera.center(request.size);
    vec![
        view[0][0],
        view[1][0],
        view[2][0],
        view[3][0],
        view[0][1],
        view[1][1],
        view[2][1],
        view[3][1],
        view[0][2],
        view[1][2],
        view[2][2],
        view[3][2],
        focal.x,
        focal.y,
        center.x,
        center.y,
        request.size.x as f32,
        request.size.y as f32,
        cfg.sigma,
        cfg.weight_floor,
        cfg.near_m,
        request.camera.position.x,
        request.camera.position.y,
        request.camera.position.z,
        request.background.x,
        request.background.y,
        request.background.z,
    ]
}

pub(crate) async fn forward_gpu(
    model: &TriangleModel,
    request: &RenderRequest,
    cfg: &RenderConfig,
) -> Result<Option<ForwardBuffers>, ReconError> {
    request.validate(Capabilities {
        depth: true,
        normals: true,
    })?;
    ReferenceConfig {
        sigma: cfg.sigma,
        weight_floor: cfg.weight_floor,
        near_m: cfg.near_m,
    }
    .validate()?;
    let initial_bytes = check_budget(model, request, cfg, 0)?;
    let f = model.num_faces();
    let v = model.num_vertices();
    if model.positions.dims() != [v.max(1), 3]
        || model.rgb_logits.dims() != [v.max(1), 3]
        || model.weight_logits.dims() != [v.max(1)]
        || model.faces.dims() != [f.max(1), 3]
    {
        return Err(ReconError::input(
            "triangle device tensor shapes differ from logical model counts",
        ));
    }
    let tile_w = request.size.x.div_ceil(16);
    let tile_h = request.size.y.div_ceil(16);
    let max_intersections = (f as u64)
        .checked_mul(u64::from(tile_w) * u64::from(tile_h))
        .ok_or_else(|| ReconError::input("tile count overflow"))?;
    if max_intersections > u32::MAX as u64 {
        return Err(ReconError::input(
            "potential intersection count exceeds u32 scan range",
        ));
    }
    if f == 0 {
        model.visible_faces.store(0, Ordering::Relaxed);
        model.intersections.store(0, Ordering::Relaxed);
        model
            .buffer_peak
            .fetch_max(initial_bytes, Ordering::Relaxed);
        return Ok(None);
    }
    let positions = into_contiguous(burn_glue::float(model.positions.clone()));
    let colors = into_contiguous(burn_glue::float(model.rgb_logits.clone()));
    let weights = into_contiguous(burn_glue::float(model.weight_logits.clone()));
    let faces = into_contiguous(burn_glue::int(model.faces.clone()));
    let device = positions.device.clone();
    let client = positions.client.clone();
    let params = create_tensor_from_slice(&uniforms(request, cfg), &device, DType::F32);
    let projected = create_tensor([f, 32], &device, DType::F32);
    let depths = create_tensor([f], &device, DType::F32);
    let ids = create_tensor([f], &device, DType::U32);
    let counts = create_tensor([f], &device, DType::U32);
    let status = CubeBackend::int_zeros([2].into(), &device, IntDType::U32);
    let dim = CubeDim::new_1d(256);
    kernels::project::project::launch(
        &client,
        calculate_cube_count_elemwise(&client, f, dim),
        dim,
        positions.into_tensor_arg(),
        faces.into_tensor_arg(),
        colors.into_tensor_arg(),
        weights.into_tensor_arg(),
        params.clone().into_tensor_arg(),
        projected.clone().into_tensor_arg(),
        depths.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        status.clone().into_tensor_arg(),
    );
    let (_, face_order) = brush_sort::radix_argsort(depths, ids, 32);
    let sorted_counts = CubeBackend::int_gather(0, counts, face_order.clone());
    let prefix = brush_scan::prefix_sum(sorted_counts);
    let last = CubeBackend::int_slice(prefix.clone(), &[(f - 1..f).into()]);
    let data = <CubeBackend as TransactionOps<CubeBackend>>::tr_execute(TransactionPrimitive::new(
        vec![],
        vec![],
        vec![status, last],
        vec![],
    ))
    .await
    .map_err(|e| ReconError::Gpu(format!("projection count readback: {e:?}")))?;
    let status = data.read_ints[0]
        .clone()
        .try_into_vec::<u32>()
        .map_err(|e| ReconError::Gpu(format!("projection status: {e:?}")))?;
    let n = data.read_ints[1]
        .clone()
        .try_into_vec::<u32>()
        .map_err(|e| ReconError::Gpu(format!("intersection count: {e:?}")))?[0]
        as usize;
    if status[1] != 0 {
        return Err(ReconError::ModelState(
            "nonfinite vertex/logit or out-of-bounds index".into(),
        ));
    }
    if status[0] as usize > f || n as u64 > max_intersections {
        return Err(ReconError::Gpu("implausible projection counts".into()));
    }
    let required = check_budget(model, request, cfg, n as u64)?;
    model.buffer_peak.fetch_max(required, Ordering::Relaxed);
    model
        .visible_faces
        .store(u64::from(status[0]), Ordering::Relaxed);
    model.intersections.store(n as u64, Ordering::Relaxed);
    if n == 0 {
        return Ok(None);
    }
    let keys = create_tensor([n], &device, DType::U32);
    let values = create_tensor([n], &device, DType::U32);
    kernels::intersections::emit::launch(
        &client,
        calculate_cube_count_elemwise(&client, f, dim),
        dim,
        projected.clone().into_tensor_arg(),
        face_order.into_tensor_arg(),
        prefix.into_tensor_arg(),
        params.clone().into_tensor_arg(),
        keys.clone().into_tensor_arg(),
        values.clone().into_tensor_arg(),
    );
    let (keys, sorted_faces) = brush_sort::radix_argsort(keys, values, 32);
    let num_tiles = (tile_w as usize) * (tile_h as usize);
    let ranges = CubeBackend::int_zeros([num_tiles, 2].into(), &device, IntDType::U32);
    kernels::ranges::ranges::launch(
        &client,
        calculate_cube_count_elemwise(&client, n, dim),
        dim,
        keys.clone().into_tensor_arg(),
        ranges.clone().into_tensor_arg(),
    );
    let output = create_tensor(
        [request.size.y as usize, request.size.x as usize, 8],
        &device,
        DType::F32,
    );
    kernels::rasterize::rasterize::launch(
        &client,
        calculate_cube_count_elemwise(&client, num_tiles * 256, dim),
        dim,
        projected.clone().into_tensor_arg(),
        sorted_faces.clone().into_tensor_arg(),
        ranges.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
    );
    Ok(Some(ForwardBuffers {
        output,
        projected,
        sorted_faces,
        tile_keys: keys,
        ranges,
        params,
    }))
}

pub(crate) fn unpack(image: Tensor<3>, request: &RenderRequest) -> RenderResult {
    let [h, w] = [request.size.y as usize, request.size.x as usize];
    RenderResult {
        rgb: image.clone().slice(s![.., .., 0..3]),
        alpha: image.clone().slice(s![.., .., 3..4]).reshape([h, w]),
        depth_m: request
            .depth
            .then(|| image.clone().slice(s![.., .., 4..5]).reshape([h, w])),
        normals: request.normals.then(|| image.slice(s![.., .., 5..8])),
    }
}

pub(crate) fn empty(model: &TriangleModel, request: &RenderRequest) -> RenderResult {
    let d = model.positions.device().inner();
    let [h, w] = [request.size.y as usize, request.size.x as usize];
    let rgb = Tensor::<1>::from_floats(request.background.to_array(), &d)
        .reshape([1, 1, 3])
        .expand([h, w, 3]);
    RenderResult {
        rgb,
        alpha: Tensor::zeros([h, w], &d),
        depth_m: request.depth.then(|| Tensor::zeros([h, w], &d)),
        normals: request.normals.then(|| Tensor::zeros([h, w, 3], &d)),
    }
}

pub async fn render_triangles(
    model: &TriangleModel,
    request: &RenderRequest,
    cfg: &RenderConfig,
) -> Result<RenderResult, ReconError> {
    let Some(forward) = forward_gpu(model, request, cfg).await? else {
        return Ok(crate::backward::empty_with_gradients(
            model,
            request,
            empty(model, request),
        ));
    };
    let image = crate::backward::attach(model, forward);
    Ok(unpack(image, request))
}
