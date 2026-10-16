//! Concrete Wgpu/Fusion backward registration, at the same dynamic Tensor boundary as Brush.
use crate::{TriangleModel, burn_glue, forward::ForwardBuffers, kernels};
use brush_cube::{CubeTensor, MainBackend, create_tensor};
use brush_recon::{RenderRequest, RenderResult};
use burn::backend::{
    Autodiff, BackendTensor, DispatchTensorKind, TensorMetadata,
    autodiff::{
        checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::FloatTensor,
};
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Tensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous};

type Ad = Autodiff<MainBackend>;

fn tracked<const D: usize>(tensor: Tensor<D>) -> FloatTensor<Ad> {
    match tensor.autodiff().into_dispatch().kind {
        DispatchTensorKind::Autodiff(inner) => match *inner {
            DispatchTensorKind::Cube(BackendTensor::Autodiff(t)) => t,
            _ => panic!("triangle backward requires the Cube autodiff backend"),
        },
        _ => panic!("triangle backward requires an autodiff tensor"),
    }
}

#[derive(Clone, Debug)]
struct State {
    buffers: ForwardBuffers,
    positions: CubeTensor,
    weights: CubeTensor,
    indices: CubeTensor,
    offsets: CubeTensor,
    corners: CubeTensor,
    reference: FloatTensor<MainBackend>,
}

#[derive(Debug)]
struct TriangleBackward;

impl Backward<MainBackend, 3> for TriangleBackward {
    type State = State;
    fn backward(self, ops: Ops<State, 3>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let state = ops.state;
        let upstream = grads.consume::<MainBackend>(&ops.node);
        let upstream = into_contiguous(
            upstream
                .client
                .clone()
                .resolve_tensor_float::<CubeBackend>(upstream),
        );
        let client = state.positions.client.clone();
        let device = state.positions.device.clone();
        let b = state.buffers;
        let f = state.indices.shape()[0];
        let v = state.positions.shape()[0];
        let n = b.sorted_faces.shape()[0];
        let isect_grads = create_tensor([n, 32], &device, DType::F32);
        let face_grads = create_tensor([f, 32], &device, DType::F32);
        let corner_grads = create_tensor([f, 21], &device, DType::F32);
        let position_grads = create_tensor([v, 3], &device, DType::F32);
        let color_grads = create_tensor([v, 3], &device, DType::F32);
        let weight_grads = create_tensor([v], &device, DType::F32);
        let dim = CubeDim::new_1d(256);
        kernels::rasterize_bwd::rasterize_bwd::launch(
            &client,
            calculate_cube_count_elemwise(&client, n * 256, dim),
            dim,
            b.projected.clone().into_tensor_arg(),
            b.sorted_faces.clone().into_tensor_arg(),
            b.tile_keys.into_tensor_arg(),
            b.ranges.clone().into_tensor_arg(),
            b.params.clone().into_tensor_arg(),
            b.output.into_tensor_arg(),
            upstream.into_tensor_arg(),
            isect_grads.clone().into_tensor_arg(),
        );
        kernels::rasterize_bwd::reduce_faces::launch(
            &client,
            calculate_cube_count_elemwise(&client, f, dim),
            dim,
            b.projected.clone().into_tensor_arg(),
            b.sorted_faces.into_tensor_arg(),
            b.ranges.into_tensor_arg(),
            b.params.clone().into_tensor_arg(),
            isect_grads.into_tensor_arg(),
            face_grads.clone().into_tensor_arg(),
        );
        kernels::project_bwd::project_bwd::launch(
            &client,
            calculate_cube_count_elemwise(&client, f, dim),
            dim,
            state.positions.into_tensor_arg(),
            state.indices.into_tensor_arg(),
            state.weights.into_tensor_arg(),
            b.projected.into_tensor_arg(),
            b.params.into_tensor_arg(),
            face_grads.into_tensor_arg(),
            corner_grads.clone().into_tensor_arg(),
        );
        kernels::vertex_reduce::vertex_reduce::launch(
            &client,
            calculate_cube_count_elemwise(&client, v, dim),
            dim,
            state.offsets.into_tensor_arg(),
            state.corners.into_tensor_arg(),
            corner_grads.into_tensor_arg(),
            position_grads.clone().into_tensor_arg(),
            color_grads.clone().into_tensor_arg(),
            weight_grads.clone().into_tensor_arg(),
        );
        let [p, c, w] = ops.parents;
        if let Some(node) = p {
            grads.register::<MainBackend>(
                node.id,
                burn_glue::bind(position_grads, state.reference.clone()),
            );
        }
        if let Some(node) = c {
            grads.register::<MainBackend>(
                node.id,
                burn_glue::bind(color_grads, state.reference.clone()),
            );
        }
        if let Some(node) = w {
            grads.register::<MainBackend>(node.id, burn_glue::bind(weight_grads, state.reference));
        }
    }
}

pub(crate) fn attach(model: &TriangleModel, buffers: ForwardBuffers) -> Tensor<3> {
    if !model.positions.device().is_autodiff()
        && !model.rgb_logits.device().is_autodiff()
        && !model.weight_logits.device().is_autodiff()
    {
        return burn_glue::wrap(buffers.output, &model.positions);
    }
    let positions = tracked(model.positions.clone());
    let colors = tracked(model.rgb_logits.clone());
    let weights = tracked(model.weight_logits.clone());
    let prep = TriangleBackward
        .prepare::<NoCheckpointing>([positions.node(), colors.node(), weights.node()])
        .compute_bound()
        .stateful();
    let reference = positions.primitive().clone();
    let output = burn_glue::bind(buffers.output.clone(), reference.clone());
    let ad = match prep {
        OpsKind::Tracked(prep) => prep.finish(
            State {
                buffers,
                positions: burn_glue::float(model.positions.clone()),
                weights: burn_glue::float(model.weight_logits.clone()),
                indices: burn_glue::int(model.faces.clone()),
                offsets: burn_glue::int(model.vertex_offsets.clone()),
                corners: burn_glue::int(model.vertex_corners.clone()),
                reference,
            },
            output,
        ),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    let mut dispatch = model.positions.clone().autodiff().into_dispatch();
    dispatch.kind = DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Cube(
        BackendTensor::Autodiff(ad),
    )));
    Tensor::from_dispatch(dispatch)
}

pub(crate) fn empty_with_gradients(
    model: &TriangleModel,
    _request: &RenderRequest,
    mut out: RenderResult,
) -> RenderResult {
    if model.positions.device().is_autodiff()
        || model.rgb_logits.device().is_autodiff()
        || model.weight_logits.device().is_autodiff()
    {
        let zero = (model.positions.clone().autodiff().sum()
            + model.rgb_logits.clone().autodiff().sum()
            + model.weight_logits.clone().autodiff().sum())
            * 0.;
        out.rgb = out.rgb.autodiff() + zero.clone().reshape([1, 1, 1]);
        out.alpha = out.alpha.autodiff() + zero.clone().reshape([1, 1]);
        out.depth_m = out
            .depth_m
            .map(|t| t.autodiff() + zero.clone().reshape([1, 1]));
        out.normals = out.normals.map(|t| t.autodiff() + zero.reshape([1, 1, 1]));
    }
    out
}
