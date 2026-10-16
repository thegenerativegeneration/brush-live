//! Resolve pending fusion inputs and bind finished custom buffers without host copies.
use brush_cube::{CubeTensor, MainBackendBase};
use burn::backend::{
    BackendTensor, DispatchTensorKind, TensorMetadata,
    ir::{BackendIr, InitOperationIr, OperationIr, OperationOutput},
};
use burn::tensor::{Int, Tensor};
use burn_fusion::{NoOp, stream::StreamId};

pub(crate) fn float<const D: usize>(tensor: Tensor<D>) -> CubeTensor {
    brush_render::burn_glue::resolve_to_cube_float(tensor.without_autodiff())
}

pub(crate) fn int<const D: usize>(tensor: Tensor<D, Int>) -> CubeTensor {
    let fusion = brush_render::burn_glue::unwrap_wgpu_int(tensor.without_autodiff());
    fusion
        .client
        .clone()
        .resolve_tensor_int::<MainBackendBase>(fusion)
}

pub(crate) fn wrap<const D: usize>(cube: CubeTensor, reference: &Tensor<2>) -> Tensor<D> {
    let mut dispatch = reference.clone().without_autodiff().into_dispatch();
    let fusion = brush_render::burn_glue::unwrap_wgpu_float(reference.clone().without_autodiff());
    let client = fusion.client.clone();
    let shape = cube.shape();
    let dtype = cube.dtype;
    let handle = MainBackendBase::float_tensor_handle(cube);
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));
    let bound = client
        .register(
            StreamId::current(),
            OperationIr::Init(desc),
            NoOp::<MainBackendBase>::new(),
        )
        .output();
    dispatch.kind = DispatchTensorKind::Cube(BackendTensor::Float(bound));
    Tensor::from_dispatch(dispatch)
}
