//! Host readback of GPU tensors.

use burn::tensor::Tensor;

/// Reads `t` back to the host as flat `f32`, row-major. Panics when the GPU read fails.
pub(crate) async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("GPU readback")
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("f32 readback")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[tokio::test]
    async fn read_f32_returns_the_tensor_values() {
        let device = Device::from(brush_cube::test_helpers::test_device().await);
        let t = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, -5.0, 6.5]], &device);
        assert_eq!(read_f32(t).await, vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.5]);
    }
}
