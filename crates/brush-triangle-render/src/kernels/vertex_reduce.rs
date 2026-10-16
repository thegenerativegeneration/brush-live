use super::*;

#[cube(launch)]
pub fn vertex_reduce(
    offsets: &Tensor<u32>,
    corners: &Tensor<u32>,
    face_grads: &Tensor<f32>,
    position_grads: &mut Tensor<f32>,
    color_grads: &mut Tensor<f32>,
    weight_grads: &mut Tensor<f32>,
) {
    let vertex = ABSOLUTE_POS;
    if vertex >= position_grads.shape(0) {
        terminate!();
    }
    let mut position = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    let mut color = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    let mut weight = 0.0f32;
    for i in offsets[vertex]..offsets[vertex + 1] {
        let corner = corners[i as usize] as usize;
        let base = (corner / 3) * 21 + (corner % 3) * 7;
        position = position.add(load3(face_grads, base));
        color = color.add(load3(face_grads, base + 3));
        weight += face_grads[base + 6];
    }
    store3(position_grads, vertex * 3, position);
    store3(color_grads, vertex * 3, color);
    weight_grads[vertex] = weight;
}
