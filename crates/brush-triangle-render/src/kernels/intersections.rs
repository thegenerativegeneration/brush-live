use super::*;

#[cube(launch)]
pub fn emit(
    projected: &Tensor<f32>,
    face_order: &Tensor<u32>,
    prefix: &Tensor<u32>,
    params: &Tensor<f32>,
    keys: &mut Tensor<u32>,
    values: &mut Tensor<u32>,
) {
    let slot = ABSOLUTE_POS;
    if slot >= face_order.len() {
        terminate!();
    }
    let id = face_order[slot] as usize;
    let mut offset = 0u32;
    if slot > 0 {
        offset = prefix[slot - 1];
    }
    if offset == prefix[slot] {
        terminate!();
    }
    let b = id * 32;
    let x0 = projected[b + 28] as u32 / 16u32;
    let y0 = projected[b + 29] as u32 / 16u32;
    let x1 = (projected[b + 30] as u32 + 15u32) / 16u32;
    let y1 = (projected[b + 31] as u32 + 15u32) / 16u32;
    let tile_w = (params[16] as u32 + 15u32) / 16u32;
    for y in y0..y1 {
        for x in x0..x1 {
            keys[offset as usize] = y * tile_w + x;
            values[offset as usize] = id as u32;
            offset += 1u32;
        }
    }
}
