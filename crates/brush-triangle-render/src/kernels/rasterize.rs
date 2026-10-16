use super::*;

#[cube(launch)]
pub fn rasterize(
    projected: &Tensor<f32>,
    faces: &Tensor<u32>,
    ranges: &Tensor<u32>,
    params: &Tensor<f32>,
    output: &mut Tensor<f32>,
) {
    let tile = CUBE_POS as u32;
    let local = UNIT_POS;
    let w = params[16] as u32;
    let h = params[17] as u32;
    let tw = (w + 15u32) / 16u32;
    let x = (tile % tw) * 16u32 + local % 16u32;
    let y = (tile / tw) * 16u32 + local / 16u32;
    if x >= w || y >= h {
        terminate!();
    }
    let mut transmission = 1.0f32;
    let mut rgb = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    let mut normal = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    let mut depth = 0.0f32;
    let start = ranges[(tile * 2u32) as usize];
    let end = ranges[(tile * 2u32 + 1u32) as usize];
    for i in start..end {
        let face = load_face(projected, faces[i as usize] as usize);
        let c = sample(face, x as f32 + 0.5f32, y as f32 + 0.5f32, params[18]);
        let blend = transmission * c.alpha;
        rgb = rgb.add(c.rgb.scale(blend));
        depth += c.depth * blend;
        normal = normal.add(face.normal.scale(blend));
        transmission *= 1.0f32 - c.alpha;
    }
    rgb = rgb.add(Vec3A::new(params[24], params[25], params[26]).scale(transmission));
    let alpha = 1.0f32 - transmission;
    if alpha > 1e-6f32 {
        depth /= alpha;
        normal = normal.scale(1.0f32 / alpha);
    } else {
        depth = 0.0f32;
        normal = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    }
    let b = ((y * w + x) * 8u32) as usize;
    store3(output, b, rgb);
    output[b + 3] = alpha;
    output[b + 4] = depth;
    store3(output, b + 5, normal);
}
