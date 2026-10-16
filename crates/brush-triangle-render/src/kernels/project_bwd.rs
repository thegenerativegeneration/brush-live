use super::dual::*;
use super::*;

#[cube(launch)]
pub fn project_bwd(
    positions: &Tensor<f32>,
    indices: &Tensor<u32>,
    weights: &Tensor<f32>,
    projected: &Tensor<f32>,
    params: &Tensor<f32>,
    g: &Tensor<f32>,
    corners: &mut Tensor<f32>,
) {
    let face = ABSOLUTE_POS;
    if face >= indices.shape(0) {
        terminate!();
    }
    for j in 0..21 {
        corners[face * 21 + j] = 0.0f32;
    }
    let b = face * 32;
    if projected[b + 25] <= 0.0f32 {
        terminate!();
    }
    let i0 = indices[face * 3] as usize;
    let i1 = indices[face * 3 + 1] as usize;
    let i2 = indices[face * 3 + 2] as usize;
    for target in 0u32..9u32 {
        let a = load_position(positions, i0, 0u32, target);
        let c1 = load_position(positions, i1, 1u32, target);
        let c2 = load_position(positions, i2, 2u32, target);
        let sa = screen_d(view_d(a, params), params);
        let sb = screen_d(view_d(c1, params), params);
        let sc = screen_d(view_d(c2, params), params);
        let area = edge_d(sa, sb, sc);
        let l0 = length_d(sa, sb);
        let l1 = length_d(sb, sc);
        let l2 = length_d(sc, sa);
        let radius = area.abs().div(l0.add(l1).add(l2));
        let cross = cross_d(c1.sub(a), c2.sub(a));
        let mut n = cross.divide(cross.length());
        // Preserve the exact discrete normal-orientation branch chosen by forward.
        if n.x.v * projected[b + 18] + n.y.v * projected[b + 19] + n.z.v * projected[b + 20]
            < 0.0f32
        {
            n = n.scale(-1.0f32);
        }
        let grad = sa.x.d * g[b]
            + sa.y.d * g[b + 1]
            + sa.z.d * g[b + 2]
            + sb.x.d * g[b + 3]
            + sb.y.d * g[b + 4]
            + sb.z.d * g[b + 5]
            + sc.x.d * g[b + 6]
            + sc.y.d * g[b + 7]
            + sc.z.d * g[b + 8]
            + n.x.d * g[b + 18]
            + n.y.d * g[b + 19]
            + n.z.d * g[b + 20]
            + l0.d * g[b + 21]
            + l1.d * g[b + 22]
            + l2.d * g[b + 23]
            + area.d * g[b + 24]
            + radius.d * g[b + 25];
        let vertex = target / 3u32;
        let axis = target % 3u32;
        corners[face * 21 + vertex as usize * 7 + axis as usize] = grad;
    }
    for local in 0..3 {
        for ch in 0..3 {
            let color = projected[b + 9 + local * 3 + ch];
            corners[face * 21 + local * 7 + 3 + ch] =
                g[b + 9 + local * 3 + ch] * color * (1.0f32 - color);
        }
    }
    let w0 = weights[i0];
    let w1 = weights[i1];
    let w2 = weights[i2];
    let mut winner = 0usize;
    let mut wi = i0;
    let mut minimum = w0;
    if w1 < minimum || (w1 == minimum && i1 < wi) {
        winner = 1;
        wi = i1;
        minimum = w1;
    }
    if w2 < minimum || (w2 == minimum && i2 < wi) {
        winner = 2;
        minimum = w2;
    }
    let activated = 1.0f32 / (1.0f32 + f32::exp(-minimum));
    corners[face * 21 + winner * 7 + 6] =
        g[b + 26] * (1.0f32 - params[19]) * activated * (1.0f32 - activated);
}
