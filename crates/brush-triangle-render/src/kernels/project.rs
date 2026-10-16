use super::*;

#[cube(launch)]
pub fn project(
    positions: &Tensor<f32>,
    faces: &Tensor<u32>,
    colors: &Tensor<f32>,
    weights: &Tensor<f32>,
    params: &Tensor<f32>,
    projected: &mut Tensor<f32>,
    depths: &mut Tensor<f32>,
    ids: &mut Tensor<u32>,
    counts: &mut Tensor<u32>,
    status: &mut Tensor<Atomic<u32>>,
) {
    let id = ABSOLUTE_POS;
    if id >= faces.shape(0) {
        terminate!();
    }
    counts[id] = 0;
    depths[id] = f32::MAX;
    ids[id] = id as u32;
    let base = id * 32;
    for i in 0..32 {
        projected[base + i] = 0.0f32;
    }
    let i0 = faces[id * 3] as usize;
    let i1 = faces[id * 3 + 1] as usize;
    let i2 = faces[id * 3 + 2] as usize;
    if i0 >= positions.shape(0) || i1 >= positions.shape(0) || i2 >= positions.shape(0) {
        Atomic::store(&status[1], 1u32);
        terminate!();
    }
    if i0 == i1 || i1 == i2 || i2 == i0 {
        terminate!();
    }
    let a = load3(positions, i0 * 3);
    let b = load3(positions, i1 * 3);
    let c = load3(positions, i2 * 3);
    let ca = load3(colors, i0 * 3);
    let cb = load3(colors, i1 * 3);
    let cc = load3(colors, i2 * 3);
    let wa = weights[i0];
    let wb = weights[i1];
    let wc = weights[i2];
    if !a.is_finite()
        || !b.is_finite()
        || !c.is_finite()
        || !ca.is_finite()
        || !cb.is_finite()
        || !cc.is_finite()
        || !is_finite_f32(wa)
        || !is_finite_f32(wb)
        || !is_finite_f32(wc)
    {
        Atomic::store(&status[1], 1u32);
        terminate!();
    }
    let cross_n = cross(b.sub(a), c.sub(a));
    let cross_length = cross_n.length();
    if cross_length * 0.5f32 < 1e-8f32 || !is_finite_f32(cross_length) {
        terminate!();
    }
    let va = view(a, params);
    let vb = view(b, params);
    let vc = view(c, params);
    if !va.is_finite()
        || !vb.is_finite()
        || !vc.is_finite()
        || min(min(va.z(), vb.z()), vc.z()) <= params[20]
    {
        terminate!();
    }
    let sa = screen(va, params);
    let sb = screen(vb, params);
    let sc = screen(vc, params);
    if !sa.is_finite() || !sb.is_finite() || !sc.is_finite() {
        terminate!();
    }
    let area = edge(sa, sb, sc.x(), sc.y());
    if f32::abs(area) * 0.5f32 < 1e-6f32 {
        terminate!();
    }
    let l0 = length2(sa, sb);
    let l1 = length2(sb, sc);
    let l2 = length2(sc, sa);
    let radius = f32::abs(area) / (l0 + l1 + l2);
    if !is_finite_f32(radius) || radius <= 0.0f32 {
        terminate!();
    }
    let x0 = f32::clamp(
        f32::floor(min(min(sa.x(), sb.x()), sc.x())),
        0.0f32,
        params[16],
    ) as u32;
    let y0 = f32::clamp(
        f32::floor(min(min(sa.y(), sb.y()), sc.y())),
        0.0f32,
        params[17],
    ) as u32;
    let x1 = f32::clamp(
        f32::ceil(max(max(sa.x(), sb.x()), sc.x())),
        0.0f32,
        params[16],
    ) as u32;
    let y1 = f32::clamp(
        f32::ceil(max(max(sa.y(), sb.y()), sc.y())),
        0.0f32,
        params[17],
    ) as u32;
    if x0 >= x1 || y0 >= y1 {
        terminate!();
    }
    let mut normal = cross_n.scale(1.0f32 / cross_length);
    let centroid = a.add(b).add(c).scale(1.0f32 / 3.0f32);
    if normal.dot(Vec3A::new(params[21], params[22], params[23]).sub(centroid)) < 0.0f32 {
        normal = normal.scale(-1.0f32);
    }
    let weight = params[19] + (1.0f32 - params[19]) / (1.0f32 + f32::exp(-min(min(wa, wb), wc)));
    store3(projected, base, sa);
    store3(projected, base + 3, sb);
    store3(projected, base + 6, sc);
    store3(projected, base + 9, activate(ca));
    store3(projected, base + 12, activate(cb));
    store3(projected, base + 15, activate(cc));
    store3(projected, base + 18, normal);
    projected[base + 21] = l0;
    projected[base + 22] = l1;
    projected[base + 23] = l2;
    projected[base + 24] = area;
    projected[base + 25] = radius;
    projected[base + 26] = weight;
    projected[base + 27] = (va.z() + vb.z() + vc.z()) / 3.0f32;
    projected[base + 28] = x0 as f32;
    projected[base + 29] = y0 as f32;
    projected[base + 30] = x1 as f32;
    projected[base + 31] = y1 as f32;
    depths[id] = projected[base + 27];
    counts[id] = ((x1 + 15u32) / 16u32 - x0 / 16u32) * ((y1 + 15u32) / 16u32 - y0 / 16u32);
    Atomic::fetch_add(&status[0], 1u32);
}
