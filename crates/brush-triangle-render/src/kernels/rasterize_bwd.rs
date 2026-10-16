//! One workgroup per face/tile intersection. Pixel VJPs reduce without float atomics.
use super::*;

#[cube]
fn add3(g: &mut Array<f32>, i: usize, v: Vec3A) {
    g[i] += v.x();
    g[i + 1] += v.y();
    g[i + 2] += v.z();
}
#[cube]
fn edge_grad(g: &mut Array<f32>, a: Vec3A, b: Vec3A, ai: usize, bi: usize, x: f32, y: f32, v: f32) {
    g[ai * 3] += (b.y() - y) * v;
    g[ai * 3 + 1] += (x - b.x()) * v;
    g[bi * 3] += (y - a.y()) * v;
    g[bi * 3 + 1] += (a.x() - x) * v;
}

#[cube(launch)]
pub fn rasterize_bwd(
    projected: &Tensor<f32>,
    faces: &Tensor<u32>,
    keys: &Tensor<u32>,
    ranges: &Tensor<u32>,
    params: &Tensor<f32>,
    output: &Tensor<f32>,
    upstream: &Tensor<f32>,
    isect_grads: &mut Tensor<f32>,
) {
    let intersection = CUBE_POS;
    if intersection >= faces.len() {
        terminate!();
    }
    let tile = keys[intersection];
    let w = params[16] as u32;
    let h = params[17] as u32;
    let tw = (w + 15u32) / 16u32;
    let x = (tile % tw) * 16u32 + UNIT_POS % 16u32;
    let y = (tile / tw) * 16u32 + UNIT_POS / 16u32;
    let mut g = Array::<f32>::new(32usize);
    for j in 0usize..32usize {
        g[j] = 0.0f32;
    }
    if x < w && y < h {
        let px = x as f32 + 0.5f32;
        let py = y as f32 + 0.5f32;
        let start = ranges[(tile * 2u32) as usize];
        let end = ranges[(tile * 2u32 + 1u32) as usize];
        let face = load_face(projected, faces[intersection] as usize);
        let e0 = edge(face.a, face.b, px, py);
        let e1 = edge(face.b, face.c, px, py);
        let e2 = edge(face.c, face.a, px, py);
        let mut sign = 1.0f32;
        if face.area < 0.0f32 {
            sign = -1.0f32;
        }
        let d0 = e0 * sign / face.l0;
        let d1 = e1 * sign / face.l1;
        let d2 = e2 * sign / face.l2;
        let distance = min(min(d0, d1), d2);
        if distance > 0.0f32 {
            let current = sample(face, px, py, params[18]);
            let mut prefix = 1.0f32;
            for i in start..intersection as u32 {
                prefix *= 1.0f32
                    - sample(
                        load_face(projected, faces[i as usize] as usize),
                        px,
                        py,
                        params[18],
                    )
                    .alpha;
            }
            let mut suffix_t = 1.0f32;
            let mut suffix_rgb = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
            let mut suffix_depth = 0.0f32;
            let mut suffix_normal = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
            for i in intersection as u32 + 1u32..end {
                let f = load_face(projected, faces[i as usize] as usize);
                let c = sample(f, px, py, params[18]);
                let blend = suffix_t * c.alpha;
                suffix_rgb = suffix_rgb.add(c.rgb.scale(blend));
                suffix_depth += c.depth * blend;
                suffix_normal = suffix_normal.add(f.normal.scale(blend));
                suffix_t *= 1.0f32 - c.alpha;
            }
            suffix_rgb =
                suffix_rgb.add(Vec3A::new(params[24], params[25], params[26]).scale(suffix_t));
            let pixel = ((y * w + x) * 8u32) as usize;
            let grgb = load3(upstream, pixel);
            let mut ga = upstream[pixel + 3];
            let mut gd = 0.0f32;
            let mut gn = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
            let final_alpha = output[pixel + 3];
            if final_alpha > 1e-6f32 {
                let inv = 1.0f32 / final_alpha;
                gd = upstream[pixel + 4] * inv;
                gn = load3(upstream, pixel + 5).scale(inv);
                ga -= gd * output[pixel + 4] + gn.dot(load3(output, pixel + 5));
            }
            let galpha = prefix
                * (grgb.dot(current.rgb.sub(suffix_rgb))
                    + gd * (current.depth - suffix_depth)
                    + gn.dot(face.normal.sub(suffix_normal))
                    + ga * suffix_t);
            let blend = prefix * current.alpha;
            let gc = grgb.scale(blend);
            let gdepth = gd * blend;
            let q0 = e1 / face.area * face.a.z();
            let q1 = e2 / face.area * face.b.z();
            let q2 = e0 / face.area * face.c.z();
            let inv = current.depth;
            let gq0 = gc.dot(face.ca.sub(current.rgb)) * inv - gdepth * inv * inv;
            let gq1 = gc.dot(face.cb.sub(current.rgb)) * inv - gdepth * inv * inv;
            let gq2 = gc.dot(face.cc.sub(current.rgb)) * inv - gdepth * inv * inv;
            add3(&mut g, 9, gc.scale(q0 * inv));
            add3(&mut g, 12, gc.scale(q1 * inv));
            add3(&mut g, 15, gc.scale(q2 * inv));
            add3(&mut g, 18, gn.scale(blend));
            g[2] += gq0 * e1 / face.area;
            g[5] += gq1 * e2 / face.area;
            g[8] += gq2 * e0 / face.area;
            g[24] -= (gq0 * q0 + gq1 * q1 + gq2 * q2) / face.area;
            let mut ge0 = gq2 * face.c.z() / face.area;
            let mut ge1 = gq0 * face.a.z() / face.area;
            let mut ge2 = gq1 * face.b.z() / face.area;
            let coverage = f32::exp(params[18] * f32::ln(distance / face.radius));
            if face.weight * coverage < 0.99f32 {
                let gdistance = galpha * face.weight * coverage * params[18] / distance;
                g[25] -= galpha * face.weight * coverage * params[18] / face.radius;
                g[26] += galpha * coverage;
                if d0 <= d1 && d0 <= d2 {
                    ge0 += gdistance * sign / face.l0;
                    g[21] -= gdistance * distance / face.l0;
                } else if d1 <= d2 {
                    ge1 += gdistance * sign / face.l1;
                    g[22] -= gdistance * distance / face.l1;
                } else {
                    ge2 += gdistance * sign / face.l2;
                    g[23] -= gdistance * distance / face.l2;
                }
            }
            edge_grad(&mut g, face.a, face.b, 0, 1, px, py, ge0);
            edge_grad(&mut g, face.b, face.c, 1, 2, px, py, ge1);
            edge_grad(&mut g, face.c, face.a, 2, 0, px, py, ge2);
        }
    }
    // Supports any hardware subgroup width >=8, not just Apple's 32 lanes.
    let mut partials = Shared::<[f32]>::new_slice(32usize * 32usize);
    let plane = UNIT_POS / PLANE_DIM;
    let planes = CUBE_DIM / PLANE_DIM;
    for component in 0usize..32usize {
        let sum = plane_sum(g[component]);
        if UNIT_POS_PLANE == 0u32 {
            partials[component * 32 + plane as usize] = sum;
        }
    }
    sync_cube();
    if UNIT_POS < 32u32 {
        let mut total = 0.0f32;
        for plane in 0u32..planes {
            total += partials[UNIT_POS as usize * 32 + plane as usize];
        }
        isect_grads[intersection * 32 + UNIT_POS as usize] = total;
    }
}

#[cube(launch)]
pub fn reduce_faces(
    projected: &Tensor<f32>,
    faces: &Tensor<u32>,
    ranges: &Tensor<u32>,
    params: &Tensor<f32>,
    isect_grads: &Tensor<f32>,
    face_grads: &mut Tensor<f32>,
) {
    let id = ABSOLUTE_POS;
    if id >= projected.shape(0) {
        terminate!();
    }
    let mut g = Array::<f32>::new(32usize);
    for i in 0usize..32usize {
        g[i] = 0.0f32;
    }
    let base = id * 32;
    let tw = (params[16] as u32 + 15u32) / 16u32;
    let x0 = projected[base + 28] as u32 / 16u32;
    let y0 = projected[base + 29] as u32 / 16u32;
    let x1 = (projected[base + 30] as u32 + 15u32) / 16u32;
    let y1 = (projected[base + 31] as u32 + 15u32) / 16u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let tile = y * tw + x;
            let start = ranges[(tile * 2u32) as usize];
            let end = ranges[(tile * 2u32 + 1u32) as usize];
            for i in start..end {
                if faces[i as usize] == id as u32 {
                    for j in 0..32 {
                        g[j] += isect_grads[i as usize * 32 + j];
                    }
                }
            }
        }
    }
    for i in 0..32 {
        face_grads[base + i] = g[i];
    }
}
