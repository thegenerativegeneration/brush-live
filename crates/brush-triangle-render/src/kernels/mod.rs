use brush_cube::{Vec3A, is_finite_f32};
use burn::cubecl;
use burn::cubecl::{cube, prelude::*};

pub mod intersections;
pub mod project;
pub mod ranges;
pub mod rasterize;

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct Face {
    pub a: Vec3A,
    pub b: Vec3A,
    pub c: Vec3A,
    pub ca: Vec3A,
    pub cb: Vec3A,
    pub cc: Vec3A,
    pub normal: Vec3A,
    pub l0: f32,
    pub l1: f32,
    pub l2: f32,
    pub area: f32,
    pub radius: f32,
    pub weight: f32,
}

#[cube]
pub fn load3(t: &Tensor<f32>, i: usize) -> Vec3A {
    Vec3A::new(t[i], t[i + 1], t[i + 2])
}
#[cube]
pub fn store3(t: &mut Tensor<f32>, i: usize, v: Vec3A) {
    t[i] = v.x();
    t[i + 1] = v.y();
    t[i + 2] = v.z();
}
#[cube]
pub fn cross(a: Vec3A, b: Vec3A) -> Vec3A {
    Vec3A::new(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )
}
#[cube]
pub fn view(v: Vec3A, p: &Tensor<f32>) -> Vec3A {
    Vec3A::new(
        p[0] * v.x() + p[1] * v.y() + p[2] * v.z() + p[3],
        p[4] * v.x() + p[5] * v.y() + p[6] * v.z() + p[7],
        p[8] * v.x() + p[9] * v.y() + p[10] * v.z() + p[11],
    )
}
#[cube]
pub fn screen(v: Vec3A, p: &Tensor<f32>) -> Vec3A {
    Vec3A::new(
        v.x() / v.z() * p[12] + p[14],
        v.y() / v.z() * p[13] + p[15],
        1.0f32 / v.z(),
    )
}
#[cube]
pub fn activate(v: Vec3A) -> Vec3A {
    Vec3A::new(
        1.0f32 / (1.0f32 + f32::exp(-v.x())),
        1.0f32 / (1.0f32 + f32::exp(-v.y())),
        1.0f32 / (1.0f32 + f32::exp(-v.z())),
    )
}
#[cube]
pub fn edge(a: Vec3A, b: Vec3A, x: f32, y: f32) -> f32 {
    (b.x() - a.x()) * (y - a.y()) - (b.y() - a.y()) * (x - a.x())
}
#[cube]
pub fn length2(a: Vec3A, b: Vec3A) -> f32 {
    f32::sqrt((a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()))
}
#[cube]
pub fn load_face(t: &Tensor<f32>, id: usize) -> Face {
    let b = id * 32;
    Face {
        a: load3(t, b),
        b: load3(t, b + 3),
        c: load3(t, b + 6),
        ca: load3(t, b + 9),
        cb: load3(t, b + 12),
        cc: load3(t, b + 15),
        normal: load3(t, b + 18),
        l0: t[b + 21],
        l1: t[b + 22],
        l2: t[b + 23],
        area: t[b + 24],
        radius: t[b + 25],
        weight: t[b + 26],
    }
}

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct Contribution {
    pub alpha: f32,
    pub rgb: Vec3A,
    pub depth: f32,
}

#[cube]
pub fn sample(face: Face, x: f32, y: f32, sigma: f32) -> Contribution {
    let e0 = edge(face.a, face.b, x, y);
    let e1 = edge(face.b, face.c, x, y);
    let e2 = edge(face.c, face.a, x, y);
    let mut sign = 1.0f32;
    if face.area < 0.0f32 {
        sign = -1.0f32;
    }
    let distance = min(
        min(e0 * sign / face.l0, e1 * sign / face.l1),
        e2 * sign / face.l2,
    );
    let mut out = Contribution {
        alpha: 0.0f32,
        rgb: Vec3A::new(0.0f32, 0.0f32, 0.0f32),
        depth: 0.0f32,
    };
    if distance > 0.0f32 {
        let q0 = e1 / face.area * face.a.z();
        let q1 = e2 / face.area * face.b.z();
        let q2 = e0 / face.area * face.c.z();
        let inv = 1.0f32 / (q0 + q1 + q2);
        out.alpha = min(
            0.99f32,
            face.weight * f32::exp(sigma * f32::ln(distance / face.radius)),
        );
        out.rgb = face
            .ca
            .scale(q0)
            .add(face.cb.scale(q1))
            .add(face.cc.scale(q2))
            .scale(inv);
        out.depth = inv;
    }
    out
}
