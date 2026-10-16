//! Scalar tangent arithmetic for the tiny per-face projection VJP; no numerical differences.
use super::*;

#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct D {
    pub v: f32,
    pub d: f32,
}
#[cube]
impl D {
    pub fn constant(v: f32) -> D {
        D { v, d: 0.0f32 }
    }
    pub fn add(self, b: D) -> D {
        D {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
    pub fn sub(self, b: D) -> D {
        D {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
    pub fn mul(self, b: D) -> D {
        D {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
    pub fn scale(self, b: f32) -> D {
        D {
            v: self.v * b,
            d: self.d * b,
        }
    }
    pub fn div(self, b: D) -> D {
        D {
            v: self.v / b.v,
            d: (self.d * b.v - self.v * b.d) / (b.v * b.v),
        }
    }
    pub fn sqrt(self) -> D {
        let v = f32::sqrt(self.v);
        D {
            v,
            d: self.d / (2.0f32 * v),
        }
    }
    pub fn abs(self) -> D {
        let mut d = self.d;
        if self.v < 0.0f32 {
            d = -d;
        }
        D {
            v: f32::abs(self.v),
            d,
        }
    }
}

#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct V {
    pub x: D,
    pub y: D,
    pub z: D,
}
#[cube]
impl V {
    pub fn sub(self, b: V) -> V {
        V {
            x: self.x.sub(b.x),
            y: self.y.sub(b.y),
            z: self.z.sub(b.z),
        }
    }
    pub fn scale(self, b: f32) -> V {
        V {
            x: self.x.scale(b),
            y: self.y.scale(b),
            z: self.z.scale(b),
        }
    }
    pub fn length(self) -> D {
        self.x
            .mul(self.x)
            .add(self.y.mul(self.y))
            .add(self.z.mul(self.z))
            .sqrt()
    }
    pub fn divide(self, b: D) -> V {
        V {
            x: self.x.div(b),
            y: self.y.div(b),
            z: self.z.div(b),
        }
    }
}
#[cube]
pub fn cross_d(a: V, b: V) -> V {
    V {
        x: a.y.mul(b.z).sub(a.z.mul(b.y)),
        y: a.z.mul(b.x).sub(a.x.mul(b.z)),
        z: a.x.mul(b.y).sub(a.y.mul(b.x)),
    }
}
#[cube]
pub fn load_position(t: &Tensor<f32>, index: usize, local: u32, target: u32) -> V {
    V {
        x: D {
            v: t[index * 3],
            d: select(target == local * 3u32, 1.0f32, 0.0f32),
        },
        y: D {
            v: t[index * 3 + 1],
            d: select(target == local * 3u32 + 1u32, 1.0f32, 0.0f32),
        },
        z: D {
            v: t[index * 3 + 2],
            d: select(target == local * 3u32 + 2u32, 1.0f32, 0.0f32),
        },
    }
}
#[cube]
pub fn view_d(a: V, p: &Tensor<f32>) -> V {
    V {
        x: a.x
            .scale(p[0])
            .add(a.y.scale(p[1]))
            .add(a.z.scale(p[2]))
            .add(D::constant(p[3])),
        y: a.x
            .scale(p[4])
            .add(a.y.scale(p[5]))
            .add(a.z.scale(p[6]))
            .add(D::constant(p[7])),
        z: a.x
            .scale(p[8])
            .add(a.y.scale(p[9]))
            .add(a.z.scale(p[10]))
            .add(D::constant(p[11])),
    }
}
#[cube]
pub fn screen_d(v: V, p: &Tensor<f32>) -> V {
    V {
        x: v.x.div(v.z).scale(p[12]).add(D::constant(p[14])),
        y: v.y.div(v.z).scale(p[13]).add(D::constant(p[15])),
        z: D::constant(1.0f32).div(v.z),
    }
}
#[cube]
pub fn edge_d(a: V, b: V, c: V) -> D {
    b.x.sub(a.x)
        .mul(c.y.sub(a.y))
        .sub(b.y.sub(a.y).mul(c.x.sub(a.x)))
}
#[cube]
pub fn length_d(a: V, b: V) -> D {
    let x = a.x.sub(b.x);
    let y = a.y.sub(b.y);
    x.mul(x).add(y.mul(y)).sqrt()
}
