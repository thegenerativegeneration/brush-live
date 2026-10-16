use brush_recon::ReconError;
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriangleSeed {
    pub positions: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub colors: Vec<[f32; 3]>,
}

impl TriangleSeed {
    pub fn validate(&self) -> Result<(), ReconError> {
        if self.positions.len() != self.colors.len()
            || self.positions.len() > i32::MAX as usize
            || self.faces.len() > i32::MAX as usize
            || self.positions.iter().flatten().any(|v| !v.is_finite())
            || self
                .colors
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || self
                .faces
                .iter()
                .flatten()
                .any(|i| *i as usize >= self.positions.len())
        {
            return Err(ReconError::input(
                "invalid triangle seed shapes, coordinates, colors or indices",
            ));
        }
        Ok(())
    }
    pub fn fingerprint(&self) -> Result<String, ReconError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| ReconError::input(format!("seed serialization: {e}")))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Clone)]
pub struct TriangleModel {
    pub positions: Tensor<2>,
    pub faces: Tensor<2, Int>,
    pub rgb_logits: Tensor<2>,
    pub weight_logits: Tensor<1>,
    vertex_count: usize,
    face_count: usize,
}

impl TriangleModel {
    pub fn from_seed(seed: &TriangleSeed, device: &Device) -> Result<Self, ReconError> {
        seed.validate()?;
        // Empty logical scenes use sentinel storage; renderers must branch on counts,
        // not dispatch zero-sized kernels (unsupported by the pinned CubeCL stack).
        let v = seed.positions.len().max(1);
        let f = seed.faces.len().max(1);
        let positions = if seed.positions.is_empty() {
            vec![0.; 3]
        } else {
            seed.positions.iter().flatten().copied().collect()
        };
        let colors = if seed.colors.is_empty() {
            vec![0.; 3]
        } else {
            seed.colors
                .iter()
                .flatten()
                .map(|c| {
                    let c = c.clamp(1e-4, 1. - 1e-4);
                    (c / (1. - c)).ln()
                })
                .collect()
        };
        let faces = if seed.faces.is_empty() {
            vec![0_i32; 3]
        } else {
            seed.faces.iter().flatten().map(|i| *i as i32).collect()
        };
        Ok(Self {
            positions: Tensor::from_data(TensorData::new(positions, [v, 3]), device),
            faces: Tensor::from_data(TensorData::new(faces, [f, 3]), device),
            rgb_logits: Tensor::from_data(TensorData::new(colors, [v, 3]), device),
            weight_logits: Tensor::from_data(
                TensorData::new(vec![(0.1_f32 / 0.9).ln(); v], [v]),
                device,
            ),
            vertex_count: seed.positions.len(),
            face_count: seed.faces.len(),
        })
    }

    pub fn num_vertices(&self) -> usize {
        self.vertex_count
    }
    pub fn num_faces(&self) -> usize {
        self.face_count
    }
}
