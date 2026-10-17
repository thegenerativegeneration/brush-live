use brush_recon::ReconError;
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RendererStats {
    pub visible_faces: u64,
    pub culled_faces: u64,
    pub intersections: u64,
    pub reserved_peak_bytes: u64,
}

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
            || self.faces.len() > i32::MAX as usize / 3
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
    pub(crate) faces: Tensor<2, Int>,
    pub(crate) vertex_offsets: Tensor<1, Int>,
    pub(crate) vertex_corners: Tensor<1, Int>,
    pub rgb_logits: Tensor<2>,
    pub weight_logits: Tensor<1>,
    vertex_count: usize,
    face_count: usize,
    pub(crate) buffer_peak: Arc<AtomicU64>,
    pub(crate) visible_faces: Arc<AtomicU64>,
    pub(crate) intersections: Arc<AtomicU64>,
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
        let mut adjacency = vec![vec![]; v];
        for (f, face) in seed.faces.iter().enumerate() {
            for (corner, &i) in face.iter().enumerate() {
                adjacency[i as usize].push((f * 3 + corner) as i32);
            }
        }
        let mut offsets = vec![0_i32];
        let mut corners = vec![];
        for list in adjacency {
            corners.extend(list);
            offsets.push(corners.len() as i32);
        }
        if corners.is_empty() {
            corners.push(0);
        }
        Ok(Self {
            positions: Tensor::from_data(TensorData::new(positions, [v, 3]), device),
            faces: Tensor::from_data(TensorData::new(faces, [f, 3]), device),
            vertex_offsets: Tensor::from_data(TensorData::new(offsets, [v + 1]), device),
            vertex_corners: Tensor::from_data(
                TensorData::new(corners.clone(), [corners.len()]),
                device,
            ),
            rgb_logits: Tensor::from_data(TensorData::new(colors, [v, 3]), device),
            weight_logits: Tensor::from_data(
                TensorData::new(vec![(0.1_f32 / 0.9).ln(); v], [v]),
                device,
            ),
            vertex_count: seed.positions.len(),
            face_count: seed.faces.len(),
            buffer_peak: Arc::new(AtomicU64::new(0)),
            visible_faces: Arc::new(AtomicU64::new(0)),
            intersections: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn num_vertices(&self) -> usize {
        self.vertex_count
    }
    /// Fixed connectivity; changing it requires rebuilding the seed/model and adjacency.
    pub fn faces(&self) -> Tensor<2, Int> {
        self.faces.clone()
    }
    pub fn num_faces(&self) -> usize {
        self.face_count
    }
    pub fn renderer_stats(&self) -> RendererStats {
        let visible_faces = self.visible_faces.load(Ordering::Relaxed);
        RendererStats {
            visible_faces,
            culled_faces: (self.face_count as u64).saturating_sub(visible_faces),
            intersections: self.intersections.load(Ordering::Relaxed),
            reserved_peak_bytes: self.buffer_peak.load(Ordering::Relaxed),
        }
    }
}
