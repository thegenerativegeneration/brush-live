//! Independent native triangle experiment; no Gaussian parameter packing or research code reuse.
mod burn_glue;
pub mod forward;
mod kernels;
pub mod model;
pub mod reference;
pub use forward::{RenderConfig, render_triangles};
pub use model::{TriangleModel, TriangleSeed};
