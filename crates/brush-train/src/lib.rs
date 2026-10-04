#![recursion_limit = "256"]

pub mod config;
pub mod eval;
pub mod evict;
pub mod lod;
pub mod msg;
pub mod profile;
pub mod sh_background;
pub mod train;

mod adam_scaled;
mod multinomial;
mod quat_vec;
mod stats;

mod splat_init;

pub use splat_init::{RandomSplatsConfig, create_random_splats, to_init_splats};
