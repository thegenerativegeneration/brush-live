//! Capture guidance on top of Brush: incremental training on posed keyframes
//! and per-voxel coverage / Fisher-uncertainty scores.

pub mod config;
pub mod keyframe;
pub mod live;
pub mod protocol;
pub mod schedule;
pub mod seed;
pub mod session;

pub mod scores;
