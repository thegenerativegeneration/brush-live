//! Offline reconstruction contracts. No live session, trainer or representation ownership.
pub mod backend;
pub mod capture;
pub mod error;
pub mod frame;

pub use backend::*;
pub use error::ReconError;
pub use frame::{MetricDepth, ReconFrame};
