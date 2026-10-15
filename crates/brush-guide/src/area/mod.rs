//! Area training: only the tiles around the newest keyframe train; tiles the capture has left are parked on disk
//! and restored when it comes back (spec `docs/superpowers/specs/2026-10-14-area-training-design.md`).

pub mod tiles;

pub use tiles::{AreaParams, AreaSummary, TileInfo, TileKey, TileRegistry, group_rows, rows_in};
