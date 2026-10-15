//! Area training: only the tiles around the newest keyframe train; tiles the capture has left are parked on disk
//! and restored when it comes back (spec `docs/superpowers/specs/2026-10-14-area-training-design.md`).

pub mod park_file;
pub mod pose_replay;
pub mod tiles;

pub use park_file::{
    ParkedTile, SlotPose, TILE_FILE_VERSION, decode_tile, encode_tile, read_tile, read_tile_count, remove_tile_files,
    tile_path, write_tile,
};
pub use pose_replay::{CameraPose, MIN_SLOT_SHARE, apply_moves, move_spread, slot_moves};
pub use tiles::{AreaParams, AreaSummary, TileInfo, TileKey, TileRegistry, group_rows, rows_in};
