//! Area training: only the tiles around the newest keyframe train; tiles the capture has left are parked on disk
//! and restored when it comes back (spec `docs/superpowers/specs/2026-10-14-area-training-design.md`).

pub mod active;
pub mod budget;
pub mod park_file;
pub mod ply_stream;
pub mod pose_replay;
pub mod tiles;
pub mod views;

pub use active::{
    ActiveRule, Candidate, Heights, SeedSort, candidate_boxes, keep_active_seeds, near_radius, near_tiles, rank_active,
    tile_heights,
    views_seeing, widen_all,
};
pub use budget::{FairShare, far_inactive_to_park, keep_under_tile_cap, row_budget, tile_counts, tile_splats};
pub use ply_stream::PlyStreamWriter;
pub use park_file::{
    ParkedTile, SlotPose, TILE_FILE_VERSION, decode_tile, encode_tile, read_tile, read_tile_count, remove_tile_files,
    tile_path, write_tile,
};
pub use pose_replay::{CameraPose, MIN_SLOT_SHARE, apply_moves, move_spread, slot_moves};
pub use tiles::{AreaParams, AreaSummary, TileInfo, TileKey, TileRegistry, group_rows, rows_in};
pub use views::{area_views, camera_sees_box, junction_views, membership_additions, projected_area};

use serde::Serialize;

/// What area training did since the model started (or since the last reset).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AreaCounters {
    /// Tiles parked.
    pub parks: u32,
    /// Tiles among `parks` parked because the rows held still in left tiles exceeded `area_held_fraction`.
    pub parks_for_cap: u32,
    /// Parks left out because they would have taken every resident splat.
    pub parks_skipped: u32,
    pub restores: u32,
    /// Restores put off because the splat budget had no room.
    pub deferrals: u32,
    /// Parked tiles dropped instead of restored.
    pub drops: u32,
    /// Milliseconds spent parking (row readback, file writes), in total.
    pub park_ms: f64,
    /// The longest single park (one refine's parked tiles together).
    pub max_park_ms: f64,
    /// Milliseconds spent restoring (file reads, room made by parking, row appends), in total.
    pub restore_ms: f64,
    /// The longest single restore event (one call's tiles together, room made by parking included).
    pub max_restore_ms: f64,
    /// Bytes of the tile files on disk.
    pub tile_bytes: u64,
    /// Changes of the active set (`area_freeze` inactive or inactive_at_cap).
    pub active_changes: u32,
    /// Changes among `active_changes` back to the set before the last one.
    pub active_reversals: u32,
    /// Tiles parked before a refine to make room for its eviction or for the held cap (`parks_for_cap` among them).
    pub parks_for_room: u32,
    /// Seeds dropped because their tile had no room under the per-tile cap.
    pub seeds_over_tile_cap: u32,
    /// Refines whose eviction widened beyond the active tiles' fair shares because those gave fewer rows than it wanted
    /// after the parks.
    pub eviction_fallbacks: u32,
}

/// Area training's state for the eval log: tile and splat counts, counters, and every tile's state.
#[derive(Clone, Debug, Serialize)]
pub struct AreaReport {
    #[serde(flatten)]
    pub summary: AreaSummary,
    #[serde(flatten)]
    pub counters: AreaCounters,
    /// `[x, z, state]` per tile ([`TileRegistry::tile_states`]).
    pub tiles: Vec<[i32; 3]>,
    /// A tile file could not be written, so nothing parks until the session resets.
    pub parking_off: bool,
    /// Rows the trainer holds still (of left tiles, or outside the active tiles), as of the last mask build.
    pub frozen_rows: u32,
    /// Splats of the resident tiles the walker has left, held or not, as of the last refine's readback.
    pub left_splats: u32,
    /// `[x, z]` of the active tiles with `area_freeze` inactive or inactive_at_cap, the tile under the newest camera
    /// first; empty otherwise.
    pub active_tiles: Vec<[i32; 2]>,
    /// Rows in the active tiles when the mask was last built (0 unless `area_freeze` is inactive or inactive_at_cap;
    /// every resident row while inactive_at_cap is below the cap).
    pub active_splats: u32,
    /// Held rows no training step ever updated, when the mask was last built (rows restored from tile files count as
    /// trained; 0 without eviction, whose state carries the flag).
    pub held_untrained_rows: u32,
    /// With `area_freeze` inactive_at_cap: the rows outside the active tiles are held (the resident rows reached
    /// `area_freeze_at` of `max_splats`).
    pub at_cap: bool,
    /// `[x, z, splats]` per tile, resident (as of the last readback) or parked ([`tile_splats`]).
    pub tile_splats: Vec<[i32; 3]>,
    /// `[x, z, rows]`: rows refines evicted per tile under a row budget since the start, tiles with evictions only.
    pub tile_evictions: Vec<[i32; 3]>,
    /// The per-tile cap in effect (`area_tile_max_fraction` of `max_splats`); 0: none. Held rows no step ever trained
    /// are eviction victims at the total cap with or without it.
    pub tile_cap: u32,
}
