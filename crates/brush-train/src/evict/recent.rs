use super::*;

/// Cells newly observed by recent keyframes (see the module docs).
#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
pub struct RecentCells {
    /// Cell → refine count when a keyframe last seeded it.
    stamps: hashbrown::HashMap<glam::IVec3, u32>,
    refines: u32,
    last_seeded: Option<u32>,
    /// Blocked split/growth demand inside recent cells since the last eviction.
    pub backlog: u32,
}

impl RecentCells {
    /// Records one keyframe's seed positions (flat xyz, before the budget
    /// drops any).
    pub fn note_keyframe(&mut self, seed_means: &[f32]) {
        let mut counts: hashbrown::HashMap<glam::IVec3, usize> = hashbrown::HashMap::new();
        for p in seed_means.as_chunks::<3>().0 {
            *counts.entry(cell_of(glam::Vec3::from_array(*p))).or_default() += 1;
        }
        for (cell, n) in counts {
            if n >= MIN_CELL_SEEDS {
                self.stamps.insert(cell, self.refines);
                self.last_seeded = Some(self.refines);
            }
        }
    }

    /// Advances to the next refine.
    pub fn tick(&mut self) {
        self.refines += 1;
    }

    /// Whether any cell was seeded within the last `window` refines.
    pub fn active(&self, window: u32) -> bool {
        self.last_seeded
            .is_some_and(|r| self.refines - r < window)
    }

    /// Whether `p` lies in a cell seeded within the last `window` refines.
    pub fn contains(&self, p: glam::Vec3, window: u32) -> bool {
        self.stamps
            .get(&cell_of(p))
            .is_some_and(|&r| self.refines - r < window)
    }

    /// Per splat (flat xyz `means`), whether it lies in a recent cell.
    pub fn mask(&self, means: &[f32], window: u32) -> Vec<bool> {
        means
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| self.contains(glam::Vec3::from_array(*p), window))
            .collect()
    }
}

/// Split and growth demand inside recent cells: the force-split candidates
/// plus `growth_select_fraction` of the high-gradient candidates, the same
/// ratio refine applies to the whole model.
pub fn recent_demand(
    in_recent: &[bool],
    oversized: &[bool],
    above_threshold: &[bool],
    growth_select_fraction: f32,
) -> u32 {
    let count = |mask: &[bool]| {
        in_recent
            .iter()
            .zip(mask)
            .filter(|&(&r, &m)| r && m)
            .count()
    };
    let grow = (count(above_threshold) as f32 * growth_select_fraction).round() as u32;
    count(oversized) as u32 + grow
}
