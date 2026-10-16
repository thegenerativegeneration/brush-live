//! Per-group splat budgets the caller sets for the next refine ([`crate::train::SplatTrainer::set_row_budget`]; area
//! training's groups are its tiles): rows of a group at its cap neither split, clone, stand in for pruned rows nor grow;
//! a group over its cap loses its least important rows down to the cap; and the budget eviction may be limited to so
//! many rows per group.

/// The group of a row that belongs to none (rows appended after the budget was set): no cap, no growth limit, and no
/// victim of a limited eviction.
pub const NO_GROUP: u32 = u32::MAX;

/// One group's budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupBudget {
    /// Rows the group keeps at most: the refine evicts its least important rows above it. `u32::MAX`: no cap.
    pub max_rows: u32,
    /// The group is at its cap: none of its rows splits, clones, stands in for a pruned row or grows.
    pub full: bool,
    /// Rows a limited eviction ([`RowBudget::limit_eviction`]) may take from the group at most.
    pub evictable: u32,
}

/// Groups of the current rows and their budgets, row-aligned with the splats: take and append keep it aligned
/// (appended rows get [`NO_GROUP`]), refine uses and drops it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowBudget {
    /// Per row, its index into `groups`, or [`NO_GROUP`].
    pub group: Vec<u32>,
    pub groups: Vec<GroupBudget>,
    /// The budget eviction takes rows only within each group's `evictable` (rows held but never trained excepted);
    /// false: anywhere, as without a budget.
    pub limit_eviction: bool,
}

impl RowBudget {
    fn of_row(&self, row: usize) -> Option<&GroupBudget> {
        self.groups.get(*self.group.get(row)? as usize)
    }

    /// Removes the rows flagged in `take` (one flag per row).
    pub(crate) fn remove(&mut self, take: &[bool]) {
        let mut flags = take.iter();
        self.group.retain(|_| !flags.next().copied().unwrap_or(false));
    }

    /// Appends `n` rows of no group.
    pub(crate) fn pad(&mut self, n: usize) {
        self.group.extend(std::iter::repeat_n(NO_GROUP, n));
    }

    /// Per row, whether its group is full; `None` when no row's group is.
    pub(crate) fn no_growth(&self) -> Option<Vec<bool>> {
        let flags: Vec<bool> = (0..self.group.len()).map(|i| self.of_row(i).is_some_and(|g| g.full)).collect();
        flags.iter().any(|&f| f).then_some(flags)
    }

    /// Per group, the rows a limited eviction may take; `None` when the eviction is not limited.
    pub(crate) fn quota(&self) -> Option<Vec<u32>> {
        self.limit_eviction.then(|| self.groups.iter().map(|g| g.evictable).collect())
    }

    /// The rows to evict so no group keeps more than its `max_rows` of the rows not flagged in `dead`: per group the
    /// lowest `scores` first, non-finite (unscored) scores last, ties by row.
    pub(crate) fn over_cap(&self, scores: &[f32], dead: &[bool]) -> Vec<usize> {
        let mut alive: Vec<Vec<usize>> = vec![Vec::new(); self.groups.len()];
        for (i, &g) in self.group.iter().enumerate() {
            if !dead.get(i).copied().unwrap_or(false)
                && let Some(rows) = alive.get_mut(g as usize)
            {
                rows.push(i);
            }
        }
        let key = |i: usize| {
            let s = scores.get(i).copied().unwrap_or(f32::INFINITY);
            if s.is_finite() { s } else { f32::INFINITY }
        };
        let mut picked = Vec::new();
        for (budget, mut rows) in self.groups.iter().zip(alive) {
            let over = rows.len().saturating_sub(budget.max_rows as usize);
            if over == 0 {
                continue;
            }
            rows.sort_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
            picked.extend_from_slice(&rows[..over]);
        }
        picked.sort_unstable();
        picked
    }

    /// How many of `rows` lie in each group.
    pub(crate) fn count_by_group(&self, rows: &[usize]) -> Vec<u32> {
        let mut counts = vec![0; self.groups.len()];
        for &i in rows {
            if let Some(c) = self.group.get(i).and_then(|&g| counts.get_mut(g as usize)) {
                *c += 1;
            }
        }
        counts
    }
}

/// A limit on which rows an eviction may take: at most `quota[g]` rows of group `g` (rows of [`NO_GROUP`] none),
/// except rows flagged in `free`, which no group limit binds.
pub(crate) struct GroupLimit<'a> {
    pub group: &'a [u32],
    pub quota: Vec<u32>,
    pub free: Vec<bool>,
}

impl GroupLimit<'_> {
    /// Whether row `i` may still go; counts it when it does.
    pub(crate) fn admits(&self, i: usize) -> bool {
        self.free.get(i).copied().unwrap_or(false)
            || self.group.get(i).and_then(|&g| self.quota.get(g as usize)).is_some_and(|&q| q > 0)
    }

    /// Counts row `i` against its group's quota.
    pub(crate) fn take(&mut self, i: usize) {
        if self.free.get(i).copied().unwrap_or(false) {
            return;
        }
        if let Some(q) = self.group.get(i).and_then(|&g| self.quota.get_mut(g as usize)) {
            *q = q.saturating_sub(1);
        }
    }
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
