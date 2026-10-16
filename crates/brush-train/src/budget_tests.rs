use super::*;
use crate::evict::pick_evictions_within;

fn group(max_rows: u32, full: bool, evictable: u32) -> GroupBudget {
    GroupBudget { max_rows, full, evictable }
}

/// Rows 0..3 in group 0, 3..5 in group 1, row 5 in none.
fn budget() -> RowBudget {
    RowBudget {
        group: vec![0, 0, 0, 1, 1, NO_GROUP],
        groups: vec![group(2, true, 1), group(u32::MAX, false, 0)],
        limit_eviction: true,
    }
}

#[test]
fn rows_taken_leave_and_appended_rows_have_no_group() {
    let mut b = budget();
    b.remove(&[false, true, false, true, false, false]);
    assert_eq!(b.group, vec![0, 0, 1, NO_GROUP]);
    b.pad(2);
    assert_eq!(b.group, vec![0, 0, 1, NO_GROUP, NO_GROUP, NO_GROUP]);
}

#[test]
fn rows_of_a_full_group_do_not_grow() {
    assert_eq!(budget().no_growth(), Some(vec![true, true, true, false, false, false]));
    let mut b = budget();
    b.groups[0].full = false;
    assert_eq!(b.no_growth(), None, "no full group, no mask");
}

#[test]
fn the_quota_is_per_group_and_only_when_eviction_is_limited() {
    assert_eq!(budget().quota(), Some(vec![1, 0]));
    assert_eq!(RowBudget { limit_eviction: false, ..budget() }.quota(), None);
}

/// A group over its cap gives its lowest-scored live rows, unscored last and ties by row; dead rows do not count.
#[test]
fn a_group_over_its_cap_gives_its_least_important_rows() {
    let b = RowBudget {
        group: vec![0, 0, 0, 0, 0, 1, 1, NO_GROUP],
        groups: vec![group(2, true, 0), group(1, true, 0)],
        limit_eviction: false,
    };
    let scores = [5.0, f32::INFINITY, 1.0, 1.0, 9.0, 3.0, 2.0, 0.0];
    // Group 0: five rows for two; rows 2 and 3 tie at 1.0, then row 0, then 4; the unscored row 1 last.
    assert_eq!(b.over_cap(&scores, &[false; 8]), vec![0, 2, 3, 6]);
    // Row 2 already dead: four live rows for two.
    let dead = [false, false, true, false, false, false, false, false];
    assert_eq!(b.over_cap(&scores, &dead), vec![0, 3, 6]);
    assert_eq!(b.count_by_group(&[0, 3, 6, 7]), vec![2, 1]);
    let no_cap = RowBudget { groups: vec![group(u32::MAX, false, 0); 2], ..b };
    assert!(no_cap.over_cap(&scores, &[false; 8]).is_empty());
}

/// A limited pick takes rows in score order only while their group may give one; free rows pass any limit and do
/// not use up their group's quota; rows of no group never go.
#[test]
fn a_limited_pick_keeps_each_groups_quota() {
    let means = [0.0f32, 0.0, 0.0].repeat(6);
    let scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let group = [0, 0, 0, 1, 1, NO_GROUP];
    let limit = GroupLimit { group: &group, quota: vec![1, 5], free: vec![false; 6] };
    assert_eq!(pick_evictions_within(&scores, &means, 6, 1.0, Some(limit)), vec![0, 3, 4]);
    let limit = GroupLimit { group: &group, quota: vec![1, 0], free: vec![false, true, false, false, false, true] };
    assert_eq!(pick_evictions_within(&scores, &means, 6, 1.0, Some(limit)), vec![0, 1, 5]);
    assert_eq!(pick_evictions_within(&scores, &means, 2, 1.0, None), vec![0, 1]);
}
