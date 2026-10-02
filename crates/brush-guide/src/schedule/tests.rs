use super::*;

#[test]
fn cadence_counts_start_to_start() {
    let mut c = Cadence::new(0.25, 2.0);
    assert!(c.due(0.0));
    // Started at 10 s and took 0.4 s: the next start is due at 12 s, not 12.4 s.
    c.record(10.0, 0.4);
    assert!(!c.due(11.99));
    assert!(c.due(12.0));
}

#[test]
fn a_slightly_late_start_keeps_the_cadence() {
    let mut c = Cadence::new(0.25, 2.0);
    c.record(10.0, 0.4);
    // Due at 12 s, noticed after a 0.3 s training step.
    c.record(12.3, 0.4);
    assert!(c.due(14.0), "anchored to the due time");
    // Far behind (a long stall): counts from the actual start.
    c.record(15.5, 0.4);
    assert!(!c.due(17.4));
    assert!(c.due(17.5));
}

#[test]
fn cadence_budget_stretches_long_rounds() {
    let mut c = Cadence::new(0.25, 2.0);
    // 1 s at 25 %: one start every 4 s.
    c.record(10.0, 1.0);
    assert!(!c.due(13.99));
    assert!(c.due(14.0));
    assert_eq!(c.interval(), 4.0);
}

fn scheduler() -> RoundScheduler {
    RoundScheduler::new(Cadence::new(0.25, 2.0), Cadence::new(0.1, 3.0), 30, 12)
}

const COST: FisherCost = FisherCost {
    per_view_s: 0.03,
    fixed_s: 0.2,
};

#[test]
fn first_fisher_pass_runs_before_the_first_voxel_round() {
    let mut s = scheduler();
    assert_eq!(s.next(0.0, None), Some(Round::Fisher(30)));
    s.fisher.record(0.0, 0.2);
    assert_eq!(s.next(0.2, Some(COST)), Some(Round::Voxel));
}

#[test]
fn budgets_are_separate() {
    let mut s = scheduler();
    s.fisher.record(0.0, 1.0); // 1 s at 10 %: next pass at 10 s
    s.voxel.record(1.0, 0.4); // next voxel round at 3 s
    // The slow Fisher pass does not stretch the voxel cadence.
    assert_eq!(s.next(2.9, Some(COST)), None);
    assert_eq!(s.next(3.0, Some(COST)), Some(Round::Voxel));
    s.voxel.record(3.0, 0.4);
    // Not yet due for Fisher, although there is room before 5 s.
    assert_eq!(s.next(3.5, Some(COST)), None);
    // Fisher is due at 10 s, and the voxel round (due 11.5 s) leaves room.
    s.voxel.record(9.5, 0.4);
    assert_eq!(s.next(9.6, Some(COST)), None, "fisher not due before 10 s");
    assert!(matches!(s.next(10.0, Some(COST)), Some(Round::Fisher(_))));
}

#[test]
fn fisher_waits_for_the_slot_after_a_voxel_round() {
    let mut s = scheduler();
    s.fisher.record(7.6, 0.2); // due at 10.6 s, starving only from 13.6 s
    s.voxel.record(10.0, 0.4); // next voxel round at 12 s; slot 1.6 s
    // At 10.6 s: 1.4 s − 0.1 margin − 0.2 fixed = 1.1 s → 36 views fit, capped at 30.
    assert_eq!(s.next(10.6, Some(COST)), Some(Round::Fisher(30)));
    // At 11.2 s only 16 would fit: wait for the slot after the next voxel round.
    assert_eq!(s.next(11.2, Some(COST)), None);
}

#[test]
fn a_slow_pass_takes_what_the_slot_holds() {
    let mut s = scheduler();
    s.fisher.record(7.6, 0.2);
    s.voxel.record(10.0, 0.4);
    let slow = FisherCost {
        per_view_s: 0.06,
        fixed_s: 0.2,
    };
    // Slot: 0.75 · 1.6 s = 1.2 s → 15 views; at 10.6 s, 18 fit.
    assert_eq!(s.next(10.6, Some(slow)), Some(Round::Fisher(18)));
    // At 11.0 s, 11: fewer than the slot's 15.
    assert_eq!(s.next(11.0, Some(slow)), None);
}

#[test]
fn starving_fisher_runs_with_the_minimum() {
    let mut s = scheduler();
    s.fisher.record(0.0, 0.2); // interval 3 s: due at 3 s, starving from 6 s
    let slow = FisherCost {
        per_view_s: 1.0,
        fixed_s: 0.0,
    };
    s.voxel.record(5.0, 0.4);
    assert_eq!(s.next(5.5, Some(slow)), None);
    s.voxel.record(6.0, 0.4);
    assert_eq!(s.next(6.5, Some(slow)), Some(Round::Fisher(12)));
}

#[test]
fn short_captures_score_every_view() {
    let v = ViewSample::new(30);
    assert_eq!(v.select(20, 3, 7), (0..20).collect::<Vec<_>>());
    assert_eq!(v.select(30, 3, 7), (0..30).collect::<Vec<_>>());
    for i in 0..30 {
        assert_eq!(v.weight(i, 30), 1.0);
    }
}

#[test]
fn long_captures_keep_newest_and_take_one_view_per_stratum() {
    let v = ViewSample::new(30);
    assert_eq!((v.recent, v.max_views), (10, 30));
    let picked = v.select(322, 5, 7);
    assert_eq!(picked.len(), 30);
    assert!(picked.windows(2).all(|w| w[0] < w[1]), "sorted, unique");
    assert!((312..322).all(|i| picked.contains(&i)), "newest 10 included");
    // 20 strata over the 312 older views: one pick in each.
    for (j, &i) in picked[..20].iter().enumerate() {
        assert!((j * 312 / 20..(j + 1) * 312 / 20).contains(&i), "{j}: {i}");
    }
    assert_eq!(picked, v.select(322, 5, 7), "deterministic");
    assert_ne!(picked, v.select(322, 6, 7), "rotates with the pass");
}

#[test]
fn rotation_visits_every_older_view_once_per_cycle() {
    let v = ViewSample::new(30);
    // 300 older views over 20 strata of exactly 15.
    let n = 310;
    for seed in [0, 1, 99] {
        let mut seen = vec![0u32; n];
        for pass in 100..115 {
            for i in v.select(n, pass, seed) {
                seen[i] += 1;
            }
        }
        assert!(seen[..300].iter().all(|&c| c == 1), "seed {seed}");
        assert!(seen[300..].iter().all(|&c| c == 15), "newest every pass");
    }
}

#[test]
fn weights_are_horvitz_thompson_per_view() {
    let v = ViewSample::new(30);
    let n = 322;
    // Exact per pass: recent weights sum to 10, older ones to 312.
    for pass in 0..40 {
        let picked = v.select(n, pass, 3);
        let w = |r: std::ops::Range<usize>| -> f32 {
            picked
                .iter()
                .filter(|i| r.contains(i))
                .map(|&i| v.weight(i, n))
                .sum()
        };
        assert_eq!(w(312..322), 10.0);
        assert!((w(0..312) - 312.0).abs() < 1e-3, "{}", w(0..312));
    }
    // Unbiased over the seeded offset: for any fixed pass, every view's
    // expected weight · picked is 1.
    let seeds = 6000;
    for pass in [0, 7] {
        let mut total = vec![0f64; n];
        for seed in 0..seeds {
            for i in v.select(n, pass, seed) {
                total[i] += f64::from(v.weight(i, n));
            }
        }
        for (i, t) in total.iter().enumerate() {
            let mean = t / f64::from(seeds as u32);
            assert!((mean - 1.0).abs() < 0.2, "pass {pass}, view {i}: {mean}");
        }
    }
}

#[test]
fn older_weight_is_the_length_of_the_stratum_holding_the_view() {
    for (n, max) in [(322, 30), (100, 30), (31, 30), (300, 12)] {
        let v = ViewSample::new(max);
        let (older, strata) = (n - v.recent, max - v.recent);
        for j in 0..strata {
            let range = j * older / strata..(j + 1) * older / strata;
            for i in range.clone() {
                assert_eq!(v.weight(i, n), range.len() as f32, "n {n}, view {i}");
            }
        }
        assert_eq!(v.weight(n - 1, n), 1.0);
    }
}

#[test]
fn trimmed_view_counts_round_to_a_tier() {
    assert_eq!(quantise_views(0), 0);
    assert_eq!(quantise_views(1), 3);
    assert_eq!(quantise_views(2), 3);
    assert_eq!(quantise_views(3), 3);
    assert_eq!(quantise_views(4), 3);
    assert_eq!(quantise_views(5), 6);
    assert_eq!(quantise_views(17), 18);
    assert_eq!(quantise_views(30), 30);
}

#[test]
fn a_trimmed_pass_keeps_stable_strata_despite_small_cost_jitter() {
    // Two cost estimates a view apart (the kind of jitter a round-to-round
    // cost estimate has) land on the same tier, so `ViewSample::new` sees
    // the same `recent`/stratum split both times.
    let mut s = scheduler();
    s.fisher.record(7.6, 0.2);
    s.voxel.record(10.0, 0.4);
    let a = FisherCost {
        per_view_s: 1.1 / 18.5,
        fixed_s: 0.2,
    };
    let b = FisherCost {
        per_view_s: 1.1 / 19.5,
        fixed_s: 0.2,
    };
    // Raw fits differ (18 vs 19) but both round to the 18-view tier.
    assert_eq!(s.next(10.6, Some(a)), Some(Round::Fisher(18)));
    assert_eq!(s.next(10.6, Some(b)), Some(Round::Fisher(18)));
}

#[test]
fn smaller_passes_keep_a_third_recent() {
    let v = ViewSample::new(12);
    assert_eq!(v.recent, 4);
    let picked = v.select(200, 0, 1);
    assert_eq!(picked.len(), 12);
    assert!((196..200).all(|i| picked.contains(&i)));
}
