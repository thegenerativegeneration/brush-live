use super::*;

/// Ruling 42.3: the interval counts start-to-start, and a budget-sized round stretches it.
#[test]
fn cadence_counts_start_to_start_and_budget_stretches_long_rounds() {
    let mut c = Cadence::new(0.25, 2.0);
    assert!(c.due(0.0));
    // Started at 10 s and took 0.4 s: the next start is due at 12 s, not 12.4 s.
    c.record(10.0, 0.4);
    assert!(!c.due(11.99));
    assert!(c.due(12.0));

    let mut c = Cadence::new(0.25, 2.0);
    // 1 s at 25 %: one start every 4 s.
    c.record(10.0, 1.0);
    assert!(!c.due(13.99));
    assert!(c.due(14.0));
    assert_eq!(c.interval(), 4.0);
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
fn uncapped_throttle_never_waits() {
    let mut t = IterThrottle::new(0.0);
    for i in 0..5 {
        assert_eq!(t.wait_s(f64::from(i) * 1e-3), None);
        t.record_step(f64::from(i) * 1e-3);
    }
}

#[test]
fn throttle_spaces_steps_by_the_cap_without_banking_time() {
    let mut t = IterThrottle::new(2.0);
    assert_eq!(t.wait_s(0.0), None);
    t.record_step(0.0);
    assert_eq!(t.wait_s(0.25), Some(0.25));
    // After a 10 s round the next step runs at once, but no burst follows.
    assert_eq!(t.wait_s(10.0), None);
    t.record_step(10.0);
    assert_eq!(t.wait_s(10.0), Some(0.5));
}
