use brush_triangle_train::schedule::*;
use std::time::Duration;

#[test]
fn schedules_are_linear_monotone_clamped_and_have_exact_endpoints() {
    let c = ScheduleConfig::default();
    for (p, s, f) in [
        (-1., 1., 0.),
        (0., 1., 0.),
        (0.5, 0.55, 0.495),
        (1., 0.1, 0.99),
        (2., 0.1, 0.99),
    ] {
        let a = schedule(p, &c).unwrap();
        assert!((a.sigma - s).abs() < 1e-6 && (a.weight_floor - f).abs() < 1e-6);
    }
    assert!(schedule(f32::NAN, &c).is_err());
    assert!(schedule(0., &ScheduleConfig { sigma_end: 0., ..c }).is_err());
}

#[test]
fn step_and_budget_contexts_use_explicit_start_of_step_progress() {
    assert_eq!(step_context(0, 200).unwrap().progress, 0.);
    assert_eq!(step_context(199, 200).unwrap().progress, 1.);
    assert!(step_context(0, 1).is_err());
    assert!(step_context(200, 200).is_err());
    assert_eq!(
        budget_context(5, Duration::from_secs(30), Duration::from_secs(60))
            .unwrap()
            .progress,
        0.5
    );
    assert_eq!(
        budget_context(9, Duration::from_secs(70), Duration::from_secs(60))
            .unwrap()
            .progress,
        1.
    );
    assert!(budget_context(0, Duration::ZERO, Duration::ZERO).is_err());
}
