use super::{Lockstep, Pace, Paced, step_target};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

const STALL: Duration = Duration::from_secs(5);

#[test]
fn step_targets_are_k_times_the_pace_rounded() {
    assert_eq!(step_target(0, 4.0), 0);
    assert_eq!(step_target(3, 4.0), 12);
    assert_eq!(step_target(10, 4.3), 43);
    assert_eq!(step_target(2, 4.3), 9, "8.6 rounds up");
}

#[tokio::test]
async fn a_keyframe_waits_until_the_server_trained_its_steps() {
    let (tx, rx) = watch::channel(7u64);
    let mut lock = Lockstep::new(rx, 4.0, STALL);
    assert_eq!(
        lock.before_send(0).await.unwrap(),
        7,
        "keyframe 0 needs no steps"
    );
    let early = tokio::time::timeout(Duration::from_millis(50), lock.before_send(2)).await;
    assert!(early.is_err(), "keyframe 2 went out at 7 of 8 steps");
    tx.send(8).unwrap();
    assert_eq!(lock.before_send(2).await.unwrap(), 8);
    assert_eq!(lock.sent_at(), &[7, 8]);
}

#[tokio::test]
async fn a_server_that_stops_training_ends_the_wait_with_an_error() {
    let (_tx, rx) = watch::channel(3u64);
    let mut lock = Lockstep::new(rx, 4.0, Duration::from_millis(50));
    let err = lock.before_send(1).await.unwrap_err().to_string();
    assert!(err.contains("no training step"), "{err}");
}

#[tokio::test]
async fn steps_that_keep_coming_never_count_as_a_stall() {
    let (tx, rx) = watch::channel(0u64);
    let mut lock = Lockstep::new(rx, 30.0, Duration::from_millis(40));
    let trainer = tokio::spawn(async move {
        for s in 1..=30 {
            tokio::time::sleep(Duration::from_millis(5)).await;
            tx.send(s).unwrap();
        }
        tx
    });
    assert_eq!(
        lock.before_send(1).await.unwrap(),
        30,
        "150 ms of steps, 5 ms apart"
    );
    drop(trainer.await.unwrap());
}

/// A fake server that trains `per_keyframe` steps, one at a time, after each keyframe it receives.
fn fake_server(
    per_keyframe: u64,
) -> (
    watch::Receiver<u64>,
    mpsc::UnboundedSender<u64>,
    Arc<Mutex<Vec<(u64, u64)>>>,
) {
    let (steps_tx, steps_rx) = watch::channel(0u64);
    let (kf_tx, mut kf_rx) = mpsc::unbounded_channel::<u64>();
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = received.clone();
    tokio::spawn(async move {
        while let Some(id) = kf_rx.recv().await {
            log.lock().unwrap().push((id, *steps_tx.borrow()));
            for _ in 0..per_keyframe {
                tokio::time::sleep(Duration::from_millis(1)).await;
                steps_tx.send_modify(|s| *s += 1);
            }
        }
    });
    (steps_rx, kf_tx, received)
}

#[tokio::test]
async fn lockstep_keeps_the_send_order_and_sends_keyframe_k_at_k_times_the_pace() {
    let (steps, server, received) = fake_server(4);
    let (items_tx, items) = mpsc::channel(4);
    tokio::spawn(async move {
        for id in [0u64, 1, 2, 3, 9, 7] {
            items_tx.send(id).await.unwrap();
        }
    });
    let mut paced = Paced::new(items, Pace::Lockstep(Lockstep::new(steps, 4.0, STALL)));
    while let Some(id) = paced.next().await {
        server.send(id.unwrap()).unwrap();
    }
    assert_eq!(paced.sent(), 6);
    tokio::time::sleep(Duration::from_millis(30)).await;
    let got = received.lock().unwrap().clone();
    assert_eq!(got, vec![(0, 0), (1, 4), (2, 8), (3, 12), (9, 16), (7, 20)]);
    let Pace::Lockstep(lock) = paced.into_pace() else {
        unreachable!()
    };
    let s = lock.summary();
    assert_eq!((s.keyframes, s.min, s.max, s.late), (6, 4, 4, 0));
    assert!((s.mean - 4.0).abs() < 1e-9);
}

#[tokio::test]
async fn wall_pacing_sends_on_the_schedule_in_order() {
    let (items_tx, items) = mpsc::channel(4);
    tokio::spawn(async move {
        for id in [5u64, 6, 7] {
            items_tx.send(id).await.unwrap();
        }
    });
    let started = tokio::time::Instant::now();
    let mut paced = Paced::new(
        items,
        Pace::Wall {
            started,
            rate: 50.0,
        },
    );
    let mut got = Vec::new();
    while let Some(id) = paced.next().await {
        got.push((id.unwrap(), started.elapsed()));
    }
    assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), [5, 6, 7]);
    assert!(
        got[2].1 >= Duration::from_millis(40),
        "third keyframe at 2 / 50 Hz, got {:?}",
        got[2].1
    );
}

#[test]
fn the_summary_counts_steps_between_sends_and_late_keyframes() {
    let (_tx, rx) = watch::channel(0u64);
    let lock = Lockstep::with_sent_at(rx, 4.0, STALL, vec![0, 4, 9, 12, 21]);
    let s = lock.summary();
    assert_eq!((s.keyframes, s.min, s.max), (5, 3, 9));
    assert!((s.mean - 21.0 / 4.0).abs() < 1e-9);
    assert_eq!(
        s.late, 1,
        "keyframe 4 went out 5 steps past its target of 16"
    );
}
